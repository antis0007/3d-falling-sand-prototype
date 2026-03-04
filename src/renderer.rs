//! Authoritative meshing pipeline for runtime rendering.
//!
//! - **Authoritative input:** meshing consumes a [`ChunkSnapshot`] built from
//!   [`ChunkStore`] via `build_chunk_snapshot`, so workers never read mutable
//!   world state directly.
//! - **Vertex coordinate space:** all meshing emits chunk-local positions
//!   (scaled by [`VOXEL_SIZE`]), independent of world placement.
//! - **Chunk transform ownership:** draw-time code applies world placement via a
//!   per-instance chunk origin buffer.
//! - **Floating-origin contract:** `origin_offset` is only for world-to-camera
//!   conversion and must never be coupled to chunk placement.

use crate::chunk_store::{ChunkBorderStrips, ChunkStore};
use crate::gpu_compute::{
    cpu_generate_material_field, run_chunk_job_on_worker, DrawIndirectArgs, GpuComputeRuntime,
    MeshPipelineBackend,
};
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

pub const VOXEL_SIZE: f32 = 0.5;
const MAX_PENDING_DIRTY_CHUNKS: usize = 16_384;
const CHUNK_SNAPSHOT_BUILD_BUDGET_MS: f32 = 1.5;
const BUSH_ID: MaterialId = 18;
const GRASS_ID: MaterialId = 19;

#[derive(Clone, Copy, Debug)]
pub enum UnknownNeighborOcclusionPolicy {
    Conservative,
    Aggressive,
}

impl Default for UnknownNeighborOcclusionPolicy {
    fn default() -> Self {
        Self::Conservative
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub(crate) pos: [f32; 3],
    pub(crate) color: [u8; 4],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    vp: [[f32; 4]; 4],
    origin_offset: [f32; 3],
    _pad: f32,
}

#[derive(Clone, Copy)]
struct BufferSuballocation {
    slab_index: usize,
    offset: u64,
    size: u64,
}

struct BufferSlab {
    buffer: wgpu::Buffer,
    capacity: u64,
    cursor: u64,
    free: Vec<BufferSuballocation>,
}

struct BufferArena {
    label: &'static str,
    usage: wgpu::BufferUsages,
    min_slab_size: u64,
    slabs: Vec<BufferSlab>,
}

impl BufferArena {
    fn new(label: &'static str, usage: wgpu::BufferUsages, min_slab_size: u64) -> Self {
        Self {
            label,
            usage,
            min_slab_size,
            slabs: Vec::new(),
        }
    }

    fn buffer(&self, slab_index: usize) -> &wgpu::Buffer {
        &self.slabs[slab_index].buffer
    }

    fn allocate(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        telemetry: &mut MeshAllocatorTelemetry,
    ) -> BufferSuballocation {
        for (slab_index, slab) in self.slabs.iter_mut().enumerate() {
            if let Some((free_idx, free_alloc)) = slab
                .free
                .iter()
                .copied()
                .enumerate()
                .find(|(_, alloc)| alloc.size >= size)
            {
                slab.free.swap_remove(free_idx);
                telemetry.bytes_reused += size as usize;
                if free_alloc.size > size {
                    slab.free.push(BufferSuballocation {
                        slab_index,
                        offset: free_alloc.offset + size,
                        size: free_alloc.size - size,
                    });
                }
                return BufferSuballocation {
                    slab_index,
                    offset: free_alloc.offset,
                    size,
                };
            }

            if slab.capacity - slab.cursor >= size {
                let offset = slab.cursor;
                slab.cursor += size;
                return BufferSuballocation {
                    slab_index,
                    offset,
                    size,
                };
            }
        }

        let last_capacity = self
            .slabs
            .last()
            .map(|slab| slab.capacity)
            .unwrap_or(self.min_slab_size);
        let new_capacity = self
            .min_slab_size
            .max(last_capacity.saturating_mul(2))
            .max(size);
        let slab_index = self.slabs.len();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: new_capacity,
            usage: self.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.slabs.push(BufferSlab {
            buffer,
            capacity: new_capacity,
            cursor: size,
            free: Vec::new(),
        });
        telemetry.bytes_allocated += new_capacity as usize;
        BufferSuballocation {
            slab_index,
            offset: 0,
            size,
        }
    }

    fn free(&mut self, allocation: BufferSuballocation) {
        if let Some(slab) = self.slabs.get_mut(allocation.slab_index) {
            slab.free.push(allocation);
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct MeshAllocatorTelemetry {
    bytes_allocated: usize,
    bytes_reused: usize,
    realloc_count: usize,
}

pub struct Camera {
    pub pos: Vec3,
    pub dir: Vec3,
    pub aspect: f32,
}

impl Camera {
    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_to_rh(self.pos, self.dir, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), self.aspect.max(0.1), 0.1, 1200.0);
        proj * view
    }

    fn view_proj_for_world_origin(&self, origin_voxel: VoxelCoord) -> Mat4 {
        let origin_world = voxel_to_world(origin_voxel);
        let world_camera = Camera {
            pos: self.pos + origin_world,
            dir: self.dir,
            aspect: self.aspect,
        };
        world_camera.view_proj()
    }
}

pub struct ChunkMesh {
    vertex_alloc: BufferSuballocation,
    chunk_origin_buf: wgpu::Buffer,
    index_alloc: BufferSuballocation,
    index_count: u32,
    debug_aabb_vb: wgpu::Buffer,
    debug_aabb_ib: wgpu::Buffer,
    debug_aabb_index_count: u32,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    chunk_origin_world: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkOriginInstance {
    chunk_origin_world: [f32; 3],
    _pad: f32,
}

impl ChunkOriginInstance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ChunkOriginInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

const DEBUG_VISIBLE_CHUNK_LOG_COUNT: usize = 8;
const DEBUG_RENDER_CHUNK_AABBS: bool = false;
const DEBUG_VALIDATE_CULL_SPACE: bool = false;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChunkLod {
    Near,
    Mid,
    Far,
    Ultra,
}

#[derive(Clone, Copy, Debug)]
pub struct LodRadii {
    pub near: i32,
    pub mid: i32,
    pub far: i32,
    pub ultra: i32,
    pub hysteresis: i32,
}

impl LodRadii {
    pub fn normalized(mut self) -> Self {
        self.near = self.near.max(0);
        self.mid = self.mid.max(self.near);
        self.far = self.far.max(self.mid);
        self.ultra = self.ultra.max(self.far);
        self.hysteresis = self.hysteresis.max(0);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LodMeshingBudgets {
    pub near: usize,
    pub mid: usize,
    pub far: usize,
    pub ultra: usize,
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,

    depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    store_meshes: HashMap<(ChunkCoord, ChunkLod), ChunkMesh>,
    vertex_arena: BufferArena,
    index_arena: BufferArena,

    dirty_queues: DirtyChunkQueues,
    dirty_near_starve_frames: u32,
    dirty_far_starve_frames: u32,
    dirty_fair_cursor: u8,
    mesh_versions: HashMap<(ChunkCoord, ChunkLod), u64>,
    meshed_versions: HashMap<ChunkCoord, u64>,
    lod_selection: HashMap<ChunkCoord, ChunkLod>,

    mesh_queue: BackgroundMeshQueue,
    completed_meshes: Vec<MeshResult>,
    origin_voxel: VoxelCoord,

    pub day: bool,
    pub mesh_backend: MeshPipelineBackend,
    settings: RendererSettings,
    allocator_telemetry: MeshAllocatorTelemetry,
}

#[derive(Clone, Copy, Debug)]
pub struct RendererSettings {
    pub frustum_culling: bool,
    pub greedy_meshing: bool,
    pub unknown_neighbor_policy: UnknownNeighborOcclusionPolicy,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            frustum_culling: true,
            greedy_meshing: true,
            unknown_neighbor_policy: UnknownNeighborOcclusionPolicy::default(),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct CullStats {
    pub drawn: usize,
    pub frustum_culled: usize,
    pub lod_filtered: usize,
    pub screen_culled: usize,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct MeshRebuildStats {
    pub max_ms: f32,
    pub total_ms: f32,
    pub mesh_count: usize,
    pub dirty_backlog: usize,
    pub meshing_queue_depth: usize,
    pub meshing_completed_depth: usize,
    pub upload_count: usize,
    pub upload_bytes: usize,
    pub upload_latency_ms: f32,
    pub stale_drop_count: usize,
    pub age_drop_count: usize,
    pub pressure_drop_count: usize,
    pub dirty_queue_drop_count: usize,
    pub dirty_urgent_depth: usize,
    pub dirty_near_depth: usize,
    pub dirty_normal_depth: usize,
    pub dirty_far_depth: usize,
    pub near_mesh_count: usize,
    pub mid_mesh_count: usize,
    pub far_mesh_count: usize,
    pub ultra_mesh_count: usize,
    pub gpu_mesh_jobs: usize,
    pub gpu_dispatch_ms: f32,
    pub gpu_readback_bytes: u64,
    pub allocator_bytes_allocated: usize,
    pub allocator_bytes_reused: usize,
    pub allocator_realloc_count: usize,
}

const COMPLETED_MESH_BACKLOG_THRESHOLD: usize = 256;
const DIRTY_NEAR_STARVE_LIMIT_FRAMES: u32 = 8;
const DIRTY_FAR_STARVE_LIMIT_FRAMES: u32 = 20;
const DIRTY_VISIBLE_URGENT_SCORE: f32 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DirtyTier {
    Urgent,
    Near,
    Normal,
    Far,
}

impl DirtyTier {
    fn priority_rank(self) -> u8 {
        match self {
            Self::Urgent => 4,
            Self::Near => 3,
            Self::Normal => 2,
            Self::Far => 1,
        }
    }
}

#[derive(Default)]
struct DirtyChunkQueues {
    urgent: VecDeque<ChunkCoord>,
    near: VecDeque<ChunkCoord>,
    normal: VecDeque<ChunkCoord>,
    far: VecDeque<ChunkCoord>,
    tiers: HashMap<ChunkCoord, DirtyTier>,
    urgent_count: usize,
    near_count: usize,
    normal_count: usize,
    far_count: usize,
}

impl DirtyChunkQueues {
    fn total_len(&self) -> usize {
        self.tiers.len()
    }

    fn tier_len(&self, tier: DirtyTier) -> usize {
        match tier {
            DirtyTier::Urgent => self.urgent_count,
            DirtyTier::Near => self.near_count,
            DirtyTier::Normal => self.normal_count,
            DirtyTier::Far => self.far_count,
        }
    }

    fn incr_tier(&mut self, tier: DirtyTier) {
        match tier {
            DirtyTier::Urgent => self.urgent_count += 1,
            DirtyTier::Near => self.near_count += 1,
            DirtyTier::Normal => self.normal_count += 1,
            DirtyTier::Far => self.far_count += 1,
        }
    }

    fn decr_tier(&mut self, tier: DirtyTier) {
        match tier {
            DirtyTier::Urgent => self.urgent_count = self.urgent_count.saturating_sub(1),
            DirtyTier::Near => self.near_count = self.near_count.saturating_sub(1),
            DirtyTier::Normal => self.normal_count = self.normal_count.saturating_sub(1),
            DirtyTier::Far => self.far_count = self.far_count.saturating_sub(1),
        }
    }

    fn queue_for_mut(&mut self, tier: DirtyTier) -> &mut VecDeque<ChunkCoord> {
        match tier {
            DirtyTier::Urgent => &mut self.urgent,
            DirtyTier::Near => &mut self.near,
            DirtyTier::Normal => &mut self.normal,
            DirtyTier::Far => &mut self.far,
        }
    }

    fn queue_coord(&mut self, coord: ChunkCoord, tier: DirtyTier) {
        let old_tier = self.tiers.get(&coord).copied();
        if let Some(old_tier) = old_tier {
            if old_tier.priority_rank() >= tier.priority_rank() {
                return;
            }
            self.decr_tier(old_tier);
        }
        self.tiers.insert(coord, tier);
        self.incr_tier(tier);
        self.queue_for_mut(tier).push_back(coord);
    }

    fn pop_front_tier(&mut self, tier: DirtyTier) -> Option<ChunkCoord> {
        loop {
            let coord = self.queue_for_mut(tier).pop_front()?;
            if self.tiers.get(&coord).copied() == Some(tier) {
                self.tiers.remove(&coord);
                self.decr_tier(tier);
                return Some(coord);
            }
        }
    }

    fn pop_back_tier(&mut self, tier: DirtyTier) -> Option<ChunkCoord> {
        loop {
            let coord = self.queue_for_mut(tier).pop_back()?;
            if self.tiers.get(&coord).copied() == Some(tier) {
                self.tiers.remove(&coord);
                self.decr_tier(tier);
                return Some(coord);
            }
        }
    }

    fn clear(&mut self) {
        self.urgent.clear();
        self.near.clear();
        self.normal.clear();
        self.far.clear();
        self.tiers.clear();
        self.urgent_count = 0;
        self.near_count = 0;
        self.normal_count = 0;
        self.far_count = 0;
    }
}

#[derive(Clone)]
pub(crate) struct ChunkSnapshot {
    pub(crate) world_min: VoxelCoord,
    pub(crate) center_voxels: Arc<[MaterialId]>,
    pub(crate) border_strips: Arc<ChunkBorderStrips>,
}

impl ChunkSnapshot {
    fn get_local(&self, local_x: i32, local_y: i32, local_z: i32) -> MaterialId {
        let side = CHUNK_SIZE_VOXELS;
        if (0..side).contains(&local_x)
            && (0..side).contains(&local_y)
            && (0..side).contains(&local_z)
        {
            let idx = ((local_z as usize * side as usize + local_y as usize) * side as usize)
                + local_x as usize;
            return self.center_voxels[idx];
        }

        let strips = self.border_strips.as_ref();
        let idx = |u: i32, v: i32| (u as usize) * side as usize + v as usize;

        if local_x == -1 && (0..side).contains(&local_y) && (0..side).contains(&local_z) {
            return strips.neg_x[idx(local_y, local_z)];
        }
        if local_x == side && (0..side).contains(&local_y) && (0..side).contains(&local_z) {
            return strips.pos_x[idx(local_y, local_z)];
        }
        if local_y == -1 && (0..side).contains(&local_x) && (0..side).contains(&local_z) {
            return strips.neg_y[idx(local_x, local_z)];
        }
        if local_y == side && (0..side).contains(&local_x) && (0..side).contains(&local_z) {
            return strips.pos_y[idx(local_x, local_z)];
        }
        if local_z == -1 && (0..side).contains(&local_x) && (0..side).contains(&local_y) {
            return strips.neg_z[idx(local_x, local_y)];
        }
        if local_z == side && (0..side).contains(&local_x) && (0..side).contains(&local_y) {
            return strips.pos_z[idx(local_x, local_y)];
        }

        EMPTY
    }

    pub(crate) fn with_center_materials(&self, materials: Vec<MaterialId>) -> Self {
        Self {
            world_min: self.world_min,
            center_voxels: Arc::from(materials),
            border_strips: Arc::clone(&self.border_strips),
        }
    }
}

#[derive(Clone)]
pub(crate) struct MeshJob {
    pub(crate) coord: ChunkCoord,
    pub(crate) lod: ChunkLod,
    pub(crate) version: u64,
    pub(crate) queued_at: Instant,
    pub(crate) snapshot: ChunkSnapshot,
    pub(crate) greedy: bool,
}

struct MeshResult {
    coord: ChunkCoord,
    lod: ChunkLod,
    version: u64,
    queued_at: Instant,
    artifact: ChunkMeshArtifact,
}

pub(crate) enum ChunkMeshArtifact {
    Cpu {
        verts: Vec<Vertex>,
        inds: Vec<u32>,
        indirect: DrawIndirectArgs,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
    },
    Gpu {
        verts: Vec<Vertex>,
        inds: Vec<u32>,
        indirect: DrawIndirectArgs,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
        dispatch_ms: f32,
        readback_bytes: u64,
    },
}

impl ChunkMeshArtifact {
    pub(crate) fn geometry(&self) -> (&[Vertex], &[u32], DrawIndirectArgs, Vec3, Vec3, Vec3) {
        match self {
            Self::Cpu {
                verts,
                inds,
                indirect,
                aabb_min,
                aabb_max,
                chunk_origin_world,
            }
            | Self::Gpu {
                verts,
                inds,
                indirect,
                aabb_min,
                aabb_max,
                chunk_origin_world,
                ..
            } => (
                verts,
                inds,
                *indirect,
                *aabb_min,
                *aabb_max,
                *chunk_origin_world,
            ),
        }
    }
}

struct BackgroundMeshQueue {
    tx: SyncSender<MeshJob>,
    rx: Receiver<MeshResult>,
    inflight: usize,
}

impl BackgroundMeshQueue {
    fn new(worker_count: usize, queue_bound: usize, mesh_backend: MeshPipelineBackend) -> Self {
        let (tx, job_rx) = sync_channel::<MeshJob>(queue_bound);
        let (result_tx, rx) = sync_channel::<MeshResult>(queue_bound);
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

        for i in 0..worker_count {
            let worker_rx = std::sync::Arc::clone(&job_rx);
            let worker_tx = result_tx.clone();
            thread::Builder::new()
                .name(format!("mesh-worker-{i}"))
                .spawn(move || loop {
                    let job = {
                        let lock = worker_rx.lock().expect("mesh worker rx lock");
                        lock.recv()
                    };
                    let Ok(job) = job else {
                        break;
                    };

                    let artifact = match mesh_backend {
                        MeshPipelineBackend::Cpu => cpu_generate_material_field(&job).mesh_artifact,
                        #[cfg(feature = "gpu-compute")]
                        MeshPipelineBackend::Gpu => run_chunk_job_on_worker(&job)
                            .map(|output| output.mesh_artifact)
                            .unwrap_or_else(|_| cpu_generate_material_field(&job).mesh_artifact),
                    };
                    if worker_tx
                        .send(MeshResult {
                            coord: job.coord,
                            lod: job.lod,
                            version: job.version,
                            queued_at: job.queued_at,
                            artifact,
                        })
                        .is_err()
                    {
                        break;
                    }
                })
                .expect("spawn mesh worker");
        }

        Self {
            tx,
            rx,
            inflight: 0,
        }
    }

    fn try_submit(&mut self, job: MeshJob) -> Result<(), TrySendError<MeshJob>> {
        match self.tx.try_send(job) {
            Ok(()) => {
                self.inflight += 1;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn try_recv(&mut self) -> Result<MeshResult, TryRecvError> {
        let result = self.rx.try_recv();
        if result.is_ok() {
            self.inflight = self.inflight.saturating_sub(1);
        }
        result
    }
}

impl Renderer {
    pub async fn new(window: &'static winit::window::Window) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let mesh_backend = if GpuComputeRuntime::runtime_supported(&adapter) {
            #[cfg(feature = "gpu-compute")]
            {
                log::info!(
                    "mesh backend selected: gpu-compute (adapter supports compute pipelines)"
                );
                MeshPipelineBackend::Gpu
            }
            #[cfg(not(feature = "gpu-compute"))]
            {
                log::warn!(
                    "mesh backend selected: cpu (adapter supports GPU compute but `gpu-compute` feature is disabled at compile time)"
                );
                MeshPipelineBackend::Cpu
            }
        } else {
            log::warn!(
                "mesh backend selected: cpu (adapter/runtime does not satisfy gpu-compute requirements)"
            );
            MeshPipelineBackend::Cpu
        };
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cam"),
            contents: bytemuck::bytes_of(&CameraUniform {
                vp: Mat4::IDENTITY.to_cols_array_2d(),
                origin_offset: [0.0, 0.0, 0.0],
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cam_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), ChunkOriginInstance::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            cam_buf,
            cam_bg,
            depth_texture,
            depth_view,
            store_meshes: HashMap::new(),
            vertex_arena: BufferArena::new(
                "chunk vertex arena",
                wgpu::BufferUsages::VERTEX,
                4 * 1024 * 1024,
            ),
            index_arena: BufferArena::new(
                "chunk index arena",
                wgpu::BufferUsages::INDEX,
                4 * 1024 * 1024,
            ),
            dirty_queues: DirtyChunkQueues::default(),
            dirty_near_starve_frames: 0,
            dirty_far_starve_frames: 0,
            dirty_fair_cursor: 0,
            mesh_versions: HashMap::new(),
            meshed_versions: HashMap::new(),
            lod_selection: HashMap::new(),
            mesh_queue: BackgroundMeshQueue::new(2, 256, mesh_backend),
            completed_meshes: Vec::new(),
            origin_voxel: VoxelCoord { x: 0, y: 0, z: 0 },
            day: true,
            mesh_backend,
            settings: RendererSettings::default(),
            allocator_telemetry: MeshAllocatorTelemetry::default(),
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.size = size;
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        (self.depth_texture, self.depth_view) = create_depth_texture(&self.device, &self.config);
    }

    pub fn settings(&self) -> RendererSettings {
        self.settings
    }

    pub fn set_settings(&mut self, settings: RendererSettings) {
        self.settings = settings;
    }

    pub fn set_origin_voxel(&mut self, new_origin: VoxelCoord) {
        self.origin_voxel = new_origin;
    }

    pub fn cull_stats(&self, camera: &Camera) -> CullStats {
        // CPU culling is evaluated in world space to match chunk AABBs.
        // Mesh vertices are chunk-local and become world-space in the shader
        // after applying `chunk_origin_world`.
        let vp_world = camera.view_proj_for_world_origin(self.origin_voxel);
        let world_camera_pos = camera_world_position(camera, self.origin_voxel);
        let mut stats = CullStats::default();
        for (&(coord, lod), mesh) in &self.store_meshes {
            if self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near)
                != lod
            {
                stats.lod_filtered += 1;
                continue;
            }
            if self.settings.frustum_culling
                && !aabb_in_view(vp_world, mesh.world_aabb_min, mesh.world_aabb_max)
            {
                stats.frustum_culled += 1;
                continue;
            }
            if !passes_screen_space_cull(
                world_camera_pos,
                lod,
                mesh.world_aabb_min,
                mesh.world_aabb_max,
                self.size.height,
            ) {
                stats.screen_culled += 1;
                continue;
            }
            stats.drawn += 1;
        }
        stats
    }

    /// Rebuild up to `budget` dirty chunks this frame, without O(N) re-marking cost.
    ///
    /// IMPORTANT: This relies on `store.take_dirty_chunks()` returning + clearing the store's dirty set.

    pub fn rebuild_dirty_store_chunks(
        &mut self,
        store: &mut ChunkStore,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
        mesh_budget: usize,
        upload_byte_budget: usize,
        lod_radii: LodRadii,
        lod_budgets: LodMeshingBudgets,
    ) -> MeshRebuildStats {
        let lod_radii = lod_radii.normalized();
        for coord in store.take_dirty_chunks() {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
            for lod in [
                ChunkLod::Near,
                ChunkLod::Mid,
                ChunkLod::Far,
                ChunkLod::Ultra,
            ] {
                let version = self.mesh_versions.entry((coord, lod)).or_insert(0);
                *version = version.saturating_add(1);
            }
        }
        let mut stats = MeshRebuildStats::default();
        stats.dirty_queue_drop_count +=
            self.enforce_dirty_queue_bound(player_chunk, chunk_priority_scores);

        let mut near_jobs = Vec::new();
        let mut mid_jobs = Vec::new();
        let mut far_jobs = Vec::new();
        let mut ultra_jobs = Vec::new();
        let chunk_snapshot_budget = mesh_budget.max(1);
        let mut snapshot_coords = self.pop_priority_dirty_chunks(
            chunk_snapshot_budget,
            player_chunk,
            chunk_priority_scores,
        );
        let snapshot_frame_start = Instant::now();
        let mut deferred_snapshot_coords = Vec::new();
        while let Some(coord) = snapshot_coords.pop() {
            let elapsed_ms = snapshot_frame_start.elapsed().as_secs_f32() * 1000.0;
            if elapsed_ms >= CHUNK_SNAPSHOT_BUILD_BUDGET_MS {
                deferred_snapshot_coords.push(coord);
                deferred_snapshot_coords.extend(snapshot_coords.into_iter().rev());
                break;
            }

            let t0 = Instant::now();
            let snapshot =
                build_chunk_snapshot(store, coord, self.settings.unknown_neighbor_policy);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            stats.total_ms += ms;
            if ms > stats.max_ms {
                stats.max_ms = ms;
            }

            let prev = self.lod_selection.get(&coord).copied();
            let primary_lod = select_lod(coord, player_chunk, lod_radii, prev);
            let fallback_lod =
                fallback_lod_near_threshold(coord, player_chunk, lod_radii, primary_lod);

            let push_job = |lod: ChunkLod, jobs: &mut Vec<MeshJob>| {
                let version = *self.mesh_versions.get(&(coord, lod)).unwrap_or(&0);
                jobs.push(MeshJob {
                    coord,
                    lod,
                    version,
                    queued_at: Instant::now(),
                    snapshot: snapshot.clone(),
                    greedy: self.settings.greedy_meshing,
                });
            };

            match primary_lod {
                ChunkLod::Near => push_job(ChunkLod::Near, &mut near_jobs),
                ChunkLod::Mid => push_job(ChunkLod::Mid, &mut mid_jobs),
                ChunkLod::Far => push_job(ChunkLod::Far, &mut far_jobs),
                ChunkLod::Ultra => push_job(ChunkLod::Ultra, &mut ultra_jobs),
            }

            if let Some(lod) = fallback_lod {
                match lod {
                    ChunkLod::Near => push_job(ChunkLod::Near, &mut near_jobs),
                    ChunkLod::Mid => push_job(ChunkLod::Mid, &mut mid_jobs),
                    ChunkLod::Far => push_job(ChunkLod::Far, &mut far_jobs),
                    ChunkLod::Ultra => push_job(ChunkLod::Ultra, &mut ultra_jobs),
                }
            }
        }

        for coord in deferred_snapshot_coords {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }

        let job_priority = |coord: ChunkCoord| {
            chunk_priority_scores
                .get(&coord)
                .copied()
                .unwrap_or_else(|| 1.0 / (1.0 + chunk_chebyshev_dist(player_chunk, coord) as f32))
        };
        far_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        near_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        mid_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));

        let far_pressure = self.dirty_queues.total_len() + self.mesh_queue.inflight;
        let far_scale = if far_pressure > 4096 {
            4
        } else if far_pressure > 1024 {
            2
        } else {
            1
        };
        let far_budget = (lod_budgets.far / far_scale).max(usize::from(!far_jobs.is_empty()));
        let ultra_budget = (lod_budgets.ultra / far_scale)
            .max(usize::from(!ultra_jobs.is_empty() && far_scale == 1));

        let sustained_pressure = far_pressure > 2048;
        if sustained_pressure {
            let far_keep = far_budget.min(2);
            if far_jobs.len() > far_keep {
                stats.pressure_drop_count += far_jobs.len() - far_keep;
                far_jobs.truncate(far_keep);
            }
            if !ultra_jobs.is_empty() {
                stats.pressure_drop_count += ultra_jobs.len();
                ultra_jobs.clear();
            }
        }

        let mut submitted = 0usize;
        let mut submit_from =
            |jobs: &mut Vec<MeshJob>, budget: usize, stats: &mut MeshRebuildStats| {
                let mut taken = 0usize;
                while taken < budget && submitted < mesh_budget {
                    let Some(job) = jobs.pop() else {
                        break;
                    };
                    let lod = job.lod;
                    match self.mesh_queue.try_submit(job) {
                        Ok(()) => {
                            taken += 1;
                            submitted += 1;
                            stats.mesh_count += 1;
                            match lod {
                                ChunkLod::Near => stats.near_mesh_count += 1,
                                ChunkLod::Mid => stats.mid_mesh_count += 1,
                                ChunkLod::Far => stats.far_mesh_count += 1,
                                ChunkLod::Ultra => stats.ultra_mesh_count += 1,
                            }
                        }
                        Err(TrySendError::Full(job)) => {
                            jobs.push(job);
                            break;
                        }
                        Err(TrySendError::Disconnected(_)) => break,
                    }
                }
            };

        submit_from(&mut near_jobs, lod_budgets.near, &mut stats);
        submit_from(&mut mid_jobs, lod_budgets.mid, &mut stats);
        submit_from(&mut far_jobs, far_budget, &mut stats);
        submit_from(&mut ultra_jobs, ultra_budget, &mut stats);
        submit_from(&mut near_jobs, mesh_budget, &mut stats);
        submit_from(&mut mid_jobs, mesh_budget, &mut stats);
        submit_from(&mut far_jobs, mesh_budget / far_scale.max(1), &mut stats);
        submit_from(
            &mut ultra_jobs,
            mesh_budget / (far_scale.saturating_mul(2)).max(1),
            &mut stats,
        );

        for job in near_jobs
            .into_iter()
            .chain(mid_jobs)
            .chain(far_jobs)
            .chain(ultra_jobs)
        {
            self.enqueue_dirty_chunk(job.coord, player_chunk, chunk_priority_scores);
        }
        stats.dirty_queue_drop_count +=
            self.enforce_dirty_queue_bound(player_chunk, chunk_priority_scores);

        while let Ok(result) = self.mesh_queue.try_recv() {
            self.completed_meshes.push(result);
        }

        self.completed_meshes.sort_by(|a, b| {
            job_priority(b.coord)
                .total_cmp(&job_priority(a.coord))
                .then_with(|| {
                    let ad = chunk_chebyshev_dist(player_chunk, a.coord);
                    let bd = chunk_chebyshev_dist(player_chunk, b.coord);
                    ad.cmp(&bd)
                })
        });

        if self.completed_meshes.len() > COMPLETED_MESH_BACKLOG_THRESHOLD {
            let mut low_priority_indices: Vec<usize> = self
                .completed_meshes
                .iter()
                .enumerate()
                .filter_map(|(idx, result)| {
                    matches!(result.lod, ChunkLod::Far | ChunkLod::Ultra).then_some(idx)
                })
                .collect();
            low_priority_indices.sort_by_key(|idx| self.completed_meshes[*idx].queued_at);

            let mut to_drop = self.completed_meshes.len() - COMPLETED_MESH_BACKLOG_THRESHOLD;
            let mut drop_mask = vec![false; self.completed_meshes.len()];
            for idx in low_priority_indices {
                if to_drop == 0 {
                    break;
                }
                drop_mask[idx] = true;
                to_drop -= 1;
            }

            if drop_mask.iter().any(|drop| *drop) {
                let mut kept = Vec::with_capacity(self.completed_meshes.len());
                for (idx, result) in self.completed_meshes.drain(..).enumerate() {
                    if drop_mask[idx] {
                        stats.age_drop_count += 1;
                    } else {
                        kept.push(result);
                    }
                }
                self.completed_meshes = kept;
            }
        }

        self.allocator_telemetry = MeshAllocatorTelemetry::default();
        let mut bytes_uploaded = 0usize;
        let mut uploaded = 0usize;
        let mut total_latency_ms = 0.0f32;
        let mut deferred = Vec::new();
        for result in self.completed_meshes.drain(..) {
            let current_version = self
                .mesh_versions
                .get(&(result.coord, result.lod))
                .copied()
                .unwrap_or(0);
            if current_version != result.version || store.is_dirty(result.coord) {
                stats.stale_drop_count += 1;
                continue;
            }

            if self
                .meshed_versions
                .get(&result.coord)
                .copied()
                .unwrap_or(0)
                != result.version
            {
                store.mark_chunk_meshed(result.coord);
                self.meshed_versions.insert(result.coord, result.version);
            }

            if let ChunkMeshArtifact::Gpu {
                dispatch_ms,
                readback_bytes,
                ..
            } = &result.artifact
            {
                stats.gpu_mesh_jobs += 1;
                stats.gpu_dispatch_ms += *dispatch_ms;
                stats.gpu_readback_bytes += *readback_bytes;
            }

            let (verts, inds, _indirect, aabb_min, aabb_max, chunk_origin_world) =
                result.artifact.geometry();
            let bytes = verts.len() * std::mem::size_of::<Vertex>()
                + inds.len() * std::mem::size_of::<u32>();
            if bytes_uploaded + bytes > upload_byte_budget {
                deferred.push(result);
                continue;
            }

            let key = (result.coord, result.lod);
            let existing = self.store_meshes.remove(&key);
            if inds.is_empty() {
                if let Some(old_mesh) = existing {
                    self.vertex_arena.free(old_mesh.vertex_alloc);
                    self.index_arena.free(old_mesh.index_alloc);
                }
            } else {
                let vertex_bytes = (verts.len() * std::mem::size_of::<Vertex>()) as u64;
                let index_bytes = (inds.len() * std::mem::size_of::<u32>()) as u64;
                let mut reallocated = false;

                let mut existing = existing;
                let mut vertex_alloc = existing.as_ref().map(|mesh| mesh.vertex_alloc);
                let mut index_alloc = existing.as_ref().map(|mesh| mesh.index_alloc);

                if let Some(alloc) = vertex_alloc {
                    if alloc.size >= vertex_bytes {
                        self.allocator_telemetry.bytes_reused += vertex_bytes as usize;
                    } else {
                        self.vertex_arena.free(alloc);
                        vertex_alloc = None;
                        reallocated = true;
                    }
                }
                if let Some(alloc) = index_alloc {
                    if alloc.size >= index_bytes {
                        self.allocator_telemetry.bytes_reused += index_bytes as usize;
                    } else {
                        self.index_arena.free(alloc);
                        index_alloc = None;
                        reallocated = true;
                    }
                }

                let vertex_alloc = vertex_alloc.unwrap_or_else(|| {
                    self.vertex_arena.allocate(
                        &self.device,
                        vertex_bytes,
                        &mut self.allocator_telemetry,
                    )
                });
                let index_alloc = index_alloc.unwrap_or_else(|| {
                    self.index_arena.allocate(
                        &self.device,
                        index_bytes,
                        &mut self.allocator_telemetry,
                    )
                });

                if reallocated {
                    self.allocator_telemetry.realloc_count += 1;
                }

                self.queue.write_buffer(
                    self.vertex_arena.buffer(vertex_alloc.slab_index),
                    vertex_alloc.offset,
                    bytemuck::cast_slice(verts),
                );
                self.queue.write_buffer(
                    self.index_arena.buffer(index_alloc.slab_index),
                    index_alloc.offset,
                    bytemuck::cast_slice(inds),
                );

                let (chunk_origin_buf, debug_aabb_vb, debug_aabb_ib, debug_aabb_index_count) =
                    if let Some(old_mesh) = existing.take() {
                        self.queue.write_buffer(
                            &old_mesh.chunk_origin_buf,
                            0,
                            bytemuck::bytes_of(&ChunkOriginInstance {
                                chunk_origin_world: chunk_origin_world.to_array(),
                                _pad: 0.0,
                            }),
                        );
                        (
                            old_mesh.chunk_origin_buf,
                            old_mesh.debug_aabb_vb,
                            old_mesh.debug_aabb_ib,
                            old_mesh.debug_aabb_index_count,
                        )
                    } else {
                        let chunk_origin_buf =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("store chunk origin instance"),
                                    contents: bytemuck::bytes_of(&ChunkOriginInstance {
                                        chunk_origin_world: chunk_origin_world.to_array(),
                                        _pad: 0.0,
                                    }),
                                    usage: wgpu::BufferUsages::VERTEX
                                        | wgpu::BufferUsages::COPY_DST,
                                });
                        let (debug_aabb_verts, debug_aabb_inds) = build_debug_aabb_mesh();
                        let debug_aabb_vb =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("store chunk debug aabb vb"),
                                    contents: bytemuck::cast_slice(&debug_aabb_verts),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });
                        let debug_aabb_ib =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("store chunk debug aabb ib"),
                                    contents: bytemuck::cast_slice(&debug_aabb_inds),
                                    usage: wgpu::BufferUsages::INDEX,
                                });
                        (
                            chunk_origin_buf,
                            debug_aabb_vb,
                            debug_aabb_ib,
                            debug_aabb_inds.len() as u32,
                        )
                    };

                self.store_meshes.insert(
                    key,
                    ChunkMesh {
                        vertex_alloc,
                        chunk_origin_buf,
                        index_alloc,
                        index_count: inds.len() as u32,
                        debug_aabb_vb,
                        debug_aabb_ib,
                        debug_aabb_index_count,
                        world_aabb_min: aabb_min,
                        world_aabb_max: aabb_max,
                        chunk_origin_world,
                    },
                );
            }

            bytes_uploaded += bytes;
            uploaded += 1;
            total_latency_ms += result.queued_at.elapsed().as_secs_f32() * 1000.0;
        }
        self.completed_meshes.extend(deferred);

        stats.upload_count = uploaded;
        stats.upload_bytes = bytes_uploaded;
        stats.upload_latency_ms = if uploaded > 0 {
            total_latency_ms / uploaded as f32
        } else {
            0.0
        };
        stats.allocator_bytes_allocated = self.allocator_telemetry.bytes_allocated;
        stats.allocator_bytes_reused = self.allocator_telemetry.bytes_reused;
        stats.allocator_realloc_count = self.allocator_telemetry.realloc_count;
        stats.dirty_backlog = self.dirty_queues.total_len();
        stats.dirty_urgent_depth = self.dirty_queues.tier_len(DirtyTier::Urgent);
        stats.dirty_near_depth = self.dirty_queues.tier_len(DirtyTier::Near);
        stats.dirty_normal_depth = self.dirty_queues.tier_len(DirtyTier::Normal);
        stats.dirty_far_depth = self.dirty_queues.tier_len(DirtyTier::Far);
        stats.meshing_queue_depth = self.dirty_queues.total_len() + self.mesh_queue.inflight;
        stats.meshing_completed_depth = self.completed_meshes.len();

        let mut drop_keys = Vec::new();
        for &(coord, _) in self.store_meshes.keys() {
            if chunk_chebyshev_dist(player_chunk, coord) > lod_radii.ultra {
                drop_keys.push(coord);
            }
        }
        for coord in drop_keys {
            for lod in [
                ChunkLod::Near,
                ChunkLod::Mid,
                ChunkLod::Far,
                ChunkLod::Ultra,
            ] {
                if let Some(mesh) = self.store_meshes.remove(&(coord, lod)) {
                    self.vertex_arena.free(mesh.vertex_alloc);
                    self.index_arena.free(mesh.index_alloc);
                }
            }
            self.lod_selection.remove(&coord);
        }

        let mut coords = HashSet::new();
        for &(coord, _) in self.store_meshes.keys() {
            coords.insert(coord);
        }
        for coord in coords {
            let prev = self.lod_selection.get(&coord).copied();
            let lod = select_lod(coord, player_chunk, lod_radii, prev);
            self.lod_selection.insert(coord, lod);
        }
        stats
    }
    pub fn clear_mesh_cache(&mut self) {
        for (_, mesh) in self.store_meshes.drain() {
            self.vertex_arena.free(mesh.vertex_alloc);
            self.index_arena.free(mesh.index_alloc);
        }
        self.dirty_queues.clear();
        self.dirty_near_starve_frames = 0;
        self.dirty_far_starve_frames = 0;
        self.dirty_fair_cursor = 0;
        self.completed_meshes.clear();
        self.mesh_versions.clear();
        self.meshed_versions.clear();
        self.lod_selection.clear();
    }
    pub fn mesh_draw_stats(&self, camera: &Camera) -> (usize, u64) {
        // Keep CPU frustum checks in world space; do not pre-apply origin
        // offsets to chunk AABBs here.
        let vp_world = camera.view_proj_for_world_origin(self.origin_voxel);
        let mut chunks = 0usize;
        let mut inds = 0u64;
        for (&(coord, lod), m) in &self.store_meshes {
            if self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near)
                != lod
            {
                continue;
            }
            if aabb_in_view(vp_world, m.world_aabb_min, m.world_aabb_max) {
                chunks += 1;
                inds += m.index_count as u64;
            }
        }
        (chunks, inds)
    }

    pub fn render_world<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        let vp_render = camera.view_proj();
        // World-space culling uses world-space camera/AABBs. Draw still uses
        // floating-origin subtraction in the shader via `origin_offset`.
        let vp_world = camera.view_proj_for_world_origin(self.origin_voxel);
        let world_camera_pos = camera_world_position(camera, self.origin_voxel);

        self.queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&CameraUniform {
                vp: vp_render.to_cols_array_2d(),
                origin_offset: [
                    self.origin_voxel.x as f32 * VOXEL_SIZE,
                    self.origin_voxel.y as f32 * VOXEL_SIZE,
                    self.origin_voxel.z as f32 * VOXEL_SIZE,
                ],
                _pad: 0.0,
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);

        let mut debug_visible_logged = 0usize;
        for (&(coord, lod), mesh) in &self.store_meshes {
            if self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near)
                != lod
            {
                continue;
            }

            let visible_world = chunk_visible_in_world_space(
                self.settings.frustum_culling,
                vp_world,
                world_camera_pos,
                lod,
                mesh.world_aabb_min,
                mesh.world_aabb_max,
                self.size.height,
            );
            if DEBUG_VALIDATE_CULL_SPACE {
                let _ = vp_render;
                let _ = camera;
                log::trace!(
                    "world-space culling active chunk={:?} lod={:?} visible={}",
                    coord,
                    lod,
                    visible_world
                );
            }

            if !visible_world {
                continue;
            }
            if debug_visible_logged < DEBUG_VISIBLE_CHUNK_LOG_COUNT {
                let origin_offset_world = voxel_to_world(self.origin_voxel);
                let computed_world_origin = voxel_to_world(chunk_to_world_min(coord));
                let render_translation = mesh.chunk_origin_world - origin_offset_world;
                log::debug!(
                    "visible chunk {:?} computed_world_origin={:?} instance_world_origin={:?} origin_offset={:?} render_translation={:?} world_aabb_min={:?} world_aabb_max={:?}",
                    coord,
                    computed_world_origin,
                    mesh.chunk_origin_world,
                    origin_offset_world,
                    render_translation,
                    mesh.world_aabb_min,
                    mesh.world_aabb_max,
                );
                debug_visible_logged += 1;
            }
            debug_assert!(mesh.chunk_origin_world.is_finite());
            debug_assert_eq!(
                mesh.chunk_origin_world,
                voxel_to_world(chunk_to_world_min(coord))
            );
            pass.set_vertex_buffer(
                0,
                self.vertex_arena
                    .buffer(mesh.vertex_alloc.slab_index)
                    .slice(
                        mesh.vertex_alloc.offset..mesh.vertex_alloc.offset + mesh.vertex_alloc.size,
                    ),
            );
            pass.set_vertex_buffer(1, mesh.chunk_origin_buf.slice(..));
            pass.set_index_buffer(
                self.index_arena.buffer(mesh.index_alloc.slab_index).slice(
                    mesh.index_alloc.offset..mesh.index_alloc.offset + mesh.index_alloc.size,
                ),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);

            if DEBUG_RENDER_CHUNK_AABBS {
                draw_debug_aabb(pass, mesh, [255, 64, 64, 140]);
            }
        }
    }
}

impl Renderer {
    fn classify_dirty_tier(
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> DirtyTier {
        let distance = chunk_chebyshev_dist(player_chunk, coord);
        let priority = dirty_coord_priority(coord, player_chunk, chunk_priority_scores);
        if distance <= 1 || priority >= DIRTY_VISIBLE_URGENT_SCORE {
            DirtyTier::Urgent
        } else if distance <= 4 {
            DirtyTier::Near
        } else if distance <= 12 {
            DirtyTier::Normal
        } else {
            DirtyTier::Far
        }
    }

    fn enqueue_dirty_chunk(
        &mut self,
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) {
        let tier = Self::classify_dirty_tier(coord, player_chunk, chunk_priority_scores);
        self.dirty_queues.queue_coord(coord, tier);
    }

    fn pop_priority_dirty_chunks(
        &mut self,
        count: usize,
        _player_chunk: ChunkCoord,
        _chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> Vec<ChunkCoord> {
        let mut snapshots = Vec::with_capacity(count);
        let had_near = self.dirty_queues.tier_len(DirtyTier::Near) > 0;
        let had_far = self.dirty_queues.tier_len(DirtyTier::Far) > 0;
        let mut popped_near = false;
        let mut popped_far = false;

        for _ in 0..count {
            if let Some(coord) = self.dirty_queues.pop_front_tier(DirtyTier::Urgent) {
                snapshots.push(coord);
                continue;
            }

            let force_near = self.dirty_near_starve_frames >= DIRTY_NEAR_STARVE_LIMIT_FRAMES;
            let force_far = self.dirty_far_starve_frames >= DIRTY_FAR_STARVE_LIMIT_FRAMES;

            let mut picked = None;
            if force_near {
                picked = self.dirty_queues.pop_front_tier(DirtyTier::Near);
                if picked.is_some() {
                    popped_near = true;
                }
            } else if force_far {
                picked = self.dirty_queues.pop_front_tier(DirtyTier::Far);
                if picked.is_some() {
                    popped_far = true;
                }
            }

            if picked.is_none() {
                let schedule = [
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Normal,
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Normal,
                    DirtyTier::Far,
                ];
                let preferred = schedule[self.dirty_fair_cursor as usize % schedule.len()];
                self.dirty_fair_cursor = self.dirty_fair_cursor.wrapping_add(1);
                let order = match preferred {
                    DirtyTier::Near => [
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Far,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Normal => [
                        DirtyTier::Normal,
                        DirtyTier::Near,
                        DirtyTier::Far,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Far => [
                        DirtyTier::Far,
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Urgent => [
                        DirtyTier::Urgent,
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Far,
                    ],
                };
                for tier in order {
                    picked = self.dirty_queues.pop_front_tier(tier);
                    if picked.is_some() {
                        if tier == DirtyTier::Near {
                            popped_near = true;
                        } else if tier == DirtyTier::Far {
                            popped_far = true;
                        }
                        break;
                    }
                }
            }

            let Some(coord) = picked else {
                break;
            };
            snapshots.push(coord);
        }

        self.dirty_near_starve_frames = if had_near && !popped_near {
            self.dirty_near_starve_frames.saturating_add(1)
        } else {
            0
        };
        self.dirty_far_starve_frames = if had_far && !popped_far {
            self.dirty_far_starve_frames.saturating_add(1)
        } else {
            0
        };

        snapshots
    }

    fn enforce_dirty_queue_bound(
        &mut self,
        _player_chunk: ChunkCoord,
        _chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> usize {
        let mut dropped = 0usize;
        while self.dirty_queues.total_len() > MAX_PENDING_DIRTY_CHUNKS {
            let removed = self
                .dirty_queues
                .pop_back_tier(DirtyTier::Far)
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Normal))
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Near))
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Urgent));
            if removed.is_none() {
                break;
            }
            dropped += 1;
        }
        dropped
    }
}

fn dirty_coord_priority(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    chunk_priority_scores: &HashMap<ChunkCoord, f32>,
) -> f32 {
    chunk_priority_scores
        .get(&coord)
        .copied()
        .unwrap_or_else(|| 1.0 / (1.0 + chunk_chebyshev_dist(player_chunk, coord) as f32))
}

fn build_chunk_snapshot(
    store: &ChunkStore,
    coord: ChunkCoord,
    policy: UnknownNeighborOcclusionPolicy,
) -> ChunkSnapshot {
    let chunk_world_min = chunk_to_world_min(coord);
    let center_voxels = store
        .get_chunk(coord)
        .map(|chunk| Arc::<[MaterialId]>::from(chunk.iter_raw()))
        .unwrap_or_else(|| Arc::from(vec![EMPTY; (CHUNK_SIZE_VOXELS.pow(3)) as usize]));

    let mut strips = store.chunk_border_strips(coord);
    if matches!(policy, UnknownNeighborOcclusionPolicy::Conservative) {
        synthesize_missing_neighbor_borders(store, coord, &mut strips);
    }

    ChunkSnapshot {
        world_min: chunk_world_min,
        center_voxels,
        border_strips: Arc::new(strips),
    }
}

fn synthesize_missing_neighbor_borders(
    store: &ChunkStore,
    coord: ChunkCoord,
    strips: &mut ChunkBorderStrips,
) {
    let Some(center) = store.get_chunk(coord) else {
        return;
    };
    let side = CHUNK_SIZE_VOXELS as usize;
    let idx = |u: usize, v: usize| u * side + v;

    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x - 1,
        y: coord.y,
        z: coord.z,
    }) {
        for y in 0..side {
            for z in 0..side {
                strips.neg_x[idx(y, z)] = center.get(0, y, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x + 1,
        y: coord.y,
        z: coord.z,
    }) {
        for y in 0..side {
            for z in 0..side {
                strips.pos_x[idx(y, z)] = center.get(side - 1, y, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y - 1,
        z: coord.z,
    }) {
        for x in 0..side {
            for z in 0..side {
                strips.neg_y[idx(x, z)] = center.get(x, 0, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y + 1,
        z: coord.z,
    }) {
        for x in 0..side {
            for z in 0..side {
                strips.pos_y[idx(x, z)] = center.get(x, side - 1, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y,
        z: coord.z - 1,
    }) {
        for x in 0..side {
            for y in 0..side {
                strips.neg_z[idx(x, y)] = center.get(x, y, 0);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y,
        z: coord.z + 1,
    }) {
        for x in 0..side {
            for y in 0..side {
                strips.pos_z[idx(x, y)] = center.get(x, y, side - 1);
            }
        }
    }
}

pub(crate) fn mesh_chunk_snapshot(
    coord: ChunkCoord,
    snapshot: &ChunkSnapshot,
    lod: ChunkLod,
    greedy: bool,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3, Vec3) {
    let chunk_world_min = snapshot.world_min;
    debug_assert_eq!(chunk_world_min, chunk_to_world_min(coord));
    let chunk_origin_world = voxel_to_world(chunk_world_min);
    let chunk_extent = Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);
    let (verts, inds) = match lod {
        ChunkLod::Near => {
            if greedy {
                mesh_chunk_voxel_faces_greedy(snapshot)
            } else {
                mesh_chunk_voxel_faces(snapshot, 1)
            }
        }
        ChunkLod::Mid => mesh_chunk_coarse_solid(snapshot, 2),
        ChunkLod::Far => mesh_chunk_coarse_solid(snapshot, 4),
        ChunkLod::Ultra => mesh_chunk_heightfield_proxy(snapshot, 8),
    };
    (
        verts,
        inds,
        chunk_origin_world,
        chunk_origin_world + chunk_extent,
        chunk_origin_world,
    )
}

fn mesh_chunk_voxel_faces(snapshot: &ChunkSnapshot, step: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();

    let mut lz = 0;
    while lz < CHUNK_SIZE_VOXELS {
        let mut ly = 0;
        while ly < CHUNK_SIZE_VOXELS {
            let mut lx = 0;
            while lx < CHUNK_SIZE_VOXELS {
                let id = snapshot.get_local(lx, ly, lz);
                if id == EMPTY {
                    lx += step;
                    continue;
                }

                if is_billboard_material(id) {
                    add_snapshot_billboard(lx, ly, lz, id, &mut verts, &mut inds);
                    lx += step;
                    continue;
                }

                let color = material(id).color;
                add_snapshot_voxel_faces(snapshot, lx, ly, lz, id, color, &mut verts, &mut inds);

                lx += step;
            }
            ly += step;
        }
        lz += step;
    }

    (verts, inds)
}

fn mesh_chunk_voxel_faces_greedy(snapshot: &ChunkSnapshot) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;

    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let id = snapshot.get_local(x, y, z);
                if is_billboard_material(id) {
                    add_snapshot_billboard(x, y, z, id, &mut verts, &mut inds);
                }
            }
        }
    }

    for y in 0..side {
        for z in 0..side {
            let mut x = 0;
            while x < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x + 1, y, z))
                {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x + 1, y, z)) {
                        break;
                    }
                    x += 1;
                }
                add_box_faces(
                    [start as f32, y as f32, z as f32],
                    (x - start) as f32,
                    1.0,
                    1.0,
                    material(id).color,
                    &[true, false, false, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for y in 0..side {
        for z in 0..side {
            let mut x = 0;
            while x < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x - 1, y, z))
                {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x - 1, y, z)) {
                        break;
                    }
                    x += 1;
                }
                add_box_faces(
                    [start as f32, y as f32, z as f32],
                    (x - start) as f32,
                    1.0,
                    1.0,
                    material(id).color,
                    &[false, true, false, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for y in 0..side {
            let mut z = 0;
            while z < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y, z + 1))
                {
                    z += 1;
                    continue;
                }
                let start = z;
                while z < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y, z + 1)) {
                        break;
                    }
                    z += 1;
                }
                add_box_faces(
                    [x as f32, y as f32, start as f32],
                    1.0,
                    1.0,
                    (z - start) as f32,
                    material(id).color,
                    &[false, false, false, false, true, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for y in 0..side {
            let mut z = 0;
            while z < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y, z - 1))
                {
                    z += 1;
                    continue;
                }
                let start = z;
                while z < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y, z - 1)) {
                        break;
                    }
                    z += 1;
                }
                add_box_faces(
                    [x as f32, y as f32, start as f32],
                    1.0,
                    1.0,
                    (z - start) as f32,
                    material(id).color,
                    &[false, false, false, false, false, true],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for z in 0..side {
            let mut y = 0;
            while y < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y + 1, z))
                {
                    y += 1;
                    continue;
                }
                let start = y;
                while y < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y + 1, z)) {
                        break;
                    }
                    y += 1;
                }
                add_box_faces(
                    [x as f32, start as f32, z as f32],
                    1.0,
                    (y - start) as f32,
                    1.0,
                    material(id).color,
                    &[false, false, true, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for z in 0..side {
            let mut y = 0;
            while y < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y - 1, z))
                {
                    y += 1;
                    continue;
                }
                let start = y;
                while y < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y - 1, z)) {
                        break;
                    }
                    y += 1;
                }
                add_box_faces(
                    [x as f32, start as f32, z as f32],
                    1.0,
                    (y - start) as f32,
                    1.0,
                    material(id).color,
                    &[false, false, false, true, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    (verts, inds)
}

fn mesh_chunk_coarse_solid(snapshot: &ChunkSnapshot, step: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;

    let is_solid =
        |x: i32, y: i32, z: i32| dominant_material_in_cell(snapshot, x, y, z, step).is_some();

    let mut z = 0;
    while z < side {
        let mut y = 0;
        while y < side {
            let mut x = 0;
            while x < side {
                let Some(id) = dominant_material_in_cell(snapshot, x, y, z, step) else {
                    x += step;
                    continue;
                };
                let color = material(id).color;
                let mut exposed = [false; 6];
                let n = [
                    (x + step, y, z),
                    (x - step, y, z),
                    (x, y + step, z),
                    (x, y - step, z),
                    (x, y, z + step),
                    (x, y, z - step),
                ];
                for (i, &(nx, ny, nz)) in n.iter().enumerate() {
                    exposed[i] = !is_solid(nx, ny, nz);
                }
                if exposed.iter().any(|e| *e) {
                    add_box_faces(
                        [x as f32, y as f32, z as f32],
                        step as f32,
                        step as f32,
                        step as f32,
                        color,
                        &exposed,
                        &mut verts,
                        &mut inds,
                    );
                }
                x += step;
            }
            y += step;
        }
        z += step;
    }
    (verts, inds)
}

fn mesh_chunk_heightfield_proxy(snapshot: &ChunkSnapshot, tile: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;
    let tiles = (side / tile) as usize;
    let mut heights = vec![None; tiles * tiles];

    for tz in 0..tiles as i32 {
        for tx in 0..tiles as i32 {
            heights[(tz as usize) * tiles + tx as usize] =
                tile_peak(snapshot, tx * tile, tz * tile, tile);
        }
    }

    for tz in 0..tiles as i32 {
        for tx in 0..tiles as i32 {
            let Some((height, id)) = heights[(tz as usize) * tiles + tx as usize] else {
                continue;
            };
            let color = material(id).color;
            let top = height + 1;
            let neighbors = [
                tx + 1 < tiles as i32
                    && heights[(tz as usize) * tiles + (tx + 1) as usize]
                        .is_some_and(|(h, _)| h >= height),
                tx > 0
                    && heights[(tz as usize) * tiles + (tx - 1) as usize]
                        .is_some_and(|(h, _)| h >= height),
                false,
                true,
                tz + 1 < tiles as i32
                    && heights[((tz + 1) as usize) * tiles + tx as usize]
                        .is_some_and(|(h, _)| h >= height),
                tz > 0
                    && heights[((tz - 1) as usize) * tiles + tx as usize]
                        .is_some_and(|(h, _)| h >= height),
            ];
            let exposed = [
                !neighbors[0],
                !neighbors[1],
                true,
                false,
                !neighbors[4],
                !neighbors[5],
            ];
            add_box_faces(
                [(tx * tile) as f32, 0.0, (tz * tile) as f32],
                tile as f32,
                top as f32,
                tile as f32,
                color,
                &exposed,
                &mut verts,
                &mut inds,
            );
        }
    }
    (verts, inds)
}

fn tile_peak(
    snapshot: &ChunkSnapshot,
    base_x: i32,
    base_z: i32,
    tile: i32,
) -> Option<(i32, MaterialId)> {
    for y in (0..CHUNK_SIZE_VOXELS).rev() {
        for z in base_z..(base_z + tile).min(CHUNK_SIZE_VOXELS) {
            for x in base_x..(base_x + tile).min(CHUNK_SIZE_VOXELS) {
                let id = snapshot.get_local(x, y, z);
                if id != EMPTY {
                    return Some((y, id));
                }
            }
        }
    }
    None
}

fn dominant_material_in_cell(
    snapshot: &ChunkSnapshot,
    base_x: i32,
    base_y: i32,
    base_z: i32,
    step: i32,
) -> Option<MaterialId> {
    if base_x < 0 || base_y < 0 || base_z < 0 {
        return None;
    }
    let mut counts = HashMap::<MaterialId, u16>::new();
    for z in base_z..(base_z + step).min(CHUNK_SIZE_VOXELS) {
        for y in base_y..(base_y + step).min(CHUNK_SIZE_VOXELS) {
            for x in base_x..(base_x + step).min(CHUNK_SIZE_VOXELS) {
                let id = snapshot.get_local(x, y, z);
                if id != EMPTY {
                    *counts.entry(id).or_default() += 1;
                }
            }
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(id, _)| id)
}

fn add_box_faces(
    local_voxel_min: [f32; 3],
    sx: f32,
    sy: f32,
    sz: f32,
    color: [u8; 4],
    exposed: &[bool; 6],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let dirs = [
        (
            [
                [1.0, 0.0, 0.0],
                [1.0, sy, 0.0],
                [1.0, sy, sz],
                [1.0, 0.0, sz],
            ],
            0.88,
        ),
        (
            [
                [0.0, 0.0, sz],
                [0.0, sy, sz],
                [0.0, sy, 0.0],
                [0.0, 0.0, 0.0],
            ],
            0.73,
        ),
        (
            [[0.0, sy, sz], [sx, sy, sz], [sx, sy, 0.0], [0.0, sy, 0.0]],
            1.0,
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [sx, 0.0, 0.0],
                [sx, 0.0, sz],
                [0.0, 0.0, sz],
            ],
            0.58,
        ),
        (
            [[sx, 0.0, sz], [sx, sy, sz], [0.0, sy, sz], [0.0, 0.0, sz]],
            0.8,
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [0.0, sy, 0.0],
                [sx, sy, 0.0],
                [sx, 0.0, 0.0],
            ],
            0.66,
        ),
    ];
    let origin = [
        local_voxel_min[0] * VOXEL_SIZE,
        local_voxel_min[1] * VOXEL_SIZE,
        local_voxel_min[2] * VOXEL_SIZE,
    ];

    for (i, (quad, shade)) in dirs.iter().enumerate() {
        if !exposed[i] {
            continue;
        }
        let b = verts.len() as u32;
        let shaded = shade_color(color, *shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    origin[0] + v[0] * VOXEL_SIZE,
                    origin[1] + v[1] * VOXEL_SIZE,
                    origin[2] + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn build_debug_aabb_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let side = CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
    let c = [200, 32, 32, 100];
    let corners = [
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side, side, 0.0],
        [0.0, side, 0.0],
        [0.0, 0.0, side],
        [side, 0.0, side],
        [side, side, side],
        [0.0, side, side],
    ];
    let mut verts = Vec::with_capacity(corners.len());
    for p in corners {
        verts.push(Vertex { pos: p, color: c });
    }
    let inds = vec![
        0, 1, 2, 0, 2, 3, // near
        4, 6, 5, 4, 7, 6, // far
        0, 4, 5, 0, 5, 1, // bottom
        3, 2, 6, 3, 6, 7, // top
        1, 5, 6, 1, 6, 2, // right
        0, 3, 7, 0, 7, 4, // left
    ];
    (verts, inds)
}

fn draw_debug_aabb<'a>(pass: &mut wgpu::RenderPass<'a>, mesh: &'a ChunkMesh, _color: [u8; 4]) {
    pass.set_vertex_buffer(0, mesh.debug_aabb_vb.slice(..));
    pass.set_vertex_buffer(1, mesh.chunk_origin_buf.slice(..));
    pass.set_index_buffer(mesh.debug_aabb_ib.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.debug_aabb_index_count, 0, 0..1);
}

fn chunk_chebyshev_dist(a: ChunkCoord, b: ChunkCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

fn select_lod(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    prev: Option<ChunkLod>,
) -> ChunkLod {
    let d = chunk_chebyshev_dist(coord, player_chunk);
    let h = radii.hysteresis;
    match prev.unwrap_or(ChunkLod::Ultra) {
        ChunkLod::Near if d <= radii.near + h => ChunkLod::Near,
        ChunkLod::Mid if d >= radii.near.saturating_sub(h) && d <= radii.mid + h => ChunkLod::Mid,
        ChunkLod::Far if d >= radii.mid.saturating_sub(h) && d <= radii.far + h => ChunkLod::Far,
        ChunkLod::Ultra if d >= radii.far.saturating_sub(h) => ChunkLod::Ultra,
        _ if d <= radii.near.saturating_sub(h) => ChunkLod::Near,
        _ if d <= radii.mid.saturating_sub(h) => ChunkLod::Mid,
        _ if d <= radii.far.saturating_sub(h) => ChunkLod::Far,
        _ => ChunkLod::Ultra,
    }
}

fn fallback_lod_near_threshold(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    primary: ChunkLod,
) -> Option<ChunkLod> {
    let d = chunk_chebyshev_dist(coord, player_chunk);
    let h = radii.hysteresis.max(1);

    let near_edge = (d - radii.near).abs() <= h;
    let mid_edge = (d - radii.mid).abs() <= h;
    let far_edge = (d - radii.far).abs() <= h;

    match primary {
        ChunkLod::Near if near_edge => Some(ChunkLod::Mid),
        ChunkLod::Mid if near_edge => Some(ChunkLod::Near),
        ChunkLod::Mid if mid_edge => Some(ChunkLod::Far),
        ChunkLod::Far if mid_edge => Some(ChunkLod::Mid),
        ChunkLod::Far if far_edge => Some(ChunkLod::Ultra),
        ChunkLod::Ultra if far_edge => Some(ChunkLod::Far),
        _ => None,
    }
}

fn camera_world_position(camera: &Camera, origin_voxel: VoxelCoord) -> Vec3 {
    camera.pos + voxel_to_world(origin_voxel)
}

fn chunk_visible_in_world_space(
    frustum_culling: bool,
    vp_world: Mat4,
    world_camera_pos: Vec3,
    lod: ChunkLod,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    screen_h: u32,
) -> bool {
    (!frustum_culling || aabb_in_view(vp_world, world_aabb_min, world_aabb_max))
        && passes_screen_space_cull(
            world_camera_pos,
            lod,
            world_aabb_min,
            world_aabb_max,
            screen_h,
        )
}

fn passes_screen_space_cull(
    camera_pos: Vec3,
    lod: ChunkLod,
    aabb_min: Vec3,
    aabb_max: Vec3,
    screen_h: u32,
) -> bool {
    let min_pixels = match lod {
        ChunkLod::Near => 0.0,
        ChunkLod::Mid => 0.5,
        ChunkLod::Far => 1.0,
        ChunkLod::Ultra => 2.0,
    };
    if min_pixels <= 0.0 {
        return true;
    }
    let center = (aabb_min + aabb_max) * 0.5;
    let radius = (aabb_max - center).length();
    let distance = (center - camera_pos).length().max(0.01);
    let focal = screen_h as f32 / (2.0 * (60f32.to_radians() * 0.5).tan());
    let pixel_radius = (radius / distance) * focal;
    pixel_radius * 2.0 >= min_pixels
}

fn add_snapshot_voxel_faces(
    snapshot: &ChunkSnapshot,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    id: MaterialId,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let dirs = [
        (
            [1, 0, 0],
            [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
            0.88,
        ),
        (
            [-1, 0, 0],
            [[0., 0., 1.], [0., 1., 1.], [0., 1., 0.], [0., 0., 0.]],
            0.73,
        ),
        (
            [0, 1, 0],
            [[0., 1., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.]],
            1.0,
        ),
        (
            [0, -1, 0],
            [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
            0.58,
        ),
        (
            [0, 0, 1],
            [[1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.]],
            0.8,
        ),
        (
            [0, 0, -1],
            [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
            0.66,
        ),
    ];

    for (d, quad, shade) in dirs {
        if is_face_occluded(
            id,
            snapshot.get_local(local_x + d[0], local_y + d[1], local_z + d[2]),
        ) {
            continue;
        }

        let b = verts.len() as u32;
        let shaded = shade_color(color, shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    local_x as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    local_y as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    local_z as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn is_billboard_material(id: MaterialId) -> bool {
    matches!(id, BUSH_ID | GRASS_ID)
}

fn add_snapshot_billboard(
    local_x: i32,
    local_y: i32,
    local_z: i32,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let color = material(id).color;
    let quads: &[[[f32; 3]; 4]] = if id == GRASS_ID {
        &[
            [
                [0.48, 0.0, 0.12],
                [0.48, 1.08, 0.12],
                [0.52, 1.08, 0.88],
                [0.52, 0.0, 0.88],
            ],
            [
                [0.12, 0.0, 0.48],
                [0.12, 1.04, 0.48],
                [0.88, 1.04, 0.52],
                [0.88, 0.0, 0.52],
            ],
        ]
    } else {
        &[
            [
                [0.14, 0.0, 0.14],
                [0.14, 0.90, 0.14],
                [0.86, 0.90, 0.86],
                [0.86, 0.0, 0.86],
            ],
            [
                [0.14, 0.0, 0.86],
                [0.14, 0.90, 0.86],
                [0.86, 0.90, 0.14],
                [0.86, 0.0, 0.14],
            ],
        ]
    };

    for quad in quads {
        let b = verts.len() as u32;
        for v in quad {
            verts.push(Vertex {
                pos: [
                    local_x as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    local_y as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    local_z as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color,
            });
        }
        inds.extend_from_slice(&[
            b,
            b + 1,
            b + 2,
            b,
            b + 2,
            b + 3,
            b,
            b + 2,
            b + 1,
            b,
            b + 3,
            b + 2,
        ]);
    }
}

fn is_face_occluded(self_id: MaterialId, neighbor_id: MaterialId) -> bool {
    if neighbor_id == EMPTY {
        return false;
    }
    if neighbor_id == self_id {
        return true;
    }
    if is_billboard_material(neighbor_id) {
        return false;
    }
    let neighbor = material(neighbor_id);
    if neighbor.color[3] < 255 {
        return false;
    }
    matches!(neighbor.phase, Phase::Solid | Phase::Powder)
}

fn shade_color(color: [u8; 4], shade: f32) -> [u8; 4] {
    [
        ((color[0] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[1] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[2] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        color[3],
    ]
}

fn voxel_to_world(voxel: VoxelCoord) -> Vec3 {
    Vec3::new(voxel.x as f32, voxel.y as f32, voxel.z as f32) * VOXEL_SIZE
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn aabb_in_view(vp: Mat4, min: Vec3, max: Vec3) -> bool {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let clips = corners.map(|corner| vp * corner.extend(1.0));

    for plane in 0..6 {
        let mut outside = 0;
        for clip in clips {
            let v = match plane {
                0 => clip.x + clip.w,
                1 => -clip.x + clip.w,
                2 => clip.y + clip.w,
                3 => -clip.y + clip.w,
                // WGPU clip-space depth is [0, w], not [-w, w].
                4 => clip.z,
                _ => -clip.z + clip.w,
            };
            if v < 0.0 {
                outside += 1;
            }
        }
        if outside == clips.len() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{Chunk, NeighborDirtyPolicy};

    fn coord() -> ChunkCoord {
        ChunkCoord { x: 0, y: 0, z: 0 }
    }

    fn chunk_with_voxel(x: usize, y: usize, z: usize, id: MaterialId) -> Chunk {
        let mut c = Chunk::new_empty();
        c.set(x, y, z, id);
        c
    }

    #[test]
    fn frustum_culls_and_accepts_expected_aabbs() {
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let vp = camera.view_proj();

        let visible_min = Vec3::new(-0.5, -0.5, -2.0);
        let visible_max = Vec3::new(0.5, 0.5, -1.0);
        assert!(aabb_in_view(vp, visible_min, visible_max));

        let behind_camera_min = Vec3::new(-0.5, -0.5, 0.5);
        let behind_camera_max = Vec3::new(0.5, 0.5, 1.5);
        assert!(!aabb_in_view(vp, behind_camera_min, behind_camera_max));

        let beyond_far_min = Vec3::new(-1.0, -1.0, -1300.0);
        let beyond_far_max = Vec3::new(1.0, 1.0, -1250.0);
        assert!(!aabb_in_view(vp, beyond_far_min, beyond_far_max));
    }

    #[test]
    fn culling_visibility_with_zero_origin_is_consistent() {
        let origin = VoxelCoord { x: 0, y: 0, z: 0 };
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = Vec3::new(-1.0, -1.0, -6.0);
        let world_max = Vec3::new(1.0, 1.0, -4.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj_for_world_origin(origin),
            camera_world_position(&camera, origin),
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );

        assert!(world_visible);
    }

    #[test]
    fn culling_visibility_with_large_non_zero_origin_is_consistent() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 1_500,
            z: -3_250_000,
        };
        let origin_world = voxel_to_world(origin);
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = origin_world + Vec3::new(-1.0, -1.0, -16.0);
        let world_max = origin_world + Vec3::new(1.0, 1.0, -8.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj_for_world_origin(origin),
            camera_world_position(&camera, origin),
            ChunkLod::Ultra,
            world_min,
            world_max,
            1080,
        );

        assert!(world_visible);
    }

    #[test]
    fn cpu_culling_is_origin_invariant_for_same_world_relationship() {
        let rebased_camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_camera = Camera {
            pos: Vec3::new(1_000_000.0, 0.0, -2_000_000.0),
            dir: rebased_camera.dir,
            aspect: rebased_camera.aspect,
        };
        let origin = VoxelCoord {
            x: (world_camera.pos.x / VOXEL_SIZE) as i32,
            y: (world_camera.pos.y / VOXEL_SIZE) as i32,
            z: (world_camera.pos.z / VOXEL_SIZE) as i32,
        };

        let world_min = world_camera.pos + Vec3::new(-2.0, -2.0, -20.0);
        let world_max = world_camera.pos + Vec3::new(2.0, 2.0, -12.0);

        let baseline_world = chunk_visible_in_world_space(
            true,
            world_camera.view_proj(),
            world_camera.pos,
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );
        let rebased_world = chunk_visible_in_world_space(
            true,
            rebased_camera.view_proj_for_world_origin(origin),
            camera_world_position(&rebased_camera, origin),
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );

        assert_eq!(baseline_world, rebased_world);
    }

    #[test]
    fn screen_space_cull_uses_world_space_distances() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 0,
            z: -2_000_000,
        };
        let world_origin = voxel_to_world(origin);
        let world_camera = world_origin + Vec3::new(0.0, 0.0, 0.0);
        let world_min = world_origin + Vec3::new(-2.0, -2.0, -24.0);
        let world_max = world_origin + Vec3::new(2.0, 2.0, -16.0);

        let consistent_world =
            passes_screen_space_cull(world_camera, ChunkLod::Ultra, world_min, world_max, 1080);

        // Intentionally incorrect mixed-space input (render-relative AABB with
        // world-space camera). This should not match correct world-space culling.
        let mixed_space = passes_screen_space_cull(
            world_camera,
            ChunkLod::Ultra,
            world_min - world_origin,
            world_max - world_origin,
            1080,
        );

        assert!(consistent_world);
        assert_ne!(consistent_world, mixed_space);
    }

    #[test]
    fn dirty_priority_prefers_near_chunks() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = ChunkCoord { x: 1, y: 0, z: 0 };
        let far = ChunkCoord { x: 40, y: 0, z: 0 };
        let scores = HashMap::new();

        assert!(
            dirty_coord_priority(near, player, &scores)
                > dirty_coord_priority(far, player, &scores)
        );
    }

    #[test]
    fn dirty_backlog_sort_perf_guard() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut backlog = Vec::with_capacity(20_000);
        for i in 0..20_000 {
            backlog.push(ChunkCoord {
                x: i % 200,
                y: (i / 200) % 10,
                z: i / 2000,
            });
        }
        let scores = HashMap::new();

        let started = Instant::now();
        backlog.sort_by(|a, b| {
            dirty_coord_priority(*a, player, &scores)
                .total_cmp(&dirty_coord_priority(*b, player, &scores))
        });
        let elapsed = started.elapsed();

        assert!(
            elapsed.as_millis() < 750,
            "backlog prioritization regressed: {:?}",
            elapsed
        );
    }
    #[test]
    fn lod_selection_uses_ultra_tier_with_hysteresis() {
        let radii = LodRadii {
            near: 4,
            mid: 8,
            far: 12,
            ultra: 20,
            hysteresis: 2,
        };
        let player = ChunkCoord { x: 0, y: 0, z: 0 };

        let near = select_lod(
            ChunkCoord { x: 3, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Near),
        );
        let far = select_lod(
            ChunkCoord { x: 11, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Far),
        );
        let ultra = select_lod(
            ChunkCoord { x: 21, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Ultra),
        );

        assert_eq!(near, ChunkLod::Near);
        assert_eq!(far, ChunkLod::Far);
        assert_eq!(ultra, ChunkLod::Ultra);
    }

    #[test]
    fn unknown_neighbor_treated_as_empty_for_boundary_faces_in_aggressive_mode() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            coord(),
            chunk_with_voxel(CHUNK_SIZE_VOXELS as usize - 1, 2, 2, 1),
            false,
            NeighborDirtyPolicy::None,
        );

        let no_neighbor =
            build_chunk_snapshot(&store, coord(), UnknownNeighborOcclusionPolicy::Aggressive);
        let (verts_a, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &no_neighbor, ChunkLod::Near, false);

        store.insert_chunk_with_policy(
            ChunkCoord { x: 1, y: 0, z: 0 },
            Chunk::new_empty(),
            false,
            NeighborDirtyPolicy::None,
        );
        let with_neighbor =
            build_chunk_snapshot(&store, coord(), UnknownNeighborOcclusionPolicy::Aggressive);
        let (verts_b, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &with_neighbor, ChunkLod::Near, false);

        assert_eq!(verts_a.len(), 24);
        assert_eq!(verts_b.len(), 24);
    }

    #[test]
    fn voxel_vertex_positions_are_chunk_local_and_world_origin_is_metadata() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            coord(),
            chunk_with_voxel(2, 3, 4, 1),
            false,
            NeighborDirtyPolicy::None,
        );

        let snapshot =
            build_chunk_snapshot(&store, coord(), UnknownNeighborOcclusionPolicy::Aggressive);
        let (verts, _, min, max, _) =
            mesh_chunk_snapshot(coord(), &snapshot, ChunkLod::Near, false);

        let expected_world_min = voxel_to_world(VoxelCoord { x: 0, y: 0, z: 0 });
        let expected_world_max =
            expected_world_min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);
        assert_eq!(min, expected_world_min);
        assert_eq!(max, expected_world_max);

        let xs: Vec<f32> = verts.iter().map(|v| v.pos[0]).collect();
        assert!(xs
            .iter()
            .all(|x| *x >= 0.0 && *x <= CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE));
    }

    #[test]
    fn adjacent_chunk_bounds_are_world_space_and_contiguous() {
        let store = ChunkStore::new();
        let left_snapshot = build_chunk_snapshot(
            &store,
            ChunkCoord { x: 0, y: 0, z: 0 },
            UnknownNeighborOcclusionPolicy::Aggressive,
        );
        let right_snapshot = build_chunk_snapshot(
            &store,
            ChunkCoord { x: 1, y: 0, z: 0 },
            UnknownNeighborOcclusionPolicy::Aggressive,
        );

        let (_, _, left_min, left_max, _) = mesh_chunk_snapshot(
            ChunkCoord { x: 0, y: 0, z: 0 },
            &left_snapshot,
            ChunkLod::Near,
            false,
        );
        let (_, _, right_min, _, _) = mesh_chunk_snapshot(
            ChunkCoord { x: 1, y: 0, z: 0 },
            &right_snapshot,
            ChunkLod::Near,
            false,
        );

        assert_eq!(left_max.x, right_min.x);
        assert_eq!(left_min.y, right_min.y);
        assert_eq!(left_min.z, right_min.z);
    }
}
