//! Authoritative meshing pipeline for runtime rendering.
//!
//! - **Authoritative input:** meshing consumes a [`ChunkSnapshot`] built from
//!   [`ChunkStore`] via `build_chunk_snapshot`, so workers never read mutable
//!   world state directly.
//! - **Vertex coordinate space:** all meshing emits chunk-local positions
//!   (scaled by [`VOXEL_SIZE`]), independent of world placement.
//! - **Chunk transform ownership:** chunk mesh vertices are authored in world
//!   space and submitted from shared GPU mesh buffers.
//! - **Floating-origin contract:** renderer APIs accept **world-space camera
//!   coordinates**. The renderer then derives both world-space culling and
//!   render-space projection from that one source of truth.

use crate::chunk_store::{ChunkBorderStrips, ChunkStore};
#[cfg(feature = "gpu-compute")]
use crate::gpu_compute::gpu_page_capacity;
use crate::gpu_compute::{
    run_chunk_job_on_worker, DrawIndirectArgs, GpuComputeRuntime, MeshPipelineBackend,
};
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, GpuPageIndex, VoxelCoord, CHUNK_SIZE_VOXELS};
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
    world_origin_offset: [f32; 3],
    _pad: f32,
}

#[derive(Clone, Copy)]
struct MeshBufferRange {
    offset: u64,
    size: u64,
}

#[derive(Clone, Copy)]
struct MeshAllocation {
    page_index: usize,
    vertex: MeshBufferRange,
    index: MeshBufferRange,
}

#[derive(Clone, Copy)]
struct MeshPageRange {
    vertex: MeshBufferRange,
    index: MeshBufferRange,
}

struct MeshPage {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    free_ranges: Vec<MeshPageRange>,
    capacity: u64,
    vertex_cursor: u64,
    index_cursor: u64,
    live_allocations: usize,
}

struct MeshPageAllocator {
    label: &'static str,
    page_size: u64,
    pages: Vec<Option<MeshPage>>,
}

impl MeshPageAllocator {
    fn new(label: &'static str, page_size: u64) -> Self {
        Self {
            label,
            page_size,
            pages: Vec::new(),
        }
    }

    fn page(&self, page_index: usize) -> Option<&MeshPage> {
        self.pages.get(page_index).and_then(|page| page.as_ref())
    }

    fn allocate(
        &mut self,
        device: &wgpu::Device,
        vertex_size: u64,
        index_size: u64,
        telemetry: &mut MeshAllocatorTelemetry,
    ) -> MeshAllocation {
        for (page_index, page) in self.pages.iter_mut().enumerate() {
            let Some(page) = page.as_mut() else {
                continue;
            };

            if let Some((free_idx, free_alloc)) =
                page.free_ranges
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, alloc)| {
                        alloc.vertex.size >= vertex_size && alloc.index.size >= index_size
                    })
            {
                page.free_ranges.swap_remove(free_idx);
                telemetry.bytes_reused += (vertex_size + index_size) as usize;
                if free_alloc.vertex.size > vertex_size && free_alloc.index.size > index_size {
                    page.free_ranges.push(MeshPageRange {
                        vertex: MeshBufferRange {
                            offset: free_alloc.vertex.offset + vertex_size,
                            size: free_alloc.vertex.size - vertex_size,
                        },
                        index: MeshBufferRange {
                            offset: free_alloc.index.offset + index_size,
                            size: free_alloc.index.size - index_size,
                        },
                    });
                }
                page.live_allocations += 1;
                return MeshAllocation {
                    page_index,
                    vertex: MeshBufferRange {
                        offset: free_alloc.vertex.offset,
                        size: vertex_size,
                    },
                    index: MeshBufferRange {
                        offset: free_alloc.index.offset,
                        size: index_size,
                    },
                };
            }

            if page.capacity - page.vertex_cursor >= vertex_size
                && page.capacity - page.index_cursor >= index_size
            {
                let vertex_offset = page.vertex_cursor;
                let index_offset = page.index_cursor;
                page.vertex_cursor += vertex_size;
                page.index_cursor += index_size;
                page.live_allocations += 1;
                return MeshAllocation {
                    page_index,
                    vertex: MeshBufferRange {
                        offset: vertex_offset,
                        size: vertex_size,
                    },
                    index: MeshBufferRange {
                        offset: index_offset,
                        size: index_size,
                    },
                };
            }
        }

        let new_capacity = self.page_size.max(vertex_size).max(index_size);
        let page_index = self
            .pages
            .iter()
            .position(|page| page.is_none())
            .unwrap_or(self.pages.len());
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} vertex page", self.label)),
            size: new_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} index page", self.label)),
            size: new_capacity,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let page = MeshPage {
            vertex_buffer,
            index_buffer,
            free_ranges: Vec::new(),
            capacity: new_capacity,
            vertex_cursor: vertex_size,
            index_cursor: index_size,
            live_allocations: 1,
        };
        if page_index == self.pages.len() {
            self.pages.push(Some(page));
        } else {
            self.pages[page_index] = Some(page);
        }
        telemetry.bytes_allocated += (new_capacity * 2) as usize;
        MeshAllocation {
            page_index,
            vertex: MeshBufferRange {
                offset: 0,
                size: vertex_size,
            },
            index: MeshBufferRange {
                offset: 0,
                size: index_size,
            },
        }
    }

    fn free(&mut self, allocation: MeshAllocation) {
        if let Some(Some(page)) = self.pages.get_mut(allocation.page_index) {
            page.free_ranges.push(MeshPageRange {
                vertex: allocation.vertex,
                index: allocation.index,
            });
            page.live_allocations = page.live_allocations.saturating_sub(1);
            if page.live_allocations == 0 {
                self.pages[allocation.page_index] = None;
            }
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
    /// Camera position in **world space**.
    ///
    /// Renderer APIs (`render_world`, `cull_stats`, `mesh_draw_stats`) expect
    /// this to be absolute world coordinates. Floating-origin rebasing is
    /// handled internally by [`Renderer`] using `origin_voxel`.
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

    fn view_proj_rebased_to_origin(&self, origin_voxel: VoxelCoord) -> Mat4 {
        let origin_world = voxel_to_world(origin_voxel);
        let rebased_camera = Camera {
            pos: self.pos - origin_world,
            dir: self.dir,
            aspect: self.aspect,
        };
        rebased_camera.view_proj()
    }
}

pub struct ChunkMesh {
    allocation: MeshAllocation,
    index_count: u32,
    debug_aabb_vb: wgpu::Buffer,
    debug_aabb_ib: wgpu::Buffer,
    debug_aabb_index_count: u32,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    chunk_origin_world: Vec3,
}

#[derive(Default)]
struct ChunkMeshCache {
    near: Option<ChunkMesh>,
    mid: Option<ChunkMesh>,
    far: Option<ChunkMesh>,
    ultra: Option<ChunkMesh>,
}

impl ChunkMeshCache {
    fn get(&self, lod: ChunkLod) -> Option<&ChunkMesh> {
        match lod {
            ChunkLod::Near => self.near.as_ref(),
            ChunkLod::Mid => self.mid.as_ref(),
            ChunkLod::Far => self.far.as_ref(),
            ChunkLod::Ultra => self.ultra.as_ref(),
        }
    }

    fn slot_mut(&mut self, lod: ChunkLod) -> &mut Option<ChunkMesh> {
        match lod {
            ChunkLod::Near => &mut self.near,
            ChunkLod::Mid => &mut self.mid,
            ChunkLod::Far => &mut self.far,
            ChunkLod::Ultra => &mut self.ultra,
        }
    }

    fn best_available(
        &self,
        selected: ChunkLod,
        chunk_distance: f32,
        near_lod_distance: f32,
    ) -> Option<(ChunkLod, &ChunkMesh)> {
        let ordered = [
            ChunkLod::Near,
            ChunkLod::Mid,
            ChunkLod::Far,
            ChunkLod::Ultra,
        ];
        let start = lod_rank(selected);
        for lod in ordered.iter().skip(start).copied() {
            if chunk_distance < near_lod_distance && lod_rank(lod) > lod_rank(ChunkLod::Mid) {
                continue;
            }
            if let Some(mesh) = self.get(lod) {
                return Some((lod, mesh));
            }
        }
        for lod in ordered.iter().take(start).copied() {
            if chunk_distance < near_lod_distance && lod_rank(lod) > lod_rank(ChunkLod::Mid) {
                continue;
            }
            if let Some(mesh) = self.get(lod) {
                return Some((lod, mesh));
            }
        }
        None
    }

    fn drain(self) -> impl Iterator<Item = ChunkMesh> {
        [self.near, self.mid, self.far, self.ultra]
            .into_iter()
            .flatten()
    }

    fn is_empty(&self) -> bool {
        self.near.is_none() && self.mid.is_none() && self.far.is_none() && self.ultra.is_none()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
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

fn lod_rank(lod: ChunkLod) -> usize {
    match lod {
        ChunkLod::Near => 0,
        ChunkLod::Mid => 1,
        ChunkLod::Far => 2,
        ChunkLod::Ultra => 3,
    }
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

    store_meshes: HashMap<ChunkCoord, ChunkMeshCache>,
    visible_gpu_chunks: HashMap<ChunkCoord, GpuChunkDraw>,
    mesh_allocator: MeshPageAllocator,
    global_gpu_vertex_buffer: wgpu::Buffer,
    global_gpu_index_buffer: wgpu::Buffer,
    global_gpu_draw_indirect_buffer: wgpu::Buffer,

    dirty_queues: DirtyChunkQueues,
    urgent_mesh_queue: VecDeque<ChunkCoord>,
    urgent_mesh_set: HashSet<ChunkCoord>,
    dirty_near_starve_frames: u32,
    dirty_far_starve_frames: u32,
    dirty_fair_cursor: u8,
    mesh_versions: HashMap<ChunkCoord, u64>,
    lod_selection: HashMap<ChunkCoord, ChunkLod>,
    pending_lod_remesh: HashSet<ChunkCoord>,
    near_lod_distance: f32,

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
    pub mesh_artifacts_received: usize,
    pub mesh_artifacts_rejected: usize,
    pub mesh_cache_entries: usize,
}

const COMPLETED_MESH_BACKLOG_THRESHOLD: usize = 256;
const DIRTY_NEAR_STARVE_LIMIT_FRAMES: u32 = 8;
const DIRTY_FAR_STARVE_LIMIT_FRAMES: u32 = 20;
const DIRTY_VISIBLE_URGENT_SCORE: f32 = 0.8;
const MAX_LOD_REMESH_PER_FRAME: usize = 64;

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

    fn remove_coord(&mut self, coord: ChunkCoord) {
        if let Some(tier) = self.tiers.remove(&coord) {
            self.decr_tier(tier);
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
        let center_voxels: Arc<[MaterialId]> = Arc::from(materials);
        let border_strips = Arc::new(self.rebuild_border_strips_from_center(&center_voxels));
        Self {
            world_min: self.world_min,
            center_voxels,
            border_strips,
        }
    }

    fn rebuild_border_strips_from_center(&self, new_center: &[MaterialId]) -> ChunkBorderStrips {
        let side = CHUNK_SIZE_VOXELS as usize;
        let mut strips = self.border_strips.as_ref().clone();
        let old_center = self.center_voxels.as_ref();
        if old_center.len() != new_center.len() || old_center.len() != side * side * side {
            return strips;
        }

        let center_idx = |x: usize, y: usize, z: usize| -> usize { (z * side + y) * side + x };
        let strip_idx = |u: usize, v: usize| -> usize { u * side + v };

        for y in 0..side {
            for z in 0..side {
                let edge_old = old_center[center_idx(0, y, z)];
                let i = strip_idx(y, z);
                if strips.neg_x[i] == edge_old {
                    strips.neg_x[i] = new_center[center_idx(0, y, z)];
                }

                let edge_old = old_center[center_idx(side - 1, y, z)];
                if strips.pos_x[i] == edge_old {
                    strips.pos_x[i] = new_center[center_idx(side - 1, y, z)];
                }
            }
        }

        for x in 0..side {
            for z in 0..side {
                let i = strip_idx(x, z);
                let edge_old = old_center[center_idx(x, 0, z)];
                if strips.neg_y[i] == edge_old {
                    strips.neg_y[i] = new_center[center_idx(x, 0, z)];
                }

                let edge_old = old_center[center_idx(x, side - 1, z)];
                if strips.pos_y[i] == edge_old {
                    strips.pos_y[i] = new_center[center_idx(x, side - 1, z)];
                }
            }
        }

        for x in 0..side {
            for y in 0..side {
                let i = strip_idx(x, y);
                let edge_old = old_center[center_idx(x, y, 0)];
                if strips.neg_z[i] == edge_old {
                    strips.neg_z[i] = new_center[center_idx(x, y, 0)];
                }

                let edge_old = old_center[center_idx(x, y, side - 1)];
                if strips.pos_z[i] == edge_old {
                    strips.pos_z[i] = new_center[center_idx(x, y, side - 1)];
                }
            }
        }

        strips
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
    pub(crate) urgent: bool,
}

struct MeshResult {
    coord: ChunkCoord,
    lod: ChunkLod,
    version: u64,
    queued_at: Instant,
    artifact: ChunkMeshArtifact,
    urgent: bool,
}

#[derive(Clone, Copy, Debug)]
struct GpuChunkDraw {
    page_index: GpuPageIndex,
    draw_indirect_index: u32,
    lod: u8,
    origin: Vec3,
}

pub(crate) enum ChunkMeshArtifact {
    #[cfg(feature = "cpu_meshing_debug")]
    Cpu {
        verts: Vec<Vertex>,
        inds: Vec<u32>,
        indirect: DrawIndirectArgs,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
    },
    Gpu {
        page_index: GpuPageIndex,
        draw_indirect_index: u32,
        lod: u8,
        verts: Vec<Vertex>,
        inds: Vec<u32>,
        indirect: DrawIndirectArgs,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
        dispatch_ms: f32,
        readback_bytes: u64,
    },
    Skipped,
}

impl ChunkMeshArtifact {
    pub(crate) fn geometry(&self) -> (&[Vertex], &[u32], DrawIndirectArgs, Vec3, Vec3, Vec3) {
        match self {
            #[cfg(feature = "cpu_meshing_debug")]
            Self::Cpu {
                verts,
                inds,
                indirect,
                aabb_min,
                aabb_max,
                chunk_origin_world,
            } => (
                verts,
                inds,
                *indirect,
                *aabb_min,
                *aabb_max,
                *chunk_origin_world,
            ),
            Self::Gpu {
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
            Self::Skipped => (
                &[],
                &[],
                DrawIndirectArgs::default(),
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
            ),
        }
    }
}

struct BackgroundMeshQueue {
    tx: SyncSender<MeshJob>,
    rx: Receiver<MeshResult>,
    inflight: usize,
}

fn build_mesh_artifact(mesh_backend: MeshPipelineBackend, job: &MeshJob) -> ChunkMeshArtifact {
    match mesh_backend {
        #[cfg(feature = "cpu_meshing_debug")]
        MeshPipelineBackend::Cpu => {
            crate::gpu_compute::cpu_generate_material_field(job).mesh_artifact
        }
        #[cfg(feature = "gpu-compute")]
        MeshPipelineBackend::Gpu => run_chunk_job_on_worker(job)
            .map(|output| output.mesh_artifact)
            .expect("gpu meshing worker failed; cpu fallback is disabled"),
    }
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

                    let artifact = build_mesh_artifact(mesh_backend, &job);
                    if worker_tx
                        .send(MeshResult {
                            coord: job.coord,
                            lod: job.lod,
                            version: job.version,
                            queued_at: job.queued_at,
                            artifact,
                            urgent: job.urgent,
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
                    "mesh backend selected: gpu-compute (adapter supports compute pipelines, page_capacity={})",
                    gpu_page_capacity()
                );
                MeshPipelineBackend::Gpu
            }
            #[cfg(not(feature = "gpu-compute"))]
            {
                anyhow::bail!(
                    "gpu meshing is required at runtime, but `gpu-compute` feature is disabled"
                );
            }
        } else {
            #[cfg(feature = "cpu_meshing_debug")]
            {
                log::warn!(
                    "mesh backend selected: cpu debug path (adapter/runtime does not satisfy gpu-compute requirements)"
                );
                MeshPipelineBackend::Cpu
            }
            #[cfg(not(feature = "cpu_meshing_debug"))]
            {
                anyhow::bail!(
                    "gpu meshing is required at runtime, but adapter/runtime does not satisfy gpu-compute requirements"
                );
            }
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
                world_origin_offset: [0.0, 0.0, 0.0],
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
                buffers: &[Vertex::desc()],
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
        let page_capacity = gpu_page_capacity() as u64;
        let verts_per_page = (CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS) as u64;
        let indices_per_page = verts_per_page * 6;
        let global_gpu_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu mesh vertex buffer"),
            size: page_capacity * verts_per_page * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let global_gpu_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu mesh index buffer"),
            size: page_capacity * indices_per_page * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let global_gpu_draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu draw indirect buffer"),
            size: page_capacity * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            visible_gpu_chunks: HashMap::new(),
            mesh_allocator: MeshPageAllocator::new("chunk mesh", 12 * 1024 * 1024),
            global_gpu_vertex_buffer,
            global_gpu_index_buffer,
            global_gpu_draw_indirect_buffer,
            dirty_queues: DirtyChunkQueues::default(),
            urgent_mesh_queue: VecDeque::new(),
            urgent_mesh_set: HashSet::new(),
            dirty_near_starve_frames: 0,
            dirty_far_starve_frames: 0,
            dirty_fair_cursor: 0,
            mesh_versions: HashMap::new(),
            lod_selection: HashMap::new(),
            pending_lod_remesh: HashSet::new(),
            near_lod_distance: 1.5,
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
        let vp_world = camera.view_proj();
        let world_camera_pos = camera_world_position(camera);
        let mut stats = CullStats::default();
        for (&coord, cache) in &self.store_meshes {
            let selected_lod = self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near);
            let distance = chunk_horizontal_distance_to_camera(coord, world_camera_pos);
            let Some((lod, mesh)) =
                cache.best_available(selected_lod, distance, self.near_lod_distance)
            else {
                continue;
            };
            if lod != selected_lod {
                stats.lod_filtered += 1;
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
        self.near_lod_distance = lod_radii.near as f32 + 0.5;
        for coord in store.take_urgent_dirty_chunks() {
            self.enqueue_urgent_mesh_chunk(coord);
        }
        for coord in store.take_dirty_chunks() {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
        let mut stats = MeshRebuildStats::default();
        stats.dirty_queue_drop_count +=
            self.enforce_dirty_queue_bound(player_chunk, chunk_priority_scores);

        let mut near_jobs = Vec::new();
        let mut mid_jobs = Vec::new();
        let mut far_jobs = Vec::new();
        let mut ultra_jobs = Vec::new();

        let mut urgent_jobs = self.pop_urgent_mesh_jobs(store, &mut stats, player_chunk, lod_radii);

        for job in urgent_jobs.drain(..) {
            let lod = job.lod;
            let queued_at = job.queued_at;
            if self.mesh_queue.inflight == 0 {
                let artifact = build_mesh_artifact(self.mesh_backend, &job);
                self.completed_meshes.push(MeshResult {
                    coord: job.coord,
                    lod,
                    version: job.version,
                    queued_at,
                    artifact,
                    urgent: true,
                });
                stats.mesh_count += 1;
                match lod {
                    ChunkLod::Near => stats.near_mesh_count += 1,
                    ChunkLod::Mid => stats.mid_mesh_count += 1,
                    ChunkLod::Far => stats.far_mesh_count += 1,
                    ChunkLod::Ultra => stats.ultra_mesh_count += 1,
                }
                continue;
            }
            match self.mesh_queue.try_submit(job) {
                Ok(()) => {
                    stats.mesh_count += 1;
                    match lod {
                        ChunkLod::Near => stats.near_mesh_count += 1,
                        ChunkLod::Mid => stats.mid_mesh_count += 1,
                        ChunkLod::Far => stats.far_mesh_count += 1,
                        ChunkLod::Ultra => stats.ultra_mesh_count += 1,
                    }
                }
                Err(TrySendError::Full(job)) => {
                    let artifact = build_mesh_artifact(self.mesh_backend, &job);
                    self.completed_meshes.push(MeshResult {
                        coord: job.coord,
                        lod,
                        version: job.version,
                        queued_at,
                        artifact,
                        urgent: job.urgent,
                    });
                    stats.mesh_count += 1;
                    match lod {
                        ChunkLod::Near => stats.near_mesh_count += 1,
                        ChunkLod::Mid => stats.mid_mesh_count += 1,
                        ChunkLod::Far => stats.far_mesh_count += 1,
                        ChunkLod::Ultra => stats.ultra_mesh_count += 1,
                    }
                }
                Err(TrySendError::Disconnected(_)) => break,
            }
        }

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
                let version = store.chunk_voxel_version(coord);
                jobs.push(MeshJob {
                    coord,
                    lod,
                    version,
                    queued_at: Instant::now(),
                    snapshot: snapshot.clone(),
                    greedy: self.settings.greedy_meshing,
                    urgent: false,
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
        near_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        mid_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        far_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        ultra_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));

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

        submit_from(
            &mut near_jobs,
            lod_budgets.near.min(mesh_budget),
            &mut stats,
        );
        submit_from(&mut mid_jobs, lod_budgets.mid.min(mesh_budget), &mut stats);
        submit_from(&mut far_jobs, far_budget.min(mesh_budget), &mut stats);
        submit_from(&mut ultra_jobs, ultra_budget.min(mesh_budget), &mut stats);

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
                    (matches!(result.lod, ChunkLod::Far | ChunkLod::Ultra) && !result.urgent)
                        .then_some(idx)
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
        let mut remesh_coords = Vec::new();
        let stale_result_coords = Vec::new();
        for result in self.completed_meshes.drain(..) {
            stats.mesh_artifacts_received += 1;
            let voxel_version = store.chunk_voxel_version(result.coord);
            // FIX 5: never upload stale geometry; requeue and skip this artifact.
            if result.version.saturating_add(1) < voxel_version {
                stats.stale_drop_count += 1;
                stats.mesh_artifacts_rejected += 1;
                remesh_coords.push(result.coord);
                continue;
            }

            // FIX 1: `Skipped` means "leave current mesh untouched"; never evict cache entries.
            if matches!(result.artifact, ChunkMeshArtifact::Skipped) {
                continue;
            }

            #[allow(irrefutable_let_patterns)]
            if let ChunkMeshArtifact::Gpu {
                page_index,
                draw_indirect_index,
                lod,
                dispatch_ms,
                readback_bytes,
                chunk_origin_world,
                ..
            } = &result.artifact
            {
                stats.gpu_mesh_jobs += 1;
                stats.gpu_dispatch_ms += *dispatch_ms;
                stats.gpu_readback_bytes += *readback_bytes;
                self.visible_gpu_chunks.insert(
                    result.coord,
                    GpuChunkDraw {
                        page_index: *page_index,
                        draw_indirect_index: *draw_indirect_index,
                        lod: *lod,
                        origin: *chunk_origin_world,
                    },
                );
                self.mesh_versions.insert(result.coord, result.version);
                store.mark_chunk_meshed(result.coord);

                let verts_per_page =
                    (CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS) as u64;
                let indices_per_page = verts_per_page * 6;
                let vertex_offset =
                    page_index.0 as u64 * verts_per_page * std::mem::size_of::<Vertex>() as u64;
                let index_offset =
                    page_index.0 as u64 * indices_per_page * std::mem::size_of::<u32>() as u64;
                let draw_offset = *draw_indirect_index as u64
                    * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64;
                let (verts, inds, indirect, ..) = result.artifact.geometry();
                if !verts.is_empty() {
                    self.queue.write_buffer(
                        &self.global_gpu_vertex_buffer,
                        vertex_offset,
                        bytemuck::cast_slice(verts),
                    );
                }
                if !inds.is_empty() {
                    self.queue.write_buffer(
                        &self.global_gpu_index_buffer,
                        index_offset,
                        bytemuck::cast_slice(inds),
                    );
                }
                let indexed_indirect = DrawIndexedIndirectCommand {
                    index_count: indirect.index_count,
                    instance_count: indirect.instance_count,
                    first_index: (index_offset / std::mem::size_of::<u32>() as u64) as u32,
                    base_vertex: (vertex_offset / std::mem::size_of::<Vertex>() as u64) as i32,
                    first_instance: 0,
                };
                self.queue.write_buffer(
                    &self.global_gpu_draw_indirect_buffer,
                    draw_offset,
                    bytemuck::bytes_of(&indexed_indirect),
                );
                continue;
            }

            let (verts, inds, _mesh_indirect, aabb_min, aabb_max, chunk_origin_world) =
                result.artifact.geometry();

            let bytes = verts.len() * std::mem::size_of::<Vertex>()
                + inds.len() * std::mem::size_of::<u32>();
            if !result.urgent && bytes_uploaded + bytes > upload_byte_budget {
                deferred.push(result);
                continue;
            }

            let chunk_is_empty = store
                .get_chunk(result.coord)
                .map(|chunk| chunk.iter_raw().iter().all(|&id| id == EMPTY))
                .unwrap_or(true);

            let cache = self.store_meshes.entry(result.coord).or_default();
            if inds.is_empty() {
                // Preserve last valid mesh for transient empty meshing artifacts.
                if !chunk_is_empty {
                    stats.mesh_artifacts_rejected += 1;
                    continue;
                }
                if let Some(old_mesh) = cache.slot_mut(result.lod).take() {
                    self.mesh_allocator.free(old_mesh.allocation);
                }
            } else {
                let vertex_bytes = (verts.len() * std::mem::size_of::<Vertex>()) as u64;
                let index_bytes = (inds.len() * std::mem::size_of::<u32>()) as u64;

                let allocation = self.mesh_allocator.allocate(
                    &self.device,
                    vertex_bytes,
                    index_bytes,
                    &mut self.allocator_telemetry,
                );
                if let Some(page) = self.mesh_allocator.page(allocation.page_index) {
                    self.queue.write_buffer(
                        &page.vertex_buffer,
                        allocation.vertex.offset,
                        bytemuck::cast_slice(verts),
                    );
                    self.queue.write_buffer(
                        &page.index_buffer,
                        allocation.index.offset,
                        bytemuck::cast_slice(inds),
                    );
                }

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

                let new_mesh = ChunkMesh {
                    allocation,
                    index_count: inds.len() as u32,
                    debug_aabb_vb,
                    debug_aabb_ib,
                    debug_aabb_index_count: debug_aabb_inds.len() as u32,
                    world_aabb_min: aabb_min,
                    world_aabb_max: aabb_max,
                    chunk_origin_world,
                };

                if let Some(old_mesh) = cache.slot_mut(result.lod).replace(new_mesh) {
                    self.mesh_allocator.free(old_mesh.allocation);
                }
            }

            if cache.is_empty() && chunk_is_empty {
                self.store_meshes.remove(&result.coord);
                self.pending_lod_remesh.remove(&result.coord);
            }

            self.mesh_versions.insert(result.coord, result.version);
            store.mark_chunk_meshed(result.coord);

            bytes_uploaded += bytes;
            uploaded += 1;
            total_latency_ms += result.queued_at.elapsed().as_secs_f32() * 1000.0;

            let mesh_version = *self.mesh_versions.get(&result.coord).unwrap_or(&0);
            if mesh_version < voxel_version {
                remesh_coords.push(result.coord);
            }
        }
        self.completed_meshes.extend(deferred);
        for coord in stale_result_coords {
            self.pending_lod_remesh.remove(&coord);
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
        for coord in remesh_coords {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }

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
        stats.mesh_cache_entries = self.store_meshes.len();
        stats.dirty_backlog = self.dirty_queues.total_len();
        stats.dirty_urgent_depth = self.dirty_queues.tier_len(DirtyTier::Urgent);
        stats.dirty_near_depth = self.dirty_queues.tier_len(DirtyTier::Near);
        stats.dirty_normal_depth = self.dirty_queues.tier_len(DirtyTier::Normal);
        stats.dirty_far_depth = self.dirty_queues.tier_len(DirtyTier::Far);
        stats.meshing_queue_depth = self.dirty_queues.total_len() + self.mesh_queue.inflight;
        stats.meshing_completed_depth = self.completed_meshes.len();

        let ultra_mesh_evict_distance =
            lod_radii.ultra.saturating_sub(lod_radii.hysteresis.max(1)) as f32;
        let far_mesh_evict_distance =
            lod_radii.far.saturating_sub(lod_radii.hysteresis.max(1)) as f32;
        let mid_mesh_evict_distance =
            lod_radii.mid.saturating_sub(lod_radii.hysteresis.max(1)) as f32;
        let mut evict_lod_slots = Vec::new();
        for &coord in self.store_meshes.keys() {
            let d = chunk_distance(player_chunk, coord);
            if d > ultra_mesh_evict_distance {
                evict_lod_slots.push((coord, ChunkLod::Ultra));
            }
            if d > far_mesh_evict_distance {
                evict_lod_slots.push((coord, ChunkLod::Far));
            }
            if d > mid_mesh_evict_distance {
                evict_lod_slots.push((coord, ChunkLod::Mid));
            }
        }
        for (coord, lod) in evict_lod_slots {
            if let Some(cache) = self.store_meshes.get_mut(&coord) {
                if let Some(mesh) = cache.slot_mut(lod).take() {
                    self.mesh_allocator.free(mesh.allocation);
                }
            }
        }

        let mut drop_keys = Vec::new();
        for &coord in self.store_meshes.keys() {
            if chunk_distance(player_chunk, coord) > lod_radii.ultra as f32 {
                drop_keys.push(coord);
            }
        }
        for coord in drop_keys {
            if let Some(cache) = self.store_meshes.remove(&coord) {
                for mesh in cache.drain() {
                    self.mesh_allocator.free(mesh.allocation);
                }
            }
            self.visible_gpu_chunks.remove(&coord);
            self.lod_selection.remove(&coord);
            self.pending_lod_remesh.remove(&coord);
        }

        let cached_coords: Vec<ChunkCoord> = self.store_meshes.keys().copied().collect();
        let mut lod_changes = Vec::new();
        for coord in cached_coords {
            let prev = self.lod_selection.get(&coord).copied();
            let lod = select_lod(coord, player_chunk, lod_radii, prev);
            if prev != Some(lod) {
                lod_changes.push(coord);
            }
            self.lod_selection.insert(coord, lod);
        }
        lod_changes.sort_by(|a, b| {
            let ad = chunk_distance(*a, player_chunk);
            let bd = chunk_distance(*b, player_chunk);
            ad.total_cmp(&bd)
        });
        for coord in lod_changes.into_iter().take(MAX_LOD_REMESH_PER_FRAME) {
            self.enqueue_lod_remesh(coord, player_chunk, chunk_priority_scores);
        }
        stats
    }
    pub fn clear_mesh_cache(&mut self) {
        for (_, cache) in self.store_meshes.drain() {
            for mesh in cache.drain() {
                self.mesh_allocator.free(mesh.allocation);
            }
        }
        self.visible_gpu_chunks.clear();
        self.dirty_queues.clear();
        self.urgent_mesh_queue.clear();
        self.urgent_mesh_set.clear();
        self.dirty_near_starve_frames = 0;
        self.dirty_far_starve_frames = 0;
        self.dirty_fair_cursor = 0;
        self.completed_meshes.clear();
        self.mesh_versions.clear();
        self.lod_selection.clear();
        self.pending_lod_remesh.clear();
    }
    pub fn mesh_draw_stats(&self, camera: &Camera) -> (usize, u64) {
        // Keep CPU frustum checks in world space; do not pre-apply origin
        // offsets to chunk AABBs here.
        let vp_world = camera.view_proj();
        let mut chunks = 0usize;
        let mut inds = 0u64;
        for (&coord, cache) in &self.store_meshes {
            let selected_lod = self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near);
            let world_camera_pos = camera_world_position(camera);
            let distance = chunk_horizontal_distance_to_camera(coord, world_camera_pos);
            let Some((_, m)) = cache.best_available(selected_lod, distance, self.near_lod_distance)
            else {
                continue;
            };
            if aabb_in_view(vp_world, m.world_aabb_min, m.world_aabb_max) {
                chunks += 1;
                inds += m.index_count as u64;
            }
        }
        (chunks, inds)
    }

    pub fn render_world<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        let vp_render = camera.view_proj_rebased_to_origin(self.origin_voxel);
        // World-space culling uses world-space camera/AABBs.
        // Draw uses rebased camera VP + shader world-origin subtraction.
        let vp_world = camera.view_proj();
        let world_camera_pos = camera_world_position(camera);
        let origin_offset_world = voxel_to_world(self.origin_voxel);

        debug_assert!(camera.pos.is_finite());
        debug_assert!(origin_offset_world.is_finite());

        self.queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&CameraUniform {
                vp: vp_render.to_cols_array_2d(),
                world_origin_offset: origin_offset_world.to_array(),
                _pad: 0.0,
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);

        let _ = vp_world;
        let _ = world_camera_pos;

        let draw_count = gpu_page_capacity();
        if draw_count > 0 {
            pass.set_vertex_buffer(0, self.global_gpu_vertex_buffer.slice(..));
            pass.set_index_buffer(
                self.global_gpu_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.multi_draw_indexed_indirect(&self.global_gpu_draw_indirect_buffer, 0, draw_count);
        }
    }
}

impl Renderer {
    fn enqueue_urgent_mesh_chunk(&mut self, coord: ChunkCoord) {
        if self.urgent_mesh_set.insert(coord) {
            self.urgent_mesh_queue.push_back(coord);
        }
        self.pending_lod_remesh.remove(&coord);
        self.dirty_queues.remove_coord(coord);
    }

    fn pop_urgent_mesh_jobs(
        &mut self,
        store: &ChunkStore,
        stats: &mut MeshRebuildStats,
        player_chunk: ChunkCoord,
        lod_radii: LodRadii,
    ) -> Vec<MeshJob> {
        let mut jobs = Vec::new();
        while let Some(coord) = self.urgent_mesh_queue.pop_front() {
            self.urgent_mesh_set.remove(&coord);
            self.pending_lod_remesh.remove(&coord);

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
            let version = store.chunk_voxel_version(coord);
            let queued_at = Instant::now();
            jobs.push(MeshJob {
                coord,
                lod: primary_lod,
                version,
                queued_at,
                snapshot: snapshot.clone(),
                greedy: self.settings.greedy_meshing,
                urgent: true,
            });
            if let Some(lod) = fallback_lod {
                jobs.push(MeshJob {
                    coord,
                    lod,
                    version,
                    queued_at,
                    snapshot: snapshot.clone(),
                    greedy: self.settings.greedy_meshing,
                    urgent: true,
                });
            }

            self.dirty_queues.remove_coord(coord);
        }
        jobs
    }

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
        if self.urgent_mesh_set.contains(&coord) {
            return;
        }
        let tier = Self::classify_dirty_tier(coord, player_chunk, chunk_priority_scores);
        self.dirty_queues.queue_coord(coord, tier);
    }

    fn enqueue_lod_remesh(
        &mut self,
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) {
        if self.pending_lod_remesh.insert(coord) {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
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
            self.pending_lod_remesh.remove(&coord);
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
            if let Some(coord) = removed {
                self.pending_lod_remesh.remove(&coord);
                dropped += 1;
            } else {
                break;
            }
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
    let (aabb_min, aabb_max) = chunk_world_aabb_from_vertices(chunk_origin_world, &verts);
    (verts, inds, aabb_min, aabb_max, chunk_origin_world)
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
    pass.set_index_buffer(mesh.debug_aabb_ib.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.debug_aabb_index_count, 0, 0..1);
}

fn chunk_chebyshev_dist(a: ChunkCoord, b: ChunkCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

fn chunk_distance(a: ChunkCoord, b: ChunkCoord) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    let dz = (a.z - b.z) as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn chunk_horizontal_distance_to_camera(coord: ChunkCoord, world_camera_pos: Vec3) -> f32 {
    let chunk_world_min = voxel_to_world(chunk_to_world_min(coord));
    let center = chunk_world_min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE * 0.5);
    let delta = center - world_camera_pos;
    (delta.x.hypot(delta.z)) / (CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE)
}

fn chunk_world_aabb_from_vertices(chunk_origin_world: Vec3, verts: &[Vertex]) -> (Vec3, Vec3) {
    if verts.is_empty() {
        return (chunk_origin_world, chunk_origin_world);
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in verts {
        let p = chunk_origin_world + Vec3::from_array(v.pos);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

fn select_lod(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    prev: Option<ChunkLod>,
) -> ChunkLod {
    let d = chunk_distance(coord, player_chunk);
    let h = radii.hysteresis.max(1);
    let near_down = (radii.near.saturating_sub(h * 5)) as f32;
    let mid_down = (radii.mid.saturating_sub(h * 10)) as f32;
    let far_down = (radii.far.saturating_sub(h * 20)) as f32;
    let near_up = radii.near as f32;
    let mid_up = radii.mid as f32;
    let far_up = radii.far as f32;

    match prev {
        Some(ChunkLod::Near) => {
            if d <= near_up {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Mid) => {
            if d <= near_down {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Far) => {
            if d <= mid_down {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Ultra) => {
            if d <= far_down {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        None => {
            if d <= near_up {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
    }
}

fn fallback_lod_near_threshold(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    primary: ChunkLod,
) -> Option<ChunkLod> {
    let d = chunk_distance(coord, player_chunk);
    let h = radii.hysteresis.max(1) as f32;

    let near_edge = (d - radii.near as f32).abs() <= h;
    let mid_edge = (d - radii.mid as f32).abs() <= h;
    let far_edge = (d - radii.far as f32).abs() <= h;

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

fn camera_world_position(camera: &Camera) -> Vec3 {
    camera.pos
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
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = Vec3::new(-1.0, -1.0, -6.0);
        let world_max = Vec3::new(1.0, 1.0, -4.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj(),
            camera_world_position(&camera),
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
            pos: origin_world,
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = origin_world + Vec3::new(-1.0, -1.0, -16.0);
        let world_max = origin_world + Vec3::new(1.0, 1.0, -8.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj(),
            camera_world_position(&camera),
            ChunkLod::Ultra,
            world_min,
            world_max,
            1080,
        );

        assert!(world_visible);
    }

    #[test]
    fn cpu_culling_is_origin_invariant_for_same_world_relationship() {
        let world_camera = Camera {
            pos: Vec3::new(1_000_000.0, 0.0, -2_000_000.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
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
            world_camera.view_proj_rebased_to_origin(origin),
            camera_world_position(&world_camera),
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );

        assert_eq!(baseline_world, rebased_world);
    }

    #[test]
    fn rebased_draw_transform_matches_world_space_projection() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 1_500,
            z: -3_250_000,
        };
        let world_camera = Camera {
            pos: voxel_to_world(origin) + Vec3::new(0.75, 1.0, 2.5),
            dir: Vec3::new(0.0, -0.1, -1.0).normalize(),
            aspect: 16.0 / 9.0,
        };
        let chunk_origin_world = voxel_to_world(VoxelCoord {
            x: origin.x + 64,
            y: origin.y + 32,
            z: origin.z - 96,
        });
        let local_vertex = Vec3::new(3.0 * VOXEL_SIZE, 5.0 * VOXEL_SIZE, 2.0 * VOXEL_SIZE);
        let world_pos = chunk_origin_world + local_vertex;

        let clip_world = world_camera.view_proj() * world_pos.extend(1.0);
        let clip_rebased = world_camera.view_proj_rebased_to_origin(origin)
            * (world_pos - voxel_to_world(origin)).extend(1.0);

        assert!(clip_world.abs_diff_eq(clip_rebased, 1e-2));
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
