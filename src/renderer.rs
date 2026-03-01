use crate::chunk_store::{ChunkBorderStrips, ChunkStore};
use crate::gpu_compute::{
    cpu_generate_material_field, rebuilt_snapshot_from_materials, GpuComputeRuntime,
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

#[derive(Clone, Copy)]
pub enum UnknownNeighborOcclusionPolicy {
    Conservative,
    Aggressive,
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawIndexedIndirectPod {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
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
}

pub struct ChunkMesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    indirect: wgpu::Buffer,
    index_count: u32,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChunkLod {
    Near,
    Mid,
    Far,
}

#[derive(Clone, Copy, Debug)]
pub struct LodRadii {
    pub near: i32,
    pub mid: i32,
    pub far: i32,
    pub hysteresis: i32,
}

impl LodRadii {
    pub fn normalized(mut self) -> Self {
        self.near = self.near.max(0);
        self.mid = self.mid.max(self.near);
        self.far = self.far.max(self.mid);
        self.hysteresis = self.hysteresis.max(0);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LodMeshingBudgets {
    pub near: usize,
    pub mid: usize,
    pub far: usize,
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

    pending_dirty: VecDeque<ChunkCoord>,
    pending_dirty_set: HashSet<ChunkCoord>,
    mesh_versions: HashMap<(ChunkCoord, ChunkLod), u64>,
    lod_selection: HashMap<ChunkCoord, ChunkLod>,

    mesh_queue: BackgroundMeshQueue,
    completed_meshes: Vec<MeshResult>,

    pub day: bool,
    pub mesh_backend: MeshPipelineBackend,
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
    pub near_mesh_count: usize,
    pub mid_mesh_count: usize,
    pub far_mesh_count: usize,
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
    pub(crate) origin_voxel: VoxelCoord,
    pub(crate) queued_at: Instant,
    pub(crate) snapshot: ChunkSnapshot,
}

struct MeshResult {
    coord: ChunkCoord,
    lod: ChunkLod,
    version: u64,
    queued_at: Instant,
    verts: Vec<Vertex>,
    inds: Vec<u32>,
    aabb_min: Vec3,
    aabb_max: Vec3,
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

                    let material_output = match mesh_backend {
                        MeshPipelineBackend::Cpu => cpu_generate_material_field(&job),
                        #[cfg(feature = "gpu-compute")]
                        MeshPipelineBackend::Gpu => {
                            crate::gpu_compute::run_chunk_job_on_worker(&job)
                                .unwrap_or_else(|_| cpu_generate_material_field(&job))
                        }
                    };

                    let snapshot =
                        rebuilt_snapshot_from_materials(&job, material_output.generated_materials);
                    let (verts, inds, aabb_min, aabb_max) =
                        mesh_chunk_snapshot(job.coord, job.origin_voxel, &snapshot, job.lod);
                    if worker_tx
                        .send(MeshResult {
                            coord: job.coord,
                            lod: job.lod,
                            version: job.version,
                            queued_at: job.queued_at,
                            verts,
                            inds,
                            aabb_min,
                            aabb_max,
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
                MeshPipelineBackend::Gpu
            }
            #[cfg(not(feature = "gpu-compute"))]
            {
                MeshPipelineBackend::Cpu
            }
        } else {
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
            pending_dirty: VecDeque::new(),
            pending_dirty_set: HashSet::new(),
            mesh_versions: HashMap::new(),
            lod_selection: HashMap::new(),
            mesh_queue: BackgroundMeshQueue::new(2, 256, mesh_backend),
            completed_meshes: Vec::new(),
            day: true,
            mesh_backend,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.size = size;
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        (self.depth_texture, self.depth_view) = create_depth_texture(&self.device, &self.config);
    }

    /// Rebuild up to `budget` dirty chunks this frame, without O(N) re-marking cost.
    ///
    /// IMPORTANT: This relies on `store.take_dirty_chunks()` returning + clearing the store's dirty set.

    pub fn rebuild_dirty_store_chunks(
        &mut self,
        store: &mut ChunkStore,
        origin_voxel: VoxelCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
        mesh_budget: usize,
        upload_byte_budget: usize,
        lod_radii: LodRadii,
        lod_budgets: LodMeshingBudgets,
    ) -> MeshRebuildStats {
        let lod_radii = lod_radii.normalized();
        for coord in store.take_dirty_chunks() {
            if self.pending_dirty_set.insert(coord) {
                self.pending_dirty.push_back(coord);
            }
            for lod in [ChunkLod::Near, ChunkLod::Mid, ChunkLod::Far] {
                let version = self.mesh_versions.entry((coord, lod)).or_insert(0);
                *version = version.saturating_add(1);
            }
        }

        let mut stats = MeshRebuildStats::default();
        let mut near_jobs = Vec::new();
        let mut mid_jobs = Vec::new();
        let mut far_jobs = Vec::new();
        while let Some(coord) = self.pending_dirty.pop_front() {
            let t0 = Instant::now();
            let snapshot = build_chunk_snapshot(store, coord);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            stats.total_ms += ms;
            if ms > stats.max_ms {
                stats.max_ms = ms;
            }
            let push_job = |lod: ChunkLod, jobs: &mut Vec<MeshJob>| {
                let version = *self.mesh_versions.get(&(coord, lod)).unwrap_or(&0);
                jobs.push(MeshJob {
                    coord,
                    lod,
                    version,
                    origin_voxel,
                    queued_at: Instant::now(),
                    snapshot: snapshot.clone(),
                });
            };
            push_job(ChunkLod::Near, &mut near_jobs);
            push_job(ChunkLod::Mid, &mut mid_jobs);
            push_job(ChunkLod::Far, &mut far_jobs);
            self.pending_dirty_set.remove(&coord);
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
        submit_from(&mut far_jobs, lod_budgets.far, &mut stats);
        submit_from(&mut mid_jobs, lod_budgets.mid, &mut stats);
        submit_from(&mut far_jobs, mesh_budget, &mut stats);
        submit_from(&mut near_jobs, mesh_budget, &mut stats);
        submit_from(&mut mid_jobs, mesh_budget, &mut stats);

        for job in near_jobs.into_iter().chain(mid_jobs).chain(far_jobs) {
            if self.pending_dirty_set.insert(job.coord) {
                self.pending_dirty.push_back(job.coord);
            }
        }

        while let Ok(result) = self.mesh_queue.try_recv() {
            self.completed_meshes.push(result);
        }

        self.completed_meshes.sort_by(|a, b| {
            job_priority(a.coord)
                .total_cmp(&job_priority(b.coord))
                .then_with(|| {
                    let ad = chunk_chebyshev_dist(player_chunk, a.coord);
                    let bd = chunk_chebyshev_dist(player_chunk, b.coord);
                    bd.cmp(&ad)
                })
        });

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

            let bytes = result.verts.len() * std::mem::size_of::<Vertex>()
                + result.inds.len() * std::mem::size_of::<u32>();
            if bytes_uploaded + bytes > upload_byte_budget {
                deferred.push(result);
                continue;
            }

            if result.inds.is_empty() {
                self.store_meshes.remove(&(result.coord, result.lod));
            } else {
                let vb = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("store chunk vb"),
                        contents: bytemuck::cast_slice(&result.verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let ib = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("store chunk ib"),
                        contents: bytemuck::cast_slice(&result.inds),
                        usage: wgpu::BufferUsages::INDEX,
                    });

                self.store_meshes.insert(
                    (result.coord, result.lod),
                    ChunkMesh {
                        vb,
                        ib,
                        indirect: self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("store chunk indirect"),
                                contents: bytemuck::bytes_of(&DrawIndexedIndirectPod {
                                    index_count: result.inds.len() as u32,
                                    instance_count: 1,
                                    first_index: 0,
                                    base_vertex: 0,
                                    first_instance: 0,
                                }),
                                usage: wgpu::BufferUsages::INDIRECT,
                            },
                        ),
                        index_count: result.inds.len() as u32,
                        aabb_min: result.aabb_min,
                        aabb_max: result.aabb_max,
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
        stats.dirty_backlog = self.pending_dirty.len();
        stats.meshing_queue_depth = self.pending_dirty.len() + self.mesh_queue.inflight;
        stats.meshing_completed_depth = self.completed_meshes.len();

        let mut drop_keys = Vec::new();
        for &(coord, _) in self.store_meshes.keys() {
            if chunk_chebyshev_dist(player_chunk, coord) > lod_radii.far {
                drop_keys.push(coord);
            }
        }
        for coord in drop_keys {
            self.store_meshes.remove(&(coord, ChunkLod::Near));
            self.store_meshes.remove(&(coord, ChunkLod::Mid));
            self.store_meshes.remove(&(coord, ChunkLod::Far));
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
        self.store_meshes.clear();
        self.pending_dirty.clear();
        self.pending_dirty_set.clear();
        self.completed_meshes.clear();
        self.mesh_versions.clear();
        self.lod_selection.clear();
    }
    pub fn mesh_draw_stats(&self, vp: glam::Mat4) -> (usize, u64) {
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
            if aabb_in_view(vp, m.aabb_min, m.aabb_max) {
                chunks += 1;
                inds += m.index_count as u64;
            }
        }
        (chunks, inds)
    }

    pub fn render_world<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        let vp = camera.view_proj();

        self.queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&CameraUniform {
                vp: vp.to_cols_array_2d(),
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);

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
            if !aabb_in_view(vp, mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vb.slice(..));
            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(&mesh.indirect, 0);
        }
    }
}

fn build_chunk_snapshot(store: &ChunkStore, coord: ChunkCoord) -> ChunkSnapshot {
    let chunk_world_min = chunk_to_world_min(coord);
    let center_voxels = store
        .get_chunk(coord)
        .map(|chunk| Arc::<[MaterialId]>::from(chunk.iter_raw().to_vec()))
        .unwrap_or_else(|| Arc::from(vec![EMPTY; (CHUNK_SIZE_VOXELS.pow(3)) as usize]));

    ChunkSnapshot {
        world_min: chunk_world_min,
        center_voxels,
        border_strips: Arc::new(store.chunk_border_strips(coord)),
    }
}

fn mesh_chunk_snapshot(
    coord: ChunkCoord,
    origin_voxel: VoxelCoord,
    snapshot: &ChunkSnapshot,
    lod: ChunkLod,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let chunk_world_min = chunk_to_world_min(coord);
    let min = relative_voxel_to_world(chunk_world_min, origin_voxel);
    let max = min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);

    let (step, top_only) = match lod {
        ChunkLod::Near => (1, false),
        ChunkLod::Mid => (2, false),
        ChunkLod::Far => (4, true),
    };

    let mut lz = 0;
    while lz < CHUNK_SIZE_VOXELS {
        let mut ly = if top_only {
            CHUNK_SIZE_VOXELS - step
        } else {
            0
        };
        while ly < CHUNK_SIZE_VOXELS {
            let mut lx = 0;
            while lx < CHUNK_SIZE_VOXELS {
                let world_voxel = VoxelCoord {
                    x: chunk_world_min.x + lx,
                    y: chunk_world_min.y + ly,
                    z: chunk_world_min.z + lz,
                };
                let id = snapshot.get_local(lx, ly, lz);
                if id == EMPTY {
                    lx += step;
                    continue;
                }

                let color = material(id).color;
                if top_only {
                    add_snapshot_voxel_top(
                        snapshot,
                        lx,
                        ly,
                        lz,
                        world_voxel,
                        origin_voxel,
                        color,
                        &mut verts,
                        &mut inds,
                    );
                } else {
                    add_snapshot_voxel_faces(
                        snapshot,
                        lx,
                        ly,
                        lz,
                        world_voxel,
                        origin_voxel,
                        id,
                        color,
                        &mut verts,
                        &mut inds,
                    );
                }

                lx += step;
            }
            ly += step;
        }
        lz += step;
    }

    (verts, inds, min, max)
}

fn add_snapshot_voxel_top(
    snapshot: &ChunkSnapshot,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    world_voxel: VoxelCoord,
    origin_voxel: VoxelCoord,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    if snapshot.get_local(local_x, local_y + 1, local_z) != EMPTY {
        return;
    }
    let b = verts.len() as u32;
    let shaded = shade_color(color, 0.9);
    let quad = [[0., 1., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.]];
    for v in quad {
        verts.push(Vertex {
            pos: [
                (world_voxel.x - origin_voxel.x) as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                (world_voxel.y - origin_voxel.y) as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                (world_voxel.z - origin_voxel.z) as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
            ],
            color: shaded,
        });
    }
    inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
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
    match prev.unwrap_or(ChunkLod::Far) {
        ChunkLod::Near if d <= radii.near + h => ChunkLod::Near,
        ChunkLod::Mid if d >= radii.near.saturating_sub(h) && d <= radii.mid + h => ChunkLod::Mid,
        ChunkLod::Far if d >= radii.mid.saturating_sub(h) => ChunkLod::Far,
        _ if d <= radii.near => ChunkLod::Near,
        _ if d <= radii.mid => ChunkLod::Mid,
        _ => ChunkLod::Far,
    }
}

fn add_snapshot_voxel_faces(
    snapshot: &ChunkSnapshot,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    world_voxel: VoxelCoord,
    origin_voxel: VoxelCoord,
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
                    (world_voxel.x - origin_voxel.x) as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    (world_voxel.y - origin_voxel.y) as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    (world_voxel.z - origin_voxel.z) as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn is_face_occluded(self_id: MaterialId, neighbor_id: MaterialId) -> bool {
    if neighbor_id == EMPTY {
        return false;
    }
    if neighbor_id == self_id {
        return true;
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

fn relative_voxel_to_world(voxel: VoxelCoord, origin_voxel: VoxelCoord) -> Vec3 {
    Vec3::new(
        (voxel.x - origin_voxel.x) as f32,
        (voxel.y - origin_voxel.y) as f32,
        (voxel.z - origin_voxel.z) as f32,
    ) * VOXEL_SIZE
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
}
