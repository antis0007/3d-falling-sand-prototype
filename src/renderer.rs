use crate::chunk_store::ChunkStore;
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
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
    index_count: u32,
    aabb_min: Vec3,
    aabb_max: Vec3,
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

    store_meshes: HashMap<ChunkCoord, ChunkMesh>,

    pending_dirty: VecDeque<ChunkCoord>,
    pending_dirty_set: HashSet<ChunkCoord>,
    mesh_versions: HashMap<ChunkCoord, u64>,

    mesh_queue: BackgroundMeshQueue,
    completed_meshes: Vec<MeshResult>,

    pub day: bool,
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
}

#[derive(Clone)]
struct ChunkSnapshot {
    world_min: VoxelCoord,
    dim: i32,
    voxels: Vec<MaterialId>,
}

impl ChunkSnapshot {
    fn get_world(&self, voxel: VoxelCoord) -> MaterialId {
        let local_x = voxel.x - self.world_min.x;
        let local_y = voxel.y - self.world_min.y;
        let local_z = voxel.z - self.world_min.z;
        if local_x < 0
            || local_y < 0
            || local_z < 0
            || local_x >= self.dim
            || local_y >= self.dim
            || local_z >= self.dim
        {
            return EMPTY;
        }
        let dim = self.dim as usize;
        let idx = (local_z as usize * dim + local_y as usize) * dim + local_x as usize;
        self.voxels[idx]
    }
}

#[derive(Clone)]
struct MeshJob {
    coord: ChunkCoord,
    version: u64,
    origin_voxel: VoxelCoord,
    queued_at: Instant,
    snapshot: ChunkSnapshot,
}

struct MeshResult {
    coord: ChunkCoord,
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
    fn new(worker_count: usize, queue_bound: usize) -> Self {
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
                    let (verts, inds, aabb_min, aabb_max) =
                        mesh_chunk_snapshot(job.coord, job.origin_voxel, &job.snapshot);
                    if worker_tx
                        .send(MeshResult {
                            coord: job.coord,
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
            mesh_queue: BackgroundMeshQueue::new(2, 256),
            completed_meshes: Vec::new(),
            day: true,
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
        mesh_budget: usize,
        upload_byte_budget: usize,
    ) -> MeshRebuildStats {
        for coord in store.take_dirty_chunks() {
            if self.pending_dirty_set.insert(coord) {
                self.pending_dirty.push_back(coord);
            }
            let version = self.mesh_versions.entry(coord).or_insert(0);
            *version = version.saturating_add(1);
        }

        let mut stats = MeshRebuildStats::default();
        for _ in 0..mesh_budget {
            let Some(coord) = self.pending_dirty.pop_front() else {
                break;
            };

            let version = *self.mesh_versions.get(&coord).unwrap_or(&0);
            let t0 = Instant::now();
            let snapshot = build_chunk_snapshot(store, coord);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            stats.total_ms += ms;
            if ms > stats.max_ms {
                stats.max_ms = ms;
            }
            stats.mesh_count += 1;

            let job = MeshJob {
                coord,
                version,
                origin_voxel,
                queued_at: Instant::now(),
                snapshot,
            };
            match self.mesh_queue.try_submit(job) {
                Ok(()) => {
                    self.pending_dirty_set.remove(&coord);
                }
                Err(TrySendError::Full(job)) => {
                    self.pending_dirty.push_front(job.coord);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.pending_dirty_set.remove(&coord);
                    break;
                }
            }
        }

        while let Ok(result) = self.mesh_queue.try_recv() {
            self.completed_meshes.push(result);
        }

        self.completed_meshes.sort_by_key(|result| {
            let dx = i64::from(result.coord.x - player_chunk.x).abs();
            let dy = i64::from(result.coord.y - player_chunk.y).abs();
            let dz = i64::from(result.coord.z - player_chunk.z).abs();
            dx + dy + dz
        });

        let mut bytes_uploaded = 0usize;
        let mut uploaded = 0usize;
        let mut total_latency_ms = 0.0f32;
        let mut deferred = Vec::new();
        for result in self.completed_meshes.drain(..) {
            let current_version = self.mesh_versions.get(&result.coord).copied().unwrap_or(0);
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
                self.store_meshes.remove(&result.coord);
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
                    result.coord,
                    ChunkMesh {
                        vb,
                        ib,
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
        stats
    }
    pub fn clear_mesh_cache(&mut self) {
        self.store_meshes.clear();
        self.pending_dirty.clear();
        self.pending_dirty_set.clear();
        self.completed_meshes.clear();
        self.mesh_versions.clear();
    }
    pub fn mesh_draw_stats(&self, vp: glam::Mat4) -> (usize, u64) {
        let mut chunks = 0usize;
        let mut inds = 0u64;
        for m in self.store_meshes.values() {
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

        for mesh in self.store_meshes.values() {
            if !aabb_in_view(vp, mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vb.slice(..));
            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}

fn build_chunk_snapshot(store: &ChunkStore, coord: ChunkCoord) -> ChunkSnapshot {
    let chunk_world_min = chunk_to_world_min(ChunkCoord {
        x: coord.x - 1,
        y: coord.y - 1,
        z: coord.z - 1,
    });
    let dim = CHUNK_SIZE_VOXELS * 3;
    let mut voxels = vec![EMPTY; (dim * dim * dim) as usize];

    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let world = VoxelCoord {
                    x: chunk_world_min.x + x,
                    y: chunk_world_min.y + y,
                    z: chunk_world_min.z + z,
                };
                let idx = ((z * dim + y) * dim + x) as usize;
                voxels[idx] = store.get_voxel(world);
            }
        }
    }

    ChunkSnapshot {
        world_min: chunk_world_min,
        dim,
        voxels,
    }
}

fn mesh_chunk_snapshot(
    coord: ChunkCoord,
    origin_voxel: VoxelCoord,
    snapshot: &ChunkSnapshot,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let chunk_world_min = chunk_to_world_min(coord);
    let min = relative_voxel_to_world(chunk_world_min, origin_voxel);
    let max = min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);

    for lz in 0..CHUNK_SIZE_VOXELS {
        for ly in 0..CHUNK_SIZE_VOXELS {
            for lx in 0..CHUNK_SIZE_VOXELS {
                let world_voxel = VoxelCoord {
                    x: chunk_world_min.x + lx,
                    y: chunk_world_min.y + ly,
                    z: chunk_world_min.z + lz,
                };
                let id = snapshot.get_world(world_voxel);
                if id == EMPTY {
                    continue;
                }

                let color = material(id).color;
                add_snapshot_voxel_faces(
                    snapshot,
                    world_voxel,
                    origin_voxel,
                    id,
                    color,
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    (verts, inds, min, max)
}

fn add_snapshot_voxel_faces(
    snapshot: &ChunkSnapshot,
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
        let neighbor = VoxelCoord {
            x: world_voxel.x + d[0],
            y: world_voxel.y + d[1],
            z: world_voxel.z + d[2],
        };
        if is_face_occluded(id, snapshot.get_world(neighbor)) {
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
