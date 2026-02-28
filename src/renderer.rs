use crate::chunk_store::ChunkStore;
use crate::meshing::mesh_chunk as mesh_store_chunk;
use crate::sim::{material, Phase};
use crate::types::{ChunkCoord, VoxelCoord};
use crate::world::{MaterialId, World, CHUNK_SIZE, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

pub const VOXEL_SIZE: f32 = 0.5;
const TURF_ID: MaterialId = 17;
const BUSH_ID: MaterialId = 18;
const GRASS_ID: MaterialId = 19;

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
        let proj = Mat4::perspective_rh_gl(60f32.to_radians(), self.aspect.max(0.1), 0.1, 1200.0);
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

struct StreamMeshEntry {
    active_version: u64,
    active_meshes: Vec<ChunkMesh>,
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
    meshes: HashMap<usize, ChunkMesh>,
    store_meshes: HashMap<ChunkCoord, ChunkMesh>,
    streamed_meshes: HashMap<[i32; 3], StreamMeshEntry>,
    stream_mesh_versions: HashMap<[i32; 3], u64>,
    dirty_scan_cursor: usize,
    mesh_frame: u64,
    dirty_since_frame: HashMap<usize, u64>,
    unknown_neighbor_policy: UnknownNeighborOcclusionPolicy,
    pub day: bool,
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
            meshes: HashMap::new(),
            store_meshes: HashMap::new(),
            streamed_meshes: HashMap::new(),
            stream_mesh_versions: HashMap::new(),
            dirty_scan_cursor: 0,
            mesh_frame: 0,
            dirty_since_frame: HashMap::new(),
            unknown_neighbor_policy: UnknownNeighborOcclusionPolicy::Conservative,
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

    pub fn rebuild_dirty_chunks_with_budget(
        &mut self,
        world: &mut World,
        budget: usize,
        world_origin: [i32; 3],
    ) {
        self.rebuild_dirty_chunks_with_budget_and_sampler(world, budget, world_origin, |_| None);
    }

    pub fn rebuild_dirty_chunks_with_budget_and_sampler<F>(
        &mut self,
        world: &mut World,
        budget: usize,
        world_origin: [i32; 3],
        mut sample_global: F,
    ) where
        F: FnMut([i32; 3]) -> Option<MaterialId>,
    {
        let budget = budget.max(1);
        self.mesh_frame = self.mesh_frame.saturating_add(1);
        for (index, chunk) in world.chunks.iter().enumerate() {
            if chunk.dirty_mesh || chunk.voxel_version != chunk.meshed_version {
                self.dirty_since_frame
                    .entry(index)
                    .or_insert(self.mesh_frame);
            }
        }

        let dirty_chunks =
            select_dirty_chunks_fair(world.chunks.len(), budget, self.dirty_scan_cursor, |idx| {
                let chunk = &world.chunks[idx];
                chunk.dirty_mesh || chunk.voxel_version != chunk.meshed_version
            });
        self.dirty_scan_cursor = dirty_chunks.next_cursor;

        for i in dirty_chunks.indices {
            let (verts, inds, aabb_min, aabb_max) =
                mesh_chunk(world, i, world_origin, |local, global| {
                    if local_in_bounds(world, local) {
                        return world.get(local[0], local[1], local[2]);
                    }
                    match sample_global(global) {
                        Some(id) => id,
                        None => unknown_neighbor_material(self.unknown_neighbor_policy),
                    }
                });
            if inds.is_empty() {
                self.meshes.remove(&i);
                world.chunks[i].dirty_mesh = false;
                world.chunks[i].meshed_version = world.chunks[i].voxel_version;
                continue;
            }
            let vb = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk ib"),
                    contents: bytemuck::cast_slice(&inds),
                    usage: wgpu::BufferUsages::INDEX,
                });
            self.meshes.insert(
                i,
                ChunkMesh {
                    vb,
                    ib,
                    index_count: inds.len() as u32,
                    aabb_min,
                    aabb_max,
                },
            );
            world.chunks[i].dirty_mesh = false;
            world.chunks[i].meshed_version = world.chunks[i].voxel_version;
            self.dirty_since_frame.remove(&i);
        }
        self.dirty_since_frame.retain(|idx, _| {
            world
                .chunks
                .get(*idx)
                .map(|chunk| chunk.dirty_mesh || chunk.voxel_version != chunk.meshed_version)
                .unwrap_or(false)
        });
    }

    pub fn rebuild_dirty_store_chunks(
        &mut self,
        store: &mut ChunkStore,
        origin_voxel: VoxelCoord,
        budget: usize,
    ) {
        let dirty = store.take_dirty_chunks();
        let mut iter = dirty.into_iter();
        for coord in iter.by_ref().take(budget) {
            let (verts, inds, aabb_min, aabb_max) = mesh_store_chunk(store, coord, origin_voxel);
            if inds.is_empty() {
                self.store_meshes.remove(&coord);
                continue;
            }
            let vb = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("store chunk vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("store chunk ib"),
                    contents: bytemuck::cast_slice(&inds),
                    usage: wgpu::BufferUsages::INDEX,
                });
            self.store_meshes.insert(
                coord,
                ChunkMesh {
                    vb,
                    ib,
                    index_count: inds.len() as u32,
                    aabb_min,
                    aabb_max,
                },
            );
        }
        for coord in iter {
            store.mark_dirty(coord);
        }
    }

    pub fn rebuild_dirty_chunks(&mut self, world: &mut World) {
        self.rebuild_dirty_chunks_with_budget(world, usize::MAX, [0, 0, 0]);
    }

    pub fn request_stream_mesh_version(&mut self, coord: [i32; 3]) -> u64 {
        let next = self
            .stream_mesh_versions
            .get(&coord)
            .map(|version| version.saturating_add(1))
            .unwrap_or(0);
        self.stream_mesh_versions.insert(coord, next);
        next
    }

    pub fn upsert_stream_mesh<F>(
        &mut self,
        coord: [i32; 3],
        version: u64,
        world: &World,
        world_origin: [i32; 3],
        mut sample_global: F,
    ) where
        F: FnMut([i32; 3]) -> Option<MaterialId>,
    {
        let mut meshes = Vec::with_capacity(world.chunks.len());
        for idx in 0..world.chunks.len() {
            if world.chunks[idx].iter_raw().iter().all(|&v| v == EMPTY) {
                continue;
            }
            let (verts, inds, aabb_min, aabb_max) =
                mesh_chunk(world, idx, world_origin, |local, global| {
                    if local_in_bounds(world, local) {
                        return world.get(local[0], local[1], local[2]);
                    }
                    match sample_global(global) {
                        Some(id) => id,
                        None => unknown_neighbor_material(self.unknown_neighbor_policy),
                    }
                });
            if inds.is_empty() {
                continue;
            }
            let vb = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("stream chunk vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("stream chunk ib"),
                    contents: bytemuck::cast_slice(&inds),
                    usage: wgpu::BufferUsages::INDEX,
                });
            meshes.push(ChunkMesh {
                vb,
                ib,
                index_count: inds.len() as u32,
                aabb_min,
                aabb_max,
            });
        }
        let expected = self.stream_mesh_versions.get(&coord).copied().unwrap_or(0);
        if version != expected {
            return;
        }
        let entry = self
            .streamed_meshes
            .entry(coord)
            .or_insert(StreamMeshEntry {
                active_version: 0,
                active_meshes: Vec::new(),
            });
        entry.active_meshes = meshes;
        entry.active_version = version;
    }

    pub fn prune_stream_meshes(&mut self, keep: &std::collections::HashSet<[i32; 3]>) {
        self.streamed_meshes.retain(|coord, _| keep.contains(coord));
        self.stream_mesh_versions
            .retain(|coord, _| keep.contains(coord));
    }

    pub fn has_stream_mesh(&self, coord: [i32; 3]) -> bool {
        self.streamed_meshes
            .get(&coord)
            .map(|entry| !entry.active_meshes.is_empty())
            .unwrap_or(false)
    }

    pub fn stream_mesh_is_transitioning(&self, coord: [i32; 3]) -> bool {
        let desired = self.stream_mesh_versions.get(&coord).copied().unwrap_or(0);
        self.streamed_meshes
            .get(&coord)
            .map(|entry| desired > entry.active_version)
            .unwrap_or(false)
    }

    pub fn clear_mesh_cache(&mut self) {
        self.meshes.clear();
        self.store_meshes.clear();
        self.streamed_meshes.clear();
        self.stream_mesh_versions.clear();
        self.dirty_since_frame.clear();
        self.dirty_scan_cursor = 0;
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
        for mesh in self.meshes.values() {
            if !aabb_in_view(vp, mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vb.slice(..));
            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        for mesh in self.store_meshes.values() {
            if !aabb_in_view(vp, mesh.aabb_min, mesh.aabb_max) {
                continue;
            }
            pass.set_vertex_buffer(0, mesh.vb.slice(..));
            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        for entry in self.streamed_meshes.values() {
            for mesh in &entry.active_meshes {
                if !aabb_in_view(vp, mesh.aabb_min, mesh.aabb_max) {
                    continue;
                }
                pass.set_vertex_buffer(0, mesh.vb.slice(..));
                pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }
}

struct DirtySelection {
    indices: Vec<usize>,
    next_cursor: usize,
}

fn select_dirty_chunks_fair<F>(
    chunk_count: usize,
    budget: usize,
    cursor: usize,
    mut is_dirty: F,
) -> DirtySelection
where
    F: FnMut(usize) -> bool,
{
    if chunk_count == 0 {
        return DirtySelection {
            indices: Vec::new(),
            next_cursor: 0,
        };
    }
    let mut indices = Vec::new();
    let mut scanned = 0usize;
    let mut index = cursor % chunk_count;

    while scanned < chunk_count && indices.len() < budget {
        if is_dirty(index) {
            indices.push(index);
        }
        index = (index + 1) % chunk_count;
        scanned += 1;
    }

    DirtySelection {
        indices,
        next_cursor: index,
    }
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
                // perspective_rh_gl uses OpenGL clip space where -w <= z <= w.
                4 => clip.z + clip.w,
                _ => -clip.z + clip.w,
            };
            if v < 0.0 {
                outside += 1;
            }
        }
        if outside == 8 {
            return false;
        }
    }
    true
}

fn unknown_neighbor_material(policy: UnknownNeighborOcclusionPolicy) -> MaterialId {
    match policy {
        // Conservative: treat unknown neighbors as solid to avoid seams/cracks.
        UnknownNeighborOcclusionPolicy::Conservative => TURF_ID,
        // Aggressive: treat unknown neighbors as empty to expose surfaces early.
        UnknownNeighborOcclusionPolicy::Aggressive => EMPTY,
    }
}

fn local_in_bounds(world: &World, local: [i32; 3]) -> bool {
    local[0] >= 0
        && local[1] >= 0
        && local[2] >= 0
        && (local[0] as usize) < world.dims[0]
        && (local[1] as usize) < world.dims[1]
        && (local[2] as usize) < world.dims[2]
}

fn mesh_chunk<F>(
    world: &World,
    idx: usize,
    world_origin: [i32; 3],
    mut sample_voxel: F,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3)
where
    F: FnMut([i32; 3], [i32; 3]) -> MaterialId,
{
    let cdx = world.chunks_dims[0];
    let cdy = world.chunks_dims[1];
    let cx = idx % cdx;
    let cy = (idx / cdx) % cdy;
    let cz = idx / (cdx * cdy);
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let chunk = &world.chunks[idx];
    if chunk.iter_raw().iter().all(|&v| v == EMPTY) {
        let min = Vec3::new(
            (world_origin[0] + (cx as i32 * CHUNK_SIZE as i32)) as f32,
            (world_origin[1] + (cy as i32 * CHUNK_SIZE as i32)) as f32,
            (world_origin[2] + (cz as i32 * CHUNK_SIZE as i32)) as f32,
        ) * VOXEL_SIZE;
        let max = min + Vec3::splat(CHUNK_SIZE as f32 * VOXEL_SIZE);
        return (verts, inds, min, max);
    }

    let ox = cx as i32 * CHUNK_SIZE as i32;
    let oy = cy as i32 * CHUNK_SIZE as i32;
    let oz = cz as i32 * CHUNK_SIZE as i32;
    for lz in 0..CHUNK_SIZE as i32 {
        for ly in 0..CHUNK_SIZE as i32 {
            for lx in 0..CHUNK_SIZE as i32 {
                let local = [ox + lx, oy + ly, oz + lz];
                let world_p = [
                    world_origin[0] + local[0],
                    world_origin[1] + local[1],
                    world_origin[2] + local[2],
                ];
                let id = chunk.get(lx as usize, ly as usize, lz as usize);
                if id == EMPTY {
                    continue;
                }
                if matches!(id, BUSH_ID | GRASS_ID) {
                    add_crossed_billboard(
                        [local[0], local[1], local[2]],
                        world_p,
                        &mut sample_voxel,
                        id,
                        &mut verts,
                        &mut inds,
                    );
                    continue;
                }
                let color = material(id).color;
                add_voxel_faces(
                    [local[0], local[1], local[2]],
                    world_p,
                    &mut sample_voxel,
                    id,
                    color,
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    let min = Vec3::new(
        (world_origin[0] + ox) as f32,
        (world_origin[1] + oy) as f32,
        (world_origin[2] + oz) as f32,
    ) * VOXEL_SIZE;
    let max = Vec3::new(
        (world_origin[0] + ox + CHUNK_SIZE as i32) as f32,
        (world_origin[1] + oy + CHUNK_SIZE as i32) as f32,
        (world_origin[2] + oz + CHUNK_SIZE as i32) as f32,
    ) * VOXEL_SIZE;
    (verts, inds, min, max)
}

fn add_voxel_faces<F>(
    local_p: [i32; 3],
    world_p: [i32; 3],
    sample_voxel: &mut F,
    id: MaterialId,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) where
    F: FnMut([i32; 3], [i32; 3]) -> MaterialId,
{
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
        let local_neighbor = [local_p[0] + d[0], local_p[1] + d[1], local_p[2] + d[2]];
        let world_neighbor = [world_p[0] + d[0], world_p[1] + d[1], world_p[2] + d[2]];
        if is_face_occluded(id, sample_voxel(local_neighbor, world_neighbor)) {
            continue;
        }
        let b = verts.len() as u32;
        let face_color = turf_face_color(id, d, color);
        let shaded = shade_color(face_color, shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    (world_p[0] as f32 + v[0]) * VOXEL_SIZE,
                    (world_p[1] as f32 + v[1]) * VOXEL_SIZE,
                    (world_p[2] as f32 + v[2]) * VOXEL_SIZE,
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

    if matches!(neighbor_id, GRASS_ID | BUSH_ID) {
        return false;
    }

    let neighbor = material(neighbor_id);
    if neighbor.color[3] < 255 {
        return false;
    }

    matches!(neighbor.phase, Phase::Solid | Phase::Powder)
}

fn turf_face_color(id: MaterialId, dir: [i32; 3], fallback: [u8; 4]) -> [u8; 4] {
    if id != TURF_ID {
        return fallback;
    }
    if dir == [0, 1, 0] {
        return [92, 171, 78, 255];
    }
    if dir == [0, -1, 0] {
        return [121, 88, 56, 255];
    }
    [116, 103, 61, 255]
}

fn add_crossed_billboard<F>(
    local_p: [i32; 3],
    world_p: [i32; 3],
    sample_voxel: &mut F,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) where
    F: FnMut([i32; 3], [i32; 3]) -> MaterialId,
{
    let above = sample_voxel(
        [local_p[0], local_p[1] + 1, local_p[2]],
        [world_p[0], world_p[1] + 1, world_p[2]],
    );
    if above != EMPTY {
        return;
    }
    let base_color = material(id).color;
    let texture_tint = plant_texture_tint(id);
    let color = [
        ((base_color[0] as u16 + texture_tint[0] as u16) / 2) as u8,
        ((base_color[1] as u16 + texture_tint[1] as u16) / 2) as u8,
        ((base_color[2] as u16 + texture_tint[2] as u16) / 2) as u8,
        base_color[3],
    ];

    let quads = [
        [
            [0.15, 0.0, 0.15],
            [0.15, 1.0, 0.15],
            [0.85, 1.0, 0.85],
            [0.85, 0.0, 0.85],
        ],
        [
            [0.15, 0.0, 0.85],
            [0.15, 1.0, 0.85],
            [0.85, 1.0, 0.15],
            [0.85, 0.0, 0.15],
        ],
    ];

    for quad in quads {
        let b = verts.len() as u32;
        for v in quad {
            verts.push(Vertex {
                pos: [
                    (world_p[0] as f32 + v[0]) * VOXEL_SIZE,
                    (world_p[1] as f32 + v[1]) * VOXEL_SIZE,
                    (world_p[2] as f32 + v[2]) * VOXEL_SIZE,
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

fn plant_texture_tint(id: MaterialId) -> [u8; 4] {
    static TINTS: OnceLock<([u8; 4], [u8; 4])> = OnceLock::new();
    let (grass, bush) = *TINTS.get_or_init(|| {
        let grass =
            load_tint_from_file(Path::new("assets/materials/grass.txt"), [90, 180, 80, 220]);
        let bush = load_tint_from_file(Path::new("assets/materials/bush.txt"), [70, 150, 65, 220]);
        (grass, bush)
    });
    if id == GRASS_ID {
        grass
    } else {
        bush
    }
}

fn load_tint_from_file(path: &Path, fallback: [u8; 4]) -> [u8; 4] {
    let Ok(raw) = fs::read_to_string(path) else {
        return fallback;
    };
    let line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'));
    let Some(line) = line else {
        return fallback;
    };
    let mut vals = line
        .split(',')
        .map(str::trim)
        .filter_map(|v| v.parse::<u8>().ok());
    let (Some(r), Some(g), Some(b), Some(a)) = (vals.next(), vals.next(), vals.next(), vals.next())
    else {
        return fallback;
    };
    [r, g, b, a]
}

fn shade_color(color: [u8; 4], shade: f32) -> [u8; 4] {
    [
        ((color[0] as f32 * shade).min(255.0)) as u8,
        ((color[1] as f32 * shade).min(255.0)) as u8,
        ((color[2] as f32 * shade).min(255.0)) as u8,
        color[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_dirty_scheduler_is_starvation_free_under_constant_dirty_load() {
        let chunk_count = 10;
        let budget = 3;
        let mut cursor = 0;
        let mut seen = [0usize; 10];

        for _frame in 0..20 {
            let selected = select_dirty_chunks_fair(chunk_count, budget, cursor, |_| true);
            cursor = selected.next_cursor;
            for idx in selected.indices {
                seen[idx] += 1;
            }
        }

        assert!(seen.iter().all(|count| *count > 0));
    }

    #[test]
    fn fair_dirty_scheduler_reaches_late_indices_when_budget_small() {
        let chunk_count = 8;
        let budget = 2;
        let mut cursor = 0;
        let mut reached = false;

        for _frame in 0..8 {
            let selected = select_dirty_chunks_fair(chunk_count, budget, cursor, |_| true);
            cursor = selected.next_cursor;
            if selected.indices.contains(&7) {
                reached = true;
                break;
            }
        }

        assert!(reached, "highest index chunk should eventually be selected");
    }

    #[test]
    fn unknown_neighbor_policy_maps_as_expected() {
        assert_eq!(
            unknown_neighbor_material(UnknownNeighborOcclusionPolicy::Conservative),
            TURF_ID
        );
        assert_eq!(
            unknown_neighbor_material(UnknownNeighborOcclusionPolicy::Aggressive),
            EMPTY
        );
    }

    #[test]
    fn crossed_billboard_builds_two_double_sided_quads() {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        add_crossed_billboard(
            [0, 0, 0],
            [0, 0, 0],
            &mut |_local, _global| EMPTY,
            GRASS_ID,
            &mut verts,
            &mut inds,
        );

        assert_eq!(verts.len(), 8);
        assert_eq!(inds.len(), 24);
        assert_eq!(&inds[0..12], &[0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]);
        assert_eq!(&inds[12..24], &[4, 5, 6, 4, 6, 7, 4, 6, 5, 4, 7, 6]);
    }

    #[test]
    fn aabb_culling_identity_view_proj() {
        let vp = Mat4::IDENTITY;
        assert!(aabb_in_view(
            vp,
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5)
        ));
        assert!(!aabb_in_view(
            vp,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::new(1001.0, 1.0, 1.0)
        ));
    }
}
