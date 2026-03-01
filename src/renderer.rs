use crate::chunk_store::ChunkStore;
use crate::meshing::mesh_chunk as mesh_store_chunk;
use crate::types::{ChunkCoord, VoxelCoord};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
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
            store_meshes: HashMap::new(),
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

    pub fn clear_mesh_cache(&mut self) {
        self.store_meshes.clear();
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
