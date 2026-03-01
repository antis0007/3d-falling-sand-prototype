use crate::renderer::MeshJob;
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use std::collections::HashMap;
#[cfg(feature = "gpu-compute")]
use std::sync::Mutex;
#[cfg(feature = "gpu-compute")]
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshPipelineBackend {
    Cpu,
    #[cfg(feature = "gpu-compute")]
    Gpu,
}

impl MeshPipelineBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "gpu-compute")]
            Self::Gpu => "gpu",
        }
    }
}

#[cfg(not(feature = "gpu-compute"))]
pub struct GpuComputeRuntime;

#[cfg(feature = "gpu-compute")]
pub struct GpuComputeRuntime {
    simulation_pipeline: wgpu::ComputePipeline,
    meshing_pipeline: wgpu::ComputePipeline,
    simulation_bgl: wgpu::BindGroupLayout,
    meshing_bgl: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu-compute")]
#[derive(Default)]
struct ChunkPageAtlas {
    page_for_chunk: HashMap<crate::types::ChunkCoord, u32>,
    version_for_chunk: HashMap<crate::types::ChunkCoord, u64>,
    next_page: u32,
}

#[cfg(feature = "gpu-compute")]
struct WorkerGpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    runtime: GpuComputeRuntime,
    atlas: Mutex<ChunkPageAtlas>,
    atlas_voxels: wgpu::Buffer,
    page_indirect: wgpu::Buffer,
    frontier: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
}

impl GpuComputeRuntime {
    pub fn runtime_supported(adapter: &wgpu::Adapter) -> bool {
        #[cfg(not(feature = "gpu-compute"))]
        {
            let _ = adapter;
            false
        }

        #[cfg(feature = "gpu-compute")]
        {
            let downlevel = adapter.get_downlevel_capabilities();
            let limits = adapter.limits();
            downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                && limits.max_storage_buffer_binding_size
                    >= (32usize * 32usize * 32usize * std::mem::size_of::<u32>() * 256) as u32
        }
    }

    pub fn new(device: &wgpu::Device) -> Option<Self> {
        #[cfg(not(feature = "gpu-compute"))]
        {
            let _ = device;
            None
        }

        #[cfg(feature = "gpu-compute")]
        {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk compute shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("compute_meshing.wgsl").into()),
            });

            let simulation_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simulation bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let meshing_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meshing bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let simulation_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("simulation layout"),
                bind_group_layouts: &[&simulation_bgl],
                push_constant_ranges: &[],
            });
            let meshing_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("meshing layout"),
                bind_group_layouts: &[&meshing_bgl],
                push_constant_ranges: &[],
            });

            let simulation_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("simulation pipeline"),
                    layout: Some(&simulation_pl),
                    module: &module,
                    entry_point: "simulation_main",
                });
            let meshing_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing pipeline"),
                    layout: Some(&meshing_pl),
                    module: &module,
                    entry_point: "meshing_main",
                });

            Some(Self {
                simulation_pipeline,
                meshing_pipeline,
                simulation_bgl,
                meshing_bgl,
            })
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn run_active_frontier(
        &self,
        state: &WorkerGpuState,
        job: &MeshJob,
    ) -> anyhow::Result<ComputedChunkArtifacts> {
        #[cfg(not(feature = "gpu-compute"))]
        {
            let _ = (device, queue, job);
            anyhow::bail!("gpu compute feature disabled")
        }

        #[cfg(feature = "gpu-compute")]
        {
            let volume = job.snapshot.center_voxels.len();
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu input voxels"),
                contents: bytemuck::cast_slice(job.snapshot.center_voxels.as_ref()),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let frontier_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu active frontier"),
                contents: bytemuck::cast_slice(&[page_index]),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let input = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu input voxels"),
                contents: bytemuck::cast_slice(&job.snapshot.voxels),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let simulation_bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simulation bg"),
            layout: &self.simulation_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.atlas_voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frontier_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: page_params_buf.as_entire_binding(),
                },
            ],
        });

        let meshing_bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshing bg"),
            layout: &self.meshing_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.atlas_voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frontier_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.page_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.diagnostics.as_entire_binding(),
                },
            ],
        });

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.simulation_pipeline);
            pass.set_bind_group(0, &simulation_bg, &[]);
            let groups = (job.snapshot.voxels.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);

            pass.set_pipeline(&self.meshing_pipeline);
            pass.set_bind_group(0, &meshing_bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let diag_readback = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diag readback"),
            size: std::mem::size_of::<DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let page_offset = (page_index as u64) * std::mem::size_of::<DrawIndirectArgs>() as u64;
        encoder.copy_buffer_to_buffer(
            &state.page_indirect,
            page_offset,
            &diag_readback,
            0,
            diag_readback.size(),
        );

        let submission = state.queue.submit(Some(encoder.finish()));
        state
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        Ok(map_readback::<DrawIndirectArgs>(&state.device, &diag_readback)?[0])
    }
}

pub(crate) fn run_chunk_job_on_worker(job: &MeshJob) -> anyhow::Result<ComputedChunkArtifacts> {
    #[cfg(not(feature = "gpu-compute"))]
    {
        let _ = job;
        anyhow::bail!("gpu compute feature disabled")
    }

    #[cfg(feature = "gpu-compute")]
    {
        use std::sync::OnceLock;

        static STATE: OnceLock<anyhow::Result<WorkerGpuState>> = OnceLock::new();
        let state = STATE.get_or_init(|| {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .context("compute adapter")?;
            let (device, queue) = pollster::block_on(
                adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
            )?;
            let runtime = GpuComputeRuntime::new(&device).context("compute runtime")?;
            let page_capacity = 256u64;
            let page_len = (32u64 * 32u64 * 32u64 * 27u64) as u64;
            let atlas_voxels = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk atlas voxels"),
                size: page_capacity * page_len * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let page_indirect = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk page indirect"),
                size: page_capacity * std::mem::size_of::<DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let frontier = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk active frontier"),
                size: page_capacity * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk diagnostics"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Ok(WorkerGpuState {
                device,
                queue,
                runtime,
                atlas: Mutex::new(ChunkPageAtlas::default()),
                atlas_voxels,
                page_indirect,
                frontier,
                diagnostics,
            })
        });
        let state = state.as_ref().map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut atlas = state.atlas.lock().expect("atlas lock");
        let page_index = *atlas.page_for_chunk.entry(job.coord).or_insert_with(|| {
            let page = atlas.next_page;
            atlas.next_page = atlas.next_page.saturating_add(1);
            page
        });
        let last_version = atlas
            .version_for_chunk
            .get(&job.coord)
            .copied()
            .unwrap_or(0);
        atlas.version_for_chunk.insert(job.coord, job.version);
        drop(atlas);

        let indirect = if last_version != job.version {
            state.runtime.run_active_frontier(state, job, page_index)?
        } else {
            DrawIndirectArgs::default()
        };

        Ok(ComputedChunkArtifacts {
            generated_materials: job.snapshot.voxels.clone(),
            mesh_indirect: indirect,
        })
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PageParams {
    page_index: u32,
    voxel_count: u32,
    _pad0: u32,
    _pad1: u32,
}

fn device_page_params(job: &MeshJob, page_index: u32) -> [PageParams; 1] {
    [PageParams {
        page_index,
        voxel_count: job.snapshot.voxels.len() as u32,
        _pad0: 0,
        _pad1: 0,
    }]
}

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

pub struct ComputedChunkArtifacts {
    pub generated_materials: Vec<MaterialId>,
    pub mesh_indirect: DrawIndirectArgs,
}

impl ComputedChunkArtifacts {
    pub fn has_any_surface(&self) -> bool {
        self.mesh_indirect.vertex_count > 0
    }
}

fn map_readback<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> anyhow::Result<Vec<T>> {
    use std::sync::mpsc;
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().context("map callback dropped")??;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}

pub(crate) fn cpu_generate_material_field(job: &MeshJob) -> ComputedChunkArtifacts {
    let mut out = vec![EMPTY; job.snapshot.center_voxels.len()];
    out.copy_from_slice(job.snapshot.center_voxels.as_ref());
    let surface = out.iter().filter(|v| **v != EMPTY).count() as u32;
    ComputedChunkArtifacts {
        generated_materials: out,
        mesh_indirect: DrawIndirectArgs {
            vertex_count: surface.saturating_mul(6),
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
    }
}

pub(crate) fn rebuilt_snapshot_from_materials(
    job: &MeshJob,
    materials: Vec<MaterialId>,
) -> crate::renderer::ChunkSnapshot {
    job.snapshot.with_center_materials(materials)
}
