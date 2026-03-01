use crate::renderer::MeshJob;
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;
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
    density_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    density_bgl: wgpu::BindGroupLayout,
    compact_bgl: wgpu::BindGroupLayout,
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
                    >= (32usize * 32usize * 32usize * std::mem::size_of::<u32>()) as u32
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

            let density_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("density bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let compact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compact bgl"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let density_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("density layout"),
                bind_group_layouts: &[&density_bgl],
                push_constant_ranges: &[],
            });
            let compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compact layout"),
                bind_group_layouts: &[&compact_bgl],
                push_constant_ranges: &[],
            });

            let density_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("density pipeline"),
                    layout: Some(&density_pl),
                    module: &module,
                    entry_point: "density_main",
                });
            let compact_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compact pipeline"),
                    layout: Some(&compact_pl),
                    module: &module,
                    entry_point: "compact_main",
                });

            Some(Self {
                density_pipeline,
                compact_pipeline,
                density_bgl,
                compact_bgl,
            })
        }
    }

    pub(crate) fn run_chunk_job(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
            let material_field = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu material field"),
                size: (volume * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let compacted_voxels = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu compacted voxels"),
                size: (volume * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu indirect args"),
                contents: bytemuck::bytes_of(&DrawIndirectArgs::default()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu chunk params"),
                contents: bytemuck::bytes_of(&ChunkParams {
                    world_min: [
                        job.snapshot.world_min.x,
                        job.snapshot.world_min.y,
                        job.snapshot.world_min.z,
                        0,
                    ],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let density_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("density bg"),
                layout: &self.density_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: material_field.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let compact_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compact bg"),
                layout: &self.compact_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_field.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: compacted_voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indirect.as_entire_binding(),
                    },
                ],
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.density_pipeline);
                pass.set_bind_group(0, &density_bg, &[]);
                let groups = (volume as u32 + 63) / 64;
                pass.dispatch_workgroups(groups, 1, 1);

                pass.set_pipeline(&self.compact_pipeline);
                pass.set_bind_group(0, &compact_bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }

            let material_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material readback"),
                size: (volume * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let indirect_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("indirect readback"),
                size: std::mem::size_of::<DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &material_field,
                0,
                &material_readback,
                0,
                material_readback.size(),
            );
            encoder.copy_buffer_to_buffer(
                &indirect,
                0,
                &indirect_readback,
                0,
                indirect_readback.size(),
            );

            let submission = queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

            let materials_u32 = map_readback::<u32>(device, &material_readback)?;
            let materials: Vec<MaterialId> =
                materials_u32.into_iter().map(|v| v as MaterialId).collect();
            let indirect = map_readback::<DrawIndirectArgs>(device, &indirect_readback)?[0];

            Ok(ComputedChunkArtifacts {
                generated_materials: materials,
                mesh_indirect: indirect,
            })
        }
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

        struct WorkerGpuState {
            device: wgpu::Device,
            queue: wgpu::Queue,
            runtime: GpuComputeRuntime,
        }

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
            Ok(WorkerGpuState {
                device,
                queue,
                runtime,
            })
        });
        let state = state.as_ref().map_err(|e| anyhow::anyhow!(e.to_string()))?;
        state
            .runtime
            .run_chunk_job(&state.device, &state.queue, job)
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ChunkParams {
    world_min: [i32; 4],
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
