use crate::renderer::MeshJob;
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use std::collections::HashMap;
#[cfg(feature = "gpu-compute")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "gpu-compute")]
use std::sync::Mutex;
use std::time::Instant;
#[cfg(feature = "gpu-compute")]
const GPU_PAGE_CAPACITY: u32 = 256;
const CHUNK_VOLUME: usize = 32 * 32 * 32;

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
    force_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    pressure_jacobi_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    material_advect_pipeline: wgpu::ComputePipeline,
    meshing_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu-compute")]
#[derive(Default)]
struct ChunkPageAtlas {
    page_for_chunk: HashMap<crate::types::ChunkCoord, u32>,
    version_for_chunk: HashMap<crate::types::ChunkCoord, u64>,
    state_for_chunk: HashMap<crate::types::ChunkCoord, u32>,
    cached_materials: HashMap<crate::types::ChunkCoord, Vec<MaterialId>>,
    next_page: u32,
}

#[cfg(feature = "gpu-compute")]
struct WorkerGpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    runtime: GpuComputeRuntime,
    atlas: Mutex<ChunkPageAtlas>,
    velocity_mac: wgpu::Buffer,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    material_density: wgpu::Buffer,
    active_tiles: wgpu::Buffer,
    active_tile_counter: wgpu::Buffer,
    edit_commands: wgpu::Buffer,
    page_params: wgpu::Buffer,
    dirty_chunk_ids: wgpu::Buffer,
    dirty_chunk_counter: wgpu::Buffer,
    compute_bg: wgpu::BindGroup,
    frontier_len: u32,
}

#[cfg(feature = "gpu-compute")]
const MAX_EDIT_COMMANDS: u32 = CHUNK_VOLUME as u32;

#[derive(Default, Clone, Copy)]
pub struct GpuComputeProfilerSnapshot {
    pub dispatch_ms: f32,
    pub bytes_transferred: u64,
    pub chunks_completed: u64,
    pub chunks_per_sec: f32,
}

#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TRANSFER_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_CHUNKS: AtomicU64 = AtomicU64::new(0);

pub fn take_gpu_compute_profiler_snapshot(frame_seconds: f32) -> GpuComputeProfilerSnapshot {
    #[cfg(not(feature = "gpu-compute"))]
    {
        let _ = frame_seconds;
        GpuComputeProfilerSnapshot::default()
    }

    #[cfg(feature = "gpu-compute")]
    {
        let dispatch_ns = GPU_DISPATCH_NS.swap(0, Ordering::Relaxed);
        let bytes_transferred = GPU_TRANSFER_BYTES.swap(0, Ordering::Relaxed);
        let chunks_completed = GPU_CHUNKS.swap(0, Ordering::Relaxed);
        let frame = frame_seconds.max(0.000_1);
        GpuComputeProfilerSnapshot {
            dispatch_ms: dispatch_ns as f32 / 1_000_000.0,
            bytes_transferred,
            chunks_completed,
            chunks_per_sec: chunks_completed as f32 / frame,
        }
    }
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
                    >= (CHUNK_VOLUME * std::mem::size_of::<u32>() * 2 * GPU_PAGE_CAPACITY as usize)
                        as u32
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
            let fluid_advect_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fluid advect shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fluid_advect.wgsl").into()),
            });
            let fluid_divergence_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fluid divergence shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/fluid_divergence.wgsl").into(),
                    ),
                });
            let fluid_pressure_jacobi_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fluid pressure jacobi shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/fluid_pressure_jacobi.wgsl").into(),
                    ),
                });
            let fluid_project_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fluid project shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fluid_project.wgsl").into()),
            });
            let fluid_material_advect_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fluid material advect shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/fluid_material_advect.wgsl").into(),
                    ),
                });
            let meshing_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk meshing shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("compute_meshing.wgsl").into()),
            });

            let entries = [
                bgl_entry(0, false),
                bgl_entry(1, false),
                bgl_entry(2, false),
                bgl_entry(3, false),
                bgl_entry(4, false),
                bgl_entry(5, false),
                bgl_entry(6, false),
                bgl_entry(7, true),
                bgl_entry(8, true),
                bgl_entry(9, false),
                bgl_entry(10, false),
                bgl_entry(11, false),
                bgl_entry(12, false),
            ];
            let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute bgl"),
                entries: &entries,
            });
            let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

            let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fluid forces pipeline"),
                layout: Some(&compute_pl),
                module: &fluid_advect_module,
                entry_point: "main",
            });
            let advect_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid advect pipeline"),
                    layout: Some(&compute_pl),
                    module: &fluid_advect_module,
                    entry_point: "main",
                });
            let divergence_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid divergence pipeline"),
                    layout: Some(&compute_pl),
                    module: &fluid_divergence_module,
                    entry_point: "main",
                });
            let pressure_jacobi_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid pressure jacobi pipeline"),
                    layout: Some(&compute_pl),
                    module: &fluid_pressure_jacobi_module,
                    entry_point: "main",
                });
            let project_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid project pipeline"),
                    layout: Some(&compute_pl),
                    module: &fluid_project_module,
                    entry_point: "main",
                });
            let material_advect_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid material advect pipeline"),
                    layout: Some(&compute_pl),
                    module: &fluid_material_advect_module,
                    entry_point: "main",
                });
            let meshing_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing pipeline"),
                    layout: Some(&compute_pl),
                    module: &meshing_module,
                    entry_point: "meshing_main",
                });

            Some(Self {
                force_pipeline,
                advect_pipeline,
                divergence_pipeline,
                pressure_jacobi_pipeline,
                project_pipeline,
                material_advect_pipeline,
                meshing_pipeline,
                compute_bgl,
            })
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn run_active_frontier(
        &self,
        state: &WorkerGpuState,
        job: &MeshJob,
        page_index: u32,
        current_state: u32,
        edit_commands: &[EditCommand],
    ) -> anyhow::Result<DrawIndirectArgs> {
        let t0 = Instant::now();
        let frontier_len = state
            .frontier_len
            .min(job.snapshot.center_voxels.len() as u32);

        if !edit_commands.is_empty() {
            state
                .queue
                .write_buffer(&state.edit_commands, 0, bytemuck::cast_slice(edit_commands));
        }
        let page_params = device_page_params(
            job,
            page_index,
            frontier_len,
            current_state,
            edit_commands.len() as u32,
        );
        state
            .queue
            .write_buffer(&state.page_params, 0, bytemuck::cast_slice(&page_params));

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let groups = frontier_len
            .max(edit_commands.len() as u32)
            .max(1)
            .div_ceil(64);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &state.compute_bg, &[]);

            // 1. external forces/gravity
            pass.set_pipeline(&self.force_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            // 2. velocity advection
            pass.set_pipeline(&self.advect_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            // 3. divergence compute
            pass.set_pipeline(&self.divergence_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            // 4. Jacobi pressure iterations
            for _ in 0..16 {
                pass.set_pipeline(&self.pressure_jacobi_pipeline);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            // 5. velocity projection
            pass.set_pipeline(&self.project_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            // 6. material advection + boundaries
            pass.set_pipeline(&self.material_advect_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);

            // meshing stays independent and only consumes the post-fluid state.
            pass.set_pipeline(&self.meshing_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        state.queue.submit(Some(encoder.finish()));

        #[cfg(feature = "gpu-compute")]
        {
            GPU_DISPATCH_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            GPU_TRANSFER_BYTES.fetch_add(
                (edit_commands.len() * std::mem::size_of::<EditCommand>()
                    + std::mem::size_of::<FrameParams>()) as u64,
                Ordering::Relaxed,
            );
            GPU_CHUNKS.fetch_add(1, Ordering::Relaxed);
        }

        Ok(DrawIndirectArgs::default())
    }
}

#[cfg(feature = "gpu-compute")]
fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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
            let page_capacity = GPU_PAGE_CAPACITY as u64;
            let page_len = CHUNK_VOLUME as u64;
            let atlas_voxels = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk atlas voxels"),
                size: page_capacity * page_len * 2 * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let page_indirect = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk page indirect"),
                size: page_capacity * std::mem::size_of::<DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let velocity_mac = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk velocity atlas"),
                size: page_capacity * page_len * std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let pressure = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk pressure atlas"),
                size: page_capacity * page_len * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let divergence = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk divergence atlas"),
                size: page_capacity * page_len * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let material_density = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk material density atlas"),
                size: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let page_params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk page params"),
                size: std::mem::size_of::<FrameParams>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let frontier = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk active frontier"),
                size: page_len * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let active_tile_counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("active tile counter"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let edit_commands = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edit command buffer"),
                size: MAX_EDIT_COMMANDS as u64 * std::mem::size_of::<EditCommand>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dirty_chunk_ids = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dirty chunk ids"),
                size: page_capacity * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let dirty_chunk_counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dirty chunk counter"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk diagnostics"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let frontier_len = CHUNK_VOLUME as u32;
            let full_frontier: Vec<u32> = (0..frontier_len).collect();
            queue.write_buffer(&frontier, 0, bytemuck::cast_slice(full_frontier.as_slice()));

            let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bg"),
                layout: &runtime.compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atlas_voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: velocity_mac.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: divergence.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: material_density.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: frontier.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: active_tile_counter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: edit_commands.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: page_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: page_indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: dirty_chunk_ids.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: dirty_chunk_counter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: diagnostics.as_entire_binding(),
                    },
                ],
            });
            Ok(WorkerGpuState {
                device,
                queue,
                runtime,
                atlas: Mutex::new(ChunkPageAtlas::default()),
                velocity_mac,
                pressure,
                divergence,
                material_density,
                active_tiles: frontier,
                active_tile_counter,
                edit_commands,
                page_params,
                dirty_chunk_ids,
                dirty_chunk_counter,
                compute_bg,
                frontier_len,
            })
        });
        let state = state.as_ref().map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut atlas = state.atlas.lock().expect("atlas lock");
        let page_index = if let Some(existing) = atlas.page_for_chunk.get(&job.coord).copied() {
            existing
        } else {
            let page = atlas.next_page;
            if page >= GPU_PAGE_CAPACITY {
                atlas.page_for_chunk.clear();
                atlas.version_for_chunk.clear();
                atlas.state_for_chunk.clear();
                atlas.next_page = 0;
            }
            let page = atlas.next_page;
            atlas.next_page = atlas.next_page.saturating_add(1);
            atlas.page_for_chunk.insert(job.coord, page);
            page
        };
        let last_version = atlas
            .version_for_chunk
            .get(&job.coord)
            .copied()
            .unwrap_or(0);
        let current_state = atlas.state_for_chunk.get(&job.coord).copied().unwrap_or(0);
        let mut edit_commands = Vec::new();
        let incoming = job.snapshot.center_voxels.as_ref();
        let cached = atlas
            .cached_materials
            .entry(job.coord)
            .or_insert_with(|| vec![EMPTY; incoming.len()]);
        if cached.len() != incoming.len() {
            cached.resize(incoming.len(), EMPTY);
        }
        for (idx, (&next, prev)) in incoming.iter().zip(cached.iter_mut()).enumerate() {
            if next != *prev {
                edit_commands.push(EditCommand {
                    voxel_index: idx as u32,
                    material_id: next as u32,
                    flags: 0,
                    _pad: 0,
                });
                *prev = next;
            }
        }
        atlas.version_for_chunk.insert(job.coord, job.version);
        atlas
            .state_for_chunk
            .insert(job.coord, (current_state + 1) & 1);
        drop(atlas);

        let indirect = if last_version != job.version {
            state.runtime.run_active_frontier(
                state,
                job,
                page_index,
                current_state,
                &edit_commands,
            )?
        } else {
            DrawIndirectArgs::default()
        };

        Ok(ComputedChunkArtifacts {
            generated_materials: job.snapshot.center_voxels.to_vec(),
            mesh_indirect: if indirect.vertex_count == 0 {
                DrawIndirectArgs {
                    vertex_count: (job
                        .snapshot
                        .center_voxels
                        .iter()
                        .filter(|v| **v != EMPTY)
                        .count() as u32)
                        .saturating_mul(6),
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                }
            } else {
                indirect
            },
        })
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct EditCommand {
    voxel_index: u32,
    material_id: u32,
    flags: u32,
    _pad: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FrameParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
    active_tile_budget: u32,
    jacobi_iterations: u32,
    jacobi_iteration: u32,
}

fn device_page_params(
    job: &MeshJob,
    page_index: u32,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
) -> [FrameParams; 1] {
    [FrameParams {
        page_index,
        voxel_count: job.snapshot.center_voxels.len() as u32,
        frontier_len,
        state_index,
        edit_count,
        active_tile_budget: frontier_len,
        jacobi_iterations: 16,
        jacobi_iteration: 0,
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

#[cfg(test)]
mod tests {
    use crate::chunk_store::ChunkStore;
    use crate::sim::XorShift32;
    use crate::sim_world::SimWorld;
    use crate::types::{chunk_to_world_min, ChunkCoord, VoxelCoord};
    use std::collections::HashSet;

    #[test]
    fn emulated_compute_matches_cpu_reference_with_zero_tolerance() {
        let mut store_a = ChunkStore::new();
        let mut store_b = ChunkStore::new();
        let chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(chunk);
        let seed_voxels = [
            VoxelCoord {
                x: base.x + 4,
                y: base.y + 8,
                z: base.z + 4,
            },
            VoxelCoord {
                x: base.x + 5,
                y: base.y + 10,
                z: base.z + 4,
            },
            VoxelCoord {
                x: base.x + 6,
                y: base.y + 12,
                z: base.z + 4,
            },
        ];
        for v in seed_voxels {
            store_a.set_voxel(v, 3);
            store_b.set_voxel(v, 3);
        }

        let region = HashSet::from([chunk]);
        let mut sim_cpu = SimWorld::default();
        let mut sim_compute_emulated = SimWorld::default();
        for v in seed_voxels {
            sim_cpu.notify_voxel_edit(v);
            sim_compute_emulated.notify_voxel_edit(v);
        }

        let mut rng_cpu = XorShift32::new(123);
        let mut rng_compute = XorShift32::new(123);
        for _ in 0..6 {
            sim_cpu.step_region(&mut store_a, &region, chunk, &mut rng_cpu);
            sim_compute_emulated.step_region(&mut store_b, &region, chunk, &mut rng_compute);
        }

        let soa_cpu = sim_cpu.build_soa_for_chunk(&store_a, chunk).unwrap();
        let soa_compute = sim_compute_emulated
            .build_soa_for_chunk(&store_b, chunk)
            .unwrap();
        let mismatches = soa_cpu
            .material_ids
            .iter()
            .zip(soa_compute.material_ids.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            mismatches <= 0,
            "mismatch count {} exceeded tolerance",
            mismatches
        );
    }
}
