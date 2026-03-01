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
use wgpu::util::DeviceExt;

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
    state_for_chunk: HashMap<crate::types::ChunkCoord, u32>,
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
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk compute shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("compute_meshing.wgsl").into()),
            });

            let entries = [
                bgl_entry(0, true),
                bgl_entry(1, false),
                bgl_entry(2, true),
                bgl_entry(3, true),
                bgl_entry(4, false),
                bgl_entry(5, false),
            ];
            let simulation_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simulation bgl"),
                    entries: &entries,
                });
            let meshing_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meshing bgl"),
                entries: &entries,
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
        page_index: u32,
        current_state: u32,
    ) -> anyhow::Result<DrawIndirectArgs> {
        let t0 = Instant::now();
        let (frontier, _) = build_frontier_with_halo(job.snapshot.center_voxels.as_ref());

        let input = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu input voxels"),
                contents: bytemuck::cast_slice(job.snapshot.center_voxels.as_ref()),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let page_params = device_page_params(job, page_index, frontier.len() as u32, current_state);
        let page_params_buf = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu page params"),
                contents: bytemuck::cast_slice(&page_params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        state.queue.write_buffer(
            &state.frontier,
            0,
            bytemuck::cast_slice(frontier.as_slice()),
        );

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
                    resource: state.frontier.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: page_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.page_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.diagnostics.as_entire_binding(),
                },
            ],
        });

        let meshing_bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshing bg"),
            layout: &self.meshing_bgl,
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
                    resource: state.frontier.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: page_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.page_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.diagnostics.as_entire_binding(),
                },
            ],
        });

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let groups = (frontier.len() as u32).max(1).div_ceil(64);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.simulation_pipeline);
            pass.set_bind_group(0, &simulation_bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);

            pass.set_pipeline(&self.meshing_pipeline);
            pass.set_bind_group(0, &meshing_bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        state.queue.submit(Some(encoder.finish()));

        #[cfg(feature = "gpu-compute")]
        {
            GPU_DISPATCH_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            GPU_TRANSFER_BYTES.fetch_add(
                ((job.snapshot.center_voxels.len() + frontier.len()) * std::mem::size_of::<u32>())
                    as u64,
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

fn build_frontier_with_halo(voxels: &[MaterialId]) -> (Vec<u32>, usize) {
    let mut mark = vec![false; CHUNK_VOLUME];
    let mut frontier = Vec::with_capacity(CHUNK_VOLUME / 2);
    let mut seeds = 0usize;
    for (idx, id) in voxels.iter().enumerate() {
        if *id == EMPTY {
            continue;
        }
        seeds += 1;
        let z = idx / (32 * 32);
        let rem = idx - z * 32 * 32;
        let y = rem / 32;
        let x = rem % 32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if !(0..32).contains(&nx) || !(0..32).contains(&ny) || !(0..32).contains(&nz) {
                        continue;
                    }
                    let nidx = (nx as usize) + (ny as usize) * 32 + (nz as usize) * 32 * 32;
                    if !mark[nidx] {
                        mark[nidx] = true;
                        frontier.push(nidx as u32);
                    }
                }
            }
        }
    }
    if frontier.is_empty() {
        frontier.push(0);
    }
    (frontier, seeds)
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
            let frontier = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk active frontier"),
                size: page_len * std::mem::size_of::<u32>() as u64,
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
        let page_index = if let Some(existing) = atlas.page_for_chunk.get(&job.coord).copied() {
            existing
        } else {
            let page = atlas.next_page;
            if page >= GPU_PAGE_CAPACITY {
                anyhow::bail!(
                    "gpu page atlas exhausted: capacity={} coord=({},{},{})",
                    GPU_PAGE_CAPACITY,
                    job.coord.x,
                    job.coord.y,
                    job.coord.z
                );
            }
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
        atlas.version_for_chunk.insert(job.coord, job.version);
        atlas
            .state_for_chunk
            .insert(job.coord, (current_state + 1) & 1);
        drop(atlas);

        let indirect = if last_version != job.version {
            state
                .runtime
                .run_active_frontier(state, job, page_index, current_state)?
        } else {
            DrawIndirectArgs::default()
        };

        Ok(ComputedChunkArtifacts {
            generated_materials: job.snapshot.center_voxels.to_vec(),
            mesh_indirect: indirect,
        })
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PageParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    state_index: u32,
}

fn device_page_params(
    job: &MeshJob,
    page_index: u32,
    frontier_len: u32,
    state_index: u32,
) -> [PageParams; 1] {
    [PageParams {
        page_index,
        voxel_count: job.snapshot.center_voxels.len() as u32,
        frontier_len,
        state_index,
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
