use crate::renderer::{mesh_chunk_snapshot, ChunkMeshArtifact, MeshJob};
use crate::types::{ChunkCoord, GpuPageIndex};
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
#[cfg(feature = "gpu-compute")]
const COMPUTE_STORAGE_BINDING_COUNT: u32 = 13;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u64 = CHUNK_VOLUME as u64;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u64 = (CHUNK_VOLUME as u64) * 6;

#[cfg(feature = "gpu-compute")]
pub const fn gpu_page_capacity() -> u32 {
    GPU_PAGE_CAPACITY
}

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
    simulation_bgl: wgpu::BindGroupLayout,
    meshing_bgl: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu-compute")]
struct SimulationBindResources<'a> {
    atlas_voxels: &'a wgpu::Buffer,
    velocity_mac: &'a wgpu::Buffer,
    pressure: &'a wgpu::Buffer,
    divergence: &'a wgpu::Buffer,
    material_density: &'a wgpu::Buffer,
    active_tiles: &'a wgpu::Buffer,
    active_tile_counter: &'a wgpu::Buffer,
    edit_commands: &'a wgpu::Buffer,
    page_params: &'a wgpu::Buffer,
    diagnostics: &'a wgpu::Buffer,
}

#[cfg(feature = "gpu-compute")]
struct MeshingBindResources<'a> {
    atlas_voxels: &'a wgpu::Buffer,
    active_tiles: &'a wgpu::Buffer,
    page_params: &'a wgpu::Buffer,
    page_indirect: &'a wgpu::Buffer,
    dirty_page_indices: &'a wgpu::Buffer,
    dirty_page_counter: &'a wgpu::Buffer,
    diagnostics: &'a wgpu::Buffer,
    chunk_vertex_buffer: &'a wgpu::Buffer,
    chunk_index_buffer: &'a wgpu::Buffer,
    draw_indirect_buffer: &'a wgpu::Buffer,
    mesh_meta_buffer: &'a wgpu::Buffer,
    vertex_counter: &'a wgpu::Buffer,
    index_counter: &'a wgpu::Buffer,
}

#[cfg(feature = "gpu-compute")]
#[derive(Default)]
struct ChunkPageAtlas {
    page_for_chunk: HashMap<ChunkCoord, GpuPageIndex>,
    chunk_for_page: HashMap<GpuPageIndex, ChunkCoord>,
    in_flight_jobs_for_page: HashMap<GpuPageIndex, u32>,
    version_for_chunk: HashMap<ChunkCoord, u64>,
    state_for_chunk: HashMap<ChunkCoord, u32>,
    frontier_len_for_chunk: HashMap<ChunkCoord, u32>,
    tick_for_chunk: HashMap<ChunkCoord, u32>,
    diagnostics_for_chunk: HashMap<ChunkCoord, ChunkSimulationDiagnostics>,
    cached_materials: HashMap<ChunkCoord, Vec<MaterialId>>,
    next_page: GpuPageIndex,
}

#[cfg(feature = "gpu-compute")]
impl ChunkPageAtlas {
    fn page_for_chunk_or_allocate(
        &mut self,
        chunk: ChunkCoord,
    ) -> anyhow::Result<(GpuPageIndex, bool)> {
        if let Some(existing) = self.page_for_chunk.get(&chunk).copied() {
            return Ok((existing, false));
        }

        if self.next_page.0 < GPU_PAGE_CAPACITY {
            let page = self.next_page;
            self.next_page = GpuPageIndex(self.next_page.0.saturating_add(1));
            self.page_for_chunk.insert(chunk, page);
            self.chunk_for_page.insert(page, chunk);
            return Ok((page, true));
        }

        let page = self.evictable_page().with_context(|| {
            format!("no reusable gpu atlas pages available for chunk {chunk:?}")
        })?;
        self.evict_page(page);
        self.page_for_chunk.insert(chunk, page);
        self.chunk_for_page.insert(page, chunk);
        Ok((page, true))
    }

    fn resolve_chunk(&self, page_index: GpuPageIndex) -> Option<ChunkCoord> {
        self.chunk_for_page.get(&page_index).copied()
    }

    fn assert_page_for_chunk(&self, chunk: ChunkCoord, page_index: GpuPageIndex) {
        let resolved = self
            .resolve_chunk(page_index)
            .expect("gpu page must resolve to a chunk");
        assert_eq!(resolved, chunk, "gpu page/chunk mapping mismatch");
    }

    fn acquire_page_for_job(&mut self, chunk: ChunkCoord, page_index: GpuPageIndex) {
        self.assert_page_for_chunk(chunk, page_index);
        *self.in_flight_jobs_for_page.entry(page_index).or_insert(0) += 1;
    }

    fn release_page_from_job(&mut self, page_index: GpuPageIndex) {
        if let Some(in_flight) = self.in_flight_jobs_for_page.get_mut(&page_index) {
            *in_flight = in_flight.saturating_sub(1);
            if *in_flight == 0 {
                self.in_flight_jobs_for_page.remove(&page_index);
            }
        }
    }

    fn evictable_page(&self) -> Option<GpuPageIndex> {
        self.chunk_for_page
            .keys()
            .copied()
            .find(|page| self.in_flight_jobs_for_page.get(page).copied().unwrap_or(0) == 0)
    }

    fn evict_page(&mut self, page_index: GpuPageIndex) {
        debug_assert_eq!(
            self.in_flight_jobs_for_page
                .get(&page_index)
                .copied()
                .unwrap_or(0),
            0,
            "attempted to evict an in-flight gpu page"
        );
        if let Some(chunk) = self.chunk_for_page.remove(&page_index) {
            self.page_for_chunk.remove(&chunk);
            self.version_for_chunk.remove(&chunk);
            self.state_for_chunk.remove(&chunk);
            self.frontier_len_for_chunk.remove(&chunk);
            self.tick_for_chunk.remove(&chunk);
            self.diagnostics_for_chunk.remove(&chunk);
            self.cached_materials.remove(&chunk);
        }
    }
}

#[cfg(feature = "gpu-compute")]
struct WorkerGpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    runtime: GpuComputeRuntime,
    gpu_job_lock: Mutex<()>,
    atlas: Mutex<ChunkPageAtlas>,
    atlas_voxels: wgpu::Buffer,
    velocity_mac: wgpu::Buffer,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    material_density: wgpu::Buffer,
    active_tiles: wgpu::Buffer,
    active_tile_counter: wgpu::Buffer,
    edit_commands: wgpu::Buffer,
    page_params: wgpu::Buffer,
    page_indirect: wgpu::Buffer,
    dirty_page_indices: wgpu::Buffer,
    dirty_page_counter: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    chunk_vertex_buffer: wgpu::Buffer,
    chunk_index_buffer: wgpu::Buffer,
    draw_indirect_buffer: wgpu::Buffer,
    mesh_meta_buffer: wgpu::Buffer,
    vertex_counter: wgpu::Buffer,
    index_counter: wgpu::Buffer,
    simulation_bg: wgpu::BindGroup,
    meshing_bg: wgpu::BindGroup,
    runtime_config: GpuSimulationRuntimeConfig,
}

#[cfg(feature = "gpu-compute")]
fn clear_page_buffers(state: &WorkerGpuState, page_index: GpuPageIndex) {
    let voxel_words = CHUNK_VOLUME * 2;
    let atlas_byte_offset =
        page_index.0 as u64 * voxel_words as u64 * std::mem::size_of::<u32>() as u64;
    let atlas_zeros = vec![0u32; voxel_words];
    state.queue.write_buffer(
        &state.atlas_voxels,
        atlas_byte_offset,
        bytemuck::cast_slice(&atlas_zeros),
    );

    let velocity_offset =
        page_index.0 as u64 * CHUNK_VOLUME as u64 * std::mem::size_of::<[f32; 4]>() as u64;
    let velocity_zeros = vec![[0.0f32; 4]; CHUNK_VOLUME];
    state.queue.write_buffer(
        &state.velocity_mac,
        velocity_offset,
        bytemuck::cast_slice(&velocity_zeros),
    );

    let scalar_offset =
        page_index.0 as u64 * CHUNK_VOLUME as u64 * std::mem::size_of::<f32>() as u64;
    let scalar_zeros = vec![0.0f32; CHUNK_VOLUME];
    state.queue.write_buffer(
        &state.pressure,
        scalar_offset,
        bytemuck::cast_slice(&scalar_zeros),
    );
    state.queue.write_buffer(
        &state.divergence,
        scalar_offset,
        bytemuck::cast_slice(&scalar_zeros),
    );

    let density_offset =
        page_index.0 as u64 * CHUNK_VOLUME as u64 * 2 * std::mem::size_of::<f32>() as u64;
    let density_zeros = vec![0.0f32; CHUNK_VOLUME * 2];
    state.queue.write_buffer(
        &state.material_density,
        density_offset,
        bytemuck::cast_slice(&density_zeros),
    );
}

#[cfg(feature = "gpu-compute")]
fn clear_job_scratch_buffers(state: &WorkerGpuState) {
    let zero_u32x4 = [0u32; 4];

    state.queue.write_buffer(
        &state.active_tile_counter,
        0,
        bytemuck::cast_slice(&zero_u32x4),
    );
    state.queue.write_buffer(
        &state.dirty_page_counter,
        0,
        bytemuck::cast_slice(&zero_u32x4),
    );
    state
        .queue
        .write_buffer(&state.diagnostics, 0, bytemuck::cast_slice(&zero_u32x4));

    let active_tile_zeros = vec![0u32; CHUNK_VOLUME];
    state.queue.write_buffer(
        &state.active_tiles,
        0,
        bytemuck::cast_slice(&active_tile_zeros),
    );

    let dirty_page_zeros = vec![0u32; GPU_PAGE_CAPACITY as usize];
    state.queue.write_buffer(
        &state.dirty_page_indices,
        0,
        bytemuck::cast_slice(&dirty_page_zeros),
    );
}

#[cfg(feature = "gpu-compute")]
fn clear_meshing_outputs_for_page(state: &WorkerGpuState, page_index: GpuPageIndex) {
    let zero_indirect = DrawIndirectArgs::default();
    let indirect_stride = std::mem::size_of::<DrawIndirectArgs>() as u64;
    let indirect_offset = page_index.0 as u64 * indirect_stride;

    state.queue.write_buffer(
        &state.page_indirect,
        indirect_offset,
        bytemuck::bytes_of(&zero_indirect),
    );
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy)]
struct GpuSimulationRuntimeConfig {
    max_jacobi_iterations: u32,
}

#[cfg(feature = "gpu-compute")]
impl Default for GpuSimulationRuntimeConfig {
    fn default() -> Self {
        Self {
            max_jacobi_iterations: 16,
        }
    }
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone)]
struct SimulationJob {
    chunk_coord: ChunkCoord,
    materials: Vec<MaterialId>,
    active_frontier_count: u32,
    simulation_tick: u32,
}

#[derive(Clone, Copy, Default)]
pub struct ChunkSimulationDiagnostics {
    pub changed_voxels: u32,
    pub dropped_frontier_writes: u32,
    pub cross_border_attempts: u32,
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
                && limits.max_storage_buffers_per_shader_stage >= COMPUTE_STORAGE_BINDING_COUNT
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
            let generated_material_ids_wgsl =
                include_str!(concat!(env!("OUT_DIR"), "/material_ids.wgsl"));
            let fluid_advect_source = format!(
                "{}\n{}",
                generated_material_ids_wgsl,
                include_str!("shaders/fluid_advect.wgsl")
            );
            let fluid_material_advect_source = format!(
                "{}\n{}",
                generated_material_ids_wgsl,
                include_str!("shaders/fluid_material_advect.wgsl")
            );
            let fluid_advect_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fluid advect shader"),
                source: wgpu::ShaderSource::Wgsl(fluid_advect_source.into()),
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
                    source: wgpu::ShaderSource::Wgsl(fluid_material_advect_source.into()),
                });
            let meshing_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk meshing shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/meshing.wgsl").into()),
            });

            let simulation_entries = [
                bgl_entry(0, false),
                bgl_entry(1, false),
                bgl_entry(2, false),
                bgl_entry(3, false),
                bgl_entry(4, false),
                bgl_entry(5, false),
                bgl_entry(6, false),
                bgl_entry(7, true),
                bgl_entry(8, true),
                bgl_entry(12, false),
            ];
            let simulation_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simulation compute bgl"),
                    entries: &simulation_entries,
                });
            let meshing_entries = [
                bgl_entry(0, false),
                bgl_entry(5, false),
                bgl_entry(8, true),
                bgl_entry(9, false),
                bgl_entry(10, false),
                bgl_entry(11, false),
                bgl_entry(12, false),
                bgl_entry(13, false),
                bgl_entry(14, false),
                bgl_entry(15, false),
                bgl_entry(16, false),
                bgl_entry(17, false),
                bgl_entry(18, false),
            ];
            let meshing_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meshing compute bgl"),
                entries: &meshing_entries,
            });
            let simulation_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("simulation compute layout"),
                bind_group_layouts: &[&simulation_bgl],
                push_constant_ranges: &[],
            });
            let meshing_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("meshing compute layout"),
                bind_group_layouts: &[&meshing_bgl],
                push_constant_ranges: &[],
            });

            let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fluid forces pipeline"),
                layout: Some(&simulation_pl),
                module: &fluid_advect_module,
                entry_point: "main",
            });
            let advect_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid advect pipeline"),
                    layout: Some(&simulation_pl),
                    module: &fluid_advect_module,
                    entry_point: "main",
                });
            let divergence_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid divergence pipeline"),
                    layout: Some(&simulation_pl),
                    module: &fluid_divergence_module,
                    entry_point: "main",
                });
            let pressure_jacobi_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid pressure jacobi pipeline"),
                    layout: Some(&simulation_pl),
                    module: &fluid_pressure_jacobi_module,
                    entry_point: "main",
                });
            let project_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid project pipeline"),
                    layout: Some(&simulation_pl),
                    module: &fluid_project_module,
                    entry_point: "main",
                });
            let material_advect_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fluid material advect pipeline"),
                    layout: Some(&simulation_pl),
                    module: &fluid_material_advect_module,
                    entry_point: "main",
                });
            let meshing_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing pipeline"),
                    layout: Some(&meshing_pl),
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
                simulation_bgl,
                meshing_bgl,
            })
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn run_active_frontier(
        &self,
        state: &WorkerGpuState,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        edit_commands: &[EditCommand],
    ) -> anyhow::Result<()> {
        let t0 = Instant::now();
        let frontier_len = sim_job
            .active_frontier_count
            .min(sim_job.materials.len() as u32);

        if !edit_commands.is_empty() {
            state
                .queue
                .write_buffer(&state.edit_commands, 0, bytemuck::cast_slice(edit_commands));
        }
        let page_params = device_page_params(
            sim_job,
            page_index,
            frontier_len,
            current_state,
            edit_commands.len() as u32,
            state.runtime_config,
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
            pass.set_bind_group(0, &state.simulation_bg, &[]);

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
            for _ in 0..state.runtime_config.max_jacobi_iterations {
                pass.set_pipeline(&self.pressure_jacobi_pipeline);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            // 5. velocity projection
            pass.set_pipeline(&self.project_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            // 6. material advection + boundaries
            pass.set_pipeline(&self.material_advect_pipeline);
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

        Ok(())
    }
    fn run_meshing_dispatch(
        &self,
        state: &WorkerGpuState,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
    ) -> anyhow::Result<DrawIndirectArgs> {
        clear_meshing_outputs_for_page(state, page_index);

        let page_params = device_page_params(
            sim_job,
            page_index,
            sim_job.active_frontier_count,
            current_state,
            0,
            state.runtime_config,
        );
        state
            .queue
            .write_buffer(&state.page_params, 0, bytemuck::cast_slice(&page_params));

        let groups = sim_job.active_frontier_count.max(1).div_ceil(64);
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &state.meshing_bg, &[]);
            pass.set_pipeline(&self.meshing_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        state.queue.submit(Some(encoder.finish()));
        Ok(DrawIndirectArgs::default())
    }
}

#[cfg(feature = "gpu-compute")]
fn readback_page_materials(
    state: &WorkerGpuState,
    page_index: GpuPageIndex,
    state_index: u32,
    voxel_count: usize,
) -> anyhow::Result<Vec<MaterialId>> {
    let atlas_offset_voxels = (page_index.0 as u64 * CHUNK_VOLUME as u64 * 2)
        + (state_index as u64 * CHUNK_VOLUME as u64);
    let byte_offset = atlas_offset_voxels * std::mem::size_of::<u32>() as u64;
    let byte_len = voxel_count as u64 * std::mem::size_of::<u32>() as u64;
    let readback = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chunk materials readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&state.atlas_voxels, byte_offset, &readback, 0, byte_len);
    state.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    state.device.poll(wgpu::Maintain::Wait);
    rx.recv().context("gpu readback completion")??;

    let bytes = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&bytes);
    let materials = words.iter().map(|m| *m as MaterialId).collect();
    drop(bytes);
    readback.unmap();
    Ok(materials)
}

#[cfg(feature = "gpu-compute")]
impl GpuComputeRuntime {
    fn create_simulation_bind_group(
        &self,
        device: &wgpu::Device,
        resources: SimulationBindResources<'_>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simulation compute bg"),
            layout: &self.simulation_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.atlas_voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.velocity_mac.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pressure.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.divergence.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.material_density.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.active_tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.active_tile_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: resources.edit_commands.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: resources.page_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: resources.diagnostics.as_entire_binding(),
                },
            ],
        })
    }

    fn create_meshing_bind_group(
        &self,
        device: &wgpu::Device,
        resources: MeshingBindResources<'_>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshing compute bg"),
            layout: &self.meshing_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.atlas_voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.active_tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: resources.page_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: resources.page_indirect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: resources.dirty_page_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: resources.dirty_page_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: resources.diagnostics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: resources.chunk_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: resources.chunk_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: resources.draw_indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: resources.mesh_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: resources.vertex_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: resources.index_counter.as_entire_binding(),
                },
            ],
        })
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
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    label: Some("gpu-compute worker device"),
                },
                None,
            ))?;
            let limits = device.limits();
            if limits.max_storage_buffers_per_shader_stage < COMPUTE_STORAGE_BINDING_COUNT {
                anyhow::bail!(
                    "adapter exposes {} storage buffers per compute stage but runtime requires {}",
                    limits.max_storage_buffers_per_shader_stage,
                    COMPUTE_STORAGE_BINDING_COUNT
                );
            }
            let runtime = GpuComputeRuntime::new(&device).context("compute runtime")?;
            let page_capacity = GPU_PAGE_CAPACITY as u64;
            let page_len = CHUNK_VOLUME as u64;
            let atlas_voxels = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk atlas voxels"),
                size: page_capacity * page_len * 2 * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let page_indirect = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk page indirect"),
                size: page_capacity * std::mem::size_of::<DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let chunk_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk mesh vertex buffer"),
                size: page_capacity
                    * GPU_MESH_VERTEX_CAPACITY_PER_PAGE
                    * std::mem::size_of::<GpuVertex>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let chunk_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk mesh index buffer"),
                size: page_capacity
                    * GPU_MESH_INDEX_CAPACITY_PER_PAGE
                    * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk draw indexed indirect buffer"),
                size: page_capacity * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mesh_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk mesh meta buffer"),
                size: page_capacity * std::mem::size_of::<ChunkMeshMeta>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let vertex_counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk mesh vertex counter"),
                size: page_capacity * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let index_counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk mesh index counter"),
                size: page_capacity * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            log::debug!(
                "gpu mesh buffers allocated: page_capacity={} vertex_bytes={} index_bytes={} indirect_bytes={} meta_bytes={} vertex_counter_bytes={} index_counter_bytes={}",
                page_capacity,
                page_capacity * GPU_MESH_VERTEX_CAPACITY_PER_PAGE * std::mem::size_of::<GpuVertex>() as u64,
                page_capacity * GPU_MESH_INDEX_CAPACITY_PER_PAGE * std::mem::size_of::<u32>() as u64,
                page_capacity * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
                page_capacity * std::mem::size_of::<ChunkMeshMeta>() as u64,
                page_capacity * std::mem::size_of::<u32>() as u64,
                page_capacity * std::mem::size_of::<u32>() as u64,
            );

            let velocity_mac = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk velocity atlas"),
                size: page_capacity * page_len * std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let pressure = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk pressure atlas"),
                size: page_capacity * page_len * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let divergence = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk divergence atlas"),
                size: page_capacity * page_len * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let material_density = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk material density atlas"),
                size: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let edit_commands = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edit command buffer"),
                size: MAX_EDIT_COMMANDS as u64 * std::mem::size_of::<EditCommand>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dirty_page_indices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dirty page indices"),
                size: page_capacity * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dirty_page_counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dirty page counter"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk diagnostics"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let simulation_resources = SimulationBindResources {
                atlas_voxels: &atlas_voxels,
                velocity_mac: &velocity_mac,
                pressure: &pressure,
                divergence: &divergence,
                material_density: &material_density,
                active_tiles: &frontier,
                active_tile_counter: &active_tile_counter,
                edit_commands: &edit_commands,
                page_params: &page_params,
                diagnostics: &diagnostics,
            };
            let simulation_bg = runtime.create_simulation_bind_group(&device, simulation_resources);
            let meshing_resources = MeshingBindResources {
                atlas_voxels: &atlas_voxels,
                active_tiles: &frontier,
                page_params: &page_params,
                page_indirect: &page_indirect,
                dirty_page_indices: &dirty_page_indices,
                dirty_page_counter: &dirty_page_counter,
                diagnostics: &diagnostics,
                chunk_vertex_buffer: &chunk_vertex_buffer,
                chunk_index_buffer: &chunk_index_buffer,
                draw_indirect_buffer: &draw_indirect_buffer,
                mesh_meta_buffer: &mesh_meta_buffer,
                vertex_counter: &vertex_counter,
                index_counter: &index_counter,
            };
            let meshing_bg = runtime.create_meshing_bind_group(&device, meshing_resources);
            Ok(WorkerGpuState {
                device,
                queue,
                runtime,
                gpu_job_lock: Mutex::new(()),
                atlas: Mutex::new(ChunkPageAtlas::default()),
                atlas_voxels,
                velocity_mac,
                pressure,
                divergence,
                material_density,
                active_tiles: frontier,
                active_tile_counter,
                edit_commands,
                page_params,
                page_indirect,
                dirty_page_indices,
                dirty_page_counter,
                diagnostics,
                chunk_vertex_buffer,
                chunk_index_buffer,
                draw_indirect_buffer,
                mesh_meta_buffer,
                vertex_counter,
                index_counter,
                simulation_bg,
                meshing_bg,
                runtime_config: GpuSimulationRuntimeConfig::default(),
            })
        });
        let state = state.as_ref().map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let incoming = job.snapshot.center_voxels.as_ref();
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        let (page_index, page_was_reassigned) = atlas.page_for_chunk_or_allocate(job.coord)?;
        atlas.assert_page_for_chunk(job.coord, page_index);
        atlas.acquire_page_for_job(job.coord, page_index);
        let last_version = atlas
            .version_for_chunk
            .get(&job.coord)
            .copied()
            .unwrap_or(0);
        let current_state = atlas.state_for_chunk.get(&job.coord).copied().unwrap_or(0);
        let previous_frontier = atlas
            .frontier_len_for_chunk
            .get(&job.coord)
            .copied()
            .unwrap_or(0);
        let tick = atlas.tick_for_chunk.get(&job.coord).copied().unwrap_or(0);
        let mut cached_before = atlas
            .cached_materials
            .get(&job.coord)
            .cloned()
            .unwrap_or_else(|| vec![EMPTY; incoming.len()]);
        if cached_before.len() != incoming.len() {
            cached_before.resize(incoming.len(), EMPTY);
        }
        drop(atlas);

        let mut edit_commands = Vec::new();
        for (idx, (&next, prev)) in incoming.iter().zip(cached_before.iter()).enumerate() {
            if next != *prev {
                edit_commands.push(EditCommand {
                    voxel_index: idx as u32,
                    material_id: next as u32,
                    flags: 0,
                    _pad: 0,
                });
            }
        }

        let seeded_active = incoming.iter().filter(|m| **m != EMPTY).count() as u32;
        let active_frontier_count = seeded_active
            .max(previous_frontier)
            .max(edit_commands.len() as u32);
        let sim_job = SimulationJob {
            chunk_coord: job.coord,
            materials: incoming.to_vec(),
            active_frontier_count,
            simulation_tick: tick,
        };
        let next_state = (current_state + 1) & 1;
        let diagnostics = ChunkSimulationDiagnostics {
            changed_voxels: edit_commands.len() as u32,
            dropped_frontier_writes: edit_commands
                .len()
                .saturating_sub(active_frontier_count as usize)
                as u32,
            cross_border_attempts: 0,
        };

        let dispatch_t0 = Instant::now();
        let job_result = (|| -> anyhow::Result<ComputedChunkArtifacts> {
            // The worker uses a shared set of writable scratch buffers/bind groups.
            // Serialize dispatch and readback to keep page_params/edit buffers/page ownership coherent
            // across the background mesh worker pool.
            let _gpu_job_guard = state.gpu_job_lock.lock().unwrap_or_else(|e| e.into_inner());

            {
                let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
                atlas.assert_page_for_chunk(job.coord, page_index);
            }
            if page_was_reassigned {
                clear_page_buffers(state, page_index);
            }
            clear_job_scratch_buffers(state);

            if active_frontier_count > 0 {
                state.runtime.run_active_frontier(
                    state,
                    &sim_job,
                    page_index,
                    current_state,
                    &edit_commands,
                )?
            }
            let indirect = if !edit_commands.is_empty() || last_version != job.version {
                state
                    .runtime
                    .run_meshing_dispatch(state, &sim_job, page_index, current_state)?
            } else {
                DrawIndirectArgs::default()
            };

            let generated_materials = readback_page_materials(
                state,
                page_index,
                next_state,
                job.snapshot.center_voxels.len(),
            )
            .unwrap_or_else(|_| job.snapshot.center_voxels.to_vec());
            let dispatch_ms = dispatch_t0.elapsed().as_secs_f32() * 1000.0;
            let snapshot = job
                .snapshot
                .with_center_materials(generated_materials.clone());
            let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
                mesh_chunk_snapshot(job.coord, &snapshot, job.lod, job.greedy);
            let mesh_indirect = DrawIndirectArgs {
                index_count: inds.len() as u32,
                instance_count: indirect.instance_count.max(1),
                first_vertex: 0,
                first_instance: 0,
            };

            Ok(ComputedChunkArtifacts {
                simulation_diagnostics: diagnostics,
                mesh_artifact: ChunkMeshArtifact::Gpu {
                    verts,
                    inds,
                    indirect: mesh_indirect,
                    aabb_min,
                    aabb_max,
                    chunk_origin_world,
                    dispatch_ms,
                    readback_bytes: generated_materials.len() as u64
                        * std::mem::size_of::<MaterialId>() as u64,
                },
            })
        })();

        if job_result.is_ok() {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            atlas.assert_page_for_chunk(job.coord, page_index);
            atlas.version_for_chunk.insert(job.coord, job.version);
            atlas.state_for_chunk.insert(job.coord, next_state);
            atlas.tick_for_chunk.insert(job.coord, tick.wrapping_add(1));
            atlas
                .frontier_len_for_chunk
                .insert(job.coord, active_frontier_count);
            atlas.diagnostics_for_chunk.insert(job.coord, diagnostics);
            atlas
                .cached_materials
                .insert(job.coord, job.snapshot.center_voxels.as_ref().to_vec());
        }

        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.release_page_from_job(page_index);
        drop(atlas);

        job_result
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
    simulation_tick: u32,
    state_index: u32,
    edit_count: u32,
    active_tile_budget: u32,
    jacobi_iterations: u32,
    jacobi_iteration: u32,
}
#[cfg(feature = "gpu-compute")]
fn device_page_params(
    sim_job: &SimulationJob,
    page_index: GpuPageIndex,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
    runtime_config: GpuSimulationRuntimeConfig,
) -> [FrameParams; 1] {
    [FrameParams {
        page_index: page_index.0,
        voxel_count: sim_job.materials.len() as u32,
        frontier_len,
        simulation_tick: sim_job.simulation_tick,
        state_index,
        edit_count,
        active_tile_budget: frontier_len,
        jacobi_iterations: runtime_config.max_jacobi_iterations,
        jacobi_iteration: 0,
    }]
}

#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub material_id: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct ChunkMeshMeta {
    pub page_index: u32,
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub _pad: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

pub struct ComputedChunkArtifacts {
    pub simulation_diagnostics: ChunkSimulationDiagnostics,
    pub(crate) mesh_artifact: ChunkMeshArtifact,
}

impl ComputedChunkArtifacts {
    pub fn has_any_surface(&self) -> bool {
        let (_, inds, _, _, _, _) = self.mesh_artifact.geometry();
        !inds.is_empty()
    }
}

pub(crate) fn cpu_generate_material_field(job: &MeshJob) -> ComputedChunkArtifacts {
    let mut out = vec![EMPTY; job.snapshot.center_voxels.len()];
    out.copy_from_slice(job.snapshot.center_voxels.as_ref());
    let snapshot = job.snapshot.with_center_materials(out);
    let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
        mesh_chunk_snapshot(job.coord, &snapshot, job.lod, job.greedy);
    let surface = inds.len() as u32;
    ComputedChunkArtifacts {
        simulation_diagnostics: ChunkSimulationDiagnostics::default(),
        mesh_artifact: ChunkMeshArtifact::Cpu {
            verts,
            inds,
            indirect: DrawIndirectArgs {
                index_count: surface,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            aabb_min,
            aabb_max,
            chunk_origin_world,
        },
    }
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
            sim_cpu.step_region(
                &mut store_a,
                &region,
                chunk,
                &mut rng_cpu,
                crate::simulation::SimulationStepMetadata::default(),
            );
            sim_compute_emulated.step_region(
                &mut store_b,
                &region,
                chunk,
                &mut rng_compute,
                crate::simulation::SimulationStepMetadata::default(),
            );
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
