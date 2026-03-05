#[cfg(not(feature = "gpu_meshing_experimental"))]
use crate::renderer::mesh_chunk_snapshot;
use crate::renderer::{ChunkMeshArtifact, MeshJob};
use crate::types::{ChunkCoord, GpuPageIndex};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "gpu-compute")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "gpu-compute")]
use std::sync::{Arc, Mutex};
use std::time::Instant;
#[cfg(feature = "gpu-compute")]
const GPU_PAGE_CAPACITY: u32 = 256;
const CHUNK_VOLUME: usize = 32 * 32 * 32;
#[cfg(feature = "gpu-compute")]
const MAC_U_COUNT: usize = (32 + 1) * 32 * 32;
#[cfg(feature = "gpu-compute")]
const MAC_V_COUNT: usize = 32 * (32 + 1) * 32;
#[cfg(feature = "gpu-compute")]
const MAC_W_COUNT: usize = 32 * 32 * (32 + 1);
#[cfg(feature = "gpu-compute")]
const MAC_TOTAL_COUNT: usize = MAC_U_COUNT + MAC_V_COUNT + MAC_W_COUNT;
#[cfg(feature = "gpu-compute")]
pub(crate) const COMPUTE_STORAGE_BINDING_COUNT: u32 = 13;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u64 = CHUNK_VOLUME as u64;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u64 = (CHUNK_VOLUME as u64) * 6;

#[cfg(feature = "gpu-compute")]
const fn atlas_voxel_size_bytes() -> u64 {
    (GPU_PAGE_CAPACITY as u64) * (CHUNK_VOLUME as u64) * 2 * std::mem::size_of::<u32>() as u64
}

#[cfg(feature = "gpu-compute")]
const fn velocity_mac_size_bytes() -> u64 {
    (GPU_PAGE_CAPACITY as u64) * (MAC_TOTAL_COUNT as u64) * 2 * std::mem::size_of::<f32>() as u64
}

#[cfg(feature = "gpu-compute")]
pub(crate) const fn required_storage_buffer_binding_size_bytes() -> u64 {
    let atlas = atlas_voxel_size_bytes();
    let velocity = velocity_mac_size_bytes();
    if atlas > velocity {
        atlas
    } else {
        velocity
    }
}

#[cfg(feature = "gpu-compute")]
pub const fn gpu_page_capacity() -> u32 {
    GPU_PAGE_CAPACITY
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshPipelineBackend {
    #[cfg(feature = "cpu_meshing_debug")]
    Cpu,
    #[cfg(feature = "gpu-compute")]
    Gpu,
}

impl MeshPipelineBackend {
    pub fn label(self) -> &'static str {
        match self {
            #[cfg(feature = "cpu_meshing_debug")]
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
#[derive(Clone)]
pub struct SharedMeshBuffers {
    pub chunk_vertex_buffer: Arc<wgpu::Buffer>,
    pub chunk_index_buffer: Arc<wgpu::Buffer>,
    pub draw_indirect_buffer: Arc<wgpu::Buffer>,
    pub page_indirect: Arc<wgpu::Buffer>,
    pub mesh_meta_buffer: Arc<wgpu::Buffer>,
    pub vertex_counter: Arc<wgpu::Buffer>,
    pub index_counter: Arc<wgpu::Buffer>,
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

    fn acquire_existing_page_for_job(&mut self, page_index: GpuPageIndex) {
        if self.resolve_chunk(page_index).is_some() {
            *self.in_flight_jobs_for_page.entry(page_index).or_insert(0) += 1;
        }
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
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    runtime: GpuComputeRuntime,
    atlas: Mutex<ChunkPageAtlas>,
    atlas_voxels: wgpu::Buffer,
    velocity_mac: wgpu::Buffer,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    material_density: wgpu::Buffer,
    page_indirect: Arc<wgpu::Buffer>,
    chunk_vertex_buffer: Arc<wgpu::Buffer>,
    chunk_index_buffer: Arc<wgpu::Buffer>,
    draw_indirect_buffer: Arc<wgpu::Buffer>,
    mesh_meta_buffer: Arc<wgpu::Buffer>,
    vertex_counter: Arc<wgpu::Buffer>,
    index_counter: Arc<wgpu::Buffer>,
    runtime_config: GpuSimulationRuntimeConfig,
}

#[cfg(feature = "gpu-compute")]
struct JobScratchBuffers {
    page_params: wgpu::Buffer,
    active_tiles: wgpu::Buffer,
    active_tile_counter: wgpu::Buffer,
    edit_commands: wgpu::Buffer,
    dirty_page_indices: wgpu::Buffer,
    dirty_page_counter: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
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
        page_index.0 as u64 * (MAC_TOTAL_COUNT * 2) as u64 * std::mem::size_of::<f32>() as u64;
    let velocity_zeros = vec![0.0f32; MAC_TOTAL_COUNT * 2];
    state.queue.write_buffer(
        &state.velocity_mac,
        velocity_offset,
        bytemuck::cast_slice(&velocity_zeros),
    );

    let scalar_offset =
        page_index.0 as u64 * (CHUNK_VOLUME * 2) as u64 * std::mem::size_of::<f32>() as u64;
    let scalar_zeros = vec![0.0f32; CHUNK_VOLUME * 2];
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
fn create_job_scratch_buffers(state: &WorkerGpuState) -> JobScratchBuffers {
    let page_len = CHUNK_VOLUME as u64;
    let page_capacity = GPU_PAGE_CAPACITY as u64;
    let device = &state.device;
    JobScratchBuffers {
        page_params: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk page params"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        active_tiles: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk active frontier"),
            size: page_len * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        active_tile_counter: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("active tile counter"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        edit_commands: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edit command buffer"),
            size: MAX_EDIT_COMMANDS as u64 * std::mem::size_of::<EditCommand>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        dirty_page_indices: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dirty page indices"),
            size: page_capacity * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        dirty_page_counter: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dirty page counter"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        diagnostics: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk diagnostics"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
    }
}

#[cfg(feature = "gpu-compute")]
fn clear_job_scratch_buffers(
    state: &WorkerGpuState,
    scratch: &JobScratchBuffers,
    _active_tile_len: usize,
    _dirty_page_len: usize,
) {
    let zero_u32x4 = [0u32; 4];

    state.queue.write_buffer(
        &scratch.active_tile_counter,
        0,
        bytemuck::cast_slice(&zero_u32x4),
    );
    state.queue.write_buffer(
        &scratch.dirty_page_counter,
        0,
        bytemuck::cast_slice(&zero_u32x4),
    );
    state
        .queue
        .write_buffer(&scratch.diagnostics, 0, bytemuck::cast_slice(&zero_u32x4));

    // FIX 7: avoid uploading large zero arrays every job; counters are authoritative.
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

    let zero_draw_indirect = DrawIndexedIndirectArgs::default();
    let draw_stride = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
    let draw_offset = page_index.0 as u64 * draw_stride;
    state.queue.write_buffer(
        &state.draw_indirect_buffer,
        draw_offset,
        bytemuck::bytes_of(&zero_draw_indirect),
    );

    let zero_meta = ChunkMeshMeta {
        page_index: page_index.0,
        ..ChunkMeshMeta::default()
    };
    let meta_stride = std::mem::size_of::<ChunkMeshMeta>() as u64;
    let meta_offset = page_index.0 as u64 * meta_stride;
    state.queue.write_buffer(
        &state.mesh_meta_buffer,
        meta_offset,
        bytemuck::bytes_of(&zero_meta),
    );

    let zero_counter = [0u32; 1];
    let counter_stride = std::mem::size_of::<u32>() as u64;
    let counter_offset = page_index.0 as u64 * counter_stride;
    state.queue.write_buffer(
        &state.vertex_counter,
        counter_offset,
        bytemuck::cast_slice(&zero_counter),
    );
    state.queue.write_buffer(
        &state.index_counter,
        counter_offset,
        bytemuck::cast_slice(&zero_counter),
    );
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy)]
struct GpuSimulationRuntimeConfig {
    max_jacobi_iterations: u32,
    cell_size: f32,
    cfl_velocity_clamp: f32,
    velocity_damping: f32,
    viscosity: f32,
}

#[cfg(feature = "gpu-compute")]
impl Default for GpuSimulationRuntimeConfig {
    fn default() -> Self {
        Self {
            max_jacobi_iterations: 32,
            cell_size: 1.0,
            cfl_velocity_clamp: 4.0,
            velocity_damping: 0.995,
            viscosity: 0.02,
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
#[cfg(feature = "gpu-compute")]
static ACTIVE_GPU_JOBS: std::sync::LazyLock<Mutex<HashSet<ChunkCoord>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashSet::new()));

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

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub struct MeshArtifactGPU {
    pub page_index: GpuPageIndex,
    pub lod: u8,
    pub draw_indirect_index: u32,
}

impl GpuComputeRuntime {
    pub fn runtime_supported(adapter: &wgpu::Adapter, effective_limits: &wgpu::Limits) -> bool {
        #[cfg(not(feature = "gpu-compute"))]
        {
            let _ = (adapter, effective_limits);
            false
        }

        #[cfg(feature = "gpu-compute")]
        {
            let downlevel = adapter.get_downlevel_capabilities();
            let required_storage_size = required_storage_buffer_binding_size_bytes();
            downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                && effective_limits.max_storage_buffers_per_shader_stage
                    >= COMPUTE_STORAGE_BINDING_COUNT
                && effective_limits.max_storage_buffer_binding_size as u64 >= required_storage_size
                && effective_limits.max_buffer_size >= required_storage_size
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
        scratch: &JobScratchBuffers,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        edit_commands: &[EditCommand],
        neighbor_pages: [u32; 6],
    ) -> anyhow::Result<()> {
        let t0 = Instant::now();
        let frontier_len = sim_job
            .active_frontier_count
            .min(sim_job.materials.len() as u32);

        if !edit_commands.is_empty() {
            state.queue.write_buffer(
                &scratch.edit_commands,
                0,
                bytemuck::cast_slice(edit_commands),
            );
        }

        let groups = frontier_len
            .max(edit_commands.len() as u32)
            .max(1)
            .div_ceil(64);
        let base_params = |jacobi_iteration: u32| {
            device_page_params(
                sim_job,
                page_index,
                frontier_len,
                current_state,
                edit_commands.len() as u32,
                state.runtime_config,
                jacobi_iteration,
                neighbor_pages,
            )
        };

        let mut staged_params =
            Vec::with_capacity(state.runtime_config.max_jacobi_iterations as usize + 5);
        staged_params.push(base_params(0)[0]); // force
        staged_params.push(base_params(0)[0]); // advect
        staged_params.push(base_params(0)[0]); // divergence
        for jacobi_iter in 0..state.runtime_config.max_jacobi_iterations {
            staged_params.push(base_params(jacobi_iter)[0]);
        }
        staged_params.push(base_params(state.runtime_config.max_jacobi_iterations)[0]); // projection
        staged_params.push(base_params(state.runtime_config.max_jacobi_iterations)[0]); // material advection

        let staged_params_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation staged frame params"),
            size: (staged_params.len() * std::mem::size_of::<FrameParams>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut mapped = staged_params_buffer.slice(..).get_mapped_range_mut();
            mapped.copy_from_slice(bytemuck::cast_slice(&staged_params));
        }
        staged_params_buffer.unmap();

        let simulation_resources = SimulationBindResources {
            atlas_voxels: &state.atlas_voxels,
            velocity_mac: &state.velocity_mac,
            pressure: &state.pressure,
            divergence: &state.divergence,
            material_density: &state.material_density,
            active_tiles: &scratch.active_tiles,
            active_tile_counter: &scratch.active_tile_counter,
            edit_commands: &scratch.edit_commands,
            page_params: &scratch.page_params,
            diagnostics: &scratch.diagnostics,
        };
        let simulation_bg = self.create_simulation_bind_group(&state.device, simulation_resources);

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let params_stride = std::mem::size_of::<FrameParams>() as u64;
        let mut stage_index = 0u64;
        let mut encode_stage = |encoder: &mut wgpu::CommandEncoder,
                                pipeline: &wgpu::ComputePipeline| {
            encoder.copy_buffer_to_buffer(
                &staged_params_buffer,
                stage_index * params_stride,
                &scratch.page_params,
                0,
                params_stride,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_bind_group(0, &simulation_bg, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            stage_index += 1;
        };

        encode_stage(&mut encoder, &self.force_pipeline);
        encode_stage(&mut encoder, &self.advect_pipeline);
        encode_stage(&mut encoder, &self.divergence_pipeline);
        for _ in 0..state.runtime_config.max_jacobi_iterations {
            encode_stage(&mut encoder, &self.pressure_jacobi_pipeline);
        }
        encode_stage(&mut encoder, &self.project_pipeline);
        encode_stage(&mut encoder, &self.material_advect_pipeline);

        state.queue.submit(Some(encoder.finish()));
        state.device.poll(wgpu::Maintain::Wait);

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
    #[cfg(feature = "gpu_meshing_experimental")]
    fn run_meshing_dispatch(
        &self,
        state: &WorkerGpuState,
        scratch: &JobScratchBuffers,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        lod: u8,
    ) -> anyhow::Result<MeshArtifactGPU> {
        clear_meshing_outputs_for_page(state, page_index);

        let page_params = device_page_params(
            sim_job,
            page_index,
            sim_job.active_frontier_count,
            current_state,
            0,
            state.runtime_config,
            0,
            [u32::MAX; 6],
        );
        state
            .queue
            .write_buffer(&scratch.page_params, 0, bytemuck::cast_slice(&page_params));

        let groups = sim_job.active_frontier_count.max(1).div_ceil(64);
        let meshing_resources = MeshingBindResources {
            atlas_voxels: &state.atlas_voxels,
            active_tiles: &scratch.active_tiles,
            page_params: &scratch.page_params,
            page_indirect: &state.page_indirect,
            dirty_page_indices: &scratch.dirty_page_indices,
            dirty_page_counter: &scratch.dirty_page_counter,
            diagnostics: &scratch.diagnostics,
            chunk_vertex_buffer: &state.chunk_vertex_buffer,
            chunk_index_buffer: &state.chunk_index_buffer,
            draw_indirect_buffer: &state.draw_indirect_buffer,
            mesh_meta_buffer: &state.mesh_meta_buffer,
            vertex_counter: &state.vertex_counter,
            index_counter: &state.index_counter,
        };
        let meshing_bg = self.create_meshing_bind_group(&state.device, meshing_resources);
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("meshing_dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("meshing_pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &meshing_bg, &[]);
            pass.set_pipeline(&self.meshing_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        state.queue.submit(Some(encoder.finish()));
        state.device.poll(wgpu::Maintain::Wait);
        Ok(MeshArtifactGPU {
            page_index,
            lod,
            draw_indirect_index: page_index.0,
        })
    }
}

#[cfg(feature = "gpu-compute")]
fn validate_storage_buffer_size(
    label: &str,
    size: u64,
    limits: &wgpu::Limits,
) -> anyhow::Result<()> {
    if size > limits.max_storage_buffer_binding_size as u64 {
        log::error!(
            "gpu buffer '{}' requires {} bytes, exceeding adapter max_storage_buffer_binding_size={}.",
            label,
            size,
            limits.max_storage_buffer_binding_size
        );
        anyhow::bail!("buffer '{}' exceeds max_storage_buffer_binding_size", label);
    }
    if size > limits.max_buffer_size {
        log::error!(
            "gpu buffer '{}' requires {} bytes, exceeding adapter max_buffer_size={}",
            label,
            size,
            limits.max_buffer_size
        );
        anyhow::bail!("buffer '{}' exceeds max_buffer_size", label);
    }
    Ok(())
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

#[cfg(feature = "gpu-compute")]
use std::sync::OnceLock;

#[cfg(feature = "gpu-compute")]
static WORKER_STATE: OnceLock<anyhow::Result<Arc<WorkerGpuState>>> = OnceLock::new();

#[cfg(feature = "gpu-compute")]
pub fn initialize_gpu_compute_worker(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    shared_mesh_buffers: SharedMeshBuffers,
) -> anyhow::Result<()> {
    let state_result = WORKER_STATE.get_or_init(|| {
        let limits = device.limits();
        log::info!(
            "gpu worker limits: storage-buffers-per-stage device={} required={}",
            limits.max_storage_buffers_per_shader_stage,
            COMPUTE_STORAGE_BINDING_COUNT
        );
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
        let atlas_voxel_size = atlas_voxel_size_bytes();
        let velocity_mac_size = velocity_mac_size_bytes();
        let required_storage_size = required_storage_buffer_binding_size_bytes();
        log::info!(
            "gpu worker storage-buffer sizes: atlas_voxels={}B velocity_mac={}B required-max={}B adapter-max-binding={}B adapter-max-buffer={}B",
            atlas_voxel_size,
            velocity_mac_size,
            required_storage_size,
            limits.max_storage_buffer_binding_size,
            limits.max_buffer_size
        );

        validate_storage_buffer_size("atlas_voxels", atlas_voxel_size, &limits)?;
        validate_storage_buffer_size("velocity_mac", velocity_mac_size, &limits)?;

        let atlas_voxels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk atlas voxels"),
            size: atlas_voxel_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocity_mac = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk velocity atlas"),
            size: velocity_mac_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk pressure atlas"),
            size: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk divergence atlas"),
            size: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_density = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk material density atlas"),
            size: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Arc::new(WorkerGpuState {
            device,
            queue,
            runtime,
            atlas: Mutex::new(ChunkPageAtlas::default()),
            atlas_voxels,
            velocity_mac,
            pressure,
            divergence,
            material_density,
            page_indirect: shared_mesh_buffers.page_indirect,
            chunk_vertex_buffer: shared_mesh_buffers.chunk_vertex_buffer,
            chunk_index_buffer: shared_mesh_buffers.chunk_index_buffer,
            draw_indirect_buffer: shared_mesh_buffers.draw_indirect_buffer,
            mesh_meta_buffer: shared_mesh_buffers.mesh_meta_buffer,
            vertex_counter: shared_mesh_buffers.vertex_counter,
            index_counter: shared_mesh_buffers.index_counter,
            runtime_config: GpuSimulationRuntimeConfig::default(),
        }))
    });

    if let Err(err) = state_result.as_ref() {
        return Err(anyhow::anyhow!(err.to_string()));
    }
    Ok(())
}

pub(crate) fn run_chunk_job_on_worker(job: &MeshJob) -> anyhow::Result<ComputedChunkArtifacts> {
    #[cfg(not(feature = "gpu-compute"))]
    {
        let _ = job;
        anyhow::bail!("gpu compute feature disabled")
    }

    #[cfg(feature = "gpu-compute")]
    {
        let state = WORKER_STATE
            .get()
            .context("gpu worker runtime is not initialized; renderer must call initialize_gpu_compute_worker")?
            .as_ref()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .clone();
        {
            let mut active_jobs = ACTIVE_GPU_JOBS.lock().unwrap_or_else(|e| e.into_inner());
            if !active_jobs.insert(job.coord) {
                // Prevent concurrent jobs for the same chunk from racing GPU artifacts.
                return Ok(ComputedChunkArtifacts {
                    simulation_diagnostics: ChunkSimulationDiagnostics::default(),
                    mesh_artifact: ChunkMeshArtifact::Skipped,
                });
            }
        }

        let incoming = job.snapshot.center_voxels.as_ref();
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        let (page_index, page_was_reassigned) = match atlas.page_for_chunk_or_allocate(job.coord) {
            Ok(value) => value,
            Err(err) => {
                drop(atlas);
                let mut active_jobs = ACTIVE_GPU_JOBS.lock().unwrap_or_else(|e| e.into_inner());
                active_jobs.remove(&job.coord);
                return Err(err);
            }
        };
        atlas.assert_page_for_chunk(job.coord, page_index);
        // FIX 6: pin the primary page until dispatch/readback completion.
        atlas.acquire_page_for_job(job.coord, page_index);
        let _last_version = atlas
            .version_for_chunk
            .get(&job.coord)
            .copied()
            .unwrap_or(0);
        let current_state = atlas.state_for_chunk.get(&job.coord).copied().unwrap_or(0);
        let tick = atlas.tick_for_chunk.get(&job.coord).copied().unwrap_or(0);
        let mut cached_before = atlas
            .cached_materials
            .get(&job.coord)
            .cloned()
            .unwrap_or_else(|| vec![EMPTY; incoming.len()]);
        if cached_before.len() != incoming.len() {
            cached_before.resize(incoming.len(), EMPTY);
        }
        // FIX 4: freeze neighbor page indices while atlas lock is held.
        let neighbor_pages = neighbor_pages_for_chunk(&atlas, job.coord);
        let mut pinned_pages = vec![page_index];
        for neighbor_page in neighbor_pages {
            if neighbor_page == u32::MAX {
                continue;
            }
            let neighbor_index = GpuPageIndex(neighbor_page);
            if !pinned_pages.contains(&neighbor_index) {
                // FIX 6: pin neighbor pages for the full job to prevent eviction races.
                atlas.acquire_existing_page_for_job(neighbor_index);
                pinned_pages.push(neighbor_index);
            }
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

        let mut active_tiles_seed = Vec::with_capacity(edit_commands.len());
        for edit in &edit_commands {
            active_tiles_seed.push(edit.voxel_index);
        }
        let active_frontier_count = active_tiles_seed.len() as u32;
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
            {
                let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
                atlas.assert_page_for_chunk(job.coord, page_index);
            }
            let scratch = create_job_scratch_buffers(&state);
            let gpu_artifact = {
                if page_was_reassigned {
                    clear_page_buffers(&state, page_index);
                }
                clear_job_scratch_buffers(&state, &scratch, active_frontier_count as usize, 1);

                if !active_tiles_seed.is_empty() {
                    state.queue.write_buffer(
                        &scratch.active_tiles,
                        0,
                        bytemuck::cast_slice(&active_tiles_seed),
                    );
                }

                if active_frontier_count > 0 {
                    state.runtime.run_active_frontier(
                        &state,
                        &scratch,
                        &sim_job,
                        page_index,
                        current_state,
                        &edit_commands,
                        neighbor_pages,
                    )?
                }
                #[cfg(feature = "gpu_meshing_experimental")]
                let gpu_artifact = if !edit_commands.is_empty() || _last_version != job.version {
                    state.runtime.run_meshing_dispatch(
                        &state,
                        &scratch,
                        &sim_job,
                        page_index,
                        current_state,
                        job.lod as u8,
                    )?
                } else {
                    MeshArtifactGPU {
                        page_index,
                        lod: job.lod as u8,
                        draw_indirect_index: page_index.0,
                    }
                };

                #[cfg(not(feature = "gpu_meshing_experimental"))]
                let gpu_artifact = MeshArtifactGPU {
                    page_index,
                    lod: job.lod as u8,
                    draw_indirect_index: page_index.0,
                };

                gpu_artifact
            };

            log::info!("[mesh-worker] gpu dispatch finished chunk={:?}", job.coord);

            let dispatch_ms = dispatch_t0.elapsed().as_secs_f32() * 1000.0;
            #[cfg(not(feature = "gpu_meshing_experimental"))]
            let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
                mesh_chunk_snapshot(job.coord, &job.snapshot, job.lod, job.greedy);
            #[cfg(feature = "gpu_meshing_experimental")]
            let (verts, inds, aabb_min, aabb_max, chunk_origin_world) = (
                Vec::new(),
                Vec::new(),
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
                glam::Vec3::new(
                    job.coord.x as f32 * 32.0,
                    job.coord.y as f32 * 32.0,
                    job.coord.z as f32 * 32.0,
                ),
            );
            log::info!("[mesh-worker] artifact created chunk={:?}", job.coord);
            Ok(ComputedChunkArtifacts {
                simulation_diagnostics: diagnostics,
                mesh_artifact: {
                    #[cfg(feature = "gpu_meshing_experimental")]
                    {
                        ChunkMeshArtifact::Gpu {
                            page_index: gpu_artifact.page_index,
                            draw_indirect_index: gpu_artifact.draw_indirect_index,
                            lod: gpu_artifact.lod,
                            verts,
                            inds,
                            aabb_min,
                            aabb_max,
                            chunk_origin_world,
                            dispatch_ms,
                        }
                    }
                    #[cfg(not(feature = "gpu_meshing_experimental"))]
                    {
                        ChunkMeshArtifact::Cpu {
                            verts,
                            inds,
                            indirect: DrawIndirectArgs::default(),
                            aabb_min,
                            aabb_max,
                            chunk_origin_world,
                        }
                    }
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
            atlas.cached_materials.insert(job.coord, incoming.to_vec());
        }

        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        for pinned_page in pinned_pages {
            atlas.release_page_from_job(pinned_page);
        }
        drop(atlas);

        let mut active_jobs = ACTIVE_GPU_JOBS.lock().unwrap_or_else(|e| e.into_inner());
        active_jobs.remove(&job.coord);

        job_result
    }
}

#[cfg(feature = "gpu-compute")]
fn neighbor_pages_for_chunk(atlas: &ChunkPageAtlas, coord: ChunkCoord) -> [u32; 6] {
    let mut pages = [u32::MAX; 6];
    let neighbors = [
        ChunkCoord {
            x: coord.x - 1,
            y: coord.y,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x + 1,
            y: coord.y,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y - 1,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y + 1,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z - 1,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z + 1,
        },
    ];
    for (i, neighbor) in neighbors.iter().enumerate() {
        if let Some(page) = atlas.page_for_chunk.get(neighbor) {
            pages[i] = page.0;
        }
    }
    pages
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
    cell_size: f32,
    max_velocity: f32,
    velocity_damping: f32,
    viscosity: f32,
    neighbor_pages: [u32; 6],
    _pad: [u32; 2],
}
#[cfg(feature = "gpu-compute")]
fn device_page_params(
    sim_job: &SimulationJob,
    page_index: GpuPageIndex,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
    runtime_config: GpuSimulationRuntimeConfig,
    jacobi_iteration: u32,
    neighbor_pages: [u32; 6],
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
        jacobi_iteration,
        cell_size: runtime_config.cell_size,
        max_velocity: runtime_config.cfl_velocity_clamp,
        velocity_damping: runtime_config.velocity_damping,
        viscosity: runtime_config.viscosity,
        neighbor_pages,
        _pad: [0; 2],
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

#[cfg(feature = "gpu-compute")]
const _: [(); 20] = [(); std::mem::size_of::<DrawIndexedIndirectArgs>()];

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

#[cfg(feature = "cpu_meshing_debug")]
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
