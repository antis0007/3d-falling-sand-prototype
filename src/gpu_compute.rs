use crate::renderer::mesh_chunk_snapshot;
use crate::renderer::{ChunkMeshArtifact, MeshJob, MeshSkipReason, VOXEL_SIZE};
use crate::types::{ChunkCoord, GpuPageIndex, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use std::collections::HashMap;
#[cfg(feature = "gpu-compute")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "gpu-compute")]
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
#[cfg(feature = "gpu-compute")]
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u64 = (CHUNK_VOLUME as u64) * 12;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u64 = (CHUNK_VOLUME as u64) * 18;
#[cfg(feature = "gpu-compute")]
pub const GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES: u64 = 512 * 1024 * 1024;
#[cfg(feature = "gpu-compute")]
pub const GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(feature = "gpu-compute")]
pub const fn mesh_pool_slot_capacity() -> u32 {
    let vertex_slots = GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES
        / (GPU_MESH_VERTEX_CAPACITY_PER_PAGE * std::mem::size_of::<GpuVertex>() as u64);
    let index_slots = GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES
        / (GPU_MESH_INDEX_CAPACITY_PER_PAGE * std::mem::size_of::<u32>() as u64);
    let slots = if vertex_slots < index_slots {
        vertex_slots
    } else {
        index_slots
    };
    slots as u32
}

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
    Disabled,
    Cpu,
    #[cfg(feature = "gpu-compute")]
    Gpu,
}

impl MeshPipelineBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
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
    detect_faces_pipeline: wgpu::ComputePipeline,
    prefix_scan_pipeline: wgpu::ComputePipeline,
    emit_mesh_pipeline: wgpu::ComputePipeline,
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
    pub chunk_origin_buffer: Arc<wgpu::Buffer>,
    pub face_mask_buffer: Arc<wgpu::Buffer>,
    pub face_offset_buffer: Arc<wgpu::Buffer>,
    pub face_count_buffer: Arc<wgpu::Buffer>,
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
    page_params: &'a wgpu::Buffer,
    face_mask_buffer: &'a wgpu::Buffer,
    face_offset_buffer: &'a wgpu::Buffer,
    face_count_buffer: &'a wgpu::Buffer,
    chunk_vertex_buffer: &'a wgpu::Buffer,
    chunk_index_buffer: &'a wgpu::Buffer,
    draw_indirect_buffer: &'a wgpu::Buffer,
    mesh_meta_buffer: &'a wgpu::Buffer,
    chunk_origin_buffer: &'a wgpu::Buffer,
}

#[cfg(feature = "gpu-compute")]
#[derive(Default)]
struct ChunkPageAtlas {
    page_for_chunk: HashMap<ChunkCoord, GpuPageIndex>,
    chunk_for_page: HashMap<GpuPageIndex, ChunkCoord>,
    page_fences: HashMap<GpuPageIndex, PageFence>,
    version_for_chunk: HashMap<ChunkCoord, u64>,
    state_for_chunk: HashMap<ChunkCoord, u32>,
    frontier_len_for_chunk: HashMap<ChunkCoord, u32>,
    tick_for_chunk: HashMap<ChunkCoord, u32>,
    diagnostics_for_chunk: HashMap<ChunkCoord, ChunkSimulationDiagnostics>,
    cached_materials: HashMap<ChunkCoord, Vec<MaterialId>>,
    mesh_slice_for_chunk: HashMap<ChunkCoord, MeshBufferSlice>,
    chunk_for_mesh_slot: HashMap<u32, ChunkCoord>,
    mesh_slot_last_used: HashMap<u32, u64>,
    mesh_slot_epoch: u64,
    next_mesh_slot: u32,
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

    fn mesh_slice_for_chunk_or_allocate(&mut self, chunk: ChunkCoord) -> Option<MeshBufferSlice> {
        if let Some(existing) = self.mesh_slice_for_chunk.get(&chunk).copied() {
            self.touch_mesh_slot(existing.slot_index);
            return Some(existing);
        }

        let slot_capacity = mesh_pool_slot_capacity();
        if slot_capacity == 0 {
            return None;
        }

        let slot = if self.next_mesh_slot < slot_capacity {
            let slot = self.next_mesh_slot;
            self.next_mesh_slot = self.next_mesh_slot.saturating_add(1);
            slot
        } else {
            self.evictable_mesh_slot(slot_capacity)?
        };

        if let Some(evicted_chunk) = self.chunk_for_mesh_slot.remove(&slot) {
            self.mesh_slice_for_chunk.remove(&evicted_chunk);
        }

        let slice = MeshBufferSlice {
            slot_index: slot,
            vertex_offset: (slot as u64 * GPU_MESH_VERTEX_CAPACITY_PER_PAGE) as u32,
            index_offset: (slot as u64 * GPU_MESH_INDEX_CAPACITY_PER_PAGE) as u32,
        };
        self.mesh_slice_for_chunk.insert(chunk, slice);
        self.chunk_for_mesh_slot.insert(slot, chunk);
        self.touch_mesh_slot(slot);
        Some(slice)
    }

    fn touch_mesh_slot(&mut self, slot: u32) {
        self.mesh_slot_epoch = self.mesh_slot_epoch.saturating_add(1);
        self.mesh_slot_last_used.insert(slot, self.mesh_slot_epoch);
    }

    fn is_mesh_slot_fence_safe(&self, slot: u32) -> bool {
        self.chunk_for_mesh_slot
            .get(&slot)
            .and_then(|owner| self.page_for_chunk.get(owner))
            .map(|page| {
                let fence = self.page_fences.get(page).copied().unwrap_or_default();
                fence.last_completed >= fence.last_submitted
            })
            .unwrap_or(true)
    }

    fn evictable_mesh_slot(&self, slot_capacity: u32) -> Option<u32> {
        let mut selected: Option<(u64, u32)> = None;
        for slot in 0..slot_capacity {
            if !self.is_mesh_slot_fence_safe(slot) {
                continue;
            }
            let age = self.mesh_slot_last_used.get(&slot).copied().unwrap_or(0);
            let candidate = (age, slot);
            if selected.map(|cur| candidate < cur).unwrap_or(true) {
                selected = Some(candidate);
            }
        }
        selected.map(|(_, slot)| slot)
    }

    fn in_flight_mesh_slot_fence_count(&self) -> u32 {
        self.chunk_for_mesh_slot
            .keys()
            .filter(|slot| !self.is_mesh_slot_fence_safe(**slot))
            .count() as u32
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

    fn mark_page_submitted(&mut self, page_index: GpuPageIndex, serial: u64) {
        let fence = self.page_fences.entry(page_index).or_default();
        fence.last_submitted = serial;
    }

    fn refresh_completed_serial(&mut self, completed: u64) {
        for fence in self.page_fences.values_mut() {
            fence.last_completed = fence.last_completed.max(completed);
        }
    }

    fn evictable_page(&self) -> Option<GpuPageIndex> {
        self.chunk_for_page.keys().copied().find(|page| {
            let fence = self.page_fences.get(page).copied().unwrap_or_default();
            fence.last_completed >= fence.last_submitted
        })
    }

    fn is_page_fence_complete(&self, page_index: GpuPageIndex) -> bool {
        let fence = self
            .page_fences
            .get(&page_index)
            .copied()
            .unwrap_or_default();
        fence.last_completed >= fence.last_submitted
    }

    fn evict_page(&mut self, page_index: GpuPageIndex) {
        if let Some(chunk) = self.chunk_for_page.remove(&page_index) {
            self.page_for_chunk.remove(&chunk);
            self.version_for_chunk.remove(&chunk);
            self.state_for_chunk.remove(&chunk);
            self.frontier_len_for_chunk.remove(&chunk);
            self.tick_for_chunk.remove(&chunk);
            self.diagnostics_for_chunk.remove(&chunk);
            self.cached_materials.remove(&chunk);
            if let Some(slice) = self.mesh_slice_for_chunk.remove(&chunk) {
                self.chunk_for_mesh_slot.remove(&slice.slot_index);
                self.mesh_slot_last_used.remove(&slice.slot_index);
            }
            self.page_fences.remove(&page_index);
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
    chunk_origin_buffer: Arc<wgpu::Buffer>,
    face_mask_buffer: Arc<wgpu::Buffer>,
    face_offset_buffer: Arc<wgpu::Buffer>,
    face_count_buffer: Arc<wgpu::Buffer>,
    runtime_config: GpuSimulationRuntimeConfig,
    scratch: GpuScratchPool,
    simulation_bg: wgpu::BindGroup,
    meshing_bg: wgpu::BindGroup,
}

#[cfg(feature = "gpu-compute")]
pub struct GpuScratchPool {
    page_params: wgpu::Buffer,
    active_tiles: wgpu::Buffer,
    active_tile_counter: wgpu::Buffer,
    edit_commands: wgpu::Buffer,
    dirty_page_indices: wgpu::Buffer,
    dirty_page_counter: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
}

#[cfg(feature = "gpu-compute")]
fn clear_page_buffers(
    encoder: &mut wgpu::CommandEncoder,
    state: &WorkerGpuState,
    page_index: GpuPageIndex,
) {
    let atlas_page = CHUNK_VOLUME as u64 * 2 * std::mem::size_of::<u32>() as u64;
    encoder.clear_buffer(
        &state.atlas_voxels,
        page_index.0 as u64 * atlas_page,
        Some(atlas_page),
    );

    let velocity_page = (MAC_TOTAL_COUNT as u64) * 2 * std::mem::size_of::<f32>() as u64;
    encoder.clear_buffer(
        &state.velocity_mac,
        page_index.0 as u64 * velocity_page,
        Some(velocity_page),
    );

    let scalar_page = CHUNK_VOLUME as u64 * 2 * std::mem::size_of::<f32>() as u64;
    encoder.clear_buffer(
        &state.pressure,
        page_index.0 as u64 * scalar_page,
        Some(scalar_page),
    );
    encoder.clear_buffer(
        &state.divergence,
        page_index.0 as u64 * scalar_page,
        Some(scalar_page),
    );
    encoder.clear_buffer(
        &state.material_density,
        page_index.0 as u64 * scalar_page,
        Some(scalar_page),
    );
}

#[cfg(feature = "gpu-compute")]
fn create_gpu_scratch_pool(device: &wgpu::Device) -> GpuScratchPool {
    let page_len = CHUNK_VOLUME as u64;
    let page_capacity = GPU_PAGE_CAPACITY as u64;
    GpuScratchPool {
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
fn clear_gpu_scratch_pool(
    state: &WorkerGpuState,
    scratch: &GpuScratchPool,
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
fn clear_meshing_outputs_for_page(
    state: &WorkerGpuState,
    page_index: GpuPageIndex,
    mesh_slice: MeshBufferSlice,
) {
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
    let draw_offset = mesh_slice.slot_index as u64 * draw_stride;
    state.queue.write_buffer(
        &state.draw_indirect_buffer,
        draw_offset,
        bytemuck::bytes_of(&zero_draw_indirect),
    );

    let zero_meta = ChunkMeshMeta {
        slot_index: mesh_slice.slot_index,
        vertex_offset: mesh_slice.vertex_offset,
        index_offset: mesh_slice.index_offset,
        _pad: 0,
    };
    let meta_stride = std::mem::size_of::<ChunkMeshMeta>() as u64;
    let meta_offset = page_index.0 as u64 * meta_stride;
    state.queue.write_buffer(
        &state.mesh_meta_buffer,
        meta_offset,
        bytemuck::bytes_of(&zero_meta),
    );

    let zero_origin = [0.0f32; 4];
    let origin_stride = std::mem::size_of::<[f32; 4]>() as u64;
    let origin_offset = page_index.0 as u64 * origin_stride;
    state.queue.write_buffer(
        &state.chunk_origin_buffer,
        origin_offset,
        bytemuck::cast_slice(&zero_origin),
    );

    state.queue.write_buffer(
        &state.face_count_buffer,
        0,
        bytemuck::cast_slice(&[0u32; 1]),
    );
    state.queue.write_buffer(
        &state.face_mask_buffer,
        0,
        bytemuck::cast_slice(&vec![0u32; CHUNK_VOLUME]),
    );
    state.queue.write_buffer(
        &state.face_offset_buffer,
        0,
        bytemuck::cast_slice(&vec![0u32; CHUNK_VOLUME]),
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
#[cfg(feature = "gpu-compute")]
const HIGH_EDIT_VOLUME_THRESHOLD: usize = 3072;
#[cfg(feature = "gpu-compute")]
const SAFE_ACTIVE_FRONTIER_LIMIT: u32 = 2048;
#[cfg(feature = "gpu-compute")]
const STARTUP_JACOBI_ITERATIONS: u32 = 8;
#[cfg(feature = "gpu-compute")]
const HIGH_PRESSURE_JACOBI_ITERATIONS: u32 = 16;
#[derive(Default, Clone, Copy)]
pub struct GpuComputeProfilerSnapshot {
    pub dispatch_ms: f32,
    pub bytes_transferred: u64,
    pub chunks_completed: u64,
    pub frontier_cap_events: u64,
    pub mesh_slot_alloc_failed: u64,
    pub chunks_per_sec: f32,
}

#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TRANSFER_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_CHUNKS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_FRONTIER_CAP_EVENTS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_SLOT_ALLOC_FAILED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_ERRORS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_STARTUP_ZERO_FRONTIER_MESH_RUNS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static ACTIVE_GPU_JOBS_BITS: std::sync::LazyLock<Vec<AtomicU64>> =
    std::sync::LazyLock::new(|| (0..1024).map(|_| AtomicU64::new(0)).collect());
#[cfg(feature = "gpu-compute")]
static GPU_TASK_TX: OnceLock<SyncSender<GpuChunkTask>> = OnceLock::new();
#[cfg(feature = "gpu-compute")]
static GPU_TASK_RX: OnceLock<Mutex<Receiver<GpuChunkTask>>> = OnceLock::new();
#[cfg(feature = "gpu-compute")]
static GPU_SUBMISSION_SERIAL: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_COMPLETED_SERIAL: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDispatchFrameStats {
    pub tasks_submitted: usize,
    pub enqueue_submit_ms: f32,
    pub wait_sync_ms: f32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone)]
pub struct GpuChunkTask {
    pub coord: ChunkCoord,
    pub page_index: GpuPageIndex,
    pub frontier_count: u32,
    pub edit_commands: Vec<EditCommand>,
    pub jacobi_iterations: u32,
    pub neighbor_pages: [u32; 6],
    pub simulation_tick: u32,
    pub current_state: u32,
    pub startup_seeding_mode: bool,
    pub mesh_slice: Option<MeshBufferSlice>,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub struct MeshBufferSlice {
    pub slot_index: u32,
    pub vertex_offset: u32,
    pub index_offset: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default)]
struct PageFence {
    last_submitted: u64,
    last_completed: u64,
}

#[cfg(feature = "gpu-compute")]
fn active_job_slot(coord: ChunkCoord) -> (usize, u64) {
    let h = (coord.x as i64).wrapping_mul(73856093)
        ^ (coord.y as i64).wrapping_mul(19349663)
        ^ (coord.z as i64).wrapping_mul(83492791);
    let idx = (h.unsigned_abs() as usize) % (ACTIVE_GPU_JOBS_BITS.len() * 64);
    (idx / 64, 1u64 << (idx % 64))
}

#[cfg(feature = "gpu-compute")]
fn try_acquire_active_job(coord: ChunkCoord) -> bool {
    let (word, mask) = active_job_slot(coord);
    let bits = &ACTIVE_GPU_JOBS_BITS[word];
    loop {
        let cur = bits.load(Ordering::Relaxed);
        if cur & mask != 0 {
            return false;
        }
        if bits
            .compare_exchange_weak(cur, cur | mask, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return true;
        }
    }
}

#[cfg(feature = "gpu-compute")]
fn release_active_job(coord: ChunkCoord) {
    let (word, mask) = active_job_slot(coord);
    ACTIVE_GPU_JOBS_BITS[word].fetch_and(!mask, Ordering::Release);
}
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
        let frontier_cap_events = GPU_FRONTIER_CAP_EVENTS.swap(0, Ordering::Relaxed);
        let mesh_slot_alloc_failed = GPU_MESH_SLOT_ALLOC_FAILED.swap(0, Ordering::Relaxed);
        let frame = frame_seconds.max(0.000_1);
        GpuComputeProfilerSnapshot {
            dispatch_ms: dispatch_ns as f32 / 1_000_000.0,
            bytes_transferred,
            chunks_completed,
            frontier_cap_events,
            mesh_slot_alloc_failed,
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
            let vertex_face_limit = GPU_MESH_VERTEX_CAPACITY_PER_PAGE / 4;
            let index_face_limit = GPU_MESH_INDEX_CAPACITY_PER_PAGE / 6;
            let max_faces = vertex_face_limit.min(index_face_limit);
            let generated_consts = format!("const GPU_MAX_FACES_PER_PAGE: u32 = {}u;", max_faces);
            let meshing_source = format!(
                "{}\n{}\n{}",
                generated_material_ids_wgsl,
                generated_consts,
                include_str!("shaders/meshing.wgsl")
            );
            let meshing_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk meshing shader"),
                source: wgpu::ShaderSource::Wgsl(meshing_source.into()),
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
                bgl_entry(0, true),
                bgl_entry(1, false),
                bgl_entry(2, false),
                bgl_entry(3, false),
                bgl_entry(4, false),
                bgl_entry(5, false),
                bgl_entry(6, true),
                bgl_entry(7, false),
                bgl_entry(8, false),
                bgl_entry(9, true),
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
            let detect_faces_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing detect faces pipeline"),
                    layout: Some(&meshing_pl),
                    module: &meshing_module,
                    entry_point: "detect_faces",
                });
            let prefix_scan_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing prefix scan pipeline"),
                    layout: Some(&meshing_pl),
                    module: &meshing_module,
                    entry_point: "prefix_scan",
                });
            let emit_mesh_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("meshing emit mesh pipeline"),
                    layout: Some(&meshing_pl),
                    module: &meshing_module,
                    entry_point: "emit_mesh",
                });

            Some(Self {
                force_pipeline,
                advect_pipeline,
                divergence_pipeline,
                pressure_jacobi_pipeline,
                project_pipeline,
                material_advect_pipeline,
                detect_faces_pipeline,
                prefix_scan_pipeline,
                emit_mesh_pipeline,
                simulation_bgl,
                meshing_bgl,
            })
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn run_active_frontier(
        &self,
        state: &WorkerGpuState,
        scratch: &GpuScratchPool,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        edit_commands: &[EditCommand],
        max_jacobi_iterations: u32,
        neighbor_pages: [u32; 6],
    ) -> anyhow::Result<()> {
        let t0 = Instant::now();
        // `SimulationJob::materials` is intentionally empty on the renderer dispatch path
        // to avoid per-dispatch CHUNK_VOLUME host allocations. Clamp by chunk volume directly.
        let frontier_len = sim_job.active_frontier_count.min(CHUNK_VOLUME as u32);

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

        let mut dispatch = |pipeline: &wgpu::ComputePipeline, jacobi_iteration: u32| {
            let params = base_params(jacobi_iteration);
            state
                .queue
                .write_buffer(&scratch.page_params, 0, bytemuck::cast_slice(&params));
            let mut encoder = state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &state.simulation_bg, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            drop(pass);
            state.queue.submit(Some(encoder.finish()));
        };

        dispatch(&self.force_pipeline, 0);
        dispatch(&self.advect_pipeline, 0);
        dispatch(&self.divergence_pipeline, 0);
        for jacobi_iter in 0..max_jacobi_iterations {
            dispatch(&self.pressure_jacobi_pipeline, jacobi_iter);
        }
        dispatch(&self.project_pipeline, max_jacobi_iterations);
        dispatch(&self.material_advect_pipeline, max_jacobi_iterations);

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
        scratch: &GpuScratchPool,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        lod: u8,
        mesh_slice: MeshBufferSlice,
    ) -> anyhow::Result<MeshArtifactGPU> {
        clear_meshing_outputs_for_page(state, page_index, mesh_slice);

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

        let chunk_span_world = CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
        let origin = [
            sim_job.chunk_coord.x as f32 * chunk_span_world,
            sim_job.chunk_coord.y as f32 * chunk_span_world,
            sim_job.chunk_coord.z as f32 * chunk_span_world,
            0.0,
        ];
        let origin_stride = std::mem::size_of::<[f32; 4]>() as u64;
        let origin_offset = page_index.0 as u64 * origin_stride;
        state.queue.write_buffer(
            &state.chunk_origin_buffer,
            origin_offset,
            bytemuck::cast_slice(&origin),
        );

        let groups = (CHUNK_VOLUME as u32).div_ceil(128);
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
            pass.set_bind_group(0, &state.meshing_bg, &[]);
            pass.set_pipeline(&self.detect_faces_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.prefix_scan_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.emit_mesh_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        state.queue.submit(Some(encoder.finish()));

        Ok(MeshArtifactGPU {
            page_index,
            lod,
            draw_indirect_index: mesh_slice.slot_index,
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
                    binding: 1,
                    resource: resources.face_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.face_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.chunk_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.chunk_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.draw_indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.page_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: resources.face_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: resources.mesh_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: resources.chunk_origin_buffer.as_entire_binding(),
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
    if GPU_TASK_TX.get().is_none() {
        let (tx, rx) = sync_channel(4096);
        let _ = GPU_TASK_TX.set(tx);
        let _ = GPU_TASK_RX.set(Mutex::new(rx));
    }
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

        let scratch = create_gpu_scratch_pool(&device);
        let simulation_bg = runtime.create_simulation_bind_group(
            &device,
            SimulationBindResources {
                atlas_voxels: &atlas_voxels,
                velocity_mac: &velocity_mac,
                pressure: &pressure,
                divergence: &divergence,
                material_density: &material_density,
                active_tiles: &scratch.active_tiles,
                active_tile_counter: &scratch.active_tile_counter,
                edit_commands: &scratch.edit_commands,
                page_params: &scratch.page_params,
                diagnostics: &scratch.diagnostics,
            },
        );
        let meshing_bg = runtime.create_meshing_bind_group(
            &device,
            MeshingBindResources {
                atlas_voxels: &atlas_voxels,
                page_params: &scratch.page_params,
                face_mask_buffer: &shared_mesh_buffers.face_mask_buffer,
                face_offset_buffer: &shared_mesh_buffers.face_offset_buffer,
                face_count_buffer: &shared_mesh_buffers.face_count_buffer,
                chunk_vertex_buffer: &shared_mesh_buffers.chunk_vertex_buffer,
                chunk_index_buffer: &shared_mesh_buffers.chunk_index_buffer,
                draw_indirect_buffer: &shared_mesh_buffers.draw_indirect_buffer,
                mesh_meta_buffer: &shared_mesh_buffers.mesh_meta_buffer,
                chunk_origin_buffer: &shared_mesh_buffers.chunk_origin_buffer,
            },
        );

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
            chunk_origin_buffer: shared_mesh_buffers.chunk_origin_buffer,
            face_mask_buffer: shared_mesh_buffers.face_mask_buffer,
            face_offset_buffer: shared_mesh_buffers.face_offset_buffer,
            face_count_buffer: shared_mesh_buffers.face_count_buffer,
            runtime_config: GpuSimulationRuntimeConfig::default(),
            scratch,
            simulation_bg,
            meshing_bg,
        }))
    });

    if let Err(err) = state_result.as_ref() {
        return Err(anyhow::anyhow!(err.to_string()));
    }
    Ok(())
}
struct ActiveJobGuard {
    coord: ChunkCoord,
}

impl Drop for ActiveJobGuard {
    fn drop(&mut self) {
        release_active_job(self.coord);
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
        let state = WORKER_STATE
            .get()
            .context(
                "gpu worker runtime is not initialized; renderer must call initialize_gpu_compute_worker",
            )?
            .as_ref()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .clone();

        if !try_acquire_active_job(job.coord) {
            anyhow::bail!("mesh job already active for {:?}", job.coord);
        }

        // Ensure the job slot is released on all exits.
        let _active_job_guard = ActiveJobGuard { coord: job.coord };

        let incoming = job.snapshot.center_voxels.as_ref();

        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());

        let (page_index, page_was_reassigned) = atlas.page_for_chunk_or_allocate(job.coord)?;
        atlas.assert_page_for_chunk(job.coord, page_index);

        let mesh_slot_capacity = mesh_pool_slot_capacity();
        let mesh_slice = atlas.mesh_slice_for_chunk_or_allocate(job.coord);
        let mesh_slot_in_flight_fences = atlas.in_flight_mesh_slot_fence_count();

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

        let neighbor_pages = neighbor_pages_for_chunk(&atlas, job.coord);

        drop(atlas);

        // Build edit commands
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

        let raw_active_frontier_count = edit_commands.len() as u32;

        let high_edit_volume = edit_commands.len() > HIGH_EDIT_VOLUME_THRESHOLD;
        let startup_seeding_mode = page_was_reassigned;
        let pressure_relief_mode = high_edit_volume;

        let active_frontier_count = if pressure_relief_mode {
            raw_active_frontier_count.min(SAFE_ACTIVE_FRONTIER_LIMIT)
        } else {
            raw_active_frontier_count
        };
        let ran_simulation = active_frontier_count > 0 || !edit_commands.is_empty();

        let jacobi_iterations = if startup_seeding_mode {
            STARTUP_JACOBI_ITERATIONS
        } else if pressure_relief_mode {
            HIGH_PRESSURE_JACOBI_ITERATIONS
        } else {
            state.runtime_config.max_jacobi_iterations
        };

        // Queue GPU task
        GPU_TASK_TX
            .get()
            .expect("gpu task queue")
            .send(GpuChunkTask {
                coord: job.coord,
                page_index,
                frontier_count: active_frontier_count,
                edit_commands,
                jacobi_iterations,
                neighbor_pages,
                simulation_tick: tick,
                current_state,
                startup_seeding_mode,
                mesh_slice,
            })
            .context("failed to send GPU chunk task")?;

        let diagnostics = ChunkSimulationDiagnostics::default();

        let next_state = if ran_simulation {
            current_state ^ 1
        } else {
            current_state
        };

        {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());

            atlas.version_for_chunk.insert(job.coord, job.version);
            atlas.state_for_chunk.insert(job.coord, next_state);
            atlas.tick_for_chunk.insert(job.coord, tick.wrapping_add(1));

            atlas
                .frontier_len_for_chunk
                .insert(job.coord, active_frontier_count);

            atlas.diagnostics_for_chunk.insert(job.coord, diagnostics);
            atlas.cached_materials.insert(job.coord, incoming.to_vec());
        }

        // CPU fallback meshing
        #[cfg(not(feature = "gpu_meshing_experimental"))]
        let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
            mesh_chunk_snapshot(job.coord, &job.snapshot, job.lod, job.greedy);

        // GPU meshing path
        #[cfg(feature = "gpu_meshing_experimental")]
        let (verts, inds, aabb_min, aabb_max, chunk_origin_world) = {
            let min = glam::Vec3::new(
                job.coord.x as f32 * CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE,
                job.coord.y as f32 * CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE,
                job.coord.z as f32 * CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE,
            );
            let extent = glam::Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);
            (Vec::new(), Vec::new(), min, min + extent, min)
        };

        Ok(ComputedChunkArtifacts {
            simulation_diagnostics: diagnostics,
            mesh_artifact: {
                #[cfg(feature = "gpu_meshing_experimental")]
                {
                    if let Some(mesh_slice) = mesh_slice {
                        ChunkMeshArtifact::Gpu {
                            page_index,
                            draw_indirect_index: mesh_slice.slot_index,
                            // Completion token for non-blocking renderer adoption.
                            // Real draw args are still sourced from the GPU indirect buffer.
                            index_count: 1,
                            lod: job.lod as u8,
                            verts,
                            inds,
                            aabb_min,
                            aabb_max,
                            chunk_origin_world,
                            dispatch_ms: 0.0,
                        }
                    } else {
                        let _ = GPU_MESH_SLOT_ALLOC_FAILED.fetch_add(1, Ordering::Relaxed);

                        log::debug!(
                            "[mesh] skipping gpu meshing for {:?}: global mesh pool exhausted (slot_capacity={}, in_flight_fences={})",
                            job.coord,
                            mesh_slot_capacity,
                            mesh_slot_in_flight_fences,
                        );

                        ChunkMeshArtifact::Skipped {
                            reason: MeshSkipReason::MeshSlotCapacitySaturated {
                                slot_capacity: mesh_slot_capacity,
                                in_flight_fences: mesh_slot_in_flight_fences,
                            },
                        }
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
    }
}
#[cfg(feature = "gpu-compute")]
pub fn dispatch_gpu_chunk_tasks_on_renderer(
    max_tasks: usize,
    max_dispatch_time: Duration,
) -> anyhow::Result<GpuDispatchFrameStats> {
    let state = WORKER_STATE
        .get()
        .context("gpu worker runtime is not initialized")?
        .as_ref()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .clone();

    let rx = GPU_TASK_RX
        .get()
        .context("gpu task receiver missing")?
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let frame_dispatch_start = Instant::now();
    let mut stats = GpuDispatchFrameStats::default();

    while stats.tasks_submitted < max_tasks {
        if frame_dispatch_start.elapsed() >= max_dispatch_time {
            break;
        }

        let task = match rx.try_recv() {
            Ok(t) => t,
            Err(_) => break,
        };

        let task_start = Instant::now();
        let mut task_wait_sync = Duration::ZERO;

        let scratch = &state.scratch;

        // Reset scratch buffers
        clear_gpu_scratch_pool(&state, scratch, task.frontier_count as usize, 1);

        // Upload edit commands
        if !task.edit_commands.is_empty() {
            state.queue.write_buffer(
                &scratch.edit_commands,
                0,
                bytemuck::cast_slice(&task.edit_commands),
            );

            let mut active = Vec::with_capacity(task.frontier_count as usize);

            for cmd in task.edit_commands.iter().take(task.frontier_count as usize) {
                active.push(cmd.voxel_index);
            }

            if !active.is_empty() {
                state
                    .queue
                    .write_buffer(&scratch.active_tiles, 0, bytemuck::cast_slice(&active));
            }
        }

        let sim_job = SimulationJob {
            chunk_coord: task.coord,
            materials: Vec::new(),
            active_frontier_count: task.frontier_count,
            simulation_tick: task.simulation_tick,
        };
        let ran_simulation = task.frontier_count > 0 || !task.edit_commands.is_empty();
        let meshing_state = if ran_simulation {
            task.current_state ^ 1
        } else {
            task.current_state
        };

        if task.startup_seeding_mode {
            let mut encoder =
                state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("gpu_page_clear"),
                    });

            clear_page_buffers(&mut encoder, &state, task.page_index);

            state.queue.submit(Some(encoder.finish()));
        }

        if task.frontier_count > 0 || !task.edit_commands.is_empty() {
            state.runtime.run_active_frontier(
                &state,
                scratch,
                &sim_job,
                task.page_index,
                task.current_state,
                &task.edit_commands,
                task.jacobi_iterations,
                task.neighbor_pages,
            )?;
        }

        #[cfg(feature = "gpu_meshing_experimental")]
        {
            if let Some(mesh_slice) = task.mesh_slice {
                if task.startup_seeding_mode && task.frontier_count == 0 {
                    let runs =
                        GPU_STARTUP_ZERO_FRONTIER_MESH_RUNS.fetch_add(1, Ordering::Relaxed) + 1;
                    log::debug!(
                        "[mesh] startup seeding ran meshing with empty frontier for {:?} (runs={}, edit_commands={})",
                        task.coord,
                        runs,
                        task.edit_commands.len(),
                    );
                }

                state.runtime.run_meshing_dispatch(
                    &state,
                    scratch,
                    &sim_job,
                    task.page_index,
                    meshing_state,
                    0,
                    mesh_slice,
                )?;

                if task.startup_seeding_mode
                    && !task.edit_commands.is_empty()
                    && task.coord.x.abs() <= 1
                    && task.coord.y.abs() <= 1
                    && task.coord.z.abs() <= 1
                {
                    let draw = read_gpu_draw_indexed_indirect(
                        &state.device,
                        &state.queue,
                        &state.draw_indirect_buffer,
                        mesh_slice.slot_index as usize,
                    )?;
                    task_wait_sync += draw.1;
                    let draw = draw.0;
                    if draw.index_count == 0 {
                        log::warn!(
                            "[mesh] startup seeding produced zero index_count for chunk {:?} page={} slot={} current_state={} meshing_state={}; scheduling immediate meshing retry after voxel edits completion",
                            task.coord,
                            task.page_index.0,
                            mesh_slice.slot_index,
                            task.current_state,
                            meshing_state,
                        );

                        state.runtime.run_meshing_dispatch(
                            &state,
                            scratch,
                            &sim_job,
                            task.page_index,
                            meshing_state,
                            0,
                            mesh_slice,
                        )?;

                        let retry_draw = read_gpu_draw_indexed_indirect(
                            &state.device,
                            &state.queue,
                            &state.draw_indirect_buffer,
                            mesh_slice.slot_index as usize,
                        )?;
                        task_wait_sync += retry_draw.1;
                        let retry_draw = retry_draw.0;
                        if retry_draw.index_count == 0 {
                            log::warn!(
                                "[mesh] startup seeding retry still zero for chunk {:?} page={} slot={} current_state={} meshing_state={}",
                                task.coord,
                                task.page_index.0,
                                mesh_slice.slot_index,
                                task.current_state,
                                meshing_state,
                            );
                        } else {
                            log::debug!(
                                "[mesh] startup seeding retry produced index_count={} for chunk {:?} page={} slot={} current_state={} meshing_state={}",
                                retry_draw.index_count,
                                task.coord,
                                task.page_index.0,
                                mesh_slice.slot_index,
                                task.current_state,
                                meshing_state,
                            );
                        }
                    } else {
                        log::debug!(
                            "[mesh] startup seeding validated indirect index_count={} for chunk {:?} page={} slot={} current_state={} meshing_state={}",
                            draw.index_count,
                            task.coord,
                            task.page_index.0,
                            mesh_slice.slot_index,
                            task.current_state,
                            meshing_state,
                        );
                    }
                }
            }
        }

        let serial = GPU_SUBMISSION_SERIAL.fetch_add(1, Ordering::Relaxed) + 1;

        {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            atlas.mark_page_submitted(task.page_index, serial);
        }

        state.queue.on_submitted_work_done(move || {
            GPU_COMPLETED_SERIAL.fetch_max(serial, Ordering::Relaxed);
        });

        stats.tasks_submitted += 1;
        stats.wait_sync_ms += task_wait_sync.as_secs_f32() * 1000.0;
        stats.enqueue_submit_ms += task_start
            .elapsed()
            .saturating_sub(task_wait_sync)
            .as_secs_f32()
            * 1000.0;
    }

    Ok(stats)
}

#[cfg(all(feature = "gpu-compute", feature = "gpu_meshing_experimental"))]
fn read_gpu_draw_indexed_indirect(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    indirect_buffer: &wgpu::Buffer,
    slot: usize,
) -> anyhow::Result<(DrawIndexedIndirectArgs, Duration)> {
    use std::sync::mpsc::channel;

    let args_size = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
    let offset = slot as u64 * args_size;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("startup_indirect_readback_staging"),
        size: args_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("startup_indirect_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(indirect_buffer, offset, &staging, 0, args_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        tx.send(v).ok();
    });
    let wait_start = Instant::now();
    device.poll(wgpu::Maintain::Wait);
    let wait_elapsed = wait_start.elapsed();

    rx.recv()
        .context("failed waiting for startup indirect readback")??;

    let data = slice.get_mapped_range();
    let args = *bytemuck::from_bytes::<DrawIndexedIndirectArgs>(&data);
    drop(data);
    staging.unmap();
    Ok((args, wait_elapsed))
}
#[cfg(feature = "gpu-compute")]
pub fn update_gpu_page_fences_on_renderer() {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let completed = GPU_COMPLETED_SERIAL.load(Ordering::Relaxed);
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.refresh_completed_serial(completed);
    }
}

#[cfg(feature = "gpu-compute")]
pub fn gpu_page_ready_for_adoption(page_index: GpuPageIndex) -> bool {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let completed = GPU_COMPLETED_SERIAL.load(Ordering::Relaxed);
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.refresh_completed_serial(completed);
        return atlas.is_page_fence_complete(page_index);
    }
    false
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
        // Renderer-side GPU dispatch does not populate `SimulationJob::materials`.
        // Always use full chunk volume for shader bounds/loops.
        voxel_count: CHUNK_VOLUME as u32,
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
    pub color: u32,
}

#[cfg(feature = "gpu-compute")]
const _: [(); std::mem::size_of::<GpuVertex>()] =
    [(); std::mem::size_of::<crate::renderer::Vertex>()];

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct ChunkMeshMeta {
    pub slot_index: u32,
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
