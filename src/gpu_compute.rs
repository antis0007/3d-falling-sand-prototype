use crate::engine::world::ChunkVersion;
use crate::mesh_layout::{
    self, MeshBufferKind, MESH_INDEX_ELEMENTS_PER_CHUNK, MESH_SLOT_COUNT,
    MESH_VERTEX_ELEMENTS_PER_CHUNK,
};
use crate::renderer::mesh_chunk_snapshot;
use crate::renderer::{ChunkMeshArtifact, MeshJob, MeshSkipReason, VOXEL_SIZE};
use crate::startup_gpu_budget::{finalize_startup_budget, StartupGpuBudget};
use crate::types::{ChunkCoord, GpuPageIndex, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use crossbeam_channel as cbc;
#[cfg(feature = "gpu-compute")]
use glam::Vec3;
#[cfg(feature = "gpu-compute")]
use std::collections::HashMap;
#[cfg(feature = "gpu-compute")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
#[cfg(feature = "gpu-compute")]
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
const CHUNK_VOLUME: usize = mesh_layout::CHUNK_VOLUME_VOXELS as usize;
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
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u64 = MESH_VERTEX_ELEMENTS_PER_CHUNK as u64;
#[cfg(feature = "gpu-compute")]
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u64 = MESH_INDEX_ELEMENTS_PER_CHUNK as u64;
#[cfg(feature = "gpu-compute")]
pub use crate::mesh_layout::{
    GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES, GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES,
};

#[cfg(feature = "gpu-compute")]
const fn atlas_voxel_size_bytes() -> u64 {
    (MESH_SLOT_COUNT as u64) * (CHUNK_VOLUME as u64) * 2 * std::mem::size_of::<u32>() as u64
}

#[cfg(feature = "gpu-compute")]
const fn velocity_mac_size_bytes() -> u64 {
    (MESH_SLOT_COUNT as u64) * (MAC_TOTAL_COUNT as u64) * 2 * std::mem::size_of::<f32>() as u64
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
    MESH_SLOT_COUNT
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshPipelineBackend {
    Disabled,
    #[cfg(feature = "gpu-compute")]
    Gpu,
}

impl MeshPipelineBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
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
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuWorkerStartupPlan {
    pub reduce_scratch_footprint: bool,
    pub disable_draw_indirect_readback: bool,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub struct GpuComputeStartupBufferSizes {
    pub atlas_voxels: u64,
    pub velocity_mac: u64,
    pub pressure: u64,
    pub divergence: u64,
    pub material_density: u64,
    pub scratch_total: u64,
    pub draw_indirect_readback: u64,
}

#[cfg(feature = "gpu-compute")]
pub fn compute_startup_buffer_sizes(plan: GpuWorkerStartupPlan) -> GpuComputeStartupBufferSizes {
    let page_len = CHUNK_VOLUME as u64;
    let page_capacity = MESH_SLOT_COUNT as u64;
    let scratch_active_tiles = page_len * std::mem::size_of::<u32>() as u64;
    let scratch_edit_commands =
        MAX_EDIT_COMMANDS as u64 * std::mem::size_of::<EditCommand>() as u64;
    let scratch_total = std::mem::size_of::<FrameParams>() as u64
        + if plan.reduce_scratch_footprint {
            scratch_active_tiles / 2
        } else {
            scratch_active_tiles
        }
        + 16
        + if plan.reduce_scratch_footprint {
            scratch_edit_commands / 2
        } else {
            scratch_edit_commands
        }
        + page_capacity * std::mem::size_of::<u32>() as u64
        + 16
        + 16;

    GpuComputeStartupBufferSizes {
        atlas_voxels: atlas_voxel_size_bytes(),
        velocity_mac: velocity_mac_size_bytes(),
        pressure: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
        divergence: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
        material_density: page_capacity * page_len * 2 * std::mem::size_of::<f32>() as u64,
        scratch_total,
        draw_indirect_readback: if plan.disable_draw_indirect_readback {
            0
        } else {
            std::mem::size_of::<DrawIndexedIndirectArgs>() as u64 * MESH_SLOT_COUNT as u64
        },
    }
}

#[cfg(feature = "gpu-compute")]
pub fn plan_gpu_worker_startup(
    device_max_buffer_size: u64,
) -> anyhow::Result<GpuWorkerStartupPlan> {
    let mut plan = GpuWorkerStartupPlan::default();
    for pass in 0..=2 {
        let sizes = compute_startup_buffer_sizes(plan);
        let mut budget = StartupGpuBudget::new(device_max_buffer_size, device_max_buffer_size / 3);
        budget.register_plan("compute_atlas", sizes.atlas_voxels);
        budget.register_plan("compute_velocity", sizes.velocity_mac);
        budget.register_plan("compute_pressure", sizes.pressure);
        budget.register_plan("compute_divergence", sizes.divergence);
        budget.register_plan("compute_material", sizes.material_density);
        budget.register_plan("compute_scratch", sizes.scratch_total);
        budget.register_plan("compute_readback", sizes.draw_indirect_readback);
        for category in [
            "compute_atlas",
            "compute_velocity",
            "compute_pressure",
            "compute_divergence",
            "compute_material",
            "compute_scratch",
            "compute_readback",
        ] {
            budget.grant_planned(category);
        }

        if let Ok(decision) = finalize_startup_budget(budget, device_max_buffer_size / 8) {
            log::info!(
                "gpu worker startup budget plan: downgraded={} planned={}B granted={}B",
                decision.downgraded,
                decision.budget.planned_total(),
                decision.budget.granted_total(),
            );
            return Ok(plan);
        }

        if pass == 0 {
            plan.reduce_scratch_footprint = true;
        } else if pass == 1 {
            plan.disable_draw_indirect_readback = true;
        }
    }
    anyhow::bail!("gpu worker startup planning exceeded budget after downgrade attempts")
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
struct ChunkPageAtlas {
    page_for_chunk: HashMap<ChunkCoord, GpuPageIndex>,
    chunk_for_page: HashMap<GpuPageIndex, ChunkCoord>,
    page_fences: HashMap<GpuPageIndex, PageFence>,
    page_last_used: HashMap<GpuPageIndex, u64>,
    page_usage_epoch: u64,
    priority_hint_for_chunk: HashMap<ChunkCoord, PagePriorityHint>,
    version_for_chunk: HashMap<ChunkCoord, u64>,
    state_for_chunk: HashMap<ChunkCoord, u32>,
    frontier_len_for_chunk: HashMap<ChunkCoord, u32>,
    tick_for_chunk: HashMap<ChunkCoord, u32>,
    diagnostics_for_chunk: HashMap<ChunkCoord, ChunkSimulationDiagnostics>,
    cached_materials: HashMap<ChunkCoord, Vec<MaterialId>>,
    mesh_slice_for_chunk: HashMap<ChunkCoord, MeshBufferSlice>,
    chunk_for_mesh_slot: HashMap<u32, ChunkCoord>,
    page_ownership_generation: HashMap<GpuPageIndex, u64>,
    mesh_slot_ownership_generation: HashMap<u32, u64>,
    mesh_slot_last_used: HashMap<u32, u64>,
    queued_page_reservations: HashMap<GpuPageIndex, u32>,
    queued_slot_reservations: HashMap<u32, u32>,
    mesh_slot_epoch: u64,
    next_mesh_slot: u32,
    mesh_slot_capacity_vertex_elements: u32,
    mesh_slot_capacity_index_elements: u32,
    next_page: GpuPageIndex,
    pending_mesh_finalize: HashMap<ChunkCoord, PendingGpuMeshFinalize>,
    queued_task_reservations: HashMap<ChunkCoord, QueuedTaskReservation>,
}

#[cfg(feature = "gpu-compute")]
impl Default for ChunkPageAtlas {
    fn default() -> Self {
        Self {
            page_for_chunk: HashMap::new(),
            chunk_for_page: HashMap::new(),
            page_fences: HashMap::new(),
            page_last_used: HashMap::new(),
            page_usage_epoch: 0,
            priority_hint_for_chunk: HashMap::new(),
            version_for_chunk: HashMap::new(),
            state_for_chunk: HashMap::new(),
            frontier_len_for_chunk: HashMap::new(),
            tick_for_chunk: HashMap::new(),
            diagnostics_for_chunk: HashMap::new(),
            cached_materials: HashMap::new(),
            mesh_slice_for_chunk: HashMap::new(),
            chunk_for_mesh_slot: HashMap::new(),
            page_ownership_generation: HashMap::new(),
            mesh_slot_ownership_generation: HashMap::new(),
            mesh_slot_last_used: HashMap::new(),
            queued_page_reservations: HashMap::new(),
            queued_slot_reservations: HashMap::new(),
            mesh_slot_epoch: 0,
            next_mesh_slot: 0,
            mesh_slot_capacity_vertex_elements: MESH_VERTEX_ELEMENTS_PER_CHUNK,
            mesh_slot_capacity_index_elements: MESH_INDEX_ELEMENTS_PER_CHUNK,
            next_page: GpuPageIndex(0),
            pending_mesh_finalize: HashMap::new(),
            queued_task_reservations: HashMap::new(),
        }
    }
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
enum MeshSliceAllocateOutcome {
    Success(MeshBufferSlice),
    Saturated,
    NoProgress,
    InvalidState,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
struct MeshPoolTelemetry {
    slot_capacity: u32,
    slots_used: u32,
    in_flight_fences: u32,
    vertex_used: u32,
    vertex_capacity: u32,
    index_used: u32,
    index_capacity: u32,
    largest_free_vertex_span: u32,
    largest_free_index_span: u32,
    vertex_usage_percent: f32,
    index_usage_percent: f32,
    slot_usage_percent: f32,
    slots_visible: u32,
    slots_pending_finalize: u32,
    reclaimable_without_fence: u32,
    lifecycle_blocked_without_fence: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PagePriorityHint {
    #[default]
    Far,
    Near,
    Visible,
}

#[cfg(feature = "gpu-compute")]
impl PagePriorityHint {
    fn eviction_rank(self) -> u8 {
        match self {
            // Lower rank is evicted earlier.
            Self::Far => 0,
            Self::Near => 1,
            Self::Visible => 2,
        }
    }
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingGpuMeshFinalize {
    version: ChunkVersion,
    task_id: u64,
    lod: u8,
    page_index: GpuPageIndex,
    draw_indirect_index: u32,
    page_generation: u64,
    slot_generation: u64,
    submission_serial: u64,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueuedTaskReservation {
    task_id: u64,
    version: ChunkVersion,
    page_index: GpuPageIndex,
    mesh_slice: Option<MeshBufferSlice>,
}

#[cfg(feature = "gpu-compute")]
impl ChunkPageAtlas {
    fn bump_page_generation(&mut self, page: GpuPageIndex) -> u64 {
        let entry = self.page_ownership_generation.entry(page).or_insert(0);
        *entry = entry.saturating_add(1);
        *entry
    }

    fn bump_mesh_slot_generation(&mut self, slot: u32) -> u64 {
        let entry = self.mesh_slot_ownership_generation.entry(slot).or_insert(0);
        *entry = entry.saturating_add(1);
        *entry
    }

    fn page_generation(&self, page: GpuPageIndex) -> u64 {
        self.page_ownership_generation
            .get(&page)
            .copied()
            .unwrap_or(0)
    }

    fn slot_generation(&self, slot: u32) -> u64 {
        self.mesh_slot_ownership_generation
            .get(&slot)
            .copied()
            .unwrap_or(0)
    }

    fn page_for_chunk_or_allocate(
        &mut self,
        chunk: ChunkCoord,
    ) -> anyhow::Result<(GpuPageIndex, bool)> {
        if let Some(existing) = self.page_for_chunk.get(&chunk).copied() {
            self.touch_page(existing);
            return Ok((existing, false));
        }

        if self.next_page.0 < MESH_SLOT_COUNT {
            let page = self.next_page;
            self.next_page = GpuPageIndex(self.next_page.0.saturating_add(1));
            self.page_for_chunk.insert(chunk, page);
            self.chunk_for_page.insert(page, chunk);
            self.bump_page_generation(page);
            self.touch_page(page);
            return Ok((page, true));
        }

        let page = self.evictable_page().with_context(|| {
            format!(
                "gpu page allocator saturated for chunk {chunk:?}: no fence-safe eviction candidate"
            )
        })?;
        self.evict_page(page);
        self.page_for_chunk.insert(chunk, page);
        self.chunk_for_page.insert(page, chunk);
        self.bump_page_generation(page);
        self.touch_page(page);
        Ok((page, true))
    }

    fn touch_page(&mut self, page: GpuPageIndex) {
        self.page_usage_epoch = self.page_usage_epoch.saturating_add(1);
        self.page_last_used.insert(page, self.page_usage_epoch);
    }

    fn touch_chunk_page(&mut self, chunk: ChunkCoord) {
        if let Some(page) = self.page_for_chunk.get(&chunk).copied() {
            self.touch_page(page);
        }
    }

    fn set_protected_chunks(&mut self, near_chunks: &[ChunkCoord], visible_chunks: &[ChunkCoord]) {
        self.priority_hint_for_chunk.clear();
        for &coord in near_chunks {
            self.priority_hint_for_chunk
                .insert(coord, PagePriorityHint::Near);
        }
        for &coord in visible_chunks {
            self.priority_hint_for_chunk
                .insert(coord, PagePriorityHint::Visible);
        }
    }

    fn mesh_slice_for_chunk_or_allocate(
        &mut self,
        chunk: ChunkCoord,
        lod: u8,
    ) -> MeshSliceAllocateOutcome {
        if let Some(existing) = self.mesh_slice_for_chunk.get(&chunk).copied() {
            self.touch_mesh_slot(existing.slot_index);
            return MeshSliceAllocateOutcome::Success(existing);
        }

        let _ = lod;
        let slot_capacity = MESH_SLOT_COUNT;
        if slot_capacity == 0 {
            return MeshSliceAllocateOutcome::InvalidState;
        }

        let mut reused_slot = false;
        let slot = if self.next_mesh_slot < slot_capacity {
            let next = self.next_mesh_slot;
            self.next_mesh_slot = self.next_mesh_slot.saturating_add(1);
            next
        } else {
            reused_slot = true;
            match self.evictable_mesh_slot(slot_capacity) {
                Some(evictable) => evictable,
                None => return MeshSliceAllocateOutcome::Saturated,
            }
        };

        if reused_slot && !self.evict_mesh_slot(slot) {
            return MeshSliceAllocateOutcome::NoProgress;
        }

        if let Some(slice) = self.try_allocate_mesh_buffers(chunk, slot) {
            return MeshSliceAllocateOutcome::Success(slice);
        }

        let Some(evict_slot) = self.evictable_mesh_slot(slot_capacity) else {
            return MeshSliceAllocateOutcome::Saturated;
        };
        if !self.evict_mesh_slot(evict_slot) {
            return MeshSliceAllocateOutcome::NoProgress;
        }

        if let Some(slice) = self.try_allocate_mesh_buffers(chunk, evict_slot) {
            return MeshSliceAllocateOutcome::Success(slice);
        }

        MeshSliceAllocateOutcome::Saturated
    }

    fn try_allocate_mesh_buffers(
        &mut self,
        chunk: ChunkCoord,
        slot: u32,
    ) -> Option<MeshBufferSlice> {
        let vertex_offset = mesh_layout::slot_base_offset_elements(MeshBufferKind::Vertex, slot)?;
        let index_offset = mesh_layout::slot_base_offset_elements(MeshBufferKind::Index, slot)?;

        let slice = MeshBufferSlice {
            slot_index: slot,
            vertex_offset,
            index_offset,
        };

        let ranges = mesh_layout::validate_slot_write_ranges(
            slot,
            self.mesh_slot_capacity_vertex_elements,
            self.mesh_slot_capacity_index_elements,
        )?;
        debug_assert_eq!(ranges.vertex.offset_elements, vertex_offset);
        debug_assert_eq!(ranges.index.offset_elements, index_offset);

        self.mesh_slice_for_chunk.insert(chunk, slice);
        self.chunk_for_mesh_slot.insert(slot, chunk);
        self.bump_mesh_slot_generation(slot);
        self.touch_mesh_slot(slot);
        self.assert_mesh_slot_chunk_mapping_invariants();
        Some(slice)
    }

    fn release_chunk_mesh_allocation(&mut self, chunk: ChunkCoord) {
        if let Some(slice) = self.mesh_slice_for_chunk.remove(&chunk) {
            self.chunk_for_mesh_slot.remove(&slice.slot_index);
            self.mesh_slot_last_used.remove(&slice.slot_index);
            self.bump_mesh_slot_generation(slice.slot_index);
        }

        self.assert_mesh_slot_chunk_mapping_invariants();
    }

    fn mesh_slot_protection_hint(&self, slot: u32) -> PagePriorityHint {
        self.chunk_for_mesh_slot
            .get(&slot)
            .and_then(|chunk| self.priority_hint_for_chunk.get(chunk))
            .copied()
            .unwrap_or_default()
    }

    fn evict_mesh_slot(&mut self, slot: u32) -> bool {
        if self.has_queued_slot_reservation(slot) {
            return false;
        }
        if let Some(chunk) = self.chunk_for_mesh_slot.get(&slot).copied() {
            match self.mesh_slot_protection_hint(slot) {
                PagePriorityHint::Far => {
                    let _ = GPU_MESH_EVICT_FAR_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                PagePriorityHint::Near => {
                    let _ = GPU_MESH_EVICT_NEAR_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                PagePriorityHint::Visible => {
                    let _ = GPU_MESH_EVICT_VISIBLE_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.release_chunk_mesh_allocation(chunk);
            return true;
        }

        false
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
        let mut selected: Option<((u8, u8, u64), u32)> = None;
        for (&slot, _) in &self.chunk_for_mesh_slot {
            if slot >= slot_capacity || self.has_queued_slot_reservation(slot) {
                continue;
            }
            let protection_rank = self.mesh_slot_protection_hint(slot).eviction_rank();
            let fence_rank = if self.is_mesh_slot_fence_safe(slot) {
                0
            } else {
                1
            };
            let recency_rank = self.mesh_slot_last_used.get(&slot).copied().unwrap_or(0);
            let candidate = ((protection_rank, fence_rank, recency_rank), slot);
            if selected
                .map(|(cur_rank, cur_slot)| {
                    candidate.0 < cur_rank || (candidate.0 == cur_rank && slot < cur_slot)
                })
                .unwrap_or(true)
            {
                selected = Some(candidate);
            }
        }
        selected.map(|(_, slot)| slot)
    }

    fn assert_mesh_slot_chunk_mapping_invariants(&self) {
        let contract = mesh_layout::gpu_mesh_slice_contract();
        assert_eq!(
            self.mesh_slice_for_chunk.len(),
            self.chunk_for_mesh_slot.len(),
            "mesh slot/chunk map size mismatch"
        );

        for (chunk, slice) in &self.mesh_slice_for_chunk {
            assert!(
                slice.slot_index < contract.slot_count,
                "mesh slot index out of bounds chunk={:?} slot={} slot_count={}",
                chunk,
                slice.slot_index,
                contract.slot_count
            );
            let owner = self
                .chunk_for_mesh_slot
                .get(&slice.slot_index)
                .expect("mesh slot must resolve to chunk owner");
            assert_eq!(owner, chunk, "mesh slot/chunk mapping mismatch");
            let expected_vertex =
                mesh_layout::slot_base_offset_elements(MeshBufferKind::Vertex, slice.slot_index)
                    .expect("vertex slot base must resolve");
            let expected_index =
                mesh_layout::slot_base_offset_elements(MeshBufferKind::Index, slice.slot_index)
                    .expect("index slot base must resolve");
            assert_eq!(
                slice.vertex_offset, expected_vertex,
                "mesh slot->vertex offset mismatch chunk={:?} slot={}",
                chunk, slice.slot_index
            );
            assert_eq!(
                slice.index_offset, expected_index,
                "mesh slot->index offset mismatch chunk={:?} slot={}",
                chunk, slice.slot_index
            );
            assert!(
                self.page_for_chunk.contains_key(chunk),
                "mesh slice owner chunk missing page ownership chunk={:?} slot={}",
                chunk,
                slice.slot_index
            );
        }

        for (slot, chunk) in &self.chunk_for_mesh_slot {
            let slice = self
                .mesh_slice_for_chunk
                .get(chunk)
                .expect("chunk owner must resolve to active mesh slice");
            assert_eq!(
                slice.slot_index, *slot,
                "chunk->mesh slice slot must match slot->chunk mapping"
            );
        }
    }

    fn in_flight_mesh_slot_fence_count(&self) -> u32 {
        self.chunk_for_mesh_slot
            .keys()
            .filter(|slot| !self.is_mesh_slot_fence_safe(**slot))
            .count() as u32
    }

    fn mesh_pool_telemetry(&self) -> MeshPoolTelemetry {
        let slot_capacity = MESH_SLOT_COUNT;
        let slots_used = self.mesh_slice_for_chunk.len() as u32;
        let vertex_capacity = MeshBufferKind::Vertex.global_capacity_elements();
        let index_capacity = MeshBufferKind::Index.global_capacity_elements();
        let slots_pending_finalize = self.pending_mesh_finalize.len() as u32;
        let slots_visible = self
            .chunk_for_mesh_slot
            .iter()
            .filter(|(_, chunk)| {
                matches!(
                    self.priority_hint_for_chunk.get(chunk),
                    Some(PagePriorityHint::Visible)
                )
            })
            .count() as u32;
        let fence_safe_slots = self
            .chunk_for_mesh_slot
            .keys()
            .filter(|slot| self.is_mesh_slot_fence_safe(**slot))
            .count() as u32;
        let reclaimable_without_fence = self
            .chunk_for_mesh_slot
            .iter()
            .filter(|(slot, chunk)| {
                self.is_mesh_slot_fence_safe(**slot)
                    && !self.pending_mesh_finalize.contains_key(chunk)
                    && matches!(
                        self.mesh_slot_protection_hint(**slot),
                        PagePriorityHint::Far
                    )
            })
            .count() as u32;
        let lifecycle_blocked_without_fence =
            fence_safe_slots.saturating_sub(reclaimable_without_fence);
        MeshPoolTelemetry {
            slot_capacity,
            slots_used,
            in_flight_fences: self.in_flight_mesh_slot_fence_count(),
            vertex_used: slots_used.saturating_mul(self.mesh_slot_capacity_vertex_elements),
            vertex_capacity,
            index_used: slots_used.saturating_mul(self.mesh_slot_capacity_index_elements),
            index_capacity,
            largest_free_vertex_span: slot_capacity
                .saturating_sub(slots_used)
                .saturating_mul(self.mesh_slot_capacity_vertex_elements),
            largest_free_index_span: slot_capacity
                .saturating_sub(slots_used)
                .saturating_mul(self.mesh_slot_capacity_index_elements),
            vertex_usage_percent: if vertex_capacity == 0 {
                0.0
            } else {
                slots_used.saturating_mul(self.mesh_slot_capacity_vertex_elements) as f32 * 100.0
                    / vertex_capacity as f32
            },
            index_usage_percent: if index_capacity == 0 {
                0.0
            } else {
                slots_used.saturating_mul(self.mesh_slot_capacity_index_elements) as f32 * 100.0
                    / index_capacity as f32
            },
            slot_usage_percent: if slot_capacity == 0 {
                0.0
            } else {
                slots_used as f32 * 100.0 / slot_capacity as f32
            },
            slots_visible,
            slots_pending_finalize,
            reclaimable_without_fence,
            lifecycle_blocked_without_fence,
        }
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

    fn has_queued_page_reservation(&self, page_index: GpuPageIndex) -> bool {
        self.queued_page_reservations
            .get(&page_index)
            .copied()
            .unwrap_or(0)
            > 0
    }

    fn has_queued_slot_reservation(&self, slot_index: u32) -> bool {
        self.queued_slot_reservations
            .get(&slot_index)
            .copied()
            .unwrap_or(0)
            > 0
    }

    fn increment_queued_task_identity(
        &mut self,
        page_index: GpuPageIndex,
        mesh_slice: Option<MeshBufferSlice>,
    ) {
        *self.queued_page_reservations.entry(page_index).or_insert(0) += 1;
        if let Some(mesh_slice) = mesh_slice {
            *self
                .queued_slot_reservations
                .entry(mesh_slice.slot_index)
                .or_insert(0) += 1;
        }
    }

    fn decrement_queued_task_identity(
        &mut self,
        page_index: GpuPageIndex,
        mesh_slice: Option<MeshBufferSlice>,
    ) {
        let remove_page = match self.queued_page_reservations.get_mut(&page_index) {
            Some(count) if *count > 1 => {
                *count -= 1;
                false
            }
            Some(_) => true,
            None => false,
        };
        if remove_page {
            self.queued_page_reservations.remove(&page_index);
        }

        if let Some(mesh_slice) = mesh_slice {
            let remove_slot = match self
                .queued_slot_reservations
                .get_mut(&mesh_slice.slot_index)
            {
                Some(count) if *count > 1 => {
                    *count -= 1;
                    false
                }
                Some(_) => true,
                None => false,
            };
            if remove_slot {
                self.queued_slot_reservations.remove(&mesh_slice.slot_index);
            }
        }
    }

    fn reserve_queued_task_identity(
        &mut self,
        coord: ChunkCoord,
        task_id: u64,
        version: ChunkVersion,
        page_index: GpuPageIndex,
        mesh_slice: Option<MeshBufferSlice>,
    ) -> (u64, Option<u64>) {
        let page_generation = self.page_generation(page_index);
        let slot_generation = mesh_slice.map(|slice| self.slot_generation(slice.slot_index));
        let next = QueuedTaskReservation {
            task_id,
            version,
            page_index,
            mesh_slice,
        };
        if let Some(previous) = self.queued_task_reservations.insert(coord, next) {
            if previous != next {
                self.decrement_queued_task_identity(previous.page_index, previous.mesh_slice);
            }
        }
        self.increment_queued_task_identity(page_index, mesh_slice);
        (page_generation, slot_generation)
    }

    fn release_queued_task_identity(&mut self, coord: ChunkCoord, task_id: u64) {
        let Some(reservation) = self.queued_task_reservations.get(&coord).copied() else {
            return;
        };
        if reservation.task_id != task_id {
            return;
        }
        self.queued_task_reservations.remove(&coord);
        self.decrement_queued_task_identity(reservation.page_index, reservation.mesh_slice);
    }

    fn release_queued_task_identity_for_coord(&mut self, coord: ChunkCoord) {
        if let Some(reservation) = self.queued_task_reservations.remove(&coord) {
            self.decrement_queued_task_identity(reservation.page_index, reservation.mesh_slice);
        }
    }

    fn stale_queued_task_reason(&self, task: &GpuChunkTask) -> Option<StaleQueuedTaskReason> {
        let Some(reservation) = self.queued_task_reservations.get(&task.coord) else {
            return Some(StaleQueuedTaskReason::PageOwnership);
        };
        if reservation.task_id != task.task_id {
            return Some(StaleQueuedTaskReason::PageGeneration);
        }
        if self.page_for_chunk.get(&task.coord).copied() != Some(task.page_index) {
            return Some(StaleQueuedTaskReason::PageOwnership);
        }
        if self.page_generation(task.page_index) != task.page_generation {
            return Some(StaleQueuedTaskReason::PageGeneration);
        }
        match (task.mesh_slice, task.slot_generation) {
            (Some(mesh_slice), Some(slot_generation)) => {
                if self
                    .chunk_for_mesh_slot
                    .get(&mesh_slice.slot_index)
                    .copied()
                    != Some(task.coord)
                {
                    return Some(StaleQueuedTaskReason::SlotOwnership);
                }
                if self.slot_generation(mesh_slice.slot_index) != slot_generation {
                    return Some(StaleQueuedTaskReason::SlotGeneration);
                }
            }
            (None, None) => {}
            _ => return Some(StaleQueuedTaskReason::SlotGeneration),
        }
        None
    }

    fn pending_finalize_identity_matches(
        &self,
        coord: ChunkCoord,
        version: ChunkVersion,
        lod: u8,
        task_id: u64,
        page_index: GpuPageIndex,
        draw_indirect_index: u32,
    ) -> bool {
        self.pending_mesh_finalize
            .get(&coord)
            .map(|pending| {
                pending.version == version
                    && pending.lod == lod
                    && pending.task_id == task_id
                    && pending.page_index == page_index
                    && pending.draw_indirect_index == draw_indirect_index
            })
            .unwrap_or(false)
    }

    fn validate_dispatch_ownership(
        &self,
        chunk: ChunkCoord,
        page_index: GpuPageIndex,
        expected_page_generation: u64,
        mesh_slice: MeshBufferSlice,
        expected_slot_generation: u64,
    ) -> anyhow::Result<()> {
        let mapped_page = self
            .page_for_chunk
            .get(&chunk)
            .copied()
            .with_context(|| format!("chunk {:?} missing page ownership", chunk))?;
        if mapped_page != page_index {
            anyhow::bail!(
                "chunk/page ownership mismatch chunk={:?} expected_page={} actual_page={}",
                chunk,
                mapped_page.0,
                page_index.0
            );
        }
        let current_page_generation = self.page_generation(page_index);
        if current_page_generation != expected_page_generation {
            anyhow::bail!(
                "chunk/page ownership generation mismatch chunk={:?} page={} expected_generation={} actual_generation={}",
                chunk,
                page_index.0,
                expected_page_generation,
                current_page_generation
            );
        }

        let mapped_chunk = self
            .chunk_for_mesh_slot
            .get(&mesh_slice.slot_index)
            .copied()
            .with_context(|| {
                format!(
                    "mesh slot {} missing owner for chunk {:?}",
                    mesh_slice.slot_index, chunk
                )
            })?;
        if mapped_chunk != chunk {
            anyhow::bail!(
                "mesh slot ownership mismatch slot={} expected_chunk={:?} actual_chunk={:?}",
                mesh_slice.slot_index,
                chunk,
                mapped_chunk
            );
        }
        let current_slot_generation = self.slot_generation(mesh_slice.slot_index);
        if current_slot_generation != expected_slot_generation {
            anyhow::bail!(
                "mesh slot ownership generation mismatch slot={} chunk={:?} expected_generation={} actual_generation={}",
                mesh_slice.slot_index,
                chunk,
                expected_slot_generation,
                current_slot_generation
            );
        }

        Ok(())
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
        let mut selected: Option<(u8, u64, GpuPageIndex)> = None;
        for &page in self.chunk_for_page.keys() {
            let fence = self.page_fences.get(&page).copied().unwrap_or_default();
            if fence.last_completed < fence.last_submitted || self.has_queued_page_reservation(page)
            {
                continue;
            }
            let rank = self
                .chunk_for_page
                .get(&page)
                .and_then(|chunk| self.priority_hint_for_chunk.get(chunk))
                .copied()
                .unwrap_or_default()
                .eviction_rank();
            let recency = self.page_last_used.get(&page).copied().unwrap_or(0);
            let replace = selected
                .map(|(cur_rank, cur_recency, _)| (rank, recency) < (cur_rank, cur_recency))
                .unwrap_or(true);
            if replace {
                selected = Some((rank, recency, page));
            }
        }
        selected.map(|(_, _, page)| page)
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
        if self.has_queued_page_reservation(page_index) {
            return;
        }
        if let Some(chunk) = self.chunk_for_page.remove(&page_index) {
            match self
                .priority_hint_for_chunk
                .get(&chunk)
                .copied()
                .unwrap_or_default()
            {
                PagePriorityHint::Far => {
                    let _ = GPU_EVICT_FAR_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                PagePriorityHint::Near => {
                    let _ = GPU_EVICT_NEAR_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                PagePriorityHint::Visible => {
                    let _ = GPU_EVICT_VISIBLE_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.page_for_chunk.remove(&chunk);
            self.bump_page_generation(page_index);
            self.version_for_chunk.remove(&chunk);
            self.state_for_chunk.remove(&chunk);
            self.frontier_len_for_chunk.remove(&chunk);
            self.tick_for_chunk.remove(&chunk);
            self.diagnostics_for_chunk.remove(&chunk);
            self.cached_materials.remove(&chunk);
            self.release_chunk_mesh_allocation(chunk);
            self.pending_mesh_finalize.remove(&chunk);
            self.release_queued_task_identity_for_coord(chunk);
            self.priority_hint_for_chunk.remove(&chunk);
            self.page_fences.remove(&page_index);
            self.page_last_used.remove(&page_index);
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
    draw_indirect_readback: Mutex<DrawIndirectReadbackState>,
    runtime_config: GpuSimulationRuntimeConfig,
    scratch: GpuScratchPool,
    simulation_bg: wgpu::BindGroup,
    meshing_bg: wgpu::BindGroup,
}

#[cfg(feature = "gpu-compute")]
struct DrawIndirectReadbackState {
    staging: wgpu::Buffer,
    map_result_rx: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    requested_serial: u64,
    ready_serial: u64,
    cached_index_counts: Vec<u32>,
}

#[cfg(feature = "gpu-compute")]
impl DrawIndirectReadbackState {
    fn new(device: &wgpu::Device, disabled: bool) -> Self {
        let draw_stride = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
        let slot_capacity = MESH_SLOT_COUNT as u64;
        let size = draw_stride * slot_capacity;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu draw indirect frame readback"),
            size: if disabled { 16 } else { size },
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            staging,
            map_result_rx: None,
            requested_serial: 0,
            ready_serial: 0,
            cached_index_counts: if disabled {
                Vec::new()
            } else {
                vec![0; MESH_SLOT_COUNT as usize]
            },
        }
    }

    fn try_collect(&mut self) {
        let Some(rx) = self.map_result_rx.take() else {
            return;
        };

        match rx.try_recv() {
            Ok(Ok(())) => {
                let mapped = self.staging.slice(..).get_mapped_range();
                let args = bytemuck::cast_slice::<u8, DrawIndexedIndirectArgs>(&mapped);
                for (slot, entry) in args.iter().enumerate() {
                    if slot < self.cached_index_counts.len() {
                        self.cached_index_counts[slot] = entry.index_count;
                    }
                }
                drop(mapped);
                self.staging.unmap();
                self.ready_serial = self.requested_serial;
            }
            Ok(Err(err)) => {
                log::warn!("[gpu-mesh] draw indirect metadata readback map failed: {err:?}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.map_result_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                log::warn!("[gpu-mesh] draw indirect metadata readback callback channel closed");
            }
        }
    }

    fn request_snapshot_if_needed(&mut self, state: &WorkerGpuState, completed_serial: u64) {
        if self.cached_index_counts.is_empty()
            || completed_serial == 0
            || self.map_result_rx.is_some()
            || self.ready_serial >= completed_serial
        {
            return;
        }

        let byte_len =
            std::mem::size_of::<DrawIndexedIndirectArgs>() as u64 * MESH_SLOT_COUNT as u64;
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu draw indirect frame readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&state.draw_indirect_buffer, 0, &self.staging, 0, byte_len);
        state.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        self.requested_serial = completed_serial;
        self.map_result_rx = Some(rx);
    }

    fn has_snapshot_for(&self, submission_serial: u64) -> bool {
        self.ready_serial >= submission_serial
    }

    fn index_count_for_slot(&self, draw_indirect_index: u32) -> Option<u32> {
        self.cached_index_counts
            .get(draw_indirect_index as usize)
            .copied()
    }
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
fn create_gpu_scratch_pool(device: &wgpu::Device, plan: GpuWorkerStartupPlan) -> GpuScratchPool {
    let page_len = CHUNK_VOLUME as u64;
    let page_capacity = MESH_SLOT_COUNT as u64;
    GpuScratchPool {
        page_params: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk page params"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        active_tiles: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk active frontier"),
            size: {
                let base = page_len * std::mem::size_of::<u32>() as u64;
                if plan.reduce_scratch_footprint {
                    base / 2
                } else {
                    base
                }
            },
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
            size: {
                let base = MAX_EDIT_COMMANDS as u64 * std::mem::size_of::<EditCommand>() as u64;
                if plan.reduce_scratch_footprint {
                    base / 2
                } else {
                    base
                }
            },
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

    let face_count_offset = face_count_offset_for_page(page_index);
    debug_assert_eq!(
        face_count_offset % std::mem::size_of::<u32>() as u64,
        0,
        "face-count clears must target a page-local slot"
    );
    state.queue.write_buffer(
        &state.face_count_buffer,
        face_count_offset,
        bytemuck::cast_slice(&[0u32; 1]),
    );
}

#[cfg(feature = "gpu-compute")]
fn face_count_offset_for_page(page_index: GpuPageIndex) -> u64 {
    page_index.0 as u64 * std::mem::size_of::<u32>() as u64
}

fn validate_mesh_slice_for_dispatch(
    coord: ChunkCoord,
    page_index: GpuPageIndex,
    mesh_slice: MeshBufferSlice,
) -> anyhow::Result<()> {
    let contract = mesh_layout::gpu_mesh_slice_contract();
    if page_index.0 >= contract.slot_count {
        anyhow::bail!(
            "invalid mesh dispatch page index for {:?}: page={} slot_capacity={}",
            coord,
            page_index.0,
            contract.slot_count
        );
    }
    if mesh_slice.slot_index >= contract.slot_count {
        anyhow::bail!(
            "invalid mesh dispatch slot index for {:?}: page={} slot={} slot_capacity={}",
            coord,
            page_index.0,
            mesh_slice.slot_index,
            contract.slot_count
        );
    }

    let Some(vertex_limit) = mesh_slice
        .vertex_offset
        .checked_add(contract.vertex_elements_per_chunk)
    else {
        anyhow::bail!(
            "invalid mesh vertex range overflow for {:?}: page={} slot={} vertex_offset_elements={} vertex_count={}",
            coord,
            page_index.0,
            mesh_slice.slot_index,
            mesh_slice.vertex_offset,
            contract.vertex_elements_per_chunk
        );
    };

    let Some(index_limit) = mesh_slice
        .index_offset
        .checked_add(contract.index_elements_per_chunk)
    else {
        anyhow::bail!(
            "invalid mesh index range overflow for {:?}: page={} slot={} index_offset_elements={} index_count={}",
            coord,
            page_index.0,
            mesh_slice.slot_index,
            mesh_slice.index_offset,
            contract.index_elements_per_chunk
        );
    };

    let Some(expected_vertex_offset) =
        mesh_layout::slot_base_offset_elements(MeshBufferKind::Vertex, mesh_slice.slot_index)
    else {
        anyhow::bail!(
            "invalid mesh slot for vertex offset {:?}: slot={}",
            coord,
            mesh_slice.slot_index,
        );
    };
    let Some(expected_index_offset) =
        mesh_layout::slot_base_offset_elements(MeshBufferKind::Index, mesh_slice.slot_index)
    else {
        anyhow::bail!(
            "invalid mesh slot for index offset {:?}: slot={}",
            coord,
            mesh_slice.slot_index,
        );
    };

    if mesh_slice.vertex_offset != expected_vertex_offset
        || mesh_slice.index_offset != expected_index_offset
    {
        anyhow::bail!(
            "invalid mesh slot/global offset mapping coord={:?} page={} slot={} vertex_offset_elements={} expected_vertex_offset_elements={} index_offset_elements={} expected_index_offset_elements={}",
            coord,
            page_index.0,
            mesh_slice.slot_index,
            mesh_slice.vertex_offset,
            expected_vertex_offset,
            mesh_slice.index_offset,
            expected_index_offset,
        );
    }

    let vertex_range = mesh_layout::validate_element_range(
        MeshBufferKind::Vertex,
        mesh_slice.vertex_offset,
        contract.vertex_elements_per_chunk,
    );
    let index_range = mesh_layout::validate_element_range(
        MeshBufferKind::Index,
        mesh_slice.index_offset,
        contract.index_elements_per_chunk,
    );

    if vertex_range.is_none() || index_range.is_none() {
        anyhow::bail!(
            "invalid mesh slice for dispatch coord={:?} page={} slot={} vertex_offset_elements={} vertex_limit_elements={} index_offset_elements={} index_limit_elements={} vertex_capacity_elements={} index_capacity_elements={} vertex_buffer_size_bytes={} index_buffer_size_bytes={}",
            coord,
            page_index.0,
            mesh_slice.slot_index,
            mesh_slice.vertex_offset,
            vertex_limit,
            mesh_slice.index_offset,
            index_limit,
            contract.vertex_global_capacity_elements,
            contract.index_global_capacity_elements,
            MeshBufferKind::Vertex.global_size_bytes(),
            MeshBufferKind::Index.global_size_bytes(),
        );
    }

    Ok(())
}

#[cfg(feature = "gpu-compute")]
fn chunk_world_bounds(coord: ChunkCoord) -> (Vec3, Vec3, Vec3) {
    let side = CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
    let origin = Vec3::new(
        coord.x as f32 * side,
        coord.y as f32 * side,
        coord.z as f32 * side,
    );
    (origin, origin, origin + Vec3::splat(side))
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
    pub mesh_finalize_waiting_on_metadata: u64,
    pub mesh_finalize_promoted_ready: u64,
    pub mesh_finalize_ownership_invalidations: u64,
    pub mesh_finalize_superseded_results: u64,
    pub mesh_allocator_pressure_events: u64,
    pub mesh_allocator_fragmentation_events: u64,
    pub mesh_resident_usage_percent: f32,
    pub evict_near_count: u64,
    pub evict_visible_count: u64,
    pub evict_far_count: u64,
    pub mesh_evict_near_count: u64,
    pub mesh_evict_visible_count: u64,
    pub mesh_evict_far_count: u64,
    pub mesh_finalize_lock_hold_ms: f32,
    pub mesh_finalize_events: u64,
    pub mesh_finalize_requeued: u64,
    pub allocator_no_progress_count: u64,
    pub allocator_saturated_count: u64,
    pub candidate_ready_count: u64,
    pub candidate_failed_count: u64,
    pub finalize_invalid_count: u64,
    pub color_contract_mismatch_count: u64,
    pub zero_or_invalid_mesh_output_count: u64,
    pub queue_full_drops: u64,
    pub queue_deferred_count: u64,
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
static GPU_EVICT_NEAR_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_EVICT_VISIBLE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_EVICT_FAR_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_EVICT_NEAR_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_EVICT_VISIBLE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_EVICT_FAR_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_DISPATCH_ERRORS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_STARTUP_ZERO_FRONTIER_MESH_RUNS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_LOCK_HOLD_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_REQUEUED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_WAITING_METADATA_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_PROMOTED_READY_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_OWNERSHIP_INVALIDATED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_FINALIZE_SUPERSEDED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_ALLOCATOR_PRESSURE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_ALLOCATOR_FRAGMENTATION_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_MESH_RESIDENT_USAGE_PERCENT_X100: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_ALLOCATOR_NO_PROGRESS_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_ALLOCATOR_SATURATED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_CANDIDATE_READY_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_CANDIDATE_FAILED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_FINALIZE_INVALID_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_COLOR_CONTRACT_MISMATCH_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_ZERO_OR_INVALID_MESH_OUTPUT_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static ACTIVE_GPU_JOBS_BITS: std::sync::LazyLock<Vec<AtomicU64>> =
    std::sync::LazyLock::new(|| (0..1024).map(|_| AtomicU64::new(0)).collect());
#[cfg(feature = "gpu-compute")]
static GPU_TASK_TX: OnceLock<cbc::Sender<GpuChunkTask>> = OnceLock::new();
#[cfg(feature = "gpu-compute")]
static GPU_TASK_RX: OnceLock<cbc::Receiver<GpuChunkTask>> = OnceLock::new();
#[cfg(feature = "gpu-compute")]
static GPU_SUBMISSION_SERIAL: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_COMPLETED_SERIAL: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TASKS_ENQUEUED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TASKS_DEQUEUED_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TASK_QUEUE_FULL_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "gpu-compute")]
static GPU_TASK_QUEUE_DEFERRED_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "gpu-compute")]
const GPU_TASK_ENQUEUE_MAX_DEFER_ATTEMPTS: u32 = 2;

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuTaskQueuePressureReason {
    QueueFull { max_defer_attempts: u32 },
    QueueDisconnected,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDispatchFrameStats {
    pub tasks_dequeued: usize,
    pub tasks_submitted: usize,
    pub enqueue_submit_ms: f32,
    pub wait_sync_ms: f32,
    pub stale_tasks_skipped: usize,
    pub ownership_validation_drops: usize,
    pub meshing_dispatch_failures: usize,
    pub tasks_without_mesh_slice: usize,
    pub queue_empty_exits: usize,
    pub queue_disconnected_exits: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuRuntimeDebugSnapshot {
    pub mesh_slots_used: usize,
    pub mesh_slot_capacity: usize,
    pub mesh_slot_in_flight_fences: usize,
    pub mesh_vertex_used: usize,
    pub mesh_vertex_capacity: usize,
    pub mesh_index_used: usize,
    pub mesh_index_capacity: usize,
    pub mesh_largest_free_vertex_span: usize,
    pub mesh_largest_free_index_span: usize,
    pub mesh_slots_visible: usize,
    pub mesh_slots_pending_finalize: usize,
    pub mesh_slots_reclaimable_without_fence: usize,
    pub mesh_slots_lifecycle_blocked_without_fence: usize,
    pub queue_enqueued: usize,
    pub queue_dequeued: usize,
    pub queue_rx_backlog: usize,
    pub queue_full_drops: u64,
    pub queue_deferred_count: u64,
}

#[cfg(feature = "gpu-compute")]
fn try_enqueue_gpu_chunk_task(
    tx: &cbc::Sender<GpuChunkTask>,
    mut task: GpuChunkTask,
) -> Result<(), GpuTaskQueuePressureReason> {
    let mut deferred_attempts = 0u32;
    loop {
        match tx.try_send(task) {
            Ok(()) => {
                GPU_TASKS_ENQUEUED_COUNT.fetch_add(1, Ordering::Relaxed);
                if deferred_attempts > 0 {
                    GPU_TASK_QUEUE_DEFERRED_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(());
            }
            Err(cbc::TrySendError::Full(returned_task)) => {
                task = returned_task;
                if deferred_attempts < GPU_TASK_ENQUEUE_MAX_DEFER_ATTEMPTS {
                    deferred_attempts += 1;
                    std::thread::yield_now();
                    continue;
                }
                GPU_TASK_QUEUE_FULL_COUNT.fetch_add(1, Ordering::Relaxed);
                return Err(GpuTaskQueuePressureReason::QueueFull {
                    max_defer_attempts: GPU_TASK_ENQUEUE_MAX_DEFER_ATTEMPTS,
                });
            }
            Err(cbc::TrySendError::Disconnected(_returned_task)) => {
                GPU_TASK_QUEUE_FULL_COUNT.fetch_add(1, Ordering::Relaxed);
                return Err(GpuTaskQueuePressureReason::QueueDisconnected);
            }
        }
    }
}

pub fn gpu_task_rx_backlog_estimate() -> usize {
    #[cfg(not(feature = "gpu-compute"))]
    {
        0
    }

    #[cfg(feature = "gpu-compute")]
    {
        GPU_TASKS_ENQUEUED_COUNT
            .load(Ordering::Relaxed)
            .saturating_sub(GPU_TASKS_DEQUEUED_COUNT.load(Ordering::Relaxed)) as usize
    }
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadyGpuMeshFinalizeWaitReason {
    WaitingOnCompletionSerial,
    WaitingOnReadbackSnapshot,
    WaitingOnMetadata,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaleQueuedTaskReason {
    PageOwnership,
    PageGeneration,
    SlotOwnership,
    SlotGeneration,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub struct ReadyGpuMeshResult {
    pub coord: ChunkCoord,
    pub version: ChunkVersion,
    pub task_id: u64,
    pub submission_serial: u64,
    pub page_index: GpuPageIndex,
    pub draw_indirect_index: u32,
    pub page_generation: u64,
    pub slot_generation: u64,
    pub index_count: Option<u32>,
    pub lod: u8,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub chunk_origin_world: Vec3,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub enum ReadyGpuMeshFinalizeStatus {
    NotReadyYet,
    ReadyAndValid,
    DroppedStaleVersion,
    DroppedInvalidMapping,
    DroppedSupersededIdentity,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
pub struct ReadyGpuMeshFinalizeEvent {
    pub result: ReadyGpuMeshResult,
    pub status: ReadyGpuMeshFinalizeStatus,
    pub wait_reason: Option<ReadyGpuMeshFinalizeWaitReason>,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone)]
pub struct GpuChunkTask {
    pub coord: ChunkCoord,
    pub page_index: GpuPageIndex,
    pub page_generation: u64,
    pub frontier_count: u32,
    pub edit_commands: Vec<EditCommand>,
    pub jacobi_iterations: u32,
    pub neighbor_pages: [u32; 6],
    pub simulation_tick: u32,
    pub current_state: u32,
    pub startup_seeding_mode: bool,
    pub mesh_slice: Option<MeshBufferSlice>,
    pub slot_generation: Option<u64>,
    pub version: ChunkVersion,
    pub task_id: u64,
    pub lod: u8,
    pub next_cached_materials: Vec<MaterialId>,
    pub next_state: u32,
    pub next_tick: u32,
    pub next_frontier_len: u32,
    pub next_diagnostics: ChunkSimulationDiagnostics,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshBufferSlice {
    pub slot_index: u32,
    /// Global vertex buffer offset in elements (not bytes).
    pub vertex_offset: u32,
    /// Global index buffer offset in elements (not bytes).
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
        let evict_near_count = GPU_EVICT_NEAR_COUNT.swap(0, Ordering::Relaxed);
        let evict_visible_count = GPU_EVICT_VISIBLE_COUNT.swap(0, Ordering::Relaxed);
        let evict_far_count = GPU_EVICT_FAR_COUNT.swap(0, Ordering::Relaxed);
        let mesh_evict_near_count = GPU_MESH_EVICT_NEAR_COUNT.swap(0, Ordering::Relaxed);
        let mesh_evict_visible_count = GPU_MESH_EVICT_VISIBLE_COUNT.swap(0, Ordering::Relaxed);
        let mesh_evict_far_count = GPU_MESH_EVICT_FAR_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_lock_hold_ns = GPU_MESH_FINALIZE_LOCK_HOLD_NS.swap(0, Ordering::Relaxed);
        let mesh_finalize_events = GPU_MESH_FINALIZE_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_requeued = GPU_MESH_FINALIZE_REQUEUED_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_waiting_on_metadata =
            GPU_MESH_FINALIZE_WAITING_METADATA_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_promoted_ready =
            GPU_MESH_FINALIZE_PROMOTED_READY_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_ownership_invalidations =
            GPU_MESH_FINALIZE_OWNERSHIP_INVALIDATED_COUNT.swap(0, Ordering::Relaxed);
        let mesh_finalize_superseded_results =
            GPU_MESH_FINALIZE_SUPERSEDED_COUNT.swap(0, Ordering::Relaxed);
        let mesh_allocator_pressure_events =
            GPU_MESH_ALLOCATOR_PRESSURE_COUNT.swap(0, Ordering::Relaxed);
        let mesh_allocator_fragmentation_events =
            GPU_MESH_ALLOCATOR_FRAGMENTATION_COUNT.swap(0, Ordering::Relaxed);
        let mesh_resident_usage_percent =
            GPU_MESH_RESIDENT_USAGE_PERCENT_X100.swap(0, Ordering::Relaxed) as f32 / 100.0;
        let allocator_no_progress_count =
            GPU_ALLOCATOR_NO_PROGRESS_COUNT.swap(0, Ordering::Relaxed);
        let allocator_saturated_count = GPU_ALLOCATOR_SATURATED_COUNT.swap(0, Ordering::Relaxed);
        let candidate_ready_count = GPU_CANDIDATE_READY_COUNT.swap(0, Ordering::Relaxed);
        let candidate_failed_count = GPU_CANDIDATE_FAILED_COUNT.swap(0, Ordering::Relaxed);
        let finalize_invalid_count = GPU_FINALIZE_INVALID_COUNT.swap(0, Ordering::Relaxed);
        let color_contract_mismatch_count =
            GPU_COLOR_CONTRACT_MISMATCH_COUNT.swap(0, Ordering::Relaxed);
        let zero_or_invalid_mesh_output_count =
            GPU_ZERO_OR_INVALID_MESH_OUTPUT_COUNT.swap(0, Ordering::Relaxed);
        let frame = frame_seconds.max(0.000_1);
        GpuComputeProfilerSnapshot {
            dispatch_ms: dispatch_ns as f32 / 1_000_000.0,
            bytes_transferred,
            chunks_completed,
            frontier_cap_events,
            mesh_slot_alloc_failed,
            mesh_finalize_waiting_on_metadata,
            mesh_finalize_promoted_ready,
            mesh_finalize_ownership_invalidations,
            mesh_finalize_superseded_results,
            mesh_allocator_pressure_events,
            mesh_allocator_fragmentation_events,
            mesh_resident_usage_percent,
            evict_near_count,
            evict_visible_count,
            evict_far_count,
            mesh_evict_near_count,
            mesh_evict_visible_count,
            mesh_evict_far_count,
            mesh_finalize_lock_hold_ms: mesh_finalize_lock_hold_ns as f32 / 1_000_000.0,
            mesh_finalize_events,
            mesh_finalize_requeued,
            allocator_no_progress_count,
            allocator_saturated_count,
            candidate_ready_count,
            candidate_failed_count,
            finalize_invalid_count,
            color_contract_mismatch_count,
            zero_or_invalid_mesh_output_count,
            chunks_per_sec: chunks_completed as f32 / frame,
            queue_full_drops: GPU_TASK_QUEUE_FULL_COUNT.swap(0, Ordering::Relaxed),
            queue_deferred_count: GPU_TASK_QUEUE_DEFERRED_COUNT.swap(0, Ordering::Relaxed),
        }
    }
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

        let dispatch = |pipeline: &wgpu::ComputePipeline, jacobi_iteration: u32| {
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
    fn run_meshing_dispatch(
        &self,
        state: &WorkerGpuState,
        scratch: &GpuScratchPool,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        lod: u8,
        mesh_slice: MeshBufferSlice,
    ) -> anyhow::Result<()> {
        validate_mesh_slice_for_dispatch(sim_job.chunk_coord, page_index, mesh_slice)?;
        clear_meshing_outputs_for_page(state, page_index, mesh_slice);
        log::trace!(
            "[gpu-mesh] meshing_dispatch coord={:?} page={} slot={} lod={}",
            sim_job.chunk_coord,
            page_index.0,
            mesh_slice.slot_index,
            lod,
        );

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

        Ok(())
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
    startup_plan: GpuWorkerStartupPlan,
) -> anyhow::Result<()> {
    if GPU_TASK_TX.get().is_none() {
        let (tx, rx) = cbc::bounded(4096);
        let _ = GPU_TASK_TX.set(tx);
        let _ = GPU_TASK_RX.set(rx);
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
        validate_gpu_vertex_contract().context("gpu mesh vertex host contract")?;
        let runtime = GpuComputeRuntime::new(&device).context("compute runtime")?;
        let page_capacity = MESH_SLOT_COUNT as u64;
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

        let scratch = create_gpu_scratch_pool(&device, startup_plan);
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
        let draw_indirect_readback = DrawIndirectReadbackState::new(
            &device,
            startup_plan.disable_draw_indirect_readback,
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
            draw_indirect_readback: Mutex::new(draw_indirect_readback),
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

        let mesh_slice_outcome = atlas.mesh_slice_for_chunk_or_allocate(job.coord, job.lod as u8);
        let mesh_slice = match mesh_slice_outcome {
            MeshSliceAllocateOutcome::Success(slice) => Some(slice),
            MeshSliceAllocateOutcome::Saturated => {
                GPU_ALLOCATOR_SATURATED_COUNT.fetch_add(1, Ordering::Relaxed);
                None
            }
            MeshSliceAllocateOutcome::NoProgress => {
                GPU_ALLOCATOR_NO_PROGRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                None
            }
            MeshSliceAllocateOutcome::InvalidState => {
                GPU_ALLOCATOR_NO_PROGRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                None
            }
        };
        let mesh_pool_telemetry = atlas.mesh_pool_telemetry();
        let resident_usage = mesh_pool_telemetry
            .vertex_usage_percent
            .max(mesh_pool_telemetry.index_usage_percent)
            .max(mesh_pool_telemetry.slot_usage_percent);
        GPU_MESH_RESIDENT_USAGE_PERCENT_X100
            .store((resident_usage * 100.0) as u64, Ordering::Relaxed);

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
        let (page_generation, slot_generation) = atlas.reserve_queued_task_identity(
            job.coord,
            job.task_id,
            job.version,
            page_index,
            mesh_slice,
        );

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
        let diagnostics = ChunkSimulationDiagnostics::default();
        let next_state = if ran_simulation {
            current_state ^ 1
        } else {
            current_state
        };

        // Queue GPU task
        let enqueue_result = try_enqueue_gpu_chunk_task(
            GPU_TASK_TX.get().expect("gpu task queue"),
            GpuChunkTask {
                coord: job.coord,
                page_index,
                page_generation,
                frontier_count: active_frontier_count,
                edit_commands,
                jacobi_iterations,
                neighbor_pages,
                simulation_tick: tick,
                current_state,
                startup_seeding_mode,
                mesh_slice,
                slot_generation,
                version: job.version,
                task_id: job.task_id,
                lod: job.lod as u8,
                next_cached_materials: incoming.to_vec(),
                next_state,
                next_tick: tick.wrapping_add(1),
                next_frontier_len: active_frontier_count,
                next_diagnostics: diagnostics,
            },
        );
        if let Err(reason) = enqueue_result {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            atlas.release_queued_task_identity(job.coord, job.task_id);
            return Err(anyhow::anyhow!(
                "failed to enqueue GPU chunk task for {:?}: {:?}",
                job.coord,
                reason
            ));
        }

        // Authoritative runtime meshing path (GPU).
        let (chunk_origin_world, aabb_min, aabb_max) = chunk_world_bounds(job.coord);

        Ok(ComputedChunkArtifacts {
            simulation_diagnostics: diagnostics,
            mesh_artifact: {
                if let Some(mesh_slice) = mesh_slice {
                    ChunkMeshArtifact::GpuPending {
                        page_index,
                        draw_indirect_index: mesh_slice.slot_index,
                        lod: job.lod as u8,
                        aabb_min,
                        aabb_max,
                        chunk_origin_world,
                    }
                } else {
                    let _ = GPU_MESH_SLOT_ALLOC_FAILED.fetch_add(1, Ordering::Relaxed);
                    let pressure = mesh_pool_telemetry.slot_usage_percent >= 95.0
                        || mesh_pool_telemetry.vertex_usage_percent >= 95.0
                        || mesh_pool_telemetry.index_usage_percent >= 95.0;
                    if pressure {
                        GPU_MESH_ALLOCATOR_PRESSURE_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    let fragmented = false;
                    if fragmented {
                        GPU_MESH_ALLOCATOR_FRAGMENTATION_COUNT.fetch_add(1, Ordering::Relaxed);
                    }

                    log::debug!(
                            "[mesh] skipping gpu meshing for {:?}: global mesh pool exhausted slots={}/{} in_flight_fences={} vertex={}/{} index={}/{} largest_free[v/i]={}/{}",
                            job.coord,
                            mesh_pool_telemetry.slots_used,
                            mesh_pool_telemetry.slot_capacity,
                            mesh_pool_telemetry.in_flight_fences,
                            mesh_pool_telemetry.vertex_used,
                            mesh_pool_telemetry.vertex_capacity,
                            mesh_pool_telemetry.index_used,
                            mesh_pool_telemetry.index_capacity,
                            mesh_pool_telemetry.largest_free_vertex_span,
                            mesh_pool_telemetry.largest_free_index_span,
                        );

                    ChunkMeshArtifact::Skipped {
                        reason: MeshSkipReason::MeshSlotCapacitySaturated {
                            slot_capacity: mesh_pool_telemetry.slot_capacity,
                            in_flight_fences: mesh_pool_telemetry.in_flight_fences,
                            slots_used: mesh_pool_telemetry.slots_used,
                            vertex_used: mesh_pool_telemetry.vertex_used,
                            vertex_capacity: mesh_pool_telemetry.vertex_capacity,
                            index_used: mesh_pool_telemetry.index_used,
                            index_capacity: mesh_pool_telemetry.index_capacity,
                            largest_free_vertex_span: mesh_pool_telemetry.largest_free_vertex_span,
                            largest_free_index_span: mesh_pool_telemetry.largest_free_index_span,
                            slots_visible: mesh_pool_telemetry.slots_visible,
                            slots_pending_finalize: mesh_pool_telemetry.slots_pending_finalize,
                            reclaimable_without_fence: mesh_pool_telemetry
                                .reclaimable_without_fence,
                            lifecycle_blocked_without_fence: mesh_pool_telemetry
                                .lifecycle_blocked_without_fence,
                        },
                    }
                }
            },
        })
    }
}

#[cfg(feature = "gpu-compute")]
fn commit_dispatched_chunk_state(atlas: &mut ChunkPageAtlas, task: &GpuChunkTask) {
    atlas
        .version_for_chunk
        .insert(task.coord, task.version.get());
    atlas.state_for_chunk.insert(task.coord, task.next_state);
    atlas.tick_for_chunk.insert(task.coord, task.next_tick);
    atlas
        .frontier_len_for_chunk
        .insert(task.coord, task.next_frontier_len);
    atlas
        .diagnostics_for_chunk
        .insert(task.coord, task.next_diagnostics);
    atlas
        .cached_materials
        .insert(task.coord, task.next_cached_materials.clone());
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

    let rx = GPU_TASK_RX.get().context("gpu task receiver missing")?;

    let frame_dispatch_start = Instant::now();
    let mut stats = GpuDispatchFrameStats::default();

    while stats.tasks_submitted < max_tasks {
        if frame_dispatch_start.elapsed() >= max_dispatch_time {
            break;
        }

        let task = match rx.try_recv() {
            Ok(t) => {
                GPU_TASKS_DEQUEUED_COUNT.fetch_add(1, Ordering::Relaxed);
                stats.tasks_dequeued += 1;
                t
            }
            Err(cbc::TryRecvError::Empty) => {
                stats.queue_empty_exits += 1;
                break;
            }
            Err(cbc::TryRecvError::Disconnected) => {
                stats.queue_disconnected_exits += 1;
                break;
            }
        };

        let task_start = Instant::now();
        let task_wait_sync = Duration::ZERO;

        struct TaskIdentityReleaseGuard {
            state: Arc<WorkerGpuState>,
            coord: ChunkCoord,
            task_id: u64,
            active: bool,
        }

        impl TaskIdentityReleaseGuard {
            fn new(state: Arc<WorkerGpuState>, coord: ChunkCoord, task_id: u64) -> Self {
                Self {
                    state,
                    coord,
                    task_id,
                    active: true,
                }
            }
            fn release_now(&mut self) {
                if !self.active {
                    return;
                }
                let mut atlas = self.state.atlas.lock().unwrap_or_else(|e| e.into_inner());
                atlas.release_queued_task_identity(self.coord, self.task_id);
                self.active = false;
            }
        }

        impl Drop for TaskIdentityReleaseGuard {
            fn drop(&mut self) {
                self.release_now();
            }
        }

        let mut task_identity_guard =
            TaskIdentityReleaseGuard::new(state.clone(), task.coord, task.task_id);

        {
            let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            if atlas.stale_queued_task_reason(&task).is_some() {
                stats.stale_tasks_skipped += 1;
                task_identity_guard.release_now();
                continue;
            }
        }

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
            if let Err(err) = state.runtime.run_active_frontier(
                &state,
                scratch,
                &sim_job,
                task.page_index,
                task.current_state,
                &task.edit_commands,
                task.jacobi_iterations,
                task.neighbor_pages,
            ) {
                stats.meshing_dispatch_failures += 1;
                log::error!(
                    "[gpu-sim] rejecting gpu simulation dispatch coord={:?} version={} task_id={} page_index={} error={:#}",
                    task.coord,
                    task.version,
                    task.task_id,
                    task.page_index.0,
                    err,
                );
                task_identity_guard.release_now();
                continue;
            }
        }

        let mut submitted_mesh_slice = None;

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

                {
                    let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(err) = atlas.validate_dispatch_ownership(
                        task.coord,
                        task.page_index,
                        task.page_generation,
                        mesh_slice,
                        task.slot_generation
                            .expect("mesh slice task missing slot generation"),
                    ) {
                        stats.ownership_validation_drops += 1;
                        task_identity_guard.release_now();
                        log::error!(
                            "[gpu-mesh] rejecting gpu mesh dispatch ownership coord={:?} version={} task_id={} page_index={} slot_index={} error={:#}",
                            task.coord,
                            task.version,
                            task.task_id,
                            task.page_index.0,
                            mesh_slice.slot_index,
                            err,
                        );
                        continue;
                    }
                }

                match state.runtime.run_meshing_dispatch(
                    &state,
                    scratch,
                    &sim_job,
                    task.page_index,
                    meshing_state,
                    task.lod,
                    mesh_slice,
                ) {
                    Ok(()) => {
                        submitted_mesh_slice = Some(mesh_slice);
                    }
                    Err(err) => {
                        stats.meshing_dispatch_failures += 1;
                        log::error!(
                            "[gpu-mesh] rejecting gpu mesh dispatch coord={:?} version={} task_id={} page_index={} slot_index={} vertex_offset={} index_offset={} vertex_capacity_elements={} index_capacity_elements={} vertex_buffer_size_bytes={} index_buffer_size_bytes={} error={:#}",
                            task.coord,
                            task.version,
                            task.task_id,
                            task.page_index.0,
                            mesh_slice.slot_index,
                            mesh_slice.vertex_offset,
                            mesh_slice.index_offset,
                            MeshBufferKind::Vertex.global_capacity_elements(),
                            MeshBufferKind::Index.global_capacity_elements(),
                            MeshBufferKind::Vertex.global_size_bytes(),
                            MeshBufferKind::Index.global_size_bytes(),
                            err,
                        );
                    }
                }
            } else {
                stats.tasks_without_mesh_slice += 1;
            }
        }

        let serial = GPU_SUBMISSION_SERIAL.fetch_add(1, Ordering::Relaxed) + 1;

        {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            atlas.mark_page_submitted(task.page_index, serial);
            atlas.touch_chunk_page(task.coord);
            commit_dispatched_chunk_state(&mut atlas, &task);
            if let Some(mesh_slice) = submitted_mesh_slice {
                let page_generation = atlas.page_generation(task.page_index);
                let slot_generation = atlas.slot_generation(mesh_slice.slot_index);
                atlas.pending_mesh_finalize.insert(
                    task.coord,
                    PendingGpuMeshFinalize {
                        version: task.version,
                        task_id: task.task_id,
                        lod: task.lod,
                        page_index: task.page_index,
                        draw_indirect_index: mesh_slice.slot_index,
                        page_generation,
                        slot_generation,
                        submission_serial: serial,
                    },
                );
            }
            atlas.release_queued_task_identity(task.coord, task.task_id);
            task_identity_guard.active = false;
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

#[cfg(feature = "gpu-compute")]
pub fn gpu_runtime_debug_snapshot() -> GpuRuntimeDebugSnapshot {
    let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    else {
        return GpuRuntimeDebugSnapshot::default();
    };
    let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
    let pool = atlas.mesh_pool_telemetry();
    GpuRuntimeDebugSnapshot {
        mesh_slots_used: pool.slots_used as usize,
        mesh_slot_capacity: pool.slot_capacity as usize,
        mesh_slot_in_flight_fences: pool.in_flight_fences as usize,
        mesh_vertex_used: pool.vertex_used as usize,
        mesh_vertex_capacity: pool.vertex_capacity as usize,
        mesh_index_used: pool.index_used as usize,
        mesh_index_capacity: pool.index_capacity as usize,
        mesh_largest_free_vertex_span: pool.largest_free_vertex_span as usize,
        mesh_largest_free_index_span: pool.largest_free_index_span as usize,
        mesh_slots_visible: pool.slots_visible as usize,
        mesh_slots_pending_finalize: pool.slots_pending_finalize as usize,
        mesh_slots_reclaimable_without_fence: pool.reclaimable_without_fence as usize,
        mesh_slots_lifecycle_blocked_without_fence: pool.lifecycle_blocked_without_fence as usize,
        queue_enqueued: GPU_TASKS_ENQUEUED_COUNT.load(Ordering::Relaxed) as usize,
        queue_dequeued: GPU_TASKS_DEQUEUED_COUNT.load(Ordering::Relaxed) as usize,
        queue_rx_backlog: gpu_task_rx_backlog_estimate(),
        queue_full_drops: GPU_TASK_QUEUE_FULL_COUNT.load(Ordering::Relaxed),
        queue_deferred_count: GPU_TASK_QUEUE_DEFERRED_COUNT.load(Ordering::Relaxed),
    }
}

#[cfg(not(feature = "gpu-compute"))]
pub fn gpu_runtime_debug_snapshot() -> GpuRuntimeDebugSnapshot {
    GpuRuntimeDebugSnapshot::default()
}

#[cfg(feature = "gpu-compute")]
pub fn take_ready_gpu_mesh_results_on_renderer() -> Vec<ReadyGpuMeshFinalizeEvent> {
    let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    else {
        return Vec::new();
    };

    #[derive(Clone, Copy)]
    struct FinalizeCandidate {
        coord: ChunkCoord,
        pending: PendingGpuMeshFinalize,
    }

    let completed = GPU_COMPLETED_SERIAL.load(Ordering::Relaxed);
    let phase1_lock_start = Instant::now();
    let candidates = {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.refresh_completed_serial(completed);

        atlas
            .pending_mesh_finalize
            .iter()
            .map(|(coord, pending)| FinalizeCandidate {
                coord: *coord,
                pending: *pending,
            })
            .collect::<Vec<_>>()
    };
    GPU_MESH_FINALIZE_LOCK_HOLD_NS.fetch_add(
        phase1_lock_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );

    let index_counts = {
        let mut readback = state
            .draw_indirect_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        readback.try_collect();
        readback.request_snapshot_if_needed(state, completed);
        candidates
            .iter()
            .map(|candidate| {
                let completed_ready = candidate.pending.submission_serial <= completed;
                let snapshot_ready = completed_ready
                    && readback.has_snapshot_for(candidate.pending.submission_serial);
                let index_count = if snapshot_ready {
                    readback.index_count_for_slot(candidate.pending.draw_indirect_index)
                } else {
                    None
                };
                (completed_ready, snapshot_ready, index_count)
            })
            .collect::<Vec<_>>()
    };

    let mut out = Vec::with_capacity(candidates.len());
    let mut finalized_count = 0u64;
    let mut requeued_count = 0u64;
    let mut waiting_metadata_count = 0u64;
    let mut promoted_ready_count = 0u64;
    let mut ownership_invalidations = 0u64;
    let mut superseded_count = 0u64;
    let mut candidate_failed_count = 0u64;
    let mut finalize_invalid_count = 0u64;
    let mut zero_or_invalid_mesh_output_count = 0u64;

    let phase3_lock_start = Instant::now();
    {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        for (candidate, (completed_ready, snapshot_ready, index_count)) in
            candidates.into_iter().zip(index_counts.into_iter())
        {
            let Some(current_pending) = atlas.pending_mesh_finalize.get(&candidate.coord).copied()
            else {
                continue;
            };
            if current_pending != candidate.pending {
                requeued_count += 1;
                superseded_count += 1;
                let (chunk_origin_world, aabb_min, aabb_max) = chunk_world_bounds(candidate.coord);
                out.push(ReadyGpuMeshFinalizeEvent {
                    result: ReadyGpuMeshResult {
                        coord: candidate.coord,
                        version: candidate.pending.version,
                        task_id: candidate.pending.task_id,
                        submission_serial: candidate.pending.submission_serial,
                        page_index: candidate.pending.page_index,
                        draw_indirect_index: candidate.pending.draw_indirect_index,
                        page_generation: candidate.pending.page_generation,
                        slot_generation: candidate.pending.slot_generation,
                        index_count,
                        lod: candidate.pending.lod,
                        aabb_min,
                        aabb_max,
                        chunk_origin_world,
                    },
                    status: ReadyGpuMeshFinalizeStatus::DroppedSupersededIdentity,
                    wait_reason: None,
                });
                finalized_count += 1;
                candidate_failed_count += 1;
                continue;
            }

            let (chunk_origin_world, aabb_min, aabb_max) = chunk_world_bounds(candidate.coord);
            let result = ReadyGpuMeshResult {
                coord: candidate.coord,
                version: candidate.pending.version,
                task_id: candidate.pending.task_id,
                submission_serial: candidate.pending.submission_serial,
                page_index: candidate.pending.page_index,
                draw_indirect_index: candidate.pending.draw_indirect_index,
                page_generation: candidate.pending.page_generation,
                slot_generation: candidate.pending.slot_generation,
                index_count,
                lod: candidate.pending.lod,
                aabb_min,
                aabb_max,
                chunk_origin_world,
            };

            if !completed_ready {
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::NotReadyYet,
                    wait_reason: Some(ReadyGpuMeshFinalizeWaitReason::WaitingOnCompletionSerial),
                });
                continue;
            }
            if !snapshot_ready {
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::NotReadyYet,
                    wait_reason: Some(ReadyGpuMeshFinalizeWaitReason::WaitingOnReadbackSnapshot),
                });
                continue;
            }
            if index_count.is_none() {
                waiting_metadata_count += 1;
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::NotReadyYet,
                    wait_reason: Some(ReadyGpuMeshFinalizeWaitReason::WaitingOnMetadata),
                });
                continue;
            }

            if atlas.version_for_chunk.get(&candidate.coord).copied()
                != Some(candidate.pending.version.get())
            {
                atlas.pending_mesh_finalize.remove(&candidate.coord);
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::DroppedStaleVersion,
                    wait_reason: None,
                });
                finalized_count += 1;
                superseded_count += 1;
                candidate_failed_count += 1;
                continue;
            }

            let current_page = atlas.page_for_chunk.get(&candidate.coord).copied();
            let current_slot = atlas
                .mesh_slice_for_chunk
                .get(&candidate.coord)
                .copied()
                .map(|slice| slice.slot_index);
            let page_gen = atlas.page_generation(candidate.pending.page_index);
            let slot_gen = atlas.slot_generation(candidate.pending.draw_indirect_index);

            let ownership_valid = current_page == Some(candidate.pending.page_index)
                && current_slot == Some(candidate.pending.draw_indirect_index)
                && page_gen == candidate.pending.page_generation
                && slot_gen == candidate.pending.slot_generation;

            if !ownership_valid {
                atlas.pending_mesh_finalize.remove(&candidate.coord);
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::DroppedInvalidMapping,
                    wait_reason: None,
                });
                finalized_count += 1;
                ownership_invalidations += 1;
                candidate_failed_count += 1;
                continue;
            }

            let Some(index_count) = result.index_count else {
                waiting_metadata_count += 1;
                continue;
            };

            if index_count == 0 {
                atlas.pending_mesh_finalize.remove(&candidate.coord);
                log::debug!(
                    "[gpu-mesh] finalize_invalid_output coord={:?} version={} task_id={} page={} slot={} serial={} lod={} reason=zero_index_count",
                    result.coord,
                    result.version,
                    result.task_id,
                    result.page_index.0,
                    result.draw_indirect_index,
                    result.submission_serial,
                    result.lod,
                );
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::DroppedInvalidMapping,
                    wait_reason: None,
                });
                finalized_count += 1;
                candidate_failed_count += 1;
                finalize_invalid_count += 1;
                zero_or_invalid_mesh_output_count += 1;
                continue;
            }

            atlas.pending_mesh_finalize.remove(&candidate.coord);
            atlas.touch_chunk_page(candidate.coord);
            out.push(ReadyGpuMeshFinalizeEvent {
                result,
                status: ReadyGpuMeshFinalizeStatus::ReadyAndValid,
                wait_reason: None,
            });
            finalized_count += 1;
            promoted_ready_count += 1;
        }
    }
    GPU_MESH_FINALIZE_LOCK_HOLD_NS.fetch_add(
        phase3_lock_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );
    GPU_MESH_FINALIZE_COUNT.fetch_add(finalized_count, Ordering::Relaxed);
    GPU_MESH_FINALIZE_REQUEUED_COUNT.fetch_add(requeued_count, Ordering::Relaxed);
    GPU_MESH_FINALIZE_WAITING_METADATA_COUNT.fetch_add(waiting_metadata_count, Ordering::Relaxed);
    GPU_MESH_FINALIZE_PROMOTED_READY_COUNT.fetch_add(promoted_ready_count, Ordering::Relaxed);
    GPU_MESH_FINALIZE_OWNERSHIP_INVALIDATED_COUNT
        .fetch_add(ownership_invalidations, Ordering::Relaxed);
    GPU_MESH_FINALIZE_SUPERSEDED_COUNT.fetch_add(superseded_count, Ordering::Relaxed);
    GPU_CANDIDATE_READY_COUNT.fetch_add(promoted_ready_count, Ordering::Relaxed);
    GPU_CANDIDATE_FAILED_COUNT.fetch_add(candidate_failed_count, Ordering::Relaxed);
    GPU_FINALIZE_INVALID_COUNT.fetch_add(finalize_invalid_count, Ordering::Relaxed);
    GPU_ZERO_OR_INVALID_MESH_OUTPUT_COUNT
        .fetch_add(zero_or_invalid_mesh_output_count, Ordering::Relaxed);

    out
}

#[cfg(feature = "gpu-compute")]
pub fn renderer_pending_finalize_identity_exists(
    coord: ChunkCoord,
    version: ChunkVersion,
    lod: u8,
    task_id: u64,
    page_index: GpuPageIndex,
    draw_indirect_index: u32,
) -> bool {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        return atlas.pending_finalize_identity_matches(
            coord,
            version,
            lod,
            task_id,
            page_index,
            draw_indirect_index,
        );
    }
    false
}

#[cfg(not(feature = "gpu-compute"))]
pub fn renderer_pending_finalize_identity_exists(
    _coord: ChunkCoord,
    _version: ChunkVersion,
    _lod: u8,
    _task_id: u64,
    _page_index: GpuPageIndex,
    _draw_indirect_index: u32,
) -> bool {
    false
}

#[cfg(feature = "gpu-compute")]
pub fn invalidate_gpu_pending_finalize_and_queued_on_renderer(
    coord: ChunkCoord,
    requested_version: Option<u64>,
    reason: &'static str,
    preserve_gpu_finalize_identity: bool,
) {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        let keep_live_finalize_identity = preserve_gpu_finalize_identity
            && atlas
                .pending_mesh_finalize
                .get(&coord)
                .map(|pending| {
                    requested_version
                        .map(|version| pending.version.get() == version)
                        .unwrap_or(true)
                })
                .unwrap_or(false);

        let removed_pending = if keep_live_finalize_identity {
            None
        } else {
            atlas.pending_mesh_finalize.remove(&coord)
        };

        let remove_queued = atlas
            .queued_task_reservations
            .get(&coord)
            .map(|reservation| {
                requested_version
                    .map(|version| reservation.version.get() <= version)
                    .unwrap_or(true)
            })
            .unwrap_or(false);
        if remove_queued {
            atlas.release_queued_task_identity_for_coord(coord);
        }

        if let Some(page_index) = atlas.page_for_chunk.get(&coord).copied() {
            let should_bump_page = removed_pending.is_some() || remove_queued;
            if should_bump_page {
                atlas.bump_page_generation(page_index);
            }
        }
        if let Some(mesh_slice) = atlas.mesh_slice_for_chunk.get(&coord).copied() {
            let should_bump_slot = removed_pending.is_some() || remove_queued;
            if should_bump_slot {
                atlas.bump_mesh_slot_generation(mesh_slice.slot_index);
            }
        }
        log::debug!(
            "[gpu-mesh] invalidate_pending_finalize_and_queued reason={} coord={:?} requested_version={:?} preserve_gpu_finalize_identity={}",
            reason,
            coord,
            requested_version,
            preserve_gpu_finalize_identity,
        );
    }
}

#[cfg(not(feature = "gpu-compute"))]
pub fn invalidate_gpu_pending_finalize_and_queued_on_renderer(
    _coord: ChunkCoord,
    _requested_version: Option<u64>,
    _reason: &'static str,
    _preserve_gpu_finalize_identity: bool,
) {
}

#[cfg(feature = "gpu-compute")]
pub fn invalidate_gpu_pending_finalize_on_renderer(
    coord: ChunkCoord,
    requested_version: Option<u64>,
    reason: &'static str,
) {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pending) = atlas.pending_mesh_finalize.remove(&coord) {
            log::debug!(
                "[gpu-mesh] invalidate_pending_finalize reason={} coord={:?} pending_version={} pending_lod={} pending_page={} pending_draw_slot={} serial={} requested_version={:?}",
                reason,
                coord,
                pending.version,
                pending.lod,
                pending.page_index.0,
                pending.draw_indirect_index,
                pending.submission_serial,
                requested_version,
            );
        }
    }
}

#[cfg(not(feature = "gpu-compute"))]
pub fn invalidate_gpu_pending_finalize_on_renderer(
    _coord: ChunkCoord,
    _requested_version: Option<u64>,
    _reason: &'static str,
) {
}

#[cfg(feature = "gpu-compute")]
pub fn set_gpu_protected_chunks_on_renderer(
    near_chunks: &[ChunkCoord],
    visible_chunks: &[ChunkCoord],
) {
    if let Some(Ok(state)) = WORKER_STATE
        .get()
        .map(|v| v.as_ref().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.set_protected_chunks(near_chunks, visible_chunks);
    }
}

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
fn gpu_vertex_color_offset_bytes() -> usize {
    let uninit = std::mem::MaybeUninit::<GpuVertex>::uninit();
    let base = uninit.as_ptr();
    // SAFETY: We only compute field addresses from an uninitialized pointer and never read data.
    unsafe { std::ptr::addr_of!((*base).color) as usize - base as usize }
}

#[cfg(feature = "gpu-compute")]
fn validate_gpu_vertex_contract() -> anyhow::Result<()> {
    let gpu_size = std::mem::size_of::<GpuVertex>();
    let renderer_size = std::mem::size_of::<crate::renderer::Vertex>();
    let gpu_align = std::mem::align_of::<GpuVertex>();
    let renderer_align = std::mem::align_of::<crate::renderer::Vertex>();
    let color_offset = gpu_vertex_color_offset_bytes();

    if gpu_size != renderer_size || gpu_align != renderer_align || color_offset != 12 {
        GPU_COLOR_CONTRACT_MISMATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        anyhow::bail!(
            "gpu vertex contract mismatch: gpu[size={},align={},color_offset={}] renderer[size={},align={}]",
            gpu_size,
            gpu_align,
            color_offset,
            renderer_size,
            renderer_align,
        );
    }

    Ok(())
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct ChunkMeshMeta {
    pub slot_index: u32,
    /// Global vertex buffer offset in elements (not bytes).
    pub vertex_offset: u32,
    /// Global index buffer offset in elements (not bytes).
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
    use super::{ChunkPageAtlas, MeshSliceAllocateOutcome, MESH_SLOT_COUNT};
    use crate::chunk_store::ChunkStore;
    use crate::engine::world::ChunkVersion;
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

    #[cfg(feature = "gpu-compute")]
    fn test_gpu_task(coord: ChunkCoord, task_id: u64) -> super::GpuChunkTask {
        super::GpuChunkTask {
            coord,
            page_index: crate::types::GpuPageIndex(0),
            page_generation: 0,
            frontier_count: 0,
            edit_commands: Vec::new(),
            jacobi_iterations: 0,
            neighbor_pages: [u32::MAX; 6],
            simulation_tick: 0,
            current_state: 0,
            startup_seeding_mode: false,
            mesh_slice: None,
            slot_generation: None,
            version: ChunkVersion(1),
            task_id,
            lod: 0,
            next_cached_materials: Vec::new(),
            next_state: 0,
            next_tick: 0,
            next_frontier_len: 0,
            next_diagnostics: super::ChunkSimulationDiagnostics::default(),
        }
    }

    #[cfg(feature = "gpu-compute")]
    #[test]
    fn face_count_clear_offset_is_page_relative() {
        let page_a = crate::types::GpuPageIndex(0);
        let page_b = crate::types::GpuPageIndex(7);

        let offset_a = super::face_count_offset_for_page(page_a);
        let offset_b = super::face_count_offset_for_page(page_b);

        assert_eq!(offset_a, 0);
        assert_eq!(
            offset_b,
            std::mem::size_of::<u32>() as u64 * page_b.0 as u64
        );
        assert_ne!(offset_a, offset_b);
    }
    #[cfg(feature = "gpu-compute")]
    #[test]
    fn enqueue_fails_fast_with_queue_full_pressure_reason_after_bounded_defers() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        tx.send(test_gpu_task(ChunkCoord { x: 9, y: 9, z: 9 }, 99))
            .expect("seed queue");

        let reason = super::try_enqueue_gpu_chunk_task(
            &tx,
            test_gpu_task(ChunkCoord { x: 1, y: 2, z: 3 }, 1),
        )
        .expect_err("full queue should fail after bounded defers");

        assert_eq!(
            reason,
            super::GpuTaskQueuePressureReason::QueueFull {
                max_defer_attempts: super::GPU_TASK_ENQUEUE_MAX_DEFER_ATTEMPTS,
            }
        );
    }

    #[cfg(feature = "gpu-compute")]
    #[test]
    fn queue_full_path_releases_queued_task_identity_reservations() {
        let mut atlas = ChunkPageAtlas::default();
        let coord = ChunkCoord { x: 2, y: 0, z: 0 };
        let task_id = 11;
        let version = ChunkVersion(2);
        let (page_index, _) = atlas.page_for_chunk_or_allocate(coord).expect("page");
        let mesh_slice = match atlas.mesh_slice_for_chunk_or_allocate(coord, 0) {
            MeshSliceAllocateOutcome::Success(slice) => Some(slice),
            other => panic!("expected mesh slice, got {other:?}"),
        };

        atlas.reserve_queued_task_identity(coord, task_id, version, page_index, mesh_slice);
        assert!(atlas.queued_task_reservations.contains_key(&coord));

        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        tx.send(test_gpu_task(ChunkCoord { x: 8, y: 8, z: 8 }, 88))
            .expect("seed queue");
        let enqueue = super::try_enqueue_gpu_chunk_task(&tx, test_gpu_task(coord, task_id));
        assert!(matches!(
            enqueue,
            Err(super::GpuTaskQueuePressureReason::QueueFull { .. })
        ));

        atlas.release_queued_task_identity(coord, task_id);
        assert!(!atlas.queued_task_reservations.contains_key(&coord));
        assert!(!atlas.has_queued_page_reservation(page_index));
        if let Some(mesh_slice) = mesh_slice {
            assert!(!atlas.has_queued_slot_reservation(mesh_slice.slot_index));
        }
    }

    #[cfg(feature = "gpu-compute")]
    #[test]
    fn reusing_slot_evicts_previous_owner_before_reallocation() {
        let mut atlas = ChunkPageAtlas::default();
        let owner_a = ChunkCoord { x: 0, y: 0, z: 0 };
        let owner_b = ChunkCoord { x: 1, y: 0, z: 0 };

        atlas
            .page_for_chunk_or_allocate(owner_a)
            .expect("owner_a page");
        atlas
            .page_for_chunk_or_allocate(owner_b)
            .expect("owner_b page");

        let allocated_a = atlas.mesh_slice_for_chunk_or_allocate(owner_a, 0);
        let slot = match allocated_a {
            MeshSliceAllocateOutcome::Success(slice) => slice.slot_index,
            other => panic!("expected initial allocation success, got {other:?}"),
        };

        atlas.next_mesh_slot = MESH_SLOT_COUNT;
        atlas.mesh_slot_last_used.insert(slot, 1);

        let allocated_b = atlas.mesh_slice_for_chunk_or_allocate(owner_b, 0);
        let slice_b = match allocated_b {
            MeshSliceAllocateOutcome::Success(slice) => slice,
            other => panic!("expected reused-slot allocation success, got {other:?}"),
        };

        assert_eq!(slice_b.slot_index, slot);
        assert!(!atlas.mesh_slice_for_chunk.contains_key(&owner_a));
        assert_eq!(atlas.chunk_for_mesh_slot.get(&slot), Some(&owner_b));
        assert_eq!(
            atlas.mesh_slice_for_chunk.len(),
            atlas.chunk_for_mesh_slot.len()
        );
        atlas.assert_mesh_slot_chunk_mapping_invariants();
    }
}
