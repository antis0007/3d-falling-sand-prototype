use crate::renderer::mesh_chunk_snapshot;
use crate::renderer::{ChunkMeshArtifact, MeshJob, MeshSkipReason, VOXEL_SIZE};
use crate::types::{ChunkCoord, GpuPageIndex, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu-compute")]
use glam::Vec3;
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
const MIN_MESH_VERTEX_CAPACITY_PER_CHUNK: u64 = 256;
#[cfg(feature = "gpu-compute")]
const MIN_MESH_INDEX_CAPACITY_PER_CHUNK: u64 = 384;

#[cfg(feature = "gpu-compute")]
pub const fn mesh_pool_slot_capacity() -> u32 {
    GPU_PAGE_CAPACITY
}

#[cfg(feature = "gpu-compute")]
pub const fn global_mesh_vertex_capacity_elements() -> u32 {
    (GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES / std::mem::size_of::<GpuVertex>() as u64) as u32
}

#[cfg(feature = "gpu-compute")]
pub const fn global_mesh_index_capacity_elements() -> u32 {
    (GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES / std::mem::size_of::<u32>() as u64) as u32
}

#[cfg(feature = "gpu-compute")]
fn lod_chunk_volume(lod: u8) -> u64 {
    let shift = (lod as u32).min(5);
    let cells_per_axis = (32u64 >> shift).max(1);
    cells_per_axis * cells_per_axis * cells_per_axis
}

#[cfg(feature = "gpu-compute")]
fn mesh_capacity_for_lod(lod: u8) -> (u32, u32) {
    let volume = lod_chunk_volume(lod);
    let vertex = (volume * 12).max(MIN_MESH_VERTEX_CAPACITY_PER_CHUNK);
    let index = (volume * 18).max(MIN_MESH_INDEX_CAPACITY_PER_CHUNK);
    (vertex as u32, index as u32)
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
    mesh_alloc_for_chunk: HashMap<ChunkCoord, MeshBufferAllocation>,
    chunk_for_mesh_slot: HashMap<u32, ChunkCoord>,
    page_ownership_generation: HashMap<GpuPageIndex, u64>,
    mesh_slot_ownership_generation: HashMap<u32, u64>,
    mesh_slot_last_used: HashMap<u32, u64>,
    mesh_slot_epoch: u64,
    next_mesh_slot: u32,
    free_vertex_ranges: Vec<MeshRange>,
    free_index_ranges: Vec<MeshRange>,
    used_vertex_elements: u32,
    used_index_elements: u32,
    next_page: GpuPageIndex,
    pending_mesh_finalize: HashMap<ChunkCoord, PendingGpuMeshFinalize>,
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
            mesh_alloc_for_chunk: HashMap::new(),
            chunk_for_mesh_slot: HashMap::new(),
            page_ownership_generation: HashMap::new(),
            mesh_slot_ownership_generation: HashMap::new(),
            mesh_slot_last_used: HashMap::new(),
            mesh_slot_epoch: 0,
            next_mesh_slot: 0,
            free_vertex_ranges: vec![MeshRange {
                start: 0,
                len: global_mesh_vertex_capacity_elements(),
            }],
            free_index_ranges: vec![MeshRange {
                start: 0,
                len: global_mesh_index_capacity_elements(),
            }],
            used_vertex_elements: 0,
            used_index_elements: 0,
            next_page: GpuPageIndex(0),
            pending_mesh_finalize: HashMap::new(),
        }
    }
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
struct MeshRange {
    start: u32,
    len: u32,
}

#[cfg(feature = "gpu-compute")]
#[derive(Clone, Copy, Debug)]
struct MeshBufferAllocation {
    vertex: MeshRange,
    index: MeshRange,
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
    version: u64,
    task_id: u64,
    lod: u8,
    page_index: GpuPageIndex,
    draw_indirect_index: u32,
    page_generation: u64,
    slot_generation: u64,
    submission_serial: u64,
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

        if self.next_page.0 < GPU_PAGE_CAPACITY {
            let page = self.next_page;
            self.next_page = GpuPageIndex(self.next_page.0.saturating_add(1));
            self.page_for_chunk.insert(chunk, page);
            self.chunk_for_page.insert(page, chunk);
            self.bump_page_generation(page);
            self.touch_page(page);
            return Ok((page, true));
        }

        let page = self.evictable_page().with_context(|| {
            format!("no reusable gpu atlas pages available for chunk {chunk:?}")
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
    ) -> Option<MeshBufferSlice> {
        if let Some(existing) = self.mesh_slice_for_chunk.get(&chunk).copied() {
            self.touch_mesh_slot(existing.slot_index);
            return Some(existing);
        }

        let (required_vertex, required_index) = mesh_capacity_for_lod(lod);
        let slot_capacity = mesh_pool_slot_capacity();
        if slot_capacity == 0 {
            return None;
        }

        let mut slot = if self.next_mesh_slot < slot_capacity {
            let next = self.next_mesh_slot;
            self.next_mesh_slot = self.next_mesh_slot.saturating_add(1);
            next
        } else {
            self.evictable_mesh_slot(slot_capacity)?
        };

        loop {
            if self
                .try_allocate_mesh_buffers(chunk, slot, required_vertex, required_index)
                .is_some()
            {
                break;
            }

            let evict_slot = self.evictable_mesh_slot(slot_capacity)?;
            if !self.evict_mesh_slot(evict_slot) {
                return None;
            }

            if self.next_mesh_slot >= slot_capacity {
                slot = evict_slot;
            }
        }

        self.mesh_slice_for_chunk.get(&chunk).copied()
    }

    fn try_allocate_mesh_buffers(
        &mut self,
        chunk: ChunkCoord,
        slot: u32,
        required_vertex: u32,
        required_index: u32,
    ) -> Option<MeshBufferSlice> {
        let vertex = take_range(&mut self.free_vertex_ranges, required_vertex)?;
        let index = take_range(&mut self.free_index_ranges, required_index).or_else(|| {
            release_range(&mut self.free_vertex_ranges, vertex);
            None
        })?;

        let slice = MeshBufferSlice {
            slot_index: slot,
            vertex_offset: vertex.start,
            index_offset: index.start,
        };
        self.mesh_alloc_for_chunk
            .insert(chunk, MeshBufferAllocation { vertex, index });
        self.mesh_slice_for_chunk.insert(chunk, slice);
        self.chunk_for_mesh_slot.insert(slot, chunk);
        self.bump_mesh_slot_generation(slot);
        self.used_vertex_elements = self.used_vertex_elements.saturating_add(vertex.len);
        self.used_index_elements = self.used_index_elements.saturating_add(index.len);
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

        if let Some(allocation) = self.mesh_alloc_for_chunk.remove(&chunk) {
            release_range(&mut self.free_vertex_ranges, allocation.vertex);
            release_range(&mut self.free_index_ranges, allocation.index);
            self.used_vertex_elements = self
                .used_vertex_elements
                .saturating_sub(allocation.vertex.len);
            self.used_index_elements = self
                .used_index_elements
                .saturating_sub(allocation.index.len);
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
            if slot >= slot_capacity {
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
        assert_eq!(
            self.mesh_slice_for_chunk.len(),
            self.chunk_for_mesh_slot.len(),
            "mesh slot/chunk map size mismatch"
        );

        for (chunk, slice) in &self.mesh_slice_for_chunk {
            let owner = self
                .chunk_for_mesh_slot
                .get(&slice.slot_index)
                .expect("mesh slot must resolve to chunk owner");
            assert_eq!(owner, chunk, "mesh slot/chunk mapping mismatch");
            assert!(
                self.mesh_alloc_for_chunk.contains_key(chunk),
                "mesh allocation missing for chunk with active mesh slice"
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
        let slot_capacity = mesh_pool_slot_capacity();
        let slots_used = self.mesh_slice_for_chunk.len() as u32;
        let vertex_capacity = global_mesh_vertex_capacity_elements();
        let index_capacity = global_mesh_index_capacity_elements();
        MeshPoolTelemetry {
            slot_capacity,
            slots_used,
            in_flight_fences: self.in_flight_mesh_slot_fence_count(),
            vertex_used: self.used_vertex_elements,
            vertex_capacity,
            index_used: self.used_index_elements,
            index_capacity,
            largest_free_vertex_span: largest_range_len(&self.free_vertex_ranges),
            largest_free_index_span: largest_range_len(&self.free_index_ranges),
            vertex_usage_percent: if vertex_capacity == 0 {
                0.0
            } else {
                self.used_vertex_elements as f32 * 100.0 / vertex_capacity as f32
            },
            index_usage_percent: if index_capacity == 0 {
                0.0
            } else {
                self.used_index_elements as f32 * 100.0 / index_capacity as f32
            },
            slot_usage_percent: if slot_capacity == 0 {
                0.0
            } else {
                slots_used as f32 * 100.0 / slot_capacity as f32
            },
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
            if fence.last_completed < fence.last_submitted {
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
            self.priority_hint_for_chunk.remove(&chunk);
            self.page_fences.remove(&page_index);
            self.page_last_used.remove(&page_index);
        }
    }
}

#[cfg(feature = "gpu-compute")]
fn take_range(free_ranges: &mut Vec<MeshRange>, required_len: u32) -> Option<MeshRange> {
    let idx = free_ranges.iter().position(|r| r.len >= required_len)?;
    let range = free_ranges[idx];
    let allocated = MeshRange {
        start: range.start,
        len: required_len,
    };
    if range.len == required_len {
        free_ranges.swap_remove(idx);
    } else {
        free_ranges[idx].start = free_ranges[idx].start.saturating_add(required_len);
        free_ranges[idx].len = free_ranges[idx].len.saturating_sub(required_len);
    }
    Some(allocated)
}

#[cfg(feature = "gpu-compute")]
fn release_range(free_ranges: &mut Vec<MeshRange>, released: MeshRange) {
    free_ranges.push(released);
    merge_free_ranges(free_ranges);
}

#[cfg(feature = "gpu-compute")]
fn merge_free_ranges(free_ranges: &mut Vec<MeshRange>) {
    free_ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<MeshRange> = Vec::with_capacity(free_ranges.len());
    for range in free_ranges.drain(..) {
        if let Some(last) = merged.last_mut() {
            let last_end = last.start.saturating_add(last.len);
            if last_end >= range.start {
                let range_end = range.start.saturating_add(range.len);
                last.len = range_end.saturating_sub(last.start).max(last.len);
                continue;
            }
        }
        merged.push(range);
    }
    *free_ranges = merged;
}

#[cfg(feature = "gpu-compute")]
fn largest_range_len(free_ranges: &[MeshRange]) -> u32 {
    free_ranges.iter().map(|range| range.len).max().unwrap_or(0)
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
    fn new(device: &wgpu::Device) -> Self {
        let draw_stride = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
        let slot_capacity = mesh_pool_slot_capacity() as u64;
        let size = draw_stride * slot_capacity;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu draw indirect frame readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            staging,
            map_result_rx: None,
            requested_serial: 0,
            ready_serial: 0,
            cached_index_counts: vec![0; mesh_pool_slot_capacity() as usize],
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
        if completed_serial == 0
            || self.map_result_rx.is_some()
            || self.ready_serial >= completed_serial
        {
            return;
        }

        let byte_len = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64
            * mesh_pool_slot_capacity() as u64;
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
#[derive(Clone, Copy, Debug)]
pub struct ReadyGpuMeshResult {
    pub coord: ChunkCoord,
    pub version: u64,
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
    pub version: u64,
    pub task_id: u64,
    pub lod: u8,
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
            chunks_per_sec: chunks_completed as f32 / frame,
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
    #[cfg(feature = "gpu_meshing_experimental")]
    fn run_meshing_dispatch(
        &self,
        state: &WorkerGpuState,
        scratch: &GpuScratchPool,
        sim_job: &SimulationJob,
        page_index: GpuPageIndex,
        current_state: u32,
        _lod: u8,
        mesh_slice: MeshBufferSlice,
    ) -> anyhow::Result<()> {
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
        let draw_indirect_readback = DrawIndirectReadbackState::new(&device);

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

        let mesh_slice = atlas.mesh_slice_for_chunk_or_allocate(job.coord, job.lod as u8);
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
                version: job.version,
                task_id: job.task_id,
                lod: job.lod as u8,
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
        let (chunk_origin_world, aabb_min, aabb_max) = chunk_world_bounds(job.coord);

        Ok(ComputedChunkArtifacts {
            simulation_diagnostics: diagnostics,
            mesh_artifact: {
                #[cfg(feature = "gpu_meshing_experimental")]
                {
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
                        let fragmented = mesh_pool_telemetry.vertex_used
                            < mesh_pool_telemetry.vertex_capacity
                            && mesh_pool_telemetry.largest_free_vertex_span
                                < MIN_MESH_VERTEX_CAPACITY_PER_CHUNK as u32
                            || mesh_pool_telemetry.index_used < mesh_pool_telemetry.index_capacity
                                && mesh_pool_telemetry.largest_free_index_span
                                    < MIN_MESH_INDEX_CAPACITY_PER_CHUNK as u32;
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
                                largest_free_vertex_span: mesh_pool_telemetry
                                    .largest_free_vertex_span,
                                largest_free_index_span: mesh_pool_telemetry
                                    .largest_free_index_span,
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
        let task_wait_sync = Duration::ZERO;

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
            }
        }

        let serial = GPU_SUBMISSION_SERIAL.fetch_add(1, Ordering::Relaxed) + 1;

        {
            let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
            atlas.mark_page_submitted(task.page_index, serial);
            atlas.touch_chunk_page(task.coord);
            if let Some(mesh_slice) = task.mesh_slice {
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
            .filter_map(|(coord, pending)| {
                (pending.submission_serial <= completed).then_some(FinalizeCandidate {
                    coord: *coord,
                    pending: *pending,
                })
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
                if readback.has_snapshot_for(candidate.pending.submission_serial) {
                    readback.index_count_for_slot(candidate.pending.draw_indirect_index)
                } else {
                    None
                }
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

    let phase3_lock_start = Instant::now();
    {
        let mut atlas = state.atlas.lock().unwrap_or_else(|e| e.into_inner());
        for (candidate, index_count) in candidates.into_iter().zip(index_counts.into_iter()) {
            let Some(current_pending) = atlas.pending_mesh_finalize.get(&candidate.coord).copied()
            else {
                continue;
            };
            if current_pending != candidate.pending {
                requeued_count += 1;
                superseded_count += 1;
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

            if atlas.version_for_chunk.get(&candidate.coord).copied()
                != Some(candidate.pending.version)
            {
                atlas.pending_mesh_finalize.remove(&candidate.coord);
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::DroppedStaleVersion,
                });
                finalized_count += 1;
                superseded_count += 1;
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
                });
                finalized_count += 1;
                ownership_invalidations += 1;
                continue;
            }

            if result.index_count.is_none() {
                out.push(ReadyGpuMeshFinalizeEvent {
                    result,
                    status: ReadyGpuMeshFinalizeStatus::NotReadyYet,
                });
                waiting_metadata_count += 1;
                continue;
            }

            atlas.pending_mesh_finalize.remove(&candidate.coord);
            atlas.touch_chunk_page(candidate.coord);
            out.push(ReadyGpuMeshFinalizeEvent {
                result,
                status: ReadyGpuMeshFinalizeStatus::ReadyAndValid,
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

    out
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
