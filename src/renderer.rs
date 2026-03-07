//! Authoritative meshing pipeline for runtime rendering.
//!
//! - **Authoritative input:** meshing consumes a [`ChunkSnapshot`] built from
//!   [`ChunkStore`] via `build_chunk_snapshot`, so workers never read mutable
//!   world state directly.
//! - **Vertex coordinate space:** both CPU and GPU meshing emit
//!   **world-space** vertex positions in meters (scaled by [`VOXEL_SIZE`]).
//! - **Chunk transform ownership:** chunk origins are authored in world space
//!   and consumed by the meshing compute shader via `chunk_origin_buffer`.
//! - **Floating-origin contract:** renderer APIs accept **world-space camera
//!   coordinates**. The renderer then derives both world-space culling and
//!   render-space projection from that one source of truth. The render shader
//!   subtracts `world_origin_offset` exactly once.

use crate::chunk_store::{ChunkBorderStrips, ChunkStore};
use crate::gpu_compute::{
    dispatch_gpu_chunk_tasks_on_renderer, initialize_gpu_compute_worker, run_chunk_job_on_worker,
    take_ready_gpu_mesh_results_on_renderer, update_gpu_page_fences_on_renderer, DrawIndirectArgs,
    GpuComputeRuntime, MeshPipelineBackend, SharedMeshBuffers,
};
#[cfg(feature = "gpu-compute")]
use crate::gpu_compute::{
    gpu_page_capacity, mesh_pool_slot_capacity, required_storage_buffer_binding_size_bytes,
    ReadyGpuMeshFinalizeEvent, ReadyGpuMeshFinalizeStatus, COMPUTE_STORAGE_BINDING_COUNT,
    GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES, GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES,
};
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, GpuPageIndex, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

pub const VOXEL_SIZE: f32 = 0.5;
const MAX_PENDING_DIRTY_CHUNKS: usize = 16_384;
const CHUNK_SNAPSHOT_BUILD_BUDGET_MS: f32 = 1.5;
const MESH_RETRY_MAX_ATTEMPTS: u32 = 6;
const MESH_RETRY_BASE_BACKOFF_FRAMES: u64 = 2;
const MESH_RETRY_SKIPPED_MAX_BACKOFF_FRAMES: u64 = 64;
const MESH_RETRY_SKIPPED_PRIORITY_BOOST_ATTEMPTS: u32 = 3;
const MESH_RETRY_SKIPPED_WARN_ATTEMPTS: u32 = 4;
const OUTCOME_TRACE_SAMPLES_PER_SECOND: u32 = 6;
const GPU_RENDER_DISPATCH_MAX_TASKS_PER_FRAME: usize = 64;
const GPU_RENDER_DISPATCH_MAX_TIME_BUDGET_MS: f32 = 1.0;
const LOD_MISMATCH_GRACE_MAX_FRAMES: u64 = 8;
const BUSH_ID: MaterialId = 18;
const GRASS_ID: MaterialId = 19;

#[derive(Clone, Copy, Debug)]
pub enum UnknownNeighborOcclusionPolicy {
    Conservative,
    Aggressive,
}

impl Default for UnknownNeighborOcclusionPolicy {
    fn default() -> Self {
        Self::Conservative
    }
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
    world_origin_offset: [f32; 3],
    _pad: f32,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
#[derive(Clone, Copy)]
struct MeshBufferRange {
    offset: u64,
    size: u64,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
#[derive(Clone, Copy)]
struct MeshAllocation {
    page_index: usize,
    vertex: MeshBufferRange,
    index: MeshBufferRange,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
#[derive(Clone, Copy)]
struct MeshPageRange {
    vertex: MeshBufferRange,
    index: MeshBufferRange,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
struct MeshPage {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    free_ranges: Vec<MeshPageRange>,
    capacity: u64,
    vertex_cursor: u64,
    index_cursor: u64,
    live_allocations: usize,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
struct MeshPageAllocator {
    label: &'static str,
    page_size: u64,
    pages: Vec<Option<MeshPage>>,
}

impl MeshPageAllocator {
    fn new(label: &'static str, page_size: u64) -> Self {
        Self {
            label,
            page_size,
            pages: Vec::new(),
        }
    }

    fn page(&self, page_index: usize) -> Option<&MeshPage> {
        self.pages.get(page_index).and_then(|page| page.as_ref())
    }

    fn allocate(
        &mut self,
        device: &wgpu::Device,
        vertex_size: u64,
        index_size: u64,
        telemetry: &mut MeshAllocatorTelemetry,
    ) -> MeshAllocation {
        for (page_index, page) in self.pages.iter_mut().enumerate() {
            let Some(page) = page.as_mut() else {
                continue;
            };

            if let Some((free_idx, free_alloc)) =
                page.free_ranges
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, alloc)| {
                        alloc.vertex.size >= vertex_size && alloc.index.size >= index_size
                    })
            {
                page.free_ranges.swap_remove(free_idx);
                telemetry.bytes_reused += (vertex_size + index_size) as usize;
                if free_alloc.vertex.size > vertex_size && free_alloc.index.size > index_size {
                    page.free_ranges.push(MeshPageRange {
                        vertex: MeshBufferRange {
                            offset: free_alloc.vertex.offset + vertex_size,
                            size: free_alloc.vertex.size - vertex_size,
                        },
                        index: MeshBufferRange {
                            offset: free_alloc.index.offset + index_size,
                            size: free_alloc.index.size - index_size,
                        },
                    });
                }
                page.live_allocations += 1;
                return MeshAllocation {
                    page_index,
                    vertex: MeshBufferRange {
                        offset: free_alloc.vertex.offset,
                        size: vertex_size,
                    },
                    index: MeshBufferRange {
                        offset: free_alloc.index.offset,
                        size: index_size,
                    },
                };
            }

            if page.capacity - page.vertex_cursor >= vertex_size
                && page.capacity - page.index_cursor >= index_size
            {
                let vertex_offset = page.vertex_cursor;
                let index_offset = page.index_cursor;
                page.vertex_cursor += vertex_size;
                page.index_cursor += index_size;
                page.live_allocations += 1;
                return MeshAllocation {
                    page_index,
                    vertex: MeshBufferRange {
                        offset: vertex_offset,
                        size: vertex_size,
                    },
                    index: MeshBufferRange {
                        offset: index_offset,
                        size: index_size,
                    },
                };
            }
        }

        let new_capacity = self.page_size.max(vertex_size).max(index_size);
        let page_index = self
            .pages
            .iter()
            .position(|page| page.is_none())
            .unwrap_or(self.pages.len());
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} vertex page", self.label)),
            size: new_capacity,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} index page", self.label)),
            size: new_capacity,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let page = MeshPage {
            vertex_buffer,
            index_buffer,
            free_ranges: Vec::new(),
            capacity: new_capacity,
            vertex_cursor: vertex_size,
            index_cursor: index_size,
            live_allocations: 1,
        };
        if page_index == self.pages.len() {
            self.pages.push(Some(page));
        } else {
            self.pages[page_index] = Some(page);
        }
        telemetry.bytes_allocated += (new_capacity * 2) as usize;
        MeshAllocation {
            page_index,
            vertex: MeshBufferRange {
                offset: 0,
                size: vertex_size,
            },
            index: MeshBufferRange {
                offset: 0,
                size: index_size,
            },
        }
    }

    fn free(&mut self, allocation: MeshAllocation) {
        if let Some(Some(page)) = self.pages.get_mut(allocation.page_index) {
            page.free_ranges.push(MeshPageRange {
                vertex: allocation.vertex,
                index: allocation.index,
            });
            page.live_allocations = page.live_allocations.saturating_sub(1);
            if page.live_allocations == 0 {
                self.pages[allocation.page_index] = None;
            }
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct MeshAllocatorTelemetry {
    bytes_allocated: usize,
    bytes_reused: usize,
    realloc_count: usize,
}

pub struct Camera {
    /// Camera position in **world space**.
    ///
    /// Renderer APIs (`render_world`, `cull_stats`, `mesh_draw_stats`) expect
    /// this to be absolute world coordinates. Floating-origin rebasing is
    /// handled internally by [`Renderer`] using `origin_voxel`.
    pub pos: Vec3,
    pub dir: Vec3,
    pub aspect: f32,
}

impl Camera {
    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_to_rh(self.pos, self.dir, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), self.aspect.max(0.1), 0.1, 1200.0);
        proj * view
    }

    fn view_proj_rebased_to_origin(&self, origin_voxel: VoxelCoord) -> Mat4 {
        let origin_world = voxel_to_world(origin_voxel);
        let rebased_camera = Camera {
            pos: self.pos - origin_world,
            dir: self.dir,
            aspect: self.aspect,
        };
        rebased_camera.view_proj()
    }
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
pub struct ChunkMesh {
    allocation: MeshAllocation,
    index_count: u32,
    debug_aabb_vb: wgpu::Buffer,
    debug_aabb_ib: wgpu::Buffer,
    debug_aabb_index_count: u32,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    chunk_origin_world: Vec3,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
#[derive(Default)]
struct ChunkMeshCache {
    near: Option<ChunkMesh>,
    mid: Option<ChunkMesh>,
    far: Option<ChunkMesh>,
    ultra: Option<ChunkMesh>,
}

impl ChunkMeshCache {
    fn get(&self, lod: ChunkLod) -> Option<&ChunkMesh> {
        match lod {
            ChunkLod::Near => self.near.as_ref(),
            ChunkLod::Mid => self.mid.as_ref(),
            ChunkLod::Far => self.far.as_ref(),
            ChunkLod::Ultra => self.ultra.as_ref(),
        }
    }

    fn slot_mut(&mut self, lod: ChunkLod) -> &mut Option<ChunkMesh> {
        match lod {
            ChunkLod::Near => &mut self.near,
            ChunkLod::Mid => &mut self.mid,
            ChunkLod::Far => &mut self.far,
            ChunkLod::Ultra => &mut self.ultra,
        }
    }

    fn best_available(
        &self,
        selected: ChunkLod,
        chunk_distance: f32,
        near_lod_distance: f32,
    ) -> Option<(ChunkLod, &ChunkMesh)> {
        let ordered = [
            ChunkLod::Near,
            ChunkLod::Mid,
            ChunkLod::Far,
            ChunkLod::Ultra,
        ];
        let start = lod_rank(selected);
        for lod in ordered.iter().skip(start).copied() {
            if chunk_distance < near_lod_distance && lod_rank(lod) > lod_rank(ChunkLod::Mid) {
                continue;
            }
            if let Some(mesh) = self.get(lod) {
                return Some((lod, mesh));
            }
        }
        for lod in ordered.iter().take(start).copied() {
            if chunk_distance < near_lod_distance && lod_rank(lod) > lod_rank(ChunkLod::Mid) {
                continue;
            }
            if let Some(mesh) = self.get(lod) {
                return Some((lod, mesh));
            }
        }
        None
    }

    fn drain(self) -> impl Iterator<Item = ChunkMesh> {
        [self.near, self.mid, self.far, self.ultra]
            .into_iter()
            .flatten()
    }

    fn is_empty(&self) -> bool {
        self.near.is_none() && self.mid.is_none() && self.far.is_none() && self.ultra.is_none()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// Deprecated: CPU mesh pipeline removed. Structures retained temporarily
// to avoid breaking references during GPU renderer migration.
#[derive(Clone, Copy, Debug)]
struct ChunkOriginInstance {
    chunk_origin_world: Vec3,
}

const DEBUG_VISIBLE_CHUNK_LOG_COUNT: usize = 8;
const DEBUG_RENDER_CHUNK_AABBS: bool = false;
const DEBUG_VALIDATE_CULL_SPACE: bool = false;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChunkLod {
    Near,
    Mid,
    Far,
    Ultra,
}

fn lod_rank(lod: ChunkLod) -> usize {
    match lod {
        ChunkLod::Near => 0,
        ChunkLod::Mid => 1,
        ChunkLod::Far => 2,
        ChunkLod::Ultra => 3,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LodRadii {
    pub near: i32,
    pub mid: i32,
    pub far: i32,
    pub ultra: i32,
    pub hysteresis: i32,
}

impl LodRadii {
    pub fn normalized(mut self) -> Self {
        self.near = self.near.max(0);
        self.mid = self.mid.max(self.near);
        self.far = self.far.max(self.mid);
        self.ultra = self.ultra.max(self.far);
        self.hysteresis = self.hysteresis.max(0);
        self
    }
}
#[derive(Clone, Copy, Debug)]
pub struct LodMeshingBudgets {
    pub near: usize,
    pub mid: usize,
    pub far: usize,
    pub ultra: usize,
}

#[derive(Clone, Copy)]
enum SubmissionTier {
    Near,
    Mid,
    Far,
    Ultra,
}

fn compute_effective_lod_submission_budgets(
    mesh_budget: usize,
    lod_budgets: LodMeshingBudgets,
    pending: [usize; 4],
) -> [usize; 4] {
    if mesh_budget == 0 {
        return [0; 4];
    }

    let configured = [
        lod_budgets.near,
        lod_budgets.mid,
        lod_budgets.far,
        lod_budgets.ultra,
    ];
    let configured_total: usize = configured.iter().sum();
    let pending_total: usize = pending.iter().sum();
    if pending_total == 0 {
        return [0; 4];
    }

    let mut effective = [0usize; 4];
    for i in 0..4 {
        if pending[i] == 0 {
            continue;
        }

        let cfg_share = if configured_total > 0 {
            (mesh_budget.saturating_mul(configured[i]) + configured_total - 1) / configured_total
        } else {
            0
        };
        let pending_share =
            (mesh_budget.saturating_mul(pending[i]) + pending_total - 1) / pending_total;
        let blended_share = (cfg_share + pending_share).div_ceil(2);
        effective[i] = blended_share.max(1).min(pending[i]);
    }

    let mut total_effective: usize = effective.iter().sum();
    while total_effective > mesh_budget {
        let mut reduced = false;
        for i in 0..4 {
            if total_effective <= mesh_budget {
                break;
            }
            let min_floor = usize::from(pending[i] > 0);
            if effective[i] > min_floor {
                effective[i] -= 1;
                total_effective -= 1;
                reduced = true;
            }
        }
        if !reduced {
            break;
        }
    }

    while total_effective < mesh_budget {
        let mut expanded = false;
        for i in 0..4 {
            if total_effective >= mesh_budget {
                break;
            }
            if effective[i] < pending[i] {
                effective[i] += 1;
                total_effective += 1;
                expanded = true;
            }
        }
        if !expanded {
            break;
        }
    }

    effective
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,

    depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    visible_gpu_chunks: HashMap<ChunkCoord, GpuChunkDraw>,
    visible_slots: HashMap<u32, ChunkCoord>,
    free_mesh_slots: Vec<u32>,
    global_gpu_vertex_buffer: Arc<wgpu::Buffer>,
    global_gpu_index_buffer: Arc<wgpu::Buffer>,
    global_gpu_draw_indirect_buffer: Arc<wgpu::Buffer>,
    global_gpu_page_indirect_buffer: Arc<wgpu::Buffer>,
    global_gpu_mesh_meta_buffer: Arc<wgpu::Buffer>,
    global_gpu_chunk_origin_buffer: Arc<wgpu::Buffer>,
    global_gpu_face_mask_buffer: Arc<wgpu::Buffer>,
    global_gpu_face_offset_buffer: Arc<wgpu::Buffer>,
    global_gpu_face_count_buffer: Arc<wgpu::Buffer>,
    supports_multi_draw_indirect: bool,

    dirty_queues: DirtyChunkQueues,
    urgent_mesh_queue: VecDeque<ChunkCoord>,
    urgent_mesh_set: HashSet<ChunkCoord>,
    dirty_near_starve_frames: u32,
    dirty_far_starve_frames: u32,
    dirty_fair_cursor: u8,
    mesh_versions: HashMap<ChunkCoord, u64>,
    lod_selection: HashMap<ChunkCoord, ChunkLod>,
    pending_lod_remesh: HashSet<ChunkCoord>,
    pending_lod_remesh_since: HashMap<ChunkCoord, u64>,
    inflight_mesh_chunks: HashSet<ChunkCoord>,
    pending_gpu_results: HashMap<(ChunkCoord, u64), PendingGpuMeshResult>,
    terminal_superseded_total: usize,
    terminal_evicted_total: usize,
    mesh_lifecycle: HashMap<ChunkCoord, MeshLifecycleState>,
    mesh_retry_state: HashMap<ChunkCoord, MeshRetryState>,
    startup_mesh_seed_state: HashMap<ChunkCoord, StartupMeshSeedState>,
    mesh_rebuild_frame_index: u64,
    near_lod_distance: f32,

    mesh_queue: BackgroundMeshQueue,
    completed_meshes: Vec<MeshResult>,
    outcome_trace_window_start: Instant,
    outcome_trace_emitted: u32,
    startup_seed_trace_window_start: Instant,
    startup_seed_trace_emitted: u32,
    startup_epoch: Instant,
    origin_voxel: VoxelCoord,

    pub day: bool,
    pub mesh_backend: MeshPipelineBackend,
    pub startup_diagnostics: StartupDiagnostics,
    settings: RendererSettings,
}

#[derive(Clone, Debug)]
pub struct StartupDiagnostics {
    pub backend_selected: MeshPipelineBackend,
    pub required_limits_summary: String,
    pub adapter_limits_summary: String,
    pub startup_error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct RendererSettings {
    pub frustum_culling: bool,
    pub greedy_meshing: bool,
    pub unknown_neighbor_policy: UnknownNeighborOcclusionPolicy,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            frustum_culling: true,
            greedy_meshing: true,
            unknown_neighbor_policy: UnknownNeighborOcclusionPolicy::default(),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct CullStats {
    pub drawn: usize,
    pub frustum_culled: usize,
    pub lod_filtered: usize,
    pub lod_mismatch_grace_drawn: usize,
    pub screen_culled: usize,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct MeshRebuildStats {
    pub max_ms: f32,
    pub total_ms: f32,
    pub mesh_count: usize,
    pub dirty_backlog: usize,
    pub meshing_queue_depth: usize,
    pub meshing_completed_depth: usize,
    pub upload_count: usize,
    pub upload_bytes: usize,
    pub upload_latency_ms: f32,
    pub stale_drop_count: usize,
    pub stale_drop_retry_enqueued: usize,
    pub age_drop_count: usize,
    pub pressure_drop_count: usize,
    pub dirty_queue_drop_count: usize,
    pub dirty_urgent_depth: usize,
    pub dirty_near_depth: usize,
    pub dirty_normal_depth: usize,
    pub dirty_far_depth: usize,
    pub near_mesh_count: usize,
    pub mid_mesh_count: usize,
    pub far_mesh_count: usize,
    pub ultra_mesh_count: usize,
    pub gpu_mesh_jobs: usize,
    pub gpu_dispatch_ms: f32,
    pub gpu_dispatch_enqueue_submit_ms: f32,
    pub gpu_dispatch_wait_sync_ms: f32,
    pub gpu_dispatch_tasks_submitted: usize,
    pub gpu_mesh_adopted_count: usize,
    pub gpu_mesh_adoption_latency_ms: f32,
    pub gpu_mesh_visible_count: usize,
    pub gpu_mesh_visible_slot_min: i32,
    pub gpu_mesh_visible_slot_max: i32,
    pub gpu_mesh_visible_slot_holes: usize,
    pub gpu_job_failures: usize,
    pub gpu_job_timeouts: usize,
    pub gpu_job_skipped: usize,
    pub gpu_mesh_slot_alloc_failed: usize,
    pub gpu_readback_bytes: u64,
    pub allocator_bytes_allocated: usize,
    pub allocator_bytes_reused: usize,
    pub allocator_realloc_count: usize,
    pub mesh_artifacts_received: usize,
    pub mesh_artifacts_rejected: usize,
    pub mesh_reject_stale: usize,
    pub mesh_reject_invalid_page: usize,
    pub mesh_reject_zero_index: usize,
    pub mesh_reject_failed: usize,
    pub mesh_reject_unhandled: usize,
    pub mesh_zero_index_soft_retries: usize,
    pub mesh_cache_entries: usize,
    pub outcome_gpu_adopted: usize,
    pub outcome_cpu_uploaded: usize,
    pub outcome_skipped_zero_geometry: usize,
    pub outcome_skipped_startup_zero_geometry: usize,
    pub outcome_skipped_no_artifact_capacity: usize,
    pub outcome_skipped_invalid_page_mapping: usize,
    pub outcome_skipped_missing_voxel_state: usize,
    pub outcome_skipped_adoption_rejected: usize,
    pub outcome_skipped_sparse_indirect_undrawable: usize,
    pub outcome_skipped_backend_contract_mismatch: usize,
    pub startup_seed_zero_count_seen: usize,
    pub startup_seed_recovered_nonzero: usize,
    pub startup_zero_near_retry_enqueued: usize,
    pub flow_received: usize,
    pub flow_adopted: usize,
    pub flow_uploaded: usize,
    pub flow_rejected: usize,
    pub resident_gpu_artifact: usize,
    pub resident_cpu_uploaded: usize,
    pub resident_stale_cached: usize,
    pub resident_fallback: usize,
    pub resident_unknown: usize,
    pub mesh_reject_no_longer_desired: usize,
    pub mesh_slot_allocation_failures: usize,
    pub mesh_pending_total: usize,
    pub mesh_pending_finalize: usize,
    pub mesh_pending_promoted_to_drawable: usize,
    pub mesh_waiting_on_fence: usize,
    pub drawable_resident_total: usize,
    pub newly_drawable_this_frame: usize,
    pub pending_finalize_total: usize,
    pub waiting_on_fence_total: usize,
    pub terminal_superseded_total: usize,
    pub terminal_evicted_total: usize,
    pub mesh_pending_superseded: usize,
    pub mesh_pending_rejected: usize,
    pub mesh_dropped_before_drawable: usize,
    pub mesh_last_good_retained: usize,
    pub mesh_visible_logical_not_drawable: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawSource {
    GpuArtifact,
    CpuUploaded,
    StaleCached,
    Fallback,
    Unknown,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct MeshDrawStats {
    pub chunks_drawn: usize,
    pub total_indices: u64,
    pub drawn_gpu_artifact_chunks: usize,
    pub drawn_gpu_artifact_indices: u64,
    pub drawn_cpu_uploaded_chunks: usize,
    pub drawn_cpu_uploaded_indices: u64,
    pub drawn_stale_cached_chunks: usize,
    pub drawn_stale_cached_indices: u64,
    pub drawn_fallback_chunks: usize,
    pub drawn_fallback_indices: u64,
    pub drawn_unknown_chunks: usize,
    pub drawn_unknown_indices: u64,
}

impl MeshDrawStats {
    fn record_draw(&mut self, source: DrawSource, index_count: u64) {
        self.chunks_drawn += 1;
        self.total_indices += index_count;
        match source {
            DrawSource::GpuArtifact => {
                self.drawn_gpu_artifact_chunks += 1;
                self.drawn_gpu_artifact_indices += index_count;
            }
            DrawSource::CpuUploaded => {
                self.drawn_cpu_uploaded_chunks += 1;
                self.drawn_cpu_uploaded_indices += index_count;
            }
            DrawSource::StaleCached => {
                self.drawn_stale_cached_chunks += 1;
                self.drawn_stale_cached_indices += index_count;
            }
            DrawSource::Fallback => {
                self.drawn_fallback_chunks += 1;
                self.drawn_fallback_indices += index_count;
            }
            DrawSource::Unknown => {
                self.drawn_unknown_chunks += 1;
                self.drawn_unknown_indices += index_count;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MeshRetryState {
    failed_attempts: u32,
    failed_next_retry_frame: u64,
    skipped_attempts: u32,
    skipped_next_retry_frame: u64,
}

#[derive(Clone, Copy)]
enum MeshRetryKind {
    Failed,
    Skipped(MeshSkipReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaleArtifactRetryPolicy {
    Urgent,
    Dirty,
}

fn stale_artifact_retry_policy(
    result_version: u64,
    voxel_version: u64,
    lod: ChunkLod,
    urgent: bool,
) -> Option<StaleArtifactRetryPolicy> {
    if result_version.saturating_add(1) >= voxel_version {
        return None;
    }

    if urgent || matches!(lod, ChunkLod::Near | ChunkLod::Mid) {
        Some(StaleArtifactRetryPolicy::Urgent)
    } else {
        Some(StaleArtifactRetryPolicy::Dirty)
    }
}

const COMPLETED_MESH_BACKLOG_THRESHOLD: usize = 256;
const DIRTY_NEAR_STARVE_LIMIT_FRAMES: u32 = 8;
const DIRTY_FAR_STARVE_LIMIT_FRAMES: u32 = 20;
const DIRTY_VISIBLE_URGENT_SCORE: f32 = 0.8;
const MAX_LOD_REMESH_PER_FRAME: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DirtyTier {
    Urgent,
    Near,
    Normal,
    Far,
}

impl DirtyTier {
    fn priority_rank(self) -> u8 {
        match self {
            Self::Urgent => 4,
            Self::Near => 3,
            Self::Normal => 2,
            Self::Far => 1,
        }
    }
}

#[derive(Default)]
struct DirtyChunkQueues {
    urgent: VecDeque<ChunkCoord>,
    near: VecDeque<ChunkCoord>,
    normal: VecDeque<ChunkCoord>,
    far: VecDeque<ChunkCoord>,
    tiers: HashMap<ChunkCoord, DirtyTier>,
    urgent_count: usize,
    near_count: usize,
    normal_count: usize,
    far_count: usize,
}

impl DirtyChunkQueues {
    fn total_len(&self) -> usize {
        self.tiers.len()
    }

    fn tier_len(&self, tier: DirtyTier) -> usize {
        match tier {
            DirtyTier::Urgent => self.urgent_count,
            DirtyTier::Near => self.near_count,
            DirtyTier::Normal => self.normal_count,
            DirtyTier::Far => self.far_count,
        }
    }

    fn incr_tier(&mut self, tier: DirtyTier) {
        match tier {
            DirtyTier::Urgent => self.urgent_count += 1,
            DirtyTier::Near => self.near_count += 1,
            DirtyTier::Normal => self.normal_count += 1,
            DirtyTier::Far => self.far_count += 1,
        }
    }

    fn decr_tier(&mut self, tier: DirtyTier) {
        match tier {
            DirtyTier::Urgent => self.urgent_count = self.urgent_count.saturating_sub(1),
            DirtyTier::Near => self.near_count = self.near_count.saturating_sub(1),
            DirtyTier::Normal => self.normal_count = self.normal_count.saturating_sub(1),
            DirtyTier::Far => self.far_count = self.far_count.saturating_sub(1),
        }
    }

    fn queue_for_mut(&mut self, tier: DirtyTier) -> &mut VecDeque<ChunkCoord> {
        match tier {
            DirtyTier::Urgent => &mut self.urgent,
            DirtyTier::Near => &mut self.near,
            DirtyTier::Normal => &mut self.normal,
            DirtyTier::Far => &mut self.far,
        }
    }

    fn queue_coord(&mut self, coord: ChunkCoord, tier: DirtyTier) {
        let old_tier = self.tiers.get(&coord).copied();
        if let Some(old_tier) = old_tier {
            if old_tier.priority_rank() >= tier.priority_rank() {
                return;
            }
            self.decr_tier(old_tier);
        }
        self.tiers.insert(coord, tier);
        self.incr_tier(tier);
        self.queue_for_mut(tier).push_back(coord);
    }

    fn pop_front_tier(&mut self, tier: DirtyTier) -> Option<ChunkCoord> {
        loop {
            let coord = self.queue_for_mut(tier).pop_front()?;
            if self.tiers.get(&coord).copied() == Some(tier) {
                self.tiers.remove(&coord);
                self.decr_tier(tier);
                return Some(coord);
            }
        }
    }

    fn pop_back_tier(&mut self, tier: DirtyTier) -> Option<ChunkCoord> {
        loop {
            let coord = self.queue_for_mut(tier).pop_back()?;
            if self.tiers.get(&coord).copied() == Some(tier) {
                self.tiers.remove(&coord);
                self.decr_tier(tier);
                return Some(coord);
            }
        }
    }

    fn remove_coord(&mut self, coord: ChunkCoord) {
        if let Some(tier) = self.tiers.remove(&coord) {
            self.decr_tier(tier);
        }
    }

    fn clear(&mut self) {
        self.urgent.clear();
        self.near.clear();
        self.normal.clear();
        self.far.clear();
        self.tiers.clear();
        self.urgent_count = 0;
        self.near_count = 0;
        self.normal_count = 0;
        self.far_count = 0;
    }
}

#[derive(Clone)]
pub(crate) struct ChunkSnapshot {
    pub(crate) world_min: VoxelCoord,
    pub(crate) center_voxels: Arc<[MaterialId]>,
    pub(crate) border_strips: Arc<ChunkBorderStrips>,
}

impl ChunkSnapshot {
    fn get_local(&self, local_x: i32, local_y: i32, local_z: i32) -> MaterialId {
        let side = CHUNK_SIZE_VOXELS;
        if (0..side).contains(&local_x)
            && (0..side).contains(&local_y)
            && (0..side).contains(&local_z)
        {
            let idx = ((local_z as usize * side as usize + local_y as usize) * side as usize)
                + local_x as usize;
            return self.center_voxels[idx];
        }

        let strips = self.border_strips.as_ref();
        let idx = |u: i32, v: i32| (u as usize) * side as usize + v as usize;

        if local_x == -1 && (0..side).contains(&local_y) && (0..side).contains(&local_z) {
            return strips.neg_x[idx(local_y, local_z)];
        }
        if local_x == side && (0..side).contains(&local_y) && (0..side).contains(&local_z) {
            return strips.pos_x[idx(local_y, local_z)];
        }
        if local_y == -1 && (0..side).contains(&local_x) && (0..side).contains(&local_z) {
            return strips.neg_y[idx(local_x, local_z)];
        }
        if local_y == side && (0..side).contains(&local_x) && (0..side).contains(&local_z) {
            return strips.pos_y[idx(local_x, local_z)];
        }
        if local_z == -1 && (0..side).contains(&local_x) && (0..side).contains(&local_y) {
            return strips.neg_z[idx(local_x, local_y)];
        }
        if local_z == side && (0..side).contains(&local_x) && (0..side).contains(&local_y) {
            return strips.pos_z[idx(local_x, local_y)];
        }

        EMPTY
    }

    pub(crate) fn with_center_materials(&self, materials: Vec<MaterialId>) -> Self {
        let center_voxels: Arc<[MaterialId]> = Arc::from(materials);
        let border_strips = Arc::new(self.rebuild_border_strips_from_center(&center_voxels));
        Self {
            world_min: self.world_min,
            center_voxels,
            border_strips,
        }
    }

    fn rebuild_border_strips_from_center(&self, new_center: &[MaterialId]) -> ChunkBorderStrips {
        let side = CHUNK_SIZE_VOXELS as usize;
        let mut strips = self.border_strips.as_ref().clone();
        let old_center = self.center_voxels.as_ref();
        if old_center.len() != new_center.len() || old_center.len() != side * side * side {
            return strips;
        }

        let center_idx = |x: usize, y: usize, z: usize| -> usize { (z * side + y) * side + x };
        let strip_idx = |u: usize, v: usize| -> usize { u * side + v };

        for y in 0..side {
            for z in 0..side {
                let edge_old = old_center[center_idx(0, y, z)];
                let i = strip_idx(y, z);
                if strips.neg_x[i] == edge_old {
                    strips.neg_x[i] = new_center[center_idx(0, y, z)];
                }

                let edge_old = old_center[center_idx(side - 1, y, z)];
                if strips.pos_x[i] == edge_old {
                    strips.pos_x[i] = new_center[center_idx(side - 1, y, z)];
                }
            }
        }

        for x in 0..side {
            for z in 0..side {
                let i = strip_idx(x, z);
                let edge_old = old_center[center_idx(x, 0, z)];
                if strips.neg_y[i] == edge_old {
                    strips.neg_y[i] = new_center[center_idx(x, 0, z)];
                }

                let edge_old = old_center[center_idx(x, side - 1, z)];
                if strips.pos_y[i] == edge_old {
                    strips.pos_y[i] = new_center[center_idx(x, side - 1, z)];
                }
            }
        }

        for x in 0..side {
            for y in 0..side {
                let i = strip_idx(x, y);
                let edge_old = old_center[center_idx(x, y, 0)];
                if strips.neg_z[i] == edge_old {
                    strips.neg_z[i] = new_center[center_idx(x, y, 0)];
                }

                let edge_old = old_center[center_idx(x, y, side - 1)];
                if strips.pos_z[i] == edge_old {
                    strips.pos_z[i] = new_center[center_idx(x, y, side - 1)];
                }
            }
        }

        strips
    }
}

#[derive(Clone)]
pub(crate) struct MeshJob {
    pub(crate) coord: ChunkCoord,
    pub(crate) lod: ChunkLod,
    pub(crate) version: u64,
    pub(crate) queued_at: Instant,
    pub(crate) snapshot: ChunkSnapshot,
    pub(crate) greedy: bool,
    pub(crate) urgent: bool,
}

struct MeshResult {
    coord: ChunkCoord,
    lod: ChunkLod,
    version: u64,
    queued_at: Instant,
    artifact: ChunkMeshArtifact,
    urgent: bool,
    from_pending_finalize: bool,
}

struct PendingGpuMeshResult {
    result: MeshResult,
    first_seen_frame: u64,
    first_seen_completed_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshLifecycleState {
    Requested,
    Meshing,
    ResultReady,
    AwaitingPage,
    AwaitingFinalize,
    Drawable,
    Superseded,
    Rejected,
    Evicted,
}

#[derive(Clone, Copy, Debug)]
enum RebuildOutcome {
    GpuAdopted,
    CpuUploaded,
    SkippedZeroGeometry,
    SkippedStartupZeroGeometry,
    SkippedNoArtifactCapacity,
    SkippedInvalidPageMapping,
    SkippedMissingVoxelState,
    SkippedAdoptionRejected,
    SkippedSparseIndirectUndrawable,
    SkippedBackendContractMismatch,
}

#[derive(Clone, Copy, Debug)]
struct GpuChunkDraw {
    page_index: GpuPageIndex,
    draw_indirect_index: u32,
    lod: u8,
    origin: Vec3,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    draw_source: DrawSource,
    // Optional debug metadata only; indirect draw args remain authoritative.
    index_count: Option<u32>,
}

#[derive(Clone, Copy)]
struct DrawVisibilityInput {
    frustum_culling: bool,
    vp_world: Mat4,
    world_camera_pos: Vec3,
    screen_h: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct StartupMeshSeedState {
    startup_seed_zero_count_seen: bool,
    startup_seed_recovered_nonzero: bool,
    first_zero_at: Option<Instant>,
    recovered_nonzero_at: Option<Instant>,
}

const GPU_MESH_VERTEX_CAPACITY_PER_SLOT: u64 =
    (CHUNK_SIZE_VOXELS as u64) * (CHUNK_SIZE_VOXELS as u64) * (CHUNK_SIZE_VOXELS as u64) * 24;
const GPU_MESH_INDEX_CAPACITY_PER_SLOT: u64 =
    (CHUNK_SIZE_VOXELS as u64) * (CHUNK_SIZE_VOXELS as u64) * (CHUNK_SIZE_VOXELS as u64) * 36;

pub(crate) enum ChunkMeshArtifact {
    Cpu {
        verts: Vec<Vertex>,
        inds: Vec<u32>,
        indirect: DrawIndirectArgs,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
    },
    GpuPending {
        page_index: GpuPageIndex,
        draw_indirect_index: u32,
        lod: u8,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
    },
    GpuReady {
        page_index: GpuPageIndex,
        draw_indirect_index: u32,
        lod: u8,
        index_count: u32,
        aabb_min: Vec3,
        aabb_max: Vec3,
        chunk_origin_world: Vec3,
        dispatch_ms: f32,
    },
    Failed {
        reason: String,
    },
    Skipped {
        reason: MeshSkipReason,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MeshSkipReason {
    BackendDisabled,
    WorkerBusy,
    MeshSlotCapacitySaturated {
        slot_capacity: u32,
        in_flight_fences: u32,
    },
    ZeroGeometry,
    StartupZeroGeometry,
    InvalidPageMapping,
    MissingVoxelState,
    AdoptionRejected,
    SparseIndirectUndrawable,
    BackendContractMismatch,
}

impl ChunkMeshArtifact {
    pub(crate) fn geometry(&self) -> (&[Vertex], &[u32], DrawIndirectArgs, Vec3, Vec3, Vec3) {
        match self {
            Self::Cpu {
                verts,
                inds,
                indirect,
                aabb_min,
                aabb_max,
                chunk_origin_world,
            } => (
                verts,
                inds,
                *indirect,
                *aabb_min,
                *aabb_max,
                *chunk_origin_world,
            ),
            Self::GpuPending {
                aabb_min,
                aabb_max,
                chunk_origin_world,
                ..
            }
            | Self::GpuReady {
                aabb_min,
                aabb_max,
                chunk_origin_world,
                ..
            } => (
                &[],
                &[],
                DrawIndirectArgs::default(),
                *aabb_min,
                *aabb_max,
                *chunk_origin_world,
            ),
            Self::Skipped { .. } => (
                &[],
                &[],
                DrawIndirectArgs::default(),
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
            ),
            Self::Failed { .. } => (
                &[],
                &[],
                DrawIndirectArgs::default(),
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
            ),
        }
    }
}

struct BackgroundMeshQueue {
    tx: SyncSender<MeshJob>,
    rx: Receiver<MeshResult>,
    inflight: usize,
}

fn build_mesh_artifact(mesh_backend: MeshPipelineBackend, job: &MeshJob) -> ChunkMeshArtifact {
    match mesh_backend {
        MeshPipelineBackend::Disabled => {
            let _ = job;
            ChunkMeshArtifact::Skipped {
                reason: MeshSkipReason::BackendDisabled,
            }
        }
        MeshPipelineBackend::Cpu => {
            crate::gpu_compute::cpu_generate_material_field(job).mesh_artifact
        }
        #[cfg(feature = "gpu-compute")]
        MeshPipelineBackend::Gpu => match run_chunk_job_on_worker(job) {
            Ok(output) => output.mesh_artifact,
            Err(err) => ChunkMeshArtifact::Failed {
                reason: format!("chunk={:?} lod={:?}: {err:#}", job.coord, job.lod),
            },
        },
    }
}

fn short_error_message(reason: &str) -> &str {
    reason.lines().next().unwrap_or("unknown")
}

fn is_gpu_timeout_reason(reason: &str) -> bool {
    reason.contains("gpu dispatch timeout")
}

impl BackgroundMeshQueue {
    fn new(worker_count: usize, queue_bound: usize, mesh_backend: MeshPipelineBackend) -> Self {
        let (tx, job_rx) = sync_channel::<MeshJob>(queue_bound);
        let (result_tx, rx) = sync_channel::<MeshResult>(queue_bound);
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

        for i in 0..worker_count {
            let worker_rx = std::sync::Arc::clone(&job_rx);
            let worker_tx = result_tx.clone();
            thread::Builder::new()
                .name(format!("mesh-worker-{i}"))
                .spawn(move || loop {
                    let job = {
                        let lock = worker_rx.lock().expect("mesh worker rx lock");
                        lock.recv()
                    };
                    let Ok(job) = job else {
                        break;
                    };
                    log::info!("[mesh-worker] picked job chunk={:?}", job.coord);

                    let artifact = catch_unwind(AssertUnwindSafe(|| {
                        build_mesh_artifact(mesh_backend, &job)
                    }))
                    .unwrap_or_else(|panic_payload| {
                        let panic_reason = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            (*s).to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        ChunkMeshArtifact::Failed {
                            reason: format!(
                                "chunk={:?} lod={:?}: worker panic while building mesh artifact: {}",
                                job.coord, job.lod, panic_reason
                            ),
                        }
                    });
                    if let ChunkMeshArtifact::Failed { reason } = &artifact {
                        log::warn!(
                            "[mesh-worker] gpu job failed chunk={:?} lod={:?} error={}",
                            job.coord,
                            job.lod,
                            short_error_message(reason)
                        );
                    }
                    let result = MeshResult {
                        coord: job.coord,
                        lod: job.lod,
                        version: job.version,
                        queued_at: job.queued_at,
                        artifact,
                        urgent: job.urgent,
                        from_pending_finalize: false,
                    };
                    log::info!("[mesh-worker] sending result chunk={:?}", result.coord);
                    if worker_tx.send(result).is_err() {
                        break;
                    }
                    log::info!("[mesh-worker] send completed chunk={:?}", job.coord);
                })
                .expect("spawn mesh worker");
        }

        Self {
            tx,
            rx,
            inflight: 0,
        }
    }

    fn try_submit(&mut self, job: MeshJob) -> Result<(), TrySendError<MeshJob>> {
        log::info!(
            "[mesh] submit job chunk={:?} version={}",
            job.coord,
            job.version
        );
        match self.tx.try_send(job) {
            Ok(()) => {
                self.inflight += 1;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn try_recv(&mut self) -> Result<MeshResult, TryRecvError> {
        let result = self.rx.try_recv();
        if result.is_ok() {
            self.inflight = self.inflight.saturating_sub(1);
        }
        result
    }
}

impl Renderer {
    pub async fn new(
        window: &'static winit::window::Window,
        require_gpu_meshing: bool,
        force_disabled_meshing: bool,
    ) -> anyhow::Result<Self> {
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

        let supported_features = adapter.features();
        let adapter_limits = adapter.limits();
        let required_features =
            wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::INDIRECT_FIRST_INSTANCE;
        let enabled_features = supported_features & required_features;
        let supports_multi_draw_indirect =
            enabled_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT);

        let mut requested_limits = adapter_limits.clone();

        #[cfg(feature = "gpu-compute")]
        {
            let required_storage_size = required_storage_buffer_binding_size_bytes();

            let requested_storage_buffers = requested_limits
                .max_storage_buffers_per_shader_stage
                .max(COMPUTE_STORAGE_BINDING_COUNT);

            requested_limits.max_storage_buffers_per_shader_stage =
                requested_storage_buffers.min(adapter_limits.max_storage_buffers_per_shader_stage);

            requested_limits.max_storage_buffer_binding_size = requested_limits
                .max_storage_buffer_binding_size
                .max(required_storage_size as u32)
                .min(adapter_limits.max_storage_buffer_binding_size);

            requested_limits.max_buffer_size = requested_limits
                .max_buffer_size
                .max(required_storage_size)
                .min(adapter_limits.max_buffer_size);

            log::info!(
                "gpu compute required limits: storage-buffers-per-stage requested={} adapter-supported={} runtime-required={} required-storage-size={}B requested-max-binding={}B adapter-max-binding={}B requested-max-buffer={}B adapter-max-buffer={}B",
                requested_limits.max_storage_buffers_per_shader_stage,
                adapter_limits.max_storage_buffers_per_shader_stage,
                COMPUTE_STORAGE_BINDING_COUNT,
                required_storage_size,
                requested_limits.max_storage_buffer_binding_size,
                adapter_limits.max_storage_buffer_binding_size,
                requested_limits.max_buffer_size,
                adapter_limits.max_buffer_size,
            );
        }

        #[cfg(not(feature = "gpu-compute"))]
        {
            log::info!(
                "device limits: storage-buffers-per-stage requested={} adapter-supported={}",
                requested_limits.max_storage_buffers_per_shader_stage,
                adapter_limits.max_storage_buffers_per_shader_stage,
            );
        }

        if !supports_multi_draw_indirect {
            log::warn!(
                "adapter does not support MULTI_DRAW_INDIRECT; falling back to per-command indirect draws"
            );
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Voxel Renderer Device"),
                    required_features: enabled_features,
                    required_limits: requested_limits.clone(),
                },
                None,
            )
            .await?;

        let device_limits = device.limits();
        let required_limits_summary = {
            #[cfg(feature = "gpu-compute")]
            {
                let required_storage_size = required_storage_buffer_binding_size_bytes();
                format!(
                    "storage_buffers/stage req={} requested={} device={} | storage_binding req={}B requested={}B device={}B | max_buffer req={}B requested={}B device={}B",
                    COMPUTE_STORAGE_BINDING_COUNT,
                    requested_limits.max_storage_buffers_per_shader_stage,
                    device_limits.max_storage_buffers_per_shader_stage,
                    required_storage_size,
                    requested_limits.max_storage_buffer_binding_size,
                    device_limits.max_storage_buffer_binding_size,
                    required_storage_size,
                    requested_limits.max_buffer_size,
                    device_limits.max_buffer_size,
                )
            }
            #[cfg(not(feature = "gpu-compute"))]
            {
                format!(
                    "storage_buffers/stage requested={} device={}",
                    requested_limits.max_storage_buffers_per_shader_stage,
                    device_limits.max_storage_buffers_per_shader_stage,
                )
            }
        };
        let adapter_limits_summary = format!(
            "storage_buffers/stage adapter={} | storage_binding adapter={}B | max_buffer adapter={}B",
            adapter_limits.max_storage_buffers_per_shader_stage,
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_buffer_size,
        );

        let (mesh_backend, startup_error) = if force_disabled_meshing {
            log::warn!("mesh backend selected: disabled (explicit CLI override)");
            (MeshPipelineBackend::Disabled, None)
        } else if GpuComputeRuntime::runtime_supported(&adapter, &device_limits) {
            #[cfg(feature = "gpu-compute")]
            {
                log::info!(
                    "mesh backend selected: gpu-compute (adapter supports compute pipelines, page_capacity={})",
                    gpu_page_capacity()
                );
                (MeshPipelineBackend::Gpu, None)
            }
            #[cfg(not(feature = "gpu-compute"))]
            {
                anyhow::bail!(
                    "gpu meshing is required at runtime, but `gpu-compute` feature is disabled"
                );
            }
        } else {
            #[cfg(feature = "gpu-compute")]
            {
                let required_storage_size = required_storage_buffer_binding_size_bytes();
                log::warn!(
                    "gpu meshing unavailable: required_storage_size={}B, storage_buffers_per_shader_stage(required={} adapter={} requested={} device={}), storage_buffer_binding_size(required={} adapter={} requested={} device={}), max_buffer_size(required={} adapter={} requested={} device={})",
                    required_storage_size,
                    COMPUTE_STORAGE_BINDING_COUNT,
                    adapter_limits.max_storage_buffers_per_shader_stage,
                    requested_limits.max_storage_buffers_per_shader_stage,
                    device_limits.max_storage_buffers_per_shader_stage,
                    required_storage_size,
                    adapter_limits.max_storage_buffer_binding_size,
                    requested_limits.max_storage_buffer_binding_size,
                    device_limits.max_storage_buffer_binding_size,
                    required_storage_size,
                    adapter_limits.max_buffer_size,
                    requested_limits.max_buffer_size,
                    device_limits.max_buffer_size,
                );
            }

            #[cfg(not(feature = "gpu-compute"))]
            {
                log::warn!(
                    "gpu meshing unavailable: `gpu-compute` feature is disabled in this build"
                );
            }

            if require_gpu_meshing {
                anyhow::bail!(
                    "gpu meshing is required (--require-gpu-meshing), but adapter/runtime does not satisfy gpu-compute requirements"
                );
            }

            log::warn!(
                "mesh backend selected: cpu fallback path (adapter/runtime does not satisfy gpu-compute requirements)"
            );
            (MeshPipelineBackend::Cpu, None)
        };
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
                world_origin_offset: [0.0, 0.0, 0.0],
                _pad: 0.0,
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
                    blend: None,
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
        let page_capacity = gpu_page_capacity() as u64;
        let mesh_slot_capacity = mesh_pool_slot_capacity() as u64;
        let global_gpu_vertex_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu mesh vertex buffer"),
            size: GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let global_gpu_index_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu mesh index buffer"),
            size: GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let global_gpu_draw_indirect_buffer =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global gpu draw indirect buffer"),
                size: mesh_slot_capacity * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
                usage: wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let global_gpu_page_indirect_buffer =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global gpu page indirect buffer"),
                size: page_capacity * std::mem::size_of::<DrawIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let global_gpu_mesh_meta_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu mesh meta buffer"),
            size: page_capacity * std::mem::size_of::<crate::gpu_compute::ChunkMeshMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let global_gpu_chunk_origin_buffer =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global gpu chunk origin buffer"),
                size: page_capacity * std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let face_entries_per_slot =
            (CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS * CHUNK_SIZE_VOXELS) as u64;
        let face_mask_bytes =
            mesh_slot_capacity * face_entries_per_slot * std::mem::size_of::<u32>() as u64;
        let face_offset_bytes =
            mesh_slot_capacity * face_entries_per_slot * std::mem::size_of::<u32>() as u64;
        let face_count_bytes = page_capacity * std::mem::size_of::<u32>() as u64;
        let global_gpu_face_mask_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global gpu face mask buffer"),
            size: face_mask_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let global_gpu_face_offset_buffer =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global gpu face offset buffer"),
                size: face_offset_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let global_gpu_face_count_buffer =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global gpu face count buffer"),
                size: face_count_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        debug_assert!(
            mesh_slot_capacity > 0,
            "mesh pool must support at least one slot"
        );
        log::info!(
            "gpu mesh pool initialized: vertex={} MiB index={} MiB slots={}",
            GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES / (1024 * 1024),
            GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES / (1024 * 1024),
            mesh_slot_capacity
        );

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        if mesh_backend == MeshPipelineBackend::Gpu {
            initialize_gpu_compute_worker(
                Arc::clone(&device),
                Arc::clone(&queue),
                SharedMeshBuffers {
                    chunk_vertex_buffer: Arc::clone(&global_gpu_vertex_buffer),
                    chunk_index_buffer: Arc::clone(&global_gpu_index_buffer),
                    draw_indirect_buffer: Arc::clone(&global_gpu_draw_indirect_buffer),
                    page_indirect: Arc::clone(&global_gpu_page_indirect_buffer),
                    mesh_meta_buffer: Arc::clone(&global_gpu_mesh_meta_buffer),
                    chunk_origin_buffer: Arc::clone(&global_gpu_chunk_origin_buffer),
                    face_mask_buffer: Arc::clone(&global_gpu_face_mask_buffer),
                    face_offset_buffer: Arc::clone(&global_gpu_face_offset_buffer),
                    face_count_buffer: Arc::clone(&global_gpu_face_count_buffer),
                },
            )?;
        }

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
            visible_gpu_chunks: HashMap::new(),
            visible_slots: HashMap::new(),
            free_mesh_slots: (0..mesh_pool_slot_capacity() as u32).collect(),
            global_gpu_vertex_buffer,
            global_gpu_index_buffer,
            global_gpu_draw_indirect_buffer,
            global_gpu_page_indirect_buffer,
            global_gpu_mesh_meta_buffer,
            global_gpu_chunk_origin_buffer,
            global_gpu_face_mask_buffer,
            global_gpu_face_offset_buffer,
            global_gpu_face_count_buffer,
            supports_multi_draw_indirect,
            dirty_queues: DirtyChunkQueues::default(),
            urgent_mesh_queue: VecDeque::new(),
            urgent_mesh_set: HashSet::new(),
            dirty_near_starve_frames: 0,
            dirty_far_starve_frames: 0,
            dirty_fair_cursor: 0,
            mesh_versions: HashMap::new(),
            lod_selection: HashMap::new(),
            pending_lod_remesh: HashSet::new(),
            pending_lod_remesh_since: HashMap::new(),
            inflight_mesh_chunks: HashSet::new(),
            pending_gpu_results: HashMap::new(),
            terminal_superseded_total: 0,
            terminal_evicted_total: 0,
            mesh_lifecycle: HashMap::new(),
            mesh_retry_state: HashMap::new(),
            startup_mesh_seed_state: HashMap::new(),
            mesh_rebuild_frame_index: 0,
            near_lod_distance: 1.5,
            mesh_queue: BackgroundMeshQueue::new(2, 256, mesh_backend),
            completed_meshes: Vec::new(),
            outcome_trace_window_start: Instant::now(),
            outcome_trace_emitted: 0,
            startup_seed_trace_window_start: Instant::now(),
            startup_seed_trace_emitted: 0,
            startup_epoch: Instant::now(),
            origin_voxel: VoxelCoord { x: 0, y: 0, z: 0 },
            day: true,
            mesh_backend,
            startup_diagnostics: StartupDiagnostics {
                backend_selected: mesh_backend,
                required_limits_summary,
                adapter_limits_summary,
                startup_error,
            },
            settings: RendererSettings::default(),
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.size = size;
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        (self.depth_texture, self.depth_view) = create_depth_texture(&self.device, &self.config);
    }

    pub fn settings(&self) -> RendererSettings {
        self.settings
    }

    pub fn set_settings(&mut self, settings: RendererSettings) {
        self.settings = settings;
    }

    pub fn set_origin_voxel(&mut self, new_origin: VoxelCoord) {
        self.origin_voxel = new_origin;
    }

    fn draw_slot_is_owned_or_free(&self, coord: ChunkCoord, draw: &GpuChunkDraw) -> bool {
        match self.visible_slots.get(&draw.draw_indirect_index).copied() {
            None => true,
            Some(owner) => owner == coord,
        }
    }

    fn draw_meets_resident_invariant(&self, coord: ChunkCoord, draw: &GpuChunkDraw) -> bool {
        self.draw_slot_is_owned_or_free(coord, draw) && draw_is_drawable(draw)
    }

    fn should_render_draw(
        &self,
        coord: ChunkCoord,
        draw: &GpuChunkDraw,
        visibility: DrawVisibilityInput,
    ) -> bool {
        chunk_passes_draw_contract(
            coord,
            draw,
            &self.visible_slots,
            &self.lod_selection,
            &self.pending_lod_remesh,
            &self.pending_lod_remesh_since,
            self.mesh_rebuild_frame_index,
            visibility,
        )
    }

    pub fn cull_stats(&self, camera: &Camera) -> CullStats {
        let visibility = DrawVisibilityInput {
            frustum_culling: self.settings.frustum_culling,
            vp_world: camera.view_proj(),
            world_camera_pos: camera_world_position(camera),
            screen_h: self.size.height,
        };
        let mut stats = CullStats::default();
        for (&coord, draw) in &self.visible_gpu_chunks {
            if !self.draw_meets_resident_invariant(coord, draw) {
                continue;
            }
            let selected_lod = self
                .lod_selection
                .get(&coord)
                .copied()
                .unwrap_or(ChunkLod::Near);
            let draw_lod = lod_from_u8(draw.lod);
            if !draw_is_drawable(draw) {
                continue;
            }
            if draw_lod != selected_lod {
                if lod_mismatch_grace_allows_draw(
                    coord,
                    &self.pending_lod_remesh,
                    &self.pending_lod_remesh_since,
                    self.mesh_rebuild_frame_index,
                ) {
                    stats.lod_mismatch_grace_drawn += 1;
                } else {
                    stats.lod_filtered += 1;
                    continue;
                }
            }
            if self.settings.frustum_culling
                && !aabb_in_view(
                    visibility.vp_world,
                    draw.world_aabb_min,
                    draw.world_aabb_max,
                )
            {
                stats.frustum_culled += 1;
                continue;
            }
            if !passes_screen_space_cull(
                visibility.world_camera_pos,
                draw_lod,
                draw.world_aabb_min,
                draw.world_aabb_max,
                visibility.screen_h,
            ) {
                stats.screen_culled += 1;
                continue;
            }
            stats.drawn += 1;
        }
        stats
    }

    pub fn cull_visible_chunks(&self, camera: &Camera) -> Vec<ChunkCoord> {
        let visibility = DrawVisibilityInput {
            frustum_culling: self.settings.frustum_culling,
            vp_world: camera.view_proj(),
            world_camera_pos: camera_world_position(camera),
            screen_h: self.size.height,
        };
        let mut visible = Vec::new();
        for (&coord, draw) in &self.visible_gpu_chunks {
            if self.should_render_draw(coord, draw, visibility) {
                visible.push(coord);
            }
        }
        visible
    }

    /// Rebuild up to `budget` dirty chunks this frame, without O(N) re-marking cost.
    ///
    /// IMPORTANT: This relies on `store.take_dirty_chunks()` returning + clearing the store's dirty set.

    pub fn rebuild_dirty_store_chunks(
        &mut self,
        store: &mut ChunkStore,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
        mesh_budget: usize,
        _upload_byte_budget: usize,
        lod_radii: LodRadii,
        lod_budgets: LodMeshingBudgets,
    ) -> MeshRebuildStats {
        self.mesh_rebuild_frame_index = self.mesh_rebuild_frame_index.saturating_add(1);
        self.process_mesh_retry_backoff(player_chunk, chunk_priority_scores);
        let lod_radii = lod_radii.normalized();
        self.near_lod_distance = lod_radii.near as f32 + 0.5;
        for coord in store.take_urgent_dirty_chunks() {
            self.enqueue_urgent_mesh_chunk(coord);
        }
        for coord in store.take_dirty_chunks() {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
        let mut stats = MeshRebuildStats::default();
        stats.dirty_queue_drop_count +=
            self.enforce_dirty_queue_bound(player_chunk, chunk_priority_scores);

        let mut near_jobs = Vec::new();
        let mut mid_jobs = Vec::new();
        let mut far_jobs = Vec::new();
        let mut ultra_jobs = Vec::new();
        let mut frame_jobs: HashMap<ChunkCoord, MeshJob> = HashMap::new();

        let mut urgent_jobs = self.pop_urgent_mesh_jobs(store, &mut stats, player_chunk, lod_radii);
        for job in urgent_jobs.drain(..) {
            Self::enqueue_frame_job(&mut frame_jobs, job);
        }

        let chunk_snapshot_budget = mesh_budget.max(1);
        let mut snapshot_coords = self.pop_priority_dirty_chunks(
            chunk_snapshot_budget,
            player_chunk,
            chunk_priority_scores,
        );
        let snapshot_frame_start = Instant::now();
        let mut deferred_snapshot_coords = Vec::new();
        while let Some(coord) = snapshot_coords.pop() {
            let elapsed_ms = snapshot_frame_start.elapsed().as_secs_f32() * 1000.0;
            if elapsed_ms >= CHUNK_SNAPSHOT_BUILD_BUDGET_MS {
                deferred_snapshot_coords.push(coord);
                deferred_snapshot_coords.extend(snapshot_coords.into_iter().rev());
                break;
            }

            let t0 = Instant::now();
            let snapshot =
                build_chunk_snapshot(store, coord, self.settings.unknown_neighbor_policy);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            stats.total_ms += ms;
            if ms > stats.max_ms {
                stats.max_ms = ms;
            }

            let prev = self.lod_selection.get(&coord).copied();
            let primary_lod = select_lod(coord, player_chunk, lod_radii, prev);
            let fallback_lod =
                fallback_lod_near_threshold(coord, player_chunk, lod_radii, primary_lod);

            let mut push_job = |lod: ChunkLod| {
                let version = store.chunk_voxel_version(coord);
                Self::enqueue_frame_job(
                    &mut frame_jobs,
                    MeshJob {
                        coord,
                        lod,
                        version,
                        queued_at: Instant::now(),
                        snapshot: snapshot.clone(),
                        greedy: self.settings.greedy_meshing,
                        urgent: false,
                    },
                );
            };

            push_job(primary_lod);

            if let Some(lod) = fallback_lod {
                push_job(lod);
            }
        }

        for coord in deferred_snapshot_coords {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }

        for job in frame_jobs.into_values() {
            match job.lod {
                ChunkLod::Near => near_jobs.push(job),
                ChunkLod::Mid => mid_jobs.push(job),
                ChunkLod::Far => far_jobs.push(job),
                ChunkLod::Ultra => ultra_jobs.push(job),
            }
        }

        let job_priority =
            |coord: ChunkCoord| urgent_job_priority(coord, player_chunk, chunk_priority_scores);

        let mut urgent_jobs = Vec::new();
        near_jobs.retain(|job| {
            if job.urgent {
                urgent_jobs.push(job.clone());
                return false;
            }
            true
        });
        mid_jobs.retain(|job| {
            if job.urgent {
                urgent_jobs.push(job.clone());
                return false;
            }
            true
        });
        far_jobs.retain(|job| {
            if job.urgent {
                urgent_jobs.push(job.clone());
                return false;
            }
            true
        });
        ultra_jobs.retain(|job| {
            if job.urgent {
                urgent_jobs.push(job.clone());
                return false;
            }
            true
        });

        urgent_jobs.sort_by(|a, b| {
            job_priority(a.coord)
                .total_cmp(&job_priority(b.coord))
                .then_with(|| {
                    chunk_chebyshev_dist(player_chunk, b.coord)
                        .cmp(&chunk_chebyshev_dist(player_chunk, a.coord))
                })
                .then_with(|| a.coord.x.cmp(&b.coord.x))
                .then_with(|| a.coord.y.cmp(&b.coord.y))
                .then_with(|| a.coord.z.cmp(&b.coord.z))
        });
        while let Some(job) = urgent_jobs.pop() {
            if self.inflight_mesh_chunks.contains(&job.coord) {
                self.enqueue_urgent_mesh_chunk(job.coord);
                continue;
            }
            match self.submit_mesh_job(job, &mut stats) {
                Ok(()) => {}
                Err(TrySendError::Full(job)) => {
                    self.enqueue_urgent_mesh_chunk(job.coord);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => break,
            }
        }
        near_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        mid_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        far_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));
        ultra_jobs.sort_by(|a, b| job_priority(a.coord).total_cmp(&job_priority(b.coord)));

        let far_pressure = self.dirty_queues.total_len() + self.mesh_queue.inflight;
        let far_scale = if far_pressure > 4096 {
            4
        } else if far_pressure > 1024 {
            2
        } else {
            1
        };
        let adjusted_lod_budgets = LodMeshingBudgets {
            near: lod_budgets.near,
            mid: lod_budgets.mid,
            far: (lod_budgets.far / far_scale).max(usize::from(!far_jobs.is_empty())),
            ultra: (lod_budgets.ultra / far_scale)
                .max(usize::from(!ultra_jobs.is_empty() && far_scale == 1)),
        };

        let sustained_pressure = far_pressure > 2048;
        if sustained_pressure {
            let far_keep = adjusted_lod_budgets.far.min(2);
            if far_jobs.len() > far_keep {
                stats.pressure_drop_count += far_jobs.len() - far_keep;
                far_jobs.truncate(far_keep);
            }
            if !ultra_jobs.is_empty() {
                stats.pressure_drop_count += ultra_jobs.len();
                ultra_jobs.clear();
            }
        }

        let pending_counts = [
            near_jobs.len(),
            mid_jobs.len(),
            far_jobs.len(),
            ultra_jobs.len(),
        ];
        let mut effective_budgets = compute_effective_lod_submission_budgets(
            mesh_budget,
            adjusted_lod_budgets,
            pending_counts,
        );

        let mut submitted = 0usize;
        let mut stalled_tiers = [false; 4];
        let mut tier_cursor = 0usize;
        let tier_order = [
            SubmissionTier::Near,
            SubmissionTier::Mid,
            SubmissionTier::Far,
            SubmissionTier::Ultra,
        ];
        while submitted < mesh_budget {
            let mut attempted_any = false;
            let mut progress_made = false;

            for offset in 0..tier_order.len() {
                let tier_idx = (tier_cursor + offset) % tier_order.len();
                if effective_budgets[tier_idx] == 0 || stalled_tiers[tier_idx] {
                    continue;
                }

                attempted_any = true;
                let jobs = match tier_order[tier_idx] {
                    SubmissionTier::Near => &mut near_jobs,
                    SubmissionTier::Mid => &mut mid_jobs,
                    SubmissionTier::Far => &mut far_jobs,
                    SubmissionTier::Ultra => &mut ultra_jobs,
                };

                let Some(job) = jobs.pop() else {
                    effective_budgets[tier_idx] = 0;
                    continue;
                };

                if self.inflight_mesh_chunks.contains(&job.coord) {
                    self.enqueue_dirty_chunk(job.coord, player_chunk, chunk_priority_scores);
                    effective_budgets[tier_idx] = effective_budgets[tier_idx].saturating_sub(1);
                    progress_made = true;
                    tier_cursor = (tier_idx + 1) % tier_order.len();
                    break;
                }

                match self.submit_mesh_job(job.clone(), &mut stats) {
                    Ok(()) => {
                        self.inflight_mesh_chunks.insert(job.coord);
                        effective_budgets[tier_idx] = effective_budgets[tier_idx].saturating_sub(1);
                        submitted += 1;
                        progress_made = true;
                        tier_cursor = (tier_idx + 1) % tier_order.len();
                        break;
                    }
                    Err(TrySendError::Full(job)) => {
                        jobs.push(job);
                        stalled_tiers[tier_idx] = true;
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        effective_budgets.fill(0);
                        break;
                    }
                }
            }

            if !attempted_any || !progress_made {
                break;
            }
        }

        for job in near_jobs
            .into_iter()
            .chain(mid_jobs)
            .chain(far_jobs)
            .chain(ultra_jobs)
        {
            if store.chunk_voxel_version(job.coord)
                > *self.mesh_versions.get(&job.coord).unwrap_or(&0)
            {
                self.enqueue_dirty_chunk(job.coord, player_chunk, chunk_priority_scores);
            }
        }
        stats.dirty_queue_drop_count +=
            self.enforce_dirty_queue_bound(player_chunk, chunk_priority_scores);

        #[cfg(feature = "gpu-compute")]
        if matches!(self.mesh_backend, MeshPipelineBackend::Gpu) {
            let dispatch_budget =
                std::time::Duration::from_secs_f32(GPU_RENDER_DISPATCH_MAX_TIME_BUDGET_MS / 1000.0);
            match dispatch_gpu_chunk_tasks_on_renderer(
                GPU_RENDER_DISPATCH_MAX_TASKS_PER_FRAME,
                dispatch_budget,
            ) {
                Ok(dispatch_stats) => {
                    stats.gpu_dispatch_enqueue_submit_ms += dispatch_stats.enqueue_submit_ms;
                    stats.gpu_dispatch_wait_sync_ms += dispatch_stats.wait_sync_ms;
                    stats.gpu_dispatch_tasks_submitted += dispatch_stats.tasks_submitted;
                    stats.gpu_dispatch_ms +=
                        dispatch_stats.enqueue_submit_ms + dispatch_stats.wait_sync_ms;
                }
                Err(err) => {
                    log::warn!(
                        "[mesh] renderer-side gpu dispatch failed before mesh adoption: {err:#}"
                    );
                }
            }
        }

        let mut pending_promoted_to_drawable = 0usize;
        let mut pending_superseded = 0usize;
        let mut pending_rejected = 0usize;

        while let Ok(result) = self.mesh_queue.try_recv() {
            log::trace!("[renderer] received mesh result chunk={:?}", result.coord);
            self.inflight_mesh_chunks.remove(&result.coord);
            self.mesh_lifecycle
                .insert(result.coord, MeshLifecycleState::ResultReady);
            self.completed_meshes.push(result);
        }

        #[cfg(feature = "gpu-compute")]
        if matches!(self.mesh_backend, MeshPipelineBackend::Gpu) {
            for ready in take_ready_gpu_mesh_results_on_renderer() {
                self.finalize_pending_gpu_result(ready);
            }
        }

        self.completed_meshes.sort_by(|a, b| {
            job_priority(b.coord)
                .total_cmp(&job_priority(a.coord))
                .then_with(|| {
                    let ad = chunk_chebyshev_dist(player_chunk, a.coord);
                    let bd = chunk_chebyshev_dist(player_chunk, b.coord);
                    ad.cmp(&bd)
                })
        });

        if self.completed_meshes.len() > COMPLETED_MESH_BACKLOG_THRESHOLD {
            let mut low_priority_indices: Vec<usize> = self
                .completed_meshes
                .iter()
                .enumerate()
                .filter_map(|(idx, result)| {
                    (matches!(result.lod, ChunkLod::Far | ChunkLod::Ultra) && !result.urgent)
                        .then_some(idx)
                })
                .collect();
            low_priority_indices.sort_by_key(|idx| self.completed_meshes[*idx].queued_at);

            let mut to_drop = self.completed_meshes.len() - COMPLETED_MESH_BACKLOG_THRESHOLD;
            let mut drop_mask = vec![false; self.completed_meshes.len()];
            for idx in low_priority_indices {
                if to_drop == 0 {
                    break;
                }
                drop_mask[idx] = true;
                to_drop -= 1;
            }

            if drop_mask.iter().any(|drop| *drop) {
                let mut kept = Vec::with_capacity(self.completed_meshes.len());
                for (idx, result) in self.completed_meshes.drain(..).enumerate() {
                    if drop_mask[idx] {
                        stats.age_drop_count += 1;
                    } else {
                        kept.push(result);
                    }
                }
                self.completed_meshes = kept;
            }
        }

        let mut bytes_uploaded = 0usize;
        let mut uploaded = 0usize;
        let mut total_latency_ms = 0.0f32;
        let mut gpu_adoption_latency_ms_total = 0.0f32;
        let mut remesh_coords = Vec::new();
        let mut failed_retry_coords = Vec::new();
        let mut skipped_retry_chunks = Vec::new();
        let completed_meshes_depth = self.completed_meshes.len();
        let completed_results: Vec<MeshResult> = self.completed_meshes.drain(..).collect();
        for (completed_index, result) in completed_results.into_iter().enumerate() {
            if let ChunkMeshArtifact::GpuPending { .. } = &result.artifact {
                stats.mesh_pending_finalize += 1;
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::AwaitingFinalize);
                self.pending_gpu_results.insert(
                    (result.coord, result.version),
                    PendingGpuMeshResult {
                        result,
                        first_seen_frame: self.mesh_rebuild_frame_index,
                        first_seen_completed_index: completed_index,
                    },
                );
                continue;
            }

            stats.mesh_artifacts_received += 1;
            let backend = Self::mesh_result_backend_label(&result);
            let (index_count, vertex_count) = Self::mesh_result_index_vertex_counts(&result);
            let desired = chunk_priority_scores.contains_key(&result.coord)
                || self.visible_gpu_chunks.contains_key(&result.coord);
            let had_prior_mesh = self.visible_gpu_chunks.contains_key(&result.coord);
            let pending_replaced = {
                let previous_len = self.pending_gpu_results.len();
                self.pending_gpu_results
                    .retain(|(coord, _), _| *coord != result.coord);
                previous_len != self.pending_gpu_results.len()
            };
            if pending_replaced {
                pending_superseded += 1;
            }
            let voxel_version = store.chunk_voxel_version(result.coord);
            log::debug!(
                "[mesh-flow] recv chunk={:?} backend={} lod={:?} version={} store_version={} desired={} prior_mesh={} index_count={} vertex_count={}",
                result.coord,
                backend,
                result.lod,
                result.version,
                voxel_version,
                desired,
                had_prior_mesh,
                index_count,
                vertex_count,
            );
            if !desired {
                stats.mesh_artifacts_rejected += 1;
                stats.mesh_reject_no_longer_desired += 1;
                if pending_replaced {
                    pending_rejected += 1;
                }
                if had_prior_mesh {
                    stats.mesh_last_good_retained += 1;
                }
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::Rejected);
                continue;
            }
            // Never upload stale geometry; schedule a retry and skip this artifact.
            if let Some(retry_policy) = stale_artifact_retry_policy(
                result.version,
                voxel_version,
                result.lod,
                result.urgent,
            ) {
                stats.stale_drop_count += 1;
                stats.mesh_artifacts_rejected += 1;
                stats.mesh_reject_stale += 1;
                if pending_replaced {
                    pending_rejected += 1;
                }
                stats.stale_drop_retry_enqueued += 1;
                match retry_policy {
                    StaleArtifactRetryPolicy::Urgent => {
                        self.enqueue_urgent_mesh_chunk(result.coord)
                    }
                    StaleArtifactRetryPolicy::Dirty => remesh_coords.push(result.coord),
                }
                continue;
            }

            // FIX 1: `Skipped` means "leave current mesh untouched"; never evict cache entries.
            if let ChunkMeshArtifact::Skipped { reason } = &result.artifact {
                stats.gpu_job_skipped += 1;
                if pending_replaced {
                    pending_rejected += 1;
                }

                log::warn!(
                    "[mesh] skipped chunk={:?} reason={:?}",
                    result.coord,
                    reason
                );

                if matches!(reason, MeshSkipReason::MeshSlotCapacitySaturated { .. }) {
                    stats.gpu_mesh_slot_alloc_failed += 1;
                    stats.mesh_slot_allocation_failures += 1;
                    Self::record_rebuild_outcome(
                        &mut stats,
                        RebuildOutcome::SkippedNoArtifactCapacity,
                    );
                    self.sampled_outcome_trace(
                        &result,
                        RebuildOutcome::SkippedNoArtifactCapacity,
                        None,
                        None,
                        None,
                        None,
                    );
                } else {
                    let outcome = match reason {
                        MeshSkipReason::ZeroGeometry => RebuildOutcome::SkippedZeroGeometry,
                        MeshSkipReason::StartupZeroGeometry => {
                            RebuildOutcome::SkippedStartupZeroGeometry
                        }
                        MeshSkipReason::InvalidPageMapping => {
                            RebuildOutcome::SkippedInvalidPageMapping
                        }
                        MeshSkipReason::MissingVoxelState => {
                            RebuildOutcome::SkippedMissingVoxelState
                        }
                        MeshSkipReason::AdoptionRejected => RebuildOutcome::SkippedAdoptionRejected,
                        MeshSkipReason::SparseIndirectUndrawable => {
                            RebuildOutcome::SkippedSparseIndirectUndrawable
                        }
                        MeshSkipReason::BackendContractMismatch => {
                            RebuildOutcome::SkippedBackendContractMismatch
                        }
                        _ => RebuildOutcome::SkippedAdoptionRejected,
                    };
                    Self::record_rebuild_outcome(&mut stats, outcome);
                    self.sampled_outcome_trace(&result, outcome, None, None, None, None);
                }

                if had_prior_mesh {
                    stats.mesh_last_good_retained += 1;
                }
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::Rejected);
                skipped_retry_chunks.push((result.coord, *reason));
                continue;
            }

            if let ChunkMeshArtifact::Failed { reason } = &result.artifact {
                stats.gpu_job_failures += 1;
                if pending_replaced {
                    pending_rejected += 1;
                }
                if is_gpu_timeout_reason(reason) {
                    stats.gpu_job_timeouts += 1;
                }
                log::warn!(
                    "[mesh] dropping failed artifact chunk={:?} lod={:?} error={}",
                    result.coord,
                    result.lod,
                    short_error_message(reason)
                );
                stats.mesh_artifacts_rejected += 1;
                stats.mesh_reject_failed += 1;
                if had_prior_mesh {
                    stats.mesh_last_good_retained += 1;
                }
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::Rejected);
                failed_retry_coords.push(result.coord);
                continue;
            }
            if let ChunkMeshArtifact::GpuPending { .. } = &result.artifact {
                stats.mesh_pending_finalize += 1;
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::AwaitingFinalize);
                self.pending_gpu_results.insert(
                    (result.coord, result.version),
                    PendingGpuMeshResult {
                        result,
                        first_seen_frame: self.mesh_rebuild_frame_index,
                        first_seen_completed_index: completed_index,
                    },
                );
                continue;
            }

            if let ChunkMeshArtifact::GpuReady {
                page_index,
                draw_indirect_index,
                lod,
                dispatch_ms,
                aabb_min,
                aabb_max,
                chunk_origin_world,
                ..
            } = &result.artifact
            {
                if page_index.0 >= gpu_page_capacity() {
                    log::warn!(
                        "[mesh] rejecting gpu artifact chunk={:?}: page index {} out of range (capacity {})",
                        result.coord,
                        page_index.0,
                        gpu_page_capacity()
                    );
                    stats.mesh_artifacts_rejected += 1;
                    stats.mesh_reject_invalid_page += 1;
                    if pending_replaced {
                        pending_rejected += 1;
                    }
                    self.mesh_lifecycle
                        .insert(result.coord, MeshLifecycleState::Rejected);
                    Self::record_rebuild_outcome(
                        &mut stats,
                        RebuildOutcome::SkippedInvalidPageMapping,
                    );
                    self.sampled_outcome_trace(
                        &result,
                        RebuildOutcome::SkippedInvalidPageMapping,
                        Some(*page_index),
                        Some(*draw_indirect_index),
                        None,
                        None,
                    );
                    remesh_coords.push(result.coord);
                    continue;
                }

                let resolved_index_count = index_count.max(1);

                let adoption_latency_ms = result.queued_at.elapsed().as_secs_f32() * 1000.0;

                stats.gpu_mesh_jobs += 1;
                stats.gpu_dispatch_ms += *dispatch_ms;
                stats.gpu_mesh_adopted_count += 1;

                gpu_adoption_latency_ms_total += adoption_latency_ms;

                let slot = *draw_indirect_index;

                let adopted = self.adopt_visible_chunk_draw(
                    result.coord,
                    GpuChunkDraw {
                        page_index: *page_index,
                        draw_indirect_index: slot,
                        lod: *lod,
                        origin: *chunk_origin_world,
                        world_aabb_min: *aabb_min,
                        world_aabb_max: *aabb_max,
                        draw_source: DrawSource::GpuArtifact,
                        index_count: Some(resolved_index_count),
                    },
                );
                if !adopted {
                    stats.mesh_artifacts_rejected += 1;
                    stats.mesh_reject_unhandled += 1;
                    self.mesh_lifecycle
                        .insert(result.coord, MeshLifecycleState::Rejected);
                    skipped_retry_chunks.push((result.coord, MeshSkipReason::AdoptionRejected));
                    continue;
                }

                self.mesh_versions.insert(result.coord, result.version);
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::Drawable);
                self.mesh_retry_state.remove(&result.coord);
                if result.from_pending_finalize {
                    pending_promoted_to_drawable += 1;
                }
                store.mark_chunk_meshed(result.coord);
                Self::record_rebuild_outcome(&mut stats, RebuildOutcome::GpuAdopted);
                self.sampled_outcome_trace(
                    &result,
                    RebuildOutcome::GpuAdopted,
                    Some(*page_index),
                    Some(slot),
                    Some(resolved_index_count),
                    Some("ready"),
                );

                total_latency_ms += adoption_latency_ms;

                continue;
            }
            if let ChunkMeshArtifact::Cpu {
                verts,
                inds,
                indirect,
                aabb_min,
                aabb_max,
                chunk_origin_world,
            } = &result.artifact
            {
                let Some(slot) = self.free_mesh_slots.pop() else {
                    stats.mesh_artifacts_rejected += 1;
                    remesh_coords.push(result.coord);
                    continue;
                };

                let vertex_offset = slot as u64
                    * GPU_MESH_VERTEX_CAPACITY_PER_SLOT
                    * std::mem::size_of::<Vertex>() as u64;

                let index_offset = slot as u64
                    * GPU_MESH_INDEX_CAPACITY_PER_SLOT
                    * std::mem::size_of::<u32>() as u64;

                let draw_offset =
                    slot as u64 * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64;

                if !verts.is_empty() {
                    self.queue.write_buffer(
                        &self.global_gpu_vertex_buffer,
                        vertex_offset,
                        bytemuck::cast_slice(verts),
                    );
                }

                if !inds.is_empty() {
                    self.queue.write_buffer(
                        &self.global_gpu_index_buffer,
                        index_offset,
                        bytemuck::cast_slice(inds),
                    );
                }

                let resolved_index_count = indirect.index_count.max(inds.len() as u32);

                let draw_command = DrawIndexedIndirectCommand {
                    index_count: resolved_index_count,
                    instance_count: 1,
                    first_index: (index_offset / std::mem::size_of::<u32>() as u64) as u32,
                    base_vertex: (vertex_offset / std::mem::size_of::<Vertex>() as u64) as i32,
                    first_instance: 0,
                };

                self.queue.write_buffer(
                    &self.global_gpu_draw_indirect_buffer,
                    draw_offset,
                    bytemuck::bytes_of(&draw_command),
                );

                let adopted = self.adopt_visible_chunk_draw(
                    result.coord,
                    GpuChunkDraw {
                        page_index: GpuPageIndex(slot),
                        draw_indirect_index: slot,
                        lod: result.lod as u8,
                        origin: *chunk_origin_world,
                        world_aabb_min: *aabb_min,
                        world_aabb_max: *aabb_max,
                        draw_source: DrawSource::CpuUploaded,
                        index_count: Some(resolved_index_count),
                    },
                );
                if !adopted {
                    self.free_mesh_slots.push(slot);
                    stats.mesh_artifacts_rejected += 1;
                    stats.mesh_reject_unhandled += 1;
                    self.mesh_lifecycle
                        .insert(result.coord, MeshLifecycleState::Rejected);
                    skipped_retry_chunks.push((result.coord, MeshSkipReason::AdoptionRejected));
                    continue;
                }

                self.mesh_versions.insert(result.coord, result.version);
                self.mesh_lifecycle
                    .insert(result.coord, MeshLifecycleState::Drawable);
                self.mesh_retry_state.remove(&result.coord);
                store.mark_chunk_meshed(result.coord);
                Self::record_rebuild_outcome(&mut stats, RebuildOutcome::CpuUploaded);
                self.sampled_outcome_trace(
                    &result,
                    RebuildOutcome::CpuUploaded,
                    Some(GpuPageIndex(slot)),
                    Some(slot),
                    Some(resolved_index_count),
                    Some("pending"),
                );

                uploaded += 1;
                bytes_uploaded += verts.len() * std::mem::size_of::<Vertex>()
                    + inds.len() * std::mem::size_of::<u32>();

                total_latency_ms += result.queued_at.elapsed().as_secs_f32() * 1000.0;

                continue;
            }
            log::warn!(
                "[mesh] rejecting unhandled artifact variant for chunk={:?}",
                result.coord
            );
            stats.mesh_artifacts_rejected += 1;
            stats.mesh_reject_unhandled += 1;
            if pending_replaced {
                pending_rejected += 1;
            }
            stats.mesh_dropped_before_drawable += 1;
            if had_prior_mesh {
                stats.mesh_last_good_retained += 1;
            }
            self.mesh_lifecycle
                .insert(result.coord, MeshLifecycleState::Rejected);
            self.pending_lod_remesh.remove(&result.coord);
            self.pending_lod_remesh_since.remove(&result.coord);
            continue;
        }

        for coord in failed_retry_coords {
            self.schedule_mesh_retry(coord, MeshRetryKind::Failed);
        }
        for (coord, reason) in skipped_retry_chunks {
            self.schedule_mesh_retry(coord, MeshRetryKind::Skipped(reason));
        }
        for coord in remesh_coords {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }

        stats.upload_count = uploaded;
        stats.upload_bytes = bytes_uploaded;
        stats.upload_latency_ms = if uploaded > 0 {
            total_latency_ms / uploaded as f32
        } else {
            0.0
        };
        stats.gpu_mesh_adoption_latency_ms = if stats.gpu_mesh_adopted_count > 0 {
            gpu_adoption_latency_ms_total / stats.gpu_mesh_adopted_count as f32
        } else {
            0.0
        };
        stats.allocator_bytes_allocated = 0;
        stats.allocator_bytes_reused = 0;
        stats.allocator_realloc_count = 0;
        stats.dirty_backlog = self.dirty_queues.total_len();
        stats.dirty_urgent_depth = self.dirty_queues.tier_len(DirtyTier::Urgent);
        stats.dirty_near_depth = self.dirty_queues.tier_len(DirtyTier::Near);
        stats.dirty_normal_depth = self.dirty_queues.tier_len(DirtyTier::Normal);
        stats.dirty_far_depth = self.dirty_queues.tier_len(DirtyTier::Far);
        stats.meshing_queue_depth = self.dirty_queues.total_len() + self.mesh_queue.inflight;
        stats.meshing_completed_depth = completed_meshes_depth;

        let mut drop_keys = Vec::new();

        for &coord in self.visible_gpu_chunks.keys() {
            if chunk_distance(player_chunk, coord) > lod_radii.ultra as f32 + 8.0 {
                drop_keys.push(coord);
            }
        }

        for coord in drop_keys {
            self.release_draw_slot_mapping(coord);
        }
        stats.mesh_cache_entries = self.visible_gpu_chunks.len();
        stats.gpu_mesh_visible_count = self.visible_gpu_chunks.len();
        stats.startup_seed_zero_count_seen = self
            .startup_mesh_seed_state
            .values()
            .filter(|state| state.startup_seed_zero_count_seen)
            .count();
        stats.startup_seed_recovered_nonzero = self
            .startup_mesh_seed_state
            .values()
            .filter(|state| state.startup_seed_recovered_nonzero)
            .count();
        let mut oldest_pending: Option<(u64, usize, ChunkCoord)> = None;
        for ((coord, _), pending) in &self.pending_gpu_results {
            let age_frames = self
                .mesh_rebuild_frame_index
                .saturating_sub(pending.first_seen_frame);
            stats.mesh_waiting_on_fence += 1;
            oldest_pending = match oldest_pending {
                Some((old_age, old_idx, old_coord)) if old_age > age_frames => {
                    Some((old_age, old_idx, old_coord))
                }
                Some((old_age, old_idx, old_coord))
                    if old_age == age_frames && old_idx <= pending.first_seen_completed_index =>
                {
                    Some((old_age, old_idx, old_coord))
                }
                _ => Some((age_frames, pending.first_seen_completed_index, *coord)),
            };
        }
        if let Some((age_frames, first_seen_completed_index, coord)) = oldest_pending {
            log::trace!(
                "[mesh-flow] pending oldest chunk={:?} age_frames={} first_seen_completed_index={}",
                coord,
                age_frames,
                first_seen_completed_index
            );
        }
        stats.mesh_pending_total = self.pending_gpu_results.len();
        stats.mesh_pending_finalize = stats.mesh_pending_total;
        stats.mesh_pending_promoted_to_drawable = pending_promoted_to_drawable;
        stats.mesh_pending_superseded = pending_superseded;
        stats.mesh_pending_rejected = pending_rejected;
        stats.drawable_resident_total = self.visible_gpu_chunks.len();
        stats.newly_drawable_this_frame = pending_promoted_to_drawable;
        stats.pending_finalize_total = self.pending_gpu_results.len();
        stats.waiting_on_fence_total = stats.mesh_waiting_on_fence;
        stats.terminal_superseded_total = self.terminal_superseded_total;
        stats.terminal_evicted_total = self.terminal_evicted_total;

        stats.flow_received = stats.mesh_artifacts_received;
        stats.flow_adopted = stats.gpu_mesh_adopted_count;
        stats.flow_uploaded = stats.upload_count;
        stats.flow_rejected = stats.mesh_artifacts_rejected;
        stats.mesh_visible_logical_not_drawable = self
            .mesh_lifecycle
            .iter()
            .filter(|(coord, state)| {
                matches!(state, MeshLifecycleState::AwaitingFinalize)
                    && !self.visible_gpu_chunks.contains_key(coord)
            })
            .count();
        for draw in self.visible_gpu_chunks.values() {
            match draw.draw_source {
                DrawSource::GpuArtifact => stats.resident_gpu_artifact += 1,
                DrawSource::CpuUploaded => stats.resident_cpu_uploaded += 1,
                DrawSource::StaleCached => stats.resident_stale_cached += 1,
                DrawSource::Fallback => stats.resident_fallback += 1,
                DrawSource::Unknown => stats.resident_unknown += 1,
            }
        }
        if self.visible_slots.is_empty() {
            stats.gpu_mesh_visible_slot_min = -1;
            stats.gpu_mesh_visible_slot_max = -1;
            stats.gpu_mesh_visible_slot_holes = 0;
        } else {
            let mut min_slot = u32::MAX;
            let mut max_slot = 0u32;
            for slot in self.visible_slots.keys().copied() {
                min_slot = min_slot.min(slot);
                max_slot = max_slot.max(slot);
            }
            let occupied = self.visible_slots.len();
            let span = (max_slot - min_slot + 1) as usize;
            stats.gpu_mesh_visible_slot_min = min_slot as i32;
            stats.gpu_mesh_visible_slot_max = max_slot as i32;
            stats.gpu_mesh_visible_slot_holes = span.saturating_sub(occupied);
        }

        let tracked_coords: Vec<ChunkCoord> = self.visible_gpu_chunks.keys().copied().collect();
        let mut lod_changes = Vec::new();
        for coord in tracked_coords {
            let prev = self.lod_selection.get(&coord).copied();
            let lod = select_lod(coord, player_chunk, lod_radii, prev);
            if prev != Some(lod) {
                lod_changes.push(coord);
            }
            self.lod_selection.insert(coord, lod);
        }
        lod_changes.sort_by(|a, b| {
            let ad = chunk_distance(*a, player_chunk);
            let bd = chunk_distance(*b, player_chunk);
            ad.total_cmp(&bd)
        });
        for coord in lod_changes.into_iter().take(MAX_LOD_REMESH_PER_FRAME) {
            self.enqueue_lod_remesh(coord, player_chunk, chunk_priority_scores);
        }
        stats
    }

    pub fn clear_mesh_cache(&mut self) {
        self.visible_gpu_chunks.clear();
        self.visible_slots.clear();
        self.free_mesh_slots.clear();
        self.free_mesh_slots.extend(0..mesh_pool_slot_capacity());
        self.dirty_queues.clear();
        self.urgent_mesh_queue.clear();
        self.urgent_mesh_set.clear();
        self.dirty_near_starve_frames = 0;
        self.dirty_far_starve_frames = 0;
        self.dirty_fair_cursor = 0;
        self.completed_meshes.clear();
        self.outcome_trace_window_start = Instant::now();
        self.outcome_trace_emitted = 0;
        self.startup_seed_trace_window_start = Instant::now();
        self.startup_seed_trace_emitted = 0;
        self.startup_epoch = Instant::now();
        self.mesh_versions.clear();
        self.lod_selection.clear();
        self.pending_lod_remesh.clear();
        self.pending_lod_remesh_since.clear();
        self.inflight_mesh_chunks.clear();
        self.pending_gpu_results.clear();
        self.terminal_superseded_total = 0;
        self.terminal_evicted_total = 0;
        self.mesh_lifecycle.clear();
        self.mesh_retry_state.clear();
        self.startup_mesh_seed_state.clear();
        self.mesh_rebuild_frame_index = 0;
    }
    pub fn mesh_draw_stats(&self, camera: &Camera) -> MeshDrawStats {
        // Keep frustum checks in world space; use GPU mesh metadata.
        let vp_world = camera.view_proj();
        let mut stats = MeshDrawStats::default();
        for (&coord, draw) in &self.visible_gpu_chunks {
            if !self.draw_meets_resident_invariant(coord, draw) {
                continue;
            }
            if aabb_in_view(vp_world, draw.world_aabb_min, draw.world_aabb_max) {
                stats.record_draw(draw.draw_source, draw.index_count.unwrap_or(0) as u64);
            }
        }
        stats
    }

    pub fn render_world<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        self.device.poll(wgpu::Maintain::Poll);
        #[cfg(feature = "gpu-compute")]
        update_gpu_page_fences_on_renderer();
        let vp_render = camera.view_proj_rebased_to_origin(self.origin_voxel);
        // World-space culling uses world-space camera/AABBs.
        // Draw uses rebased camera VP + shader world-origin subtraction.
        let vp_world = camera.view_proj();
        let world_camera_pos = camera_world_position(camera);
        let origin_offset_world = voxel_to_world(self.origin_voxel);

        debug_assert!(camera.pos.is_finite());
        debug_assert!(origin_offset_world.is_finite());

        self.queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&CameraUniform {
                vp: vp_render.to_cols_array_2d(),
                world_origin_offset: origin_offset_world.to_array(),
                _pad: 0.0,
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);

        let visibility = DrawVisibilityInput {
            frustum_culling: self.settings.frustum_culling,
            vp_world,
            world_camera_pos,
            screen_h: self.size.height,
        };

        let mut drawable_chunks: Vec<(ChunkCoord, GpuChunkDraw)> = self
            .visible_gpu_chunks
            .iter()
            .filter_map(|(&coord, draw)| {
                if self.should_render_draw(coord, draw, visibility) {
                    Some((coord, *draw))
                } else {
                    None
                }
            })
            .collect();
        drawable_chunks.sort_by_key(|(coord, _)| (coord.x, coord.y, coord.z));

        if !drawable_chunks.is_empty() {
            pass.set_vertex_buffer(0, self.global_gpu_vertex_buffer.slice(..));
            pass.set_index_buffer(
                self.global_gpu_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            let stride = std::mem::size_of::<DrawIndexedIndirectCommand>() as u64;
            let mut draw_stats = MeshDrawStats::default();
            let mut drawn_slots = HashSet::with_capacity(drawable_chunks.len());
            for (_, draw) in drawable_chunks {
                if !drawn_slots.insert(draw.draw_indirect_index) {
                    continue;
                }
                draw_stats.record_draw(draw.draw_source, draw.index_count.unwrap_or(0) as u64);
                pass.draw_indexed_indirect(
                    &self.global_gpu_draw_indirect_buffer,
                    draw.draw_indirect_index as u64 * stride,
                );
            }
            let _ = draw_stats;
        }
    }
}

impl Renderer {
    #[cfg(feature = "gpu-compute")]
    fn finalize_pending_gpu_result(&mut self, ready: ReadyGpuMeshFinalizeEvent) {
        let key = (ready.result.coord, ready.result.version);
        let Some(mut pending) = self.pending_gpu_results.remove(&key) else {
            return;
        };

        match ready.status {
            ReadyGpuMeshFinalizeStatus::ReadyAndValid => {
                pending.result.from_pending_finalize = true;
                pending.result.artifact = ChunkMeshArtifact::GpuReady {
                    page_index: ready.result.page_index,
                    draw_indirect_index: ready.result.draw_indirect_index,
                    lod: ready.result.lod,
                    index_count: 1,
                    aabb_min: ready.result.aabb_min,
                    aabb_max: ready.result.aabb_max,
                    chunk_origin_world: ready.result.chunk_origin_world,
                    dispatch_ms: 0.0,
                };
                self.completed_meshes.push(pending.result);
            }
            ReadyGpuMeshFinalizeStatus::DroppedStaleVersion => {
                self.mesh_lifecycle
                    .insert(ready.result.coord, MeshLifecycleState::Superseded);
                self.terminal_superseded_total += 1;
            }
            ReadyGpuMeshFinalizeStatus::DroppedInvalidMapping => {
                self.mesh_lifecycle
                    .insert(ready.result.coord, MeshLifecycleState::Rejected);
            }
        }
    }

    fn release_draw_slot_mapping(&mut self, coord: ChunkCoord) {
        if let Some(old) = self.visible_gpu_chunks.remove(&coord) {
            if self.visible_slots.get(&old.draw_indirect_index) == Some(&coord) {
                self.visible_slots.remove(&old.draw_indirect_index);
            }
            if !matches!(old.draw_source, DrawSource::GpuArtifact) {
                self.free_mesh_slots.push(old.draw_indirect_index);
            }
            self.mesh_lifecycle
                .insert(coord, MeshLifecycleState::Evicted);
            self.terminal_evicted_total += 1;
        }
    }

    fn adopt_visible_chunk_draw(&mut self, coord: ChunkCoord, draw: GpuChunkDraw) -> bool {
        if !self.draw_meets_resident_invariant(coord, &draw) {
            return false;
        }
        self.release_draw_slot_mapping(coord);
        if let Some(previous_coord) = self.visible_slots.insert(draw.draw_indirect_index, coord) {
            if previous_coord != coord {
                self.release_draw_slot_mapping(previous_coord);
                self.mesh_lifecycle
                    .insert(previous_coord, MeshLifecycleState::Superseded);
                self.terminal_superseded_total += 1;
            }
        }
        self.visible_gpu_chunks.insert(coord, draw);
        self.mesh_lifecycle
            .insert(coord, MeshLifecycleState::Drawable);
        true
    }

    fn mesh_result_backend_label(result: &MeshResult) -> &'static str {
        match result.artifact {
            ChunkMeshArtifact::GpuPending { .. } | ChunkMeshArtifact::GpuReady { .. } => "gpu",
            ChunkMeshArtifact::Cpu { .. } => "cpu",
            ChunkMeshArtifact::Failed { .. } => "failed",
            ChunkMeshArtifact::Skipped { .. } => "skipped",
        }
    }

    fn mesh_result_index_vertex_counts(result: &MeshResult) -> (u32, u32) {
        match &result.artifact {
            ChunkMeshArtifact::GpuReady { index_count, .. } => (*index_count, 0),
            ChunkMeshArtifact::Cpu { inds, verts, .. } => (inds.len() as u32, verts.len() as u32),
            _ => (0, 0),
        }
    }

    fn record_rebuild_outcome(stats: &mut MeshRebuildStats, outcome: RebuildOutcome) {
        match outcome {
            RebuildOutcome::GpuAdopted => stats.outcome_gpu_adopted += 1,
            RebuildOutcome::CpuUploaded => stats.outcome_cpu_uploaded += 1,
            RebuildOutcome::SkippedZeroGeometry => stats.outcome_skipped_zero_geometry += 1,
            RebuildOutcome::SkippedStartupZeroGeometry => {
                stats.outcome_skipped_startup_zero_geometry += 1
            }
            RebuildOutcome::SkippedNoArtifactCapacity => {
                stats.outcome_skipped_no_artifact_capacity += 1
            }
            RebuildOutcome::SkippedInvalidPageMapping => {
                stats.outcome_skipped_invalid_page_mapping += 1
            }
            RebuildOutcome::SkippedMissingVoxelState => {
                stats.outcome_skipped_missing_voxel_state += 1
            }
            RebuildOutcome::SkippedAdoptionRejected => stats.outcome_skipped_adoption_rejected += 1,
            RebuildOutcome::SkippedSparseIndirectUndrawable => {
                stats.outcome_skipped_sparse_indirect_undrawable += 1
            }
            RebuildOutcome::SkippedBackendContractMismatch => {
                stats.outcome_skipped_backend_contract_mismatch += 1
            }
        }
    }

    fn sampled_outcome_trace(
        &mut self,
        result: &MeshResult,
        outcome: RebuildOutcome,
        page_index: Option<GpuPageIndex>,
        slot_index: Option<u32>,
        index_count: Option<u32>,
        cull_result: Option<&'static str>,
    ) {
        if self.outcome_trace_window_start.elapsed().as_secs_f32() >= 1.0 {
            self.outcome_trace_window_start = Instant::now();
            self.outcome_trace_emitted = 0;
        }
        if self.outcome_trace_emitted >= OUTCOME_TRACE_SAMPLES_PER_SECOND {
            return;
        }
        self.outcome_trace_emitted += 1;
        log::info!(
            "[mesh-trace] coord={:?} lod={:?} version={} page={:?} slot={:?} index_count={:?} cull={:?} outcome={:?}",
            result.coord,
            result.lod,
            result.version,
            page_index.map(|p| p.0),
            slot_index,
            index_count,
            cull_result,
            outcome
        );
    }

    fn mark_startup_seed_zero_seen(&mut self, coord: ChunkCoord) -> StartupMeshSeedState {
        let state = self.startup_mesh_seed_state.entry(coord).or_default();
        if !state.startup_seed_zero_count_seen {
            state.startup_seed_zero_count_seen = true;
            state.first_zero_at = Some(Instant::now());
        }
        *state
    }

    fn mark_startup_seed_recovered(&mut self, coord: ChunkCoord) -> Option<StartupMeshSeedState> {
        let state = self.startup_mesh_seed_state.get_mut(&coord)?;
        if state.startup_seed_zero_count_seen && !state.startup_seed_recovered_nonzero {
            state.startup_seed_recovered_nonzero = true;
            state.recovered_nonzero_at = Some(Instant::now());
            return Some(*state);
        }
        None
    }

    fn sampled_startup_seed_trace(
        &mut self,
        coord: ChunkCoord,
        page_index: GpuPageIndex,
        slot_index: u32,
        first_zero_at: Option<Instant>,
        recovered_nonzero_at: Option<Instant>,
        index_count: u32,
        event: &'static str,
    ) {
        if self.startup_seed_trace_window_start.elapsed().as_secs_f32() >= 1.0 {
            self.startup_seed_trace_window_start = Instant::now();
            self.startup_seed_trace_emitted = 0;
        }
        if self.startup_seed_trace_emitted >= OUTCOME_TRACE_SAMPLES_PER_SECOND {
            return;
        }
        self.startup_seed_trace_emitted += 1;

        let first_zero_ms = first_zero_at.map(|ts| {
            ts.saturating_duration_since(self.startup_epoch)
                .as_secs_f32()
                * 1000.0
        });
        let recovered_ms = recovered_nonzero_at.map(|ts| {
            ts.saturating_duration_since(self.startup_epoch)
                .as_secs_f32()
                * 1000.0
        });
        log::info!(
            "[mesh-startup-seed] event={} coord={:?} page={} slot={} first_zero_ms={:?} recovered_ms={:?} index_count={}",
            event,
            coord,
            page_index.0,
            slot_index,
            first_zero_ms,
            recovered_ms,
            index_count
        );
    }

    fn should_replace_frame_job(candidate: &MeshJob, current: &MeshJob) -> bool {
        if candidate.urgent != current.urgent {
            return candidate.urgent;
        }
        lod_rank(candidate.lod) < lod_rank(current.lod)
    }

    fn submit_mesh_job(
        &mut self,
        job: MeshJob,
        stats: &mut MeshRebuildStats,
    ) -> Result<(), TrySendError<MeshJob>> {
        let coord = job.coord;
        let lod = job.lod;
        match self.mesh_queue.try_submit(job) {
            Ok(()) => {
                self.inflight_mesh_chunks.insert(coord);
                self.mesh_lifecycle
                    .insert(coord, MeshLifecycleState::Meshing);
                stats.mesh_count += 1;
                match lod {
                    ChunkLod::Near => stats.near_mesh_count += 1,
                    ChunkLod::Mid => stats.mid_mesh_count += 1,
                    ChunkLod::Far => stats.far_mesh_count += 1,
                    ChunkLod::Ultra => stats.ultra_mesh_count += 1,
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn enqueue_frame_job(frame_jobs: &mut HashMap<ChunkCoord, MeshJob>, job: MeshJob) {
        match frame_jobs.get(&job.coord) {
            Some(current) if !Self::should_replace_frame_job(&job, current) => {}
            _ => {
                frame_jobs.insert(job.coord, job);
            }
        }
    }

    fn enqueue_urgent_mesh_chunk(&mut self, coord: ChunkCoord) {
        if self.urgent_mesh_set.insert(coord) {
            self.urgent_mesh_queue.push_back(coord);
        }
        self.mesh_lifecycle
            .insert(coord, MeshLifecycleState::Requested);
        self.pending_lod_remesh.remove(&coord);
        self.pending_lod_remesh_since.remove(&coord);
        self.dirty_queues.remove_coord(coord);
    }

    fn pop_urgent_mesh_jobs(
        &mut self,
        store: &ChunkStore,
        stats: &mut MeshRebuildStats,
        player_chunk: ChunkCoord,
        lod_radii: LodRadii,
    ) -> Vec<MeshJob> {
        let mut jobs = Vec::new();
        while let Some(coord) = self.urgent_mesh_queue.pop_front() {
            self.urgent_mesh_set.remove(&coord);
            self.pending_lod_remesh.remove(&coord);
            self.pending_lod_remesh_since.remove(&coord);

            let t0 = Instant::now();
            let snapshot =
                build_chunk_snapshot(store, coord, self.settings.unknown_neighbor_policy);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            stats.total_ms += ms;
            if ms > stats.max_ms {
                stats.max_ms = ms;
            }

            let prev = self.lod_selection.get(&coord).copied();
            let primary_lod = select_lod(coord, player_chunk, lod_radii, prev);
            let fallback_lod =
                fallback_lod_near_threshold(coord, player_chunk, lod_radii, primary_lod);
            let version = store.chunk_voxel_version(coord);
            let queued_at = Instant::now();
            jobs.push(MeshJob {
                coord,
                lod: primary_lod,
                version,
                queued_at,
                snapshot: snapshot.clone(),
                greedy: self.settings.greedy_meshing,
                urgent: true,
            });
            if let Some(lod) = fallback_lod {
                jobs.push(MeshJob {
                    coord,
                    lod,
                    version,
                    queued_at,
                    snapshot: snapshot.clone(),
                    greedy: self.settings.greedy_meshing,
                    urgent: true,
                });
            }

            self.dirty_queues.remove_coord(coord);
        }
        jobs
    }

    fn classify_dirty_tier(
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> DirtyTier {
        let distance = chunk_chebyshev_dist(player_chunk, coord);
        let priority = dirty_coord_priority(coord, player_chunk, chunk_priority_scores);
        if distance <= 1 || priority >= DIRTY_VISIBLE_URGENT_SCORE {
            DirtyTier::Urgent
        } else if distance <= 4 {
            DirtyTier::Near
        } else if distance <= 12 {
            DirtyTier::Normal
        } else {
            DirtyTier::Far
        }
    }

    fn enqueue_dirty_chunk(
        &mut self,
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) {
        if self.urgent_mesh_set.contains(&coord) {
            return;
        }
        let tier = Self::classify_dirty_tier(coord, player_chunk, chunk_priority_scores);
        self.dirty_queues.queue_coord(coord, tier);
        self.mesh_lifecycle
            .insert(coord, MeshLifecycleState::Requested);
    }

    fn enqueue_lod_remesh(
        &mut self,
        coord: ChunkCoord,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) {
        if self.pending_lod_remesh.insert(coord) {
            self.pending_lod_remesh_since
                .entry(coord)
                .or_insert(self.mesh_rebuild_frame_index);
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
    }

    fn schedule_mesh_retry(&mut self, coord: ChunkCoord, kind: MeshRetryKind) {
        let state = self.mesh_retry_state.entry(coord).or_default();
        let (attempts, next_retry_frame) = match kind {
            MeshRetryKind::Failed => (
                &mut state.failed_attempts,
                &mut state.failed_next_retry_frame,
            ),
            MeshRetryKind::Skipped(_) => (
                &mut state.skipped_attempts,
                &mut state.skipped_next_retry_frame,
            ),
        };
        if *attempts >= MESH_RETRY_MAX_ATTEMPTS {
            return;
        }
        *attempts += 1;
        let backoff_frames = match kind {
            MeshRetryKind::Failed => {
                let exp = (*attempts).saturating_sub(1).min(8);
                MESH_RETRY_BASE_BACKOFF_FRAMES.saturating_mul(1u64 << exp)
            }
            MeshRetryKind::Skipped(reason) => {
                let exp = (*attempts).saturating_sub(1).min(6);
                let raw = MESH_RETRY_BASE_BACKOFF_FRAMES.saturating_mul(1u64 << exp);
                let bounded = raw.min(MESH_RETRY_SKIPPED_MAX_BACKOFF_FRAMES);
                if *attempts >= MESH_RETRY_SKIPPED_WARN_ATTEMPTS {
                    if let MeshSkipReason::MeshSlotCapacitySaturated {
                        slot_capacity,
                        in_flight_fences,
                    } = reason
                    {
                        log::warn!(
                            "[mesh] repeated skipped retries chunk={coord:?} attempts={} slot_capacity={} in_flight_fences={}",
                            *attempts,
                            slot_capacity,
                            in_flight_fences
                        );
                    }
                }
                bounded
            }
        };
        *next_retry_frame = self.mesh_rebuild_frame_index.saturating_add(backoff_frames);
    }

    fn process_mesh_retry_backoff(
        &mut self,
        player_chunk: ChunkCoord,
        chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) {
        let mut ready = Vec::new();
        let mut urgent_boost = Vec::new();
        self.mesh_retry_state.retain(|coord, state| {
            let failed_ready = state.failed_attempts < MESH_RETRY_MAX_ATTEMPTS
                && state.failed_next_retry_frame <= self.mesh_rebuild_frame_index;
            let skipped_ready = state.skipped_attempts < MESH_RETRY_MAX_ATTEMPTS
                && state.skipped_next_retry_frame <= self.mesh_rebuild_frame_index;
            if failed_ready || skipped_ready {
                ready.push(*coord);
            }
            if failed_ready {
                state.failed_attempts = 0;
                state.failed_next_retry_frame = u64::MAX;
            }
            if skipped_ready {
                let skipped_attempts = state.skipped_attempts;
                state.skipped_attempts = 0;
                state.skipped_next_retry_frame = u64::MAX;
                if skipped_attempts >= MESH_RETRY_SKIPPED_PRIORITY_BOOST_ATTEMPTS {
                    urgent_boost.push(*coord);
                }
            }

            state.failed_attempts > 0 || state.skipped_attempts > 0
        });
        for coord in urgent_boost {
            self.dirty_queues.queue_coord(coord, DirtyTier::Urgent);
        }
        for coord in ready {
            self.enqueue_dirty_chunk(coord, player_chunk, chunk_priority_scores);
        }
    }

    fn pop_priority_dirty_chunks(
        &mut self,
        count: usize,
        _player_chunk: ChunkCoord,
        _chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> Vec<ChunkCoord> {
        let mut snapshots = Vec::with_capacity(count);
        let had_near = self.dirty_queues.tier_len(DirtyTier::Near) > 0;
        let had_far = self.dirty_queues.tier_len(DirtyTier::Far) > 0;
        let mut popped_near = false;
        let mut popped_far = false;

        for _ in 0..count {
            if let Some(coord) = self.dirty_queues.pop_front_tier(DirtyTier::Urgent) {
                snapshots.push(coord);
                continue;
            }

            let force_near = self.dirty_near_starve_frames >= DIRTY_NEAR_STARVE_LIMIT_FRAMES;
            let force_far = self.dirty_far_starve_frames >= DIRTY_FAR_STARVE_LIMIT_FRAMES;

            let mut picked = None;
            if force_near {
                picked = self.dirty_queues.pop_front_tier(DirtyTier::Near);
                if picked.is_some() {
                    popped_near = true;
                }
            } else if force_far {
                picked = self.dirty_queues.pop_front_tier(DirtyTier::Far);
                if picked.is_some() {
                    popped_far = true;
                }
            }

            if picked.is_none() {
                let schedule = [
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Normal,
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Near,
                    DirtyTier::Normal,
                    DirtyTier::Far,
                ];
                let preferred = schedule[self.dirty_fair_cursor as usize % schedule.len()];
                self.dirty_fair_cursor = self.dirty_fair_cursor.wrapping_add(1);
                let order = match preferred {
                    DirtyTier::Near => [
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Far,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Normal => [
                        DirtyTier::Normal,
                        DirtyTier::Near,
                        DirtyTier::Far,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Far => [
                        DirtyTier::Far,
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Urgent,
                    ],
                    DirtyTier::Urgent => [
                        DirtyTier::Urgent,
                        DirtyTier::Near,
                        DirtyTier::Normal,
                        DirtyTier::Far,
                    ],
                };
                for tier in order {
                    picked = self.dirty_queues.pop_front_tier(tier);
                    if picked.is_some() {
                        if tier == DirtyTier::Near {
                            popped_near = true;
                        } else if tier == DirtyTier::Far {
                            popped_far = true;
                        }
                        break;
                    }
                }
            }

            let Some(coord) = picked else {
                break;
            };
            self.pending_lod_remesh.remove(&coord);
            self.pending_lod_remesh_since.remove(&coord);
            snapshots.push(coord);
        }

        self.dirty_near_starve_frames = if had_near && !popped_near {
            self.dirty_near_starve_frames.saturating_add(1)
        } else {
            0
        };
        self.dirty_far_starve_frames = if had_far && !popped_far {
            self.dirty_far_starve_frames.saturating_add(1)
        } else {
            0
        };

        snapshots
    }

    fn enforce_dirty_queue_bound(
        &mut self,
        _player_chunk: ChunkCoord,
        _chunk_priority_scores: &HashMap<ChunkCoord, f32>,
    ) -> usize {
        let mut dropped = 0usize;
        while self.dirty_queues.total_len() > MAX_PENDING_DIRTY_CHUNKS {
            let removed = self
                .dirty_queues
                .pop_back_tier(DirtyTier::Far)
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Normal))
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Near))
                .or_else(|| self.dirty_queues.pop_back_tier(DirtyTier::Urgent));
            if let Some(coord) = removed {
                self.pending_lod_remesh.remove(&coord);
                self.pending_lod_remesh_since.remove(&coord);
                dropped += 1;
            } else {
                break;
            }
        }
        dropped
    }
}

fn dirty_coord_priority(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    chunk_priority_scores: &HashMap<ChunkCoord, f32>,
) -> f32 {
    urgent_job_priority(coord, player_chunk, chunk_priority_scores)
}

fn urgent_job_priority(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    chunk_priority_scores: &HashMap<ChunkCoord, f32>,
) -> f32 {
    let fallback = 1.0 / (1.0 + chunk_chebyshev_dist(player_chunk, coord) as f32);
    let priority = chunk_priority_scores
        .get(&coord)
        .copied()
        .unwrap_or(fallback);
    if priority.is_finite() {
        priority
    } else {
        fallback
    }
}

fn build_chunk_snapshot(
    store: &ChunkStore,
    coord: ChunkCoord,
    policy: UnknownNeighborOcclusionPolicy,
) -> ChunkSnapshot {
    let chunk_world_min = chunk_to_world_min(coord);
    let center_voxels = store
        .get_chunk(coord)
        .map(|chunk| Arc::<[MaterialId]>::from(chunk.iter_raw()))
        .unwrap_or_else(|| Arc::from(vec![EMPTY; (CHUNK_SIZE_VOXELS.pow(3)) as usize]));

    let mut strips = store.chunk_border_strips(coord);
    if matches!(policy, UnknownNeighborOcclusionPolicy::Conservative) {
        synthesize_missing_neighbor_borders(store, coord, &mut strips);
    }

    ChunkSnapshot {
        world_min: chunk_world_min,
        center_voxels,
        border_strips: Arc::new(strips),
    }
}

fn synthesize_missing_neighbor_borders(
    store: &ChunkStore,
    coord: ChunkCoord,
    strips: &mut ChunkBorderStrips,
) {
    let Some(center) = store.get_chunk(coord) else {
        return;
    };
    let side = CHUNK_SIZE_VOXELS as usize;
    let idx = |u: usize, v: usize| u * side + v;

    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x - 1,
        y: coord.y,
        z: coord.z,
    }) {
        for y in 0..side {
            for z in 0..side {
                strips.neg_x[idx(y, z)] = center.get(0, y, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x + 1,
        y: coord.y,
        z: coord.z,
    }) {
        for y in 0..side {
            for z in 0..side {
                strips.pos_x[idx(y, z)] = center.get(side - 1, y, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y - 1,
        z: coord.z,
    }) {
        for x in 0..side {
            for z in 0..side {
                strips.neg_y[idx(x, z)] = center.get(x, 0, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y + 1,
        z: coord.z,
    }) {
        for x in 0..side {
            for z in 0..side {
                strips.pos_y[idx(x, z)] = center.get(x, side - 1, z);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y,
        z: coord.z - 1,
    }) {
        for x in 0..side {
            for y in 0..side {
                strips.neg_z[idx(x, y)] = center.get(x, y, 0);
            }
        }
    }
    if !store.is_chunk_loaded(ChunkCoord {
        x: coord.x,
        y: coord.y,
        z: coord.z + 1,
    }) {
        for x in 0..side {
            for y in 0..side {
                strips.pos_z[idx(x, y)] = center.get(x, y, side - 1);
            }
        }
    }
}

pub(crate) fn mesh_chunk_snapshot(
    coord: ChunkCoord,
    snapshot: &ChunkSnapshot,
    lod: ChunkLod,
    greedy: bool,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3, Vec3) {
    let chunk_world_min = snapshot.world_min;
    debug_assert_eq!(chunk_world_min, chunk_to_world_min(coord));
    let chunk_origin_world = voxel_to_world(chunk_world_min);
    let (mut verts, inds) = match lod {
        ChunkLod::Near => {
            if greedy {
                mesh_chunk_voxel_faces_greedy(snapshot)
            } else {
                mesh_chunk_voxel_faces(snapshot, 1)
            }
        }
        ChunkLod::Mid => mesh_chunk_coarse_solid(snapshot, 2),
        ChunkLod::Far => mesh_chunk_coarse_solid(snapshot, 4),
        ChunkLod::Ultra => mesh_chunk_heightfield_proxy(snapshot, 8),
    };
    for v in &mut verts {
        let p = chunk_origin_world + Vec3::from_array(v.pos);
        v.pos = p.to_array();
    }
    let (aabb_min, aabb_max) = chunk_world_aabb_from_vertices(chunk_origin_world, &verts);
    (verts, inds, aabb_min, aabb_max, chunk_origin_world)
}

fn mesh_chunk_voxel_faces(snapshot: &ChunkSnapshot, step: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();

    let mut lz = 0;
    while lz < CHUNK_SIZE_VOXELS {
        let mut ly = 0;
        while ly < CHUNK_SIZE_VOXELS {
            let mut lx = 0;
            while lx < CHUNK_SIZE_VOXELS {
                let id = snapshot.get_local(lx, ly, lz);
                if id == EMPTY {
                    lx += step;
                    continue;
                }

                if is_billboard_material(id) {
                    add_snapshot_billboard(lx, ly, lz, id, &mut verts, &mut inds);
                    lx += step;
                    continue;
                }

                let color = material(id).color;
                add_snapshot_voxel_faces(snapshot, lx, ly, lz, id, color, &mut verts, &mut inds);

                lx += step;
            }
            ly += step;
        }
        lz += step;
    }

    (verts, inds)
}

fn mesh_chunk_voxel_faces_greedy(snapshot: &ChunkSnapshot) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;

    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let id = snapshot.get_local(x, y, z);
                if is_billboard_material(id) {
                    add_snapshot_billboard(x, y, z, id, &mut verts, &mut inds);
                }
            }
        }
    }

    for y in 0..side {
        for z in 0..side {
            let mut x = 0;
            while x < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x + 1, y, z))
                {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x + 1, y, z)) {
                        break;
                    }
                    x += 1;
                }
                add_box_faces(
                    [start as f32, y as f32, z as f32],
                    (x - start) as f32,
                    1.0,
                    1.0,
                    material(id).color,
                    &[true, false, false, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for y in 0..side {
        for z in 0..side {
            let mut x = 0;
            while x < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x - 1, y, z))
                {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x - 1, y, z)) {
                        break;
                    }
                    x += 1;
                }
                add_box_faces(
                    [start as f32, y as f32, z as f32],
                    (x - start) as f32,
                    1.0,
                    1.0,
                    material(id).color,
                    &[false, true, false, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for y in 0..side {
            let mut z = 0;
            while z < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y, z + 1))
                {
                    z += 1;
                    continue;
                }
                let start = z;
                while z < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y, z + 1)) {
                        break;
                    }
                    z += 1;
                }
                add_box_faces(
                    [x as f32, y as f32, start as f32],
                    1.0,
                    1.0,
                    (z - start) as f32,
                    material(id).color,
                    &[false, false, false, false, true, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for y in 0..side {
            let mut z = 0;
            while z < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y, z - 1))
                {
                    z += 1;
                    continue;
                }
                let start = z;
                while z < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y, z - 1)) {
                        break;
                    }
                    z += 1;
                }
                add_box_faces(
                    [x as f32, y as f32, start as f32],
                    1.0,
                    1.0,
                    (z - start) as f32,
                    material(id).color,
                    &[false, false, false, false, false, true],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for z in 0..side {
            let mut y = 0;
            while y < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y + 1, z))
                {
                    y += 1;
                    continue;
                }
                let start = y;
                while y < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y + 1, z)) {
                        break;
                    }
                    y += 1;
                }
                add_box_faces(
                    [x as f32, start as f32, z as f32],
                    1.0,
                    (y - start) as f32,
                    1.0,
                    material(id).color,
                    &[false, false, true, false, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    for x in 0..side {
        for z in 0..side {
            let mut y = 0;
            while y < side {
                let id = snapshot.get_local(x, y, z);
                if id == EMPTY
                    || is_billboard_material(id)
                    || is_face_occluded(id, snapshot.get_local(x, y - 1, z))
                {
                    y += 1;
                    continue;
                }
                let start = y;
                while y < side {
                    let cur = snapshot.get_local(x, y, z);
                    if cur != id || is_face_occluded(cur, snapshot.get_local(x, y - 1, z)) {
                        break;
                    }
                    y += 1;
                }
                add_box_faces(
                    [x as f32, start as f32, z as f32],
                    1.0,
                    (y - start) as f32,
                    1.0,
                    material(id).color,
                    &[false, false, false, true, false, false],
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    (verts, inds)
}

fn mesh_chunk_coarse_solid(snapshot: &ChunkSnapshot, step: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;

    let is_solid =
        |x: i32, y: i32, z: i32| dominant_material_in_cell(snapshot, x, y, z, step).is_some();

    let mut z = 0;
    while z < side {
        let mut y = 0;
        while y < side {
            let mut x = 0;
            while x < side {
                let Some(id) = dominant_material_in_cell(snapshot, x, y, z, step) else {
                    x += step;
                    continue;
                };
                let color = material(id).color;
                let mut exposed = [false; 6];
                let n = [
                    (x + step, y, z),
                    (x - step, y, z),
                    (x, y + step, z),
                    (x, y - step, z),
                    (x, y, z + step),
                    (x, y, z - step),
                ];
                for (i, &(nx, ny, nz)) in n.iter().enumerate() {
                    exposed[i] = !is_solid(nx, ny, nz);
                }
                if exposed.iter().any(|e| *e) {
                    add_box_faces(
                        [x as f32, y as f32, z as f32],
                        step as f32,
                        step as f32,
                        step as f32,
                        color,
                        &exposed,
                        &mut verts,
                        &mut inds,
                    );
                }
                x += step;
            }
            y += step;
        }
        z += step;
    }
    (verts, inds)
}

fn mesh_chunk_heightfield_proxy(snapshot: &ChunkSnapshot, tile: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let side = CHUNK_SIZE_VOXELS;
    let tiles = (side / tile) as usize;
    let mut heights = vec![None; tiles * tiles];

    for tz in 0..tiles as i32 {
        for tx in 0..tiles as i32 {
            heights[(tz as usize) * tiles + tx as usize] =
                tile_peak(snapshot, tx * tile, tz * tile, tile);
        }
    }

    for tz in 0..tiles as i32 {
        for tx in 0..tiles as i32 {
            let Some((height, id)) = heights[(tz as usize) * tiles + tx as usize] else {
                continue;
            };
            let color = material(id).color;
            let top = height + 1;
            let neighbors = [
                tx + 1 < tiles as i32
                    && heights[(tz as usize) * tiles + (tx + 1) as usize]
                        .is_some_and(|(h, _)| h >= height),
                tx > 0
                    && heights[(tz as usize) * tiles + (tx - 1) as usize]
                        .is_some_and(|(h, _)| h >= height),
                false,
                true,
                tz + 1 < tiles as i32
                    && heights[((tz + 1) as usize) * tiles + tx as usize]
                        .is_some_and(|(h, _)| h >= height),
                tz > 0
                    && heights[((tz - 1) as usize) * tiles + tx as usize]
                        .is_some_and(|(h, _)| h >= height),
            ];
            let exposed = [
                !neighbors[0],
                !neighbors[1],
                true,
                false,
                !neighbors[4],
                !neighbors[5],
            ];
            add_box_faces(
                [(tx * tile) as f32, 0.0, (tz * tile) as f32],
                tile as f32,
                top as f32,
                tile as f32,
                color,
                &exposed,
                &mut verts,
                &mut inds,
            );
        }
    }
    (verts, inds)
}

fn tile_peak(
    snapshot: &ChunkSnapshot,
    base_x: i32,
    base_z: i32,
    tile: i32,
) -> Option<(i32, MaterialId)> {
    for y in (0..CHUNK_SIZE_VOXELS).rev() {
        for z in base_z..(base_z + tile).min(CHUNK_SIZE_VOXELS) {
            for x in base_x..(base_x + tile).min(CHUNK_SIZE_VOXELS) {
                let id = snapshot.get_local(x, y, z);
                if id != EMPTY {
                    return Some((y, id));
                }
            }
        }
    }
    None
}

fn sample_material_for_coarse_cell(snapshot: &ChunkSnapshot, x: i32, y: i32, z: i32) -> MaterialId {
    let side = CHUNK_SIZE_VOXELS;
    let clamp_axis = |v: i32| {
        if v < 0 {
            -1
        } else if v >= side {
            side
        } else {
            v
        }
    };
    let sx = clamp_axis(x);
    let sy = clamp_axis(y);
    let sz = clamp_axis(z);

    let out_axes = u8::from(sx != x) + u8::from(sy != y) + u8::from(sz != z);
    if out_axes > 1 {
        return EMPTY;
    }

    snapshot.get_local(sx, sy, sz)
}

fn dominant_material_in_cell(
    snapshot: &ChunkSnapshot,
    base_x: i32,
    base_y: i32,
    base_z: i32,
    step: i32,
) -> Option<MaterialId> {
    let mut counts = HashMap::<MaterialId, u16>::new();
    for z in base_z..(base_z + step).min(CHUNK_SIZE_VOXELS) {
        for y in base_y..(base_y + step).min(CHUNK_SIZE_VOXELS) {
            for x in base_x..(base_x + step).min(CHUNK_SIZE_VOXELS) {
                let id = sample_material_for_coarse_cell(snapshot, x, y, z);
                if id != EMPTY {
                    *counts.entry(id).or_default() += 1;
                }
            }
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(id, _)| id)
}

fn add_box_faces(
    local_voxel_min: [f32; 3],
    sx: f32,
    sy: f32,
    sz: f32,
    color: [u8; 4],
    exposed: &[bool; 6],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let dirs = [
        (
            [
                [1.0, 0.0, 0.0],
                [1.0, sy, 0.0],
                [1.0, sy, sz],
                [1.0, 0.0, sz],
            ],
            0.88,
        ),
        (
            [
                [0.0, 0.0, sz],
                [0.0, sy, sz],
                [0.0, sy, 0.0],
                [0.0, 0.0, 0.0],
            ],
            0.73,
        ),
        (
            [[0.0, sy, sz], [sx, sy, sz], [sx, sy, 0.0], [0.0, sy, 0.0]],
            1.0,
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [sx, 0.0, 0.0],
                [sx, 0.0, sz],
                [0.0, 0.0, sz],
            ],
            0.58,
        ),
        (
            [[sx, 0.0, sz], [sx, sy, sz], [0.0, sy, sz], [0.0, 0.0, sz]],
            0.8,
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [0.0, sy, 0.0],
                [sx, sy, 0.0],
                [sx, 0.0, 0.0],
            ],
            0.66,
        ),
    ];
    let origin = [
        local_voxel_min[0] * VOXEL_SIZE,
        local_voxel_min[1] * VOXEL_SIZE,
        local_voxel_min[2] * VOXEL_SIZE,
    ];

    for (i, (quad, shade)) in dirs.iter().enumerate() {
        if !exposed[i] {
            continue;
        }
        let b = verts.len() as u32;
        let shaded = shade_color(color, *shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    origin[0] + v[0] * VOXEL_SIZE,
                    origin[1] + v[1] * VOXEL_SIZE,
                    origin[2] + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn build_debug_aabb_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let side = CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
    let c = [200, 32, 32, 100];
    let corners = [
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side, side, 0.0],
        [0.0, side, 0.0],
        [0.0, 0.0, side],
        [side, 0.0, side],
        [side, side, side],
        [0.0, side, side],
    ];
    let mut verts = Vec::with_capacity(corners.len());
    for p in corners {
        verts.push(Vertex { pos: p, color: c });
    }
    let inds = vec![
        0, 1, 2, 0, 2, 3, // near
        4, 6, 5, 4, 7, 6, // far
        0, 4, 5, 0, 5, 1, // bottom
        3, 2, 6, 3, 6, 7, // top
        1, 5, 6, 1, 6, 2, // right
        0, 3, 7, 0, 7, 4, // left
    ];
    (verts, inds)
}

fn draw_debug_aabb<'a>(pass: &mut wgpu::RenderPass<'a>, mesh: &'a ChunkMesh, _color: [u8; 4]) {
    pass.set_vertex_buffer(0, mesh.debug_aabb_vb.slice(..));
    pass.set_index_buffer(mesh.debug_aabb_ib.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.debug_aabb_index_count, 0, 0..1);
}

fn chunk_chebyshev_dist(a: ChunkCoord, b: ChunkCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

fn chunk_distance(a: ChunkCoord, b: ChunkCoord) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    let dz = (a.z - b.z) as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn chunk_horizontal_distance_to_camera(coord: ChunkCoord, world_camera_pos: Vec3) -> f32 {
    let chunk_world_min = voxel_to_world(chunk_to_world_min(coord));
    let center = chunk_world_min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE * 0.5);
    let delta = center - world_camera_pos;
    (delta.x.hypot(delta.z)) / (CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE)
}

fn chunk_world_aabb_from_vertices(chunk_origin_world: Vec3, verts: &[Vertex]) -> (Vec3, Vec3) {
    if verts.is_empty() {
        return (chunk_origin_world, chunk_origin_world);
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in verts {
        let p = Vec3::from_array(v.pos);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

fn select_lod(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    prev: Option<ChunkLod>,
) -> ChunkLod {
    let d = chunk_distance(coord, player_chunk);
    let h = radii.hysteresis.max(1);
    let near_down = (radii.near.saturating_sub(h * 5)) as f32;
    let mid_down = (radii.mid.saturating_sub(h * 10)) as f32;
    let far_down = (radii.far.saturating_sub(h * 20)) as f32;
    let near_up = radii.near as f32;
    let mid_up = radii.mid as f32;
    let far_up = radii.far as f32;

    match prev {
        Some(ChunkLod::Near) => {
            if d <= near_up {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Mid) => {
            if d <= near_down {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Far) => {
            if d <= mid_down {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        Some(ChunkLod::Ultra) => {
            if d <= far_down {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
        None => {
            if d <= near_up {
                ChunkLod::Near
            } else if d <= mid_up {
                ChunkLod::Mid
            } else if d <= far_up {
                ChunkLod::Far
            } else {
                ChunkLod::Ultra
            }
        }
    }
}

fn fallback_lod_near_threshold(
    coord: ChunkCoord,
    player_chunk: ChunkCoord,
    radii: LodRadii,
    primary: ChunkLod,
) -> Option<ChunkLod> {
    let d = chunk_distance(coord, player_chunk);
    let h = radii.hysteresis.max(1) as f32;

    let near_edge = (d - radii.near as f32).abs() <= h;
    let mid_edge = (d - radii.mid as f32).abs() <= h;
    let far_edge = (d - radii.far as f32).abs() <= h;

    match primary {
        ChunkLod::Near if near_edge => Some(ChunkLod::Mid),
        ChunkLod::Mid if near_edge => Some(ChunkLod::Near),
        ChunkLod::Mid if mid_edge => Some(ChunkLod::Far),
        ChunkLod::Far if mid_edge => Some(ChunkLod::Mid),
        ChunkLod::Far if far_edge => Some(ChunkLod::Ultra),
        ChunkLod::Ultra if far_edge => Some(ChunkLod::Far),
        _ => None,
    }
}

fn camera_world_position(camera: &Camera) -> Vec3 {
    camera.pos
}

fn lod_from_u8(value: u8) -> ChunkLod {
    match value {
        0 => ChunkLod::Near,
        1 => ChunkLod::Mid,
        2 => ChunkLod::Far,
        _ => ChunkLod::Ultra,
    }
}

fn draw_is_drawable(draw: &GpuChunkDraw) -> bool {
    draw.index_count.unwrap_or(0) > 0
}

fn lod_mismatch_grace_allows_draw(
    coord: ChunkCoord,
    pending_lod_remesh: &HashSet<ChunkCoord>,
    pending_lod_remesh_since: &HashMap<ChunkCoord, u64>,
    mesh_rebuild_frame_index: u64,
) -> bool {
    if !pending_lod_remesh.contains(&coord) {
        return false;
    }
    let Some(first_pending_frame) = pending_lod_remesh_since.get(&coord).copied() else {
        return false;
    };
    mesh_rebuild_frame_index.saturating_sub(first_pending_frame) <= LOD_MISMATCH_GRACE_MAX_FRAMES
}

fn chunk_passes_draw_contract(
    coord: ChunkCoord,
    draw: &GpuChunkDraw,
    visible_slots: &HashMap<u32, ChunkCoord>,
    lod_selection: &HashMap<ChunkCoord, ChunkLod>,
    pending_lod_remesh: &HashSet<ChunkCoord>,
    pending_lod_remesh_since: &HashMap<ChunkCoord, u64>,
    mesh_rebuild_frame_index: u64,
    visibility: DrawVisibilityInput,
) -> bool {
    if visible_slots.get(&draw.draw_indirect_index) != Some(&coord) {
        return false;
    }
    let selected_lod = lod_selection.get(&coord).copied().unwrap_or(ChunkLod::Near);
    let draw_lod = lod_from_u8(draw.lod);
    let lod_matches_selection = draw_lod == selected_lod;
    if !lod_matches_selection
        && !lod_mismatch_grace_allows_draw(
            coord,
            pending_lod_remesh,
            pending_lod_remesh_since,
            mesh_rebuild_frame_index,
        )
    {
        return false;
    }
    if !draw_is_drawable(draw) {
        return false;
    }
    chunk_visible_in_world_space(
        visibility.frustum_culling,
        visibility.vp_world,
        visibility.world_camera_pos,
        draw_lod,
        draw.world_aabb_min,
        draw.world_aabb_max,
        visibility.screen_h,
    )
}

fn chunk_visible_in_world_space(
    frustum_culling: bool,
    vp_world: Mat4,
    world_camera_pos: Vec3,
    lod: ChunkLod,
    world_aabb_min: Vec3,
    world_aabb_max: Vec3,
    screen_h: u32,
) -> bool {
    (!frustum_culling || aabb_in_view(vp_world, world_aabb_min, world_aabb_max))
        && passes_screen_space_cull(
            world_camera_pos,
            lod,
            world_aabb_min,
            world_aabb_max,
            screen_h,
        )
}

fn passes_screen_space_cull(
    camera_pos: Vec3,
    lod: ChunkLod,
    aabb_min: Vec3,
    aabb_max: Vec3,
    screen_h: u32,
) -> bool {
    let min_pixels = match lod {
        ChunkLod::Near => 0.0,
        ChunkLod::Mid => 0.5,
        ChunkLod::Far => 1.0,
        ChunkLod::Ultra => 2.0,
    };
    if min_pixels <= 0.0 {
        return true;
    }
    let center = (aabb_min + aabb_max) * 0.5;
    let radius = (aabb_max - center).length();
    let distance = (center - camera_pos).length().max(0.01);
    let focal = screen_h as f32 / (2.0 * (60f32.to_radians() * 0.5).tan());
    let pixel_radius = (radius / distance) * focal;
    pixel_radius * 2.0 >= min_pixels
}

fn add_snapshot_voxel_faces(
    snapshot: &ChunkSnapshot,
    local_x: i32,
    local_y: i32,
    local_z: i32,
    id: MaterialId,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let dirs = [
        (
            [1, 0, 0],
            [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
            0.88,
        ),
        (
            [-1, 0, 0],
            [[0., 0., 1.], [0., 1., 1.], [0., 1., 0.], [0., 0., 0.]],
            0.73,
        ),
        (
            [0, 1, 0],
            [[0., 1., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.]],
            1.0,
        ),
        (
            [0, -1, 0],
            [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
            0.58,
        ),
        (
            [0, 0, 1],
            [[1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.]],
            0.8,
        ),
        (
            [0, 0, -1],
            [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
            0.66,
        ),
    ];

    for (d, quad, shade) in dirs {
        if is_face_occluded(
            id,
            snapshot.get_local(local_x + d[0], local_y + d[1], local_z + d[2]),
        ) {
            continue;
        }

        let b = verts.len() as u32;
        let shaded = shade_color(color, shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    local_x as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    local_y as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    local_z as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn is_billboard_material(id: MaterialId) -> bool {
    matches!(id, BUSH_ID | GRASS_ID)
}

fn add_snapshot_billboard(
    local_x: i32,
    local_y: i32,
    local_z: i32,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let color = material(id).color;
    let quads: &[[[f32; 3]; 4]] = if id == GRASS_ID {
        &[
            [
                [0.48, 0.0, 0.12],
                [0.48, 1.08, 0.12],
                [0.52, 1.08, 0.88],
                [0.52, 0.0, 0.88],
            ],
            [
                [0.12, 0.0, 0.48],
                [0.12, 1.04, 0.48],
                [0.88, 1.04, 0.52],
                [0.88, 0.0, 0.52],
            ],
        ]
    } else {
        &[
            [
                [0.14, 0.0, 0.14],
                [0.14, 0.90, 0.14],
                [0.86, 0.90, 0.86],
                [0.86, 0.0, 0.86],
            ],
            [
                [0.14, 0.0, 0.86],
                [0.14, 0.90, 0.86],
                [0.86, 0.90, 0.14],
                [0.86, 0.0, 0.14],
            ],
        ]
    };

    for quad in quads {
        let b = verts.len() as u32;
        for v in quad {
            verts.push(Vertex {
                pos: [
                    local_x as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    local_y as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    local_z as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color,
            });
        }
        inds.extend_from_slice(&[
            b,
            b + 1,
            b + 2,
            b,
            b + 2,
            b + 3,
            b,
            b + 2,
            b + 1,
            b,
            b + 3,
            b + 2,
        ]);
    }
}

fn is_face_occluded(self_id: MaterialId, neighbor_id: MaterialId) -> bool {
    if neighbor_id == EMPTY {
        return false;
    }
    if neighbor_id == self_id {
        return true;
    }
    if is_billboard_material(neighbor_id) {
        return false;
    }
    let neighbor = material(neighbor_id);
    if neighbor.color[3] < 255 {
        return false;
    }
    matches!(neighbor.phase, Phase::Solid | Phase::Powder)
}

fn shade_color(color: [u8; 4], shade: f32) -> [u8; 4] {
    [
        ((color[0] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[1] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[2] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        color[3],
    ]
}

fn voxel_to_world(voxel: VoxelCoord) -> Vec3 {
    Vec3::new(voxel.x as f32, voxel.y as f32, voxel.z as f32) * VOXEL_SIZE
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
                // WGPU clip-space depth is [0, w], not [-w, w].
                4 => clip.z,
                _ => -clip.z + clip.w,
            };
            if v < 0.0 {
                outside += 1;
            }
        }
        if outside == clips.len() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{Chunk, NeighborDirtyPolicy};

    fn coord() -> ChunkCoord {
        ChunkCoord { x: 0, y: 0, z: 0 }
    }

    fn chunk_with_voxel(x: usize, y: usize, z: usize, id: MaterialId) -> Chunk {
        let mut c = Chunk::new_empty();
        c.set(x, y, z, id);
        c
    }

    #[test]
    fn frustum_culls_and_accepts_expected_aabbs() {
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let vp = camera.view_proj();

        let visible_min = Vec3::new(-0.5, -0.5, -2.0);
        let visible_max = Vec3::new(0.5, 0.5, -1.0);
        assert!(aabb_in_view(vp, visible_min, visible_max));

        let behind_camera_min = Vec3::new(-0.5, -0.5, 0.5);
        let behind_camera_max = Vec3::new(0.5, 0.5, 1.5);
        assert!(!aabb_in_view(vp, behind_camera_min, behind_camera_max));

        let beyond_far_min = Vec3::new(-1.0, -1.0, -1300.0);
        let beyond_far_max = Vec3::new(1.0, 1.0, -1250.0);
        assert!(!aabb_in_view(vp, beyond_far_min, beyond_far_max));
    }

    #[test]
    fn culling_visibility_with_zero_origin_is_consistent() {
        let camera = Camera {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = Vec3::new(-1.0, -1.0, -6.0);
        let world_max = Vec3::new(1.0, 1.0, -4.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj(),
            camera_world_position(&camera),
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );

        assert!(world_visible);
    }

    #[test]
    fn culling_visibility_with_large_non_zero_origin_is_consistent() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 1_500,
            z: -3_250_000,
        };
        let origin_world = voxel_to_world(origin);
        let camera = Camera {
            pos: origin_world,
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let world_min = origin_world + Vec3::new(-1.0, -1.0, -16.0);
        let world_max = origin_world + Vec3::new(1.0, 1.0, -8.0);

        let world_visible = chunk_visible_in_world_space(
            true,
            camera.view_proj(),
            camera_world_position(&camera),
            ChunkLod::Ultra,
            world_min,
            world_max,
            1080,
        );

        assert!(world_visible);
    }

    #[test]
    fn cpu_culling_is_origin_invariant_for_same_world_relationship() {
        let world_camera = Camera {
            pos: Vec3::new(1_000_000.0, 0.0, -2_000_000.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let origin = VoxelCoord {
            x: (world_camera.pos.x / VOXEL_SIZE) as i32,
            y: (world_camera.pos.y / VOXEL_SIZE) as i32,
            z: (world_camera.pos.z / VOXEL_SIZE) as i32,
        };

        let world_min = world_camera.pos + Vec3::new(-2.0, -2.0, -20.0);
        let world_max = world_camera.pos + Vec3::new(2.0, 2.0, -12.0);

        let baseline_world = chunk_visible_in_world_space(
            true,
            world_camera.view_proj(),
            world_camera.pos,
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );
        let rebased_world = chunk_visible_in_world_space(
            true,
            world_camera.view_proj_rebased_to_origin(origin),
            camera_world_position(&world_camera),
            ChunkLod::Far,
            world_min,
            world_max,
            1080,
        );

        assert!(baseline_world);
        assert_ne!(baseline_world, rebased_world);
    }

    #[test]
    fn rebased_draw_transform_matches_world_space_projection() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 1_500,
            z: -3_250_000,
        };
        let world_camera = Camera {
            pos: voxel_to_world(origin) + Vec3::new(0.75, 1.0, 2.5),
            dir: Vec3::new(0.0, -0.1, -1.0).normalize(),
            aspect: 16.0 / 9.0,
        };
        let chunk_origin_world = voxel_to_world(VoxelCoord {
            x: origin.x + 64,
            y: origin.y + 32,
            z: origin.z - 96,
        });
        let local_vertex = Vec3::new(3.0 * VOXEL_SIZE, 5.0 * VOXEL_SIZE, 2.0 * VOXEL_SIZE);
        let world_pos = chunk_origin_world + local_vertex;

        let clip_world = world_camera.view_proj() * world_pos.extend(1.0);
        let clip_rebased = world_camera.view_proj_rebased_to_origin(origin)
            * (world_pos - voxel_to_world(origin)).extend(1.0);

        assert!(clip_world.abs_diff_eq(clip_rebased, 1.0));
    }

    #[test]
    fn screen_space_cull_uses_world_space_distances() {
        let origin = VoxelCoord {
            x: 4_000_000,
            y: 0,
            z: -2_000_000,
        };
        let world_origin = voxel_to_world(origin);
        let world_camera = world_origin + Vec3::new(0.0, 0.0, 0.0);
        let world_min = world_origin + Vec3::new(-2.0, -2.0, -24.0);
        let world_max = world_origin + Vec3::new(2.0, 2.0, -16.0);

        let consistent_world =
            passes_screen_space_cull(world_camera, ChunkLod::Ultra, world_min, world_max, 1080);

        // Intentionally incorrect mixed-space input (render-relative AABB with
        // world-space camera). This should not match correct world-space culling.
        let mixed_space = passes_screen_space_cull(
            world_camera,
            ChunkLod::Ultra,
            world_min - world_origin,
            world_max - world_origin,
            1080,
        );

        assert!(consistent_world);
        assert_ne!(consistent_world, mixed_space);
    }

    #[test]
    fn dirty_priority_prefers_near_chunks() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = ChunkCoord { x: 1, y: 0, z: 0 };
        let far = ChunkCoord { x: 40, y: 0, z: 0 };
        let scores = HashMap::new();

        assert!(
            dirty_coord_priority(near, player, &scores)
                > dirty_coord_priority(far, player, &scores)
        );
    }

    #[test]
    fn urgent_submission_prefers_higher_generation_priority() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let high = ChunkCoord { x: 12, y: 0, z: 0 };
        let low = ChunkCoord { x: 1, y: 0, z: 0 };
        let mut scores = HashMap::new();
        scores.insert(high, 0.95);
        scores.insert(low, 0.10);

        let snapshot = build_chunk_snapshot(
            &ChunkStore::new(),
            coord(),
            UnknownNeighborOcclusionPolicy::Aggressive,
        );

        let mut jobs = vec![
            MeshJob {
                coord: low,
                lod: ChunkLod::Near,
                version: 0,
                queued_at: Instant::now(),
                snapshot: snapshot.clone(),
                greedy: false,
                urgent: true,
            },
            MeshJob {
                coord: high,
                lod: ChunkLod::Near,
                version: 0,
                queued_at: Instant::now(),
                snapshot: snapshot.clone(),
                greedy: false,
                urgent: true,
            },
        ];

        jobs.sort_by(|a, b| {
            urgent_job_priority(a.coord, player, &scores)
                .total_cmp(&urgent_job_priority(b.coord, player, &scores))
                .then_with(|| {
                    chunk_chebyshev_dist(player, b.coord)
                        .cmp(&chunk_chebyshev_dist(player, a.coord))
                })
                .then_with(|| a.coord.x.cmp(&b.coord.x))
                .then_with(|| a.coord.y.cmp(&b.coord.y))
                .then_with(|| a.coord.z.cmp(&b.coord.z))
        });
        jobs.reverse();

        let order: Vec<ChunkCoord> = jobs.drain(..).map(|job| job.coord).collect();
        assert_eq!(order, vec![high, low]);
    }

    #[test]
    fn urgent_submission_tie_break_prefers_closer_chunks() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = ChunkCoord { x: 1, y: 0, z: 0 };
        let far = ChunkCoord { x: 4, y: 0, z: 0 };
        let scores = HashMap::from([(near, 0.5), (far, 0.5)]);

        let snapshot = build_chunk_snapshot(
            &ChunkStore::new(),
            coord(),
            UnknownNeighborOcclusionPolicy::Aggressive,
        );

        let mut jobs = vec![
            MeshJob {
                coord: far,
                lod: ChunkLod::Near,
                version: 0,
                queued_at: Instant::now(),
                snapshot: snapshot.clone(),
                greedy: false,
                urgent: true,
            },
            MeshJob {
                coord: near,
                lod: ChunkLod::Near,
                version: 0,
                queued_at: Instant::now(),
                snapshot: snapshot.clone(),
                greedy: false,
                urgent: true,
            },
        ];

        jobs.sort_by(|a, b| {
            urgent_job_priority(a.coord, player, &scores)
                .total_cmp(&urgent_job_priority(b.coord, player, &scores))
                .then_with(|| {
                    chunk_chebyshev_dist(player, b.coord)
                        .cmp(&chunk_chebyshev_dist(player, a.coord))
                })
                .then_with(|| a.coord.x.cmp(&b.coord.x))
                .then_with(|| a.coord.y.cmp(&b.coord.y))
                .then_with(|| a.coord.z.cmp(&b.coord.z))
        });
        jobs.reverse();

        assert_eq!(jobs.drain(..).next().map(|job| job.coord), Some(near));
    }

    #[test]
    fn urgent_priority_non_finite_scores_fall_back_to_distance() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = ChunkCoord { x: 1, y: 0, z: 0 };
        let far = ChunkCoord { x: 8, y: 0, z: 0 };
        let mut scores = HashMap::new();
        scores.insert(near, f32::NAN);
        scores.insert(far, f32::NAN);

        assert!(
            urgent_job_priority(near, player, &scores) > urgent_job_priority(far, player, &scores)
        );
    }

    #[test]
    fn dirty_backlog_sort_perf_guard() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut backlog = Vec::with_capacity(20_000);
        for i in 0..20_000 {
            backlog.push(ChunkCoord {
                x: i % 200,
                y: (i / 200) % 10,
                z: i / 2000,
            });
        }
        let scores = HashMap::new();

        let started = Instant::now();
        backlog.sort_by(|a, b| {
            dirty_coord_priority(*a, player, &scores)
                .total_cmp(&dirty_coord_priority(*b, player, &scores))
        });
        let elapsed = started.elapsed();

        assert!(
            elapsed.as_millis() < 750,
            "backlog prioritization regressed: {:?}",
            elapsed
        );
    }

    #[test]
    fn lod_submission_scheduler_guarantees_far_and_ultra_progress_under_backlog() {
        let mesh_budget = 8;
        let lod_budgets = LodMeshingBudgets {
            near: 6,
            mid: 3,
            far: 1,
            ultra: 1,
        };
        let pending = [64, 48, 40, 24];

        let effective = compute_effective_lod_submission_budgets(mesh_budget, lod_budgets, pending);

        assert!(effective[2] >= 1, "far tier should retain throughput");
        assert!(effective[3] >= 1, "ultra tier should retain throughput");
        assert!(effective.iter().sum::<usize>() <= mesh_budget);
    }

    #[test]
    fn lod_selection_uses_ultra_tier_with_hysteresis() {
        let radii = LodRadii {
            near: 4,
            mid: 8,
            far: 12,
            ultra: 20,
            hysteresis: 2,
        };
        let player = ChunkCoord { x: 0, y: 0, z: 0 };

        let near = select_lod(
            ChunkCoord { x: 3, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Near),
        );
        let far = select_lod(
            ChunkCoord { x: 11, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Far),
        );
        let ultra = select_lod(
            ChunkCoord { x: 21, y: 0, z: 0 },
            player,
            radii,
            Some(ChunkLod::Ultra),
        );

        assert_eq!(near, ChunkLod::Near);
        assert_eq!(far, ChunkLod::Far);
        assert_eq!(ultra, ChunkLod::Ultra);
    }

    #[test]
    fn missing_neighbors_keep_boundary_faces_visible_in_default_mode() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            coord(),
            chunk_with_voxel(CHUNK_SIZE_VOXELS as usize - 1, 2, 2, 1),
            false,
            NeighborDirtyPolicy::None,
        );

        let aggressive_snapshot =
            build_chunk_snapshot(&store, coord(), UnknownNeighborOcclusionPolicy::Aggressive);
        let conservative_snapshot = build_chunk_snapshot(
            &store,
            coord(),
            UnknownNeighborOcclusionPolicy::Conservative,
        );

        let (aggressive_verts, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &aggressive_snapshot, ChunkLod::Near, false);
        let (conservative_verts, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &conservative_snapshot, ChunkLod::Near, false);

        assert_eq!(aggressive_verts.len(), 24);
        assert_eq!(conservative_verts.len(), 20);

        store.insert_chunk_with_policy(
            ChunkCoord { x: 1, y: 0, z: 0 },
            Chunk::new_empty(),
            false,
            NeighborDirtyPolicy::None,
        );
        let aggressive_with_neighbor =
            build_chunk_snapshot(&store, coord(), UnknownNeighborOcclusionPolicy::Aggressive);
        let conservative_with_neighbor = build_chunk_snapshot(
            &store,
            coord(),
            UnknownNeighborOcclusionPolicy::Conservative,
        );

        let (aggressive_neighbor_verts, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &aggressive_with_neighbor, ChunkLod::Near, false);
        let (conservative_neighbor_verts, _, _, _, _) =
            mesh_chunk_snapshot(coord(), &conservative_with_neighbor, ChunkLod::Near, false);

        assert_eq!(aggressive_neighbor_verts.len(), 24);
        assert_eq!(conservative_neighbor_verts.len(), 24);
    }

    #[test]
    fn voxel_vertex_positions_are_world_space_and_match_chunk_origin_metadata() {
        let mut store = ChunkStore::new();
        let shifted = ChunkCoord { x: 3, y: 0, z: -2 };
        store.insert_chunk_with_policy(
            shifted,
            chunk_with_voxel(2, 3, 4, 1),
            false,
            NeighborDirtyPolicy::None,
        );

        let snapshot =
            build_chunk_snapshot(&store, shifted, UnknownNeighborOcclusionPolicy::Aggressive);
        let (verts, _, min, max, chunk_origin_world) =
            mesh_chunk_snapshot(shifted, &snapshot, ChunkLod::Near, false);

        let expected_chunk_origin = voxel_to_world(chunk_to_world_min(shifted));
        let expected_world_max =
            expected_chunk_origin + Vec3::new(3.0 * VOXEL_SIZE, 4.0 * VOXEL_SIZE, 5.0 * VOXEL_SIZE);
        let expected_world_min =
            expected_chunk_origin + Vec3::new(2.0 * VOXEL_SIZE, 3.0 * VOXEL_SIZE, 4.0 * VOXEL_SIZE);
        assert_eq!(chunk_origin_world, expected_chunk_origin);
        assert_eq!(min, expected_world_min);
        assert_eq!(max, expected_world_max);

        assert!(verts.iter().all(|v| {
            v.pos[0] >= expected_world_min.x
                && v.pos[0] <= expected_world_max.x
                && v.pos[1] >= expected_world_min.y
                && v.pos[1] <= expected_world_max.y
                && v.pos[2] >= expected_world_min.z
                && v.pos[2] <= expected_world_max.z
        }));
    }

    #[test]
    fn gpu_and_cpu_draw_records_share_world_space_contract() {
        let coord = ChunkCoord { x: 2, y: 0, z: -1 };
        let mut visible_slots = HashMap::new();
        visible_slots.insert(7, coord);
        let mut lod_selection = HashMap::new();
        lod_selection.insert(coord, ChunkLod::Mid);
        let aabb_min = Vec3::new(30.0, 0.0, -20.0);
        let aabb_max = Vec3::new(46.0, 16.0, -4.0);
        let camera = Camera {
            pos: Vec3::new(32.0, 8.0, 8.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let visibility = DrawVisibilityInput {
            frustum_culling: true,
            vp_world: camera.view_proj(),
            world_camera_pos: camera_world_position(&camera),
            screen_h: 1080,
        };

        for source in [DrawSource::CpuUploaded, DrawSource::GpuArtifact] {
            let draw = GpuChunkDraw {
                page_index: GpuPageIndex(7),
                draw_indirect_index: 7,
                lod: ChunkLod::Mid as u8,
                origin: Vec3::new(32.0, 0.0, -16.0),
                world_aabb_min: aabb_min,
                world_aabb_max: aabb_max,
                draw_source: source,
                index_count: Some(12),
            };
            assert!(chunk_passes_draw_contract(
                coord,
                &draw,
                &visible_slots,
                &lod_selection,
                &HashSet::new(),
                &HashMap::new(),
                0,
                visibility,
            ));
        }
    }

    #[test]
    fn lod_mismatch_pending_remesh_keeps_chunk_visible_during_grace_window() {
        let coord = ChunkCoord { x: 1, y: 0, z: 0 };
        let mut visible_slots = HashMap::new();
        visible_slots.insert(3, coord);
        let mut lod_selection = HashMap::new();
        lod_selection.insert(coord, ChunkLod::Near);
        let mut pending_lod_remesh = HashSet::new();
        pending_lod_remesh.insert(coord);
        let mut pending_lod_remesh_since = HashMap::new();
        pending_lod_remesh_since.insert(coord, 10);

        let draw = GpuChunkDraw {
            page_index: GpuPageIndex(3),
            draw_indirect_index: 3,
            lod: ChunkLod::Mid as u8,
            origin: Vec3::ZERO,
            world_aabb_min: Vec3::new(-1.0, -1.0, -4.0),
            world_aabb_max: Vec3::new(1.0, 1.0, -2.0),
            draw_source: DrawSource::CpuUploaded,
            index_count: Some(18),
        };
        let camera = Camera {
            pos: Vec3::ZERO,
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let visibility = DrawVisibilityInput {
            frustum_culling: true,
            vp_world: camera.view_proj(),
            world_camera_pos: camera_world_position(&camera),
            screen_h: 1080,
        };

        assert!(chunk_passes_draw_contract(
            coord,
            &draw,
            &visible_slots,
            &lod_selection,
            &pending_lod_remesh,
            &pending_lod_remesh_since,
            16,
            visibility,
        ));
    }

    #[test]
    fn lod_mismatch_grace_expires_without_replacement_artifact() {
        let coord = ChunkCoord { x: 1, y: 0, z: 0 };
        let mut pending_lod_remesh = HashSet::new();
        pending_lod_remesh.insert(coord);
        let mut pending_lod_remesh_since = HashMap::new();
        pending_lod_remesh_since.insert(coord, 1);

        assert!(!lod_mismatch_grace_allows_draw(
            coord,
            &pending_lod_remesh,
            &pending_lod_remesh_since,
            1 + LOD_MISMATCH_GRACE_MAX_FRAMES + 1,
        ));
    }
    #[test]
    fn stale_artifacts_are_always_marked_for_retry() {
        assert_eq!(
            stale_artifact_retry_policy(2, 5, ChunkLod::Near, false),
            Some(StaleArtifactRetryPolicy::Urgent)
        );
        assert_eq!(
            stale_artifact_retry_policy(1, 4, ChunkLod::Far, false),
            Some(StaleArtifactRetryPolicy::Dirty)
        );
        assert_eq!(
            stale_artifact_retry_policy(3, 6, ChunkLod::Ultra, true),
            Some(StaleArtifactRetryPolicy::Urgent)
        );
    }

    #[test]
    fn stale_retry_policy_allows_adoption_once_versions_catch_up() {
        assert_eq!(
            stale_artifact_retry_policy(4, 5, ChunkLod::Far, false),
            None
        );
        assert_eq!(
            stale_artifact_retry_policy(5, 5, ChunkLod::Far, false),
            None
        );
    }
    #[test]
    fn adjacent_chunk_bounds_are_world_space_and_contiguous() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            ChunkCoord { x: 0, y: 0, z: 0 },
            chunk_with_voxel(CHUNK_SIZE_VOXELS as usize - 1, 0, 0, 1),
            false,
            NeighborDirtyPolicy::None,
        );
        store.insert_chunk_with_policy(
            ChunkCoord { x: 1, y: 0, z: 0 },
            chunk_with_voxel(0, 0, 0, 1),
            false,
            NeighborDirtyPolicy::None,
        );
        let left_snapshot = build_chunk_snapshot(
            &store,
            ChunkCoord { x: 0, y: 0, z: 0 },
            UnknownNeighborOcclusionPolicy::Aggressive,
        );
        let right_snapshot = build_chunk_snapshot(
            &store,
            ChunkCoord { x: 1, y: 0, z: 0 },
            UnknownNeighborOcclusionPolicy::Aggressive,
        );

        let (_, _, left_min, left_max, _) = mesh_chunk_snapshot(
            ChunkCoord { x: 0, y: 0, z: 0 },
            &left_snapshot,
            ChunkLod::Near,
            false,
        );
        let (_, _, right_min, _, _) = mesh_chunk_snapshot(
            ChunkCoord { x: 1, y: 0, z: 0 },
            &right_snapshot,
            ChunkLod::Near,
            false,
        );

        assert_eq!(left_max.x, right_min.x);
        assert_eq!(left_min.y, right_min.y);
        assert_eq!(left_min.z, right_min.z);
    }

    #[test]
    fn mid_lod_coarse_sampling_uses_border_for_negative_step_neighbor() {
        let mut store = ChunkStore::new();
        let mut west = Chunk::new_empty();
        let mut center = Chunk::new_empty();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                west.set(CHUNK_SIZE_VOXELS as usize - 1, y, z, 1);
                center.set(0, y, z, 1);
            }
        }
        store.insert_chunk_with_policy(
            ChunkCoord { x: -1, y: 0, z: 0 },
            west,
            false,
            NeighborDirtyPolicy::None,
        );
        store.insert_chunk_with_policy(
            ChunkCoord { x: 0, y: 0, z: 0 },
            center,
            false,
            NeighborDirtyPolicy::None,
        );

        let snapshot = build_chunk_snapshot(
            &store,
            ChunkCoord { x: 0, y: 0, z: 0 },
            UnknownNeighborOcclusionPolicy::Aggressive,
        );

        assert!(dominant_material_in_cell(&snapshot, -2, 0, 0, 2).is_some());
    }

    #[test]
    fn render_selection_contract_matches_culling_contract() {
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let draw = GpuChunkDraw {
            page_index: GpuPageIndex(0),
            draw_indirect_index: 0,
            lod: ChunkLod::Near as u8,
            origin: Vec3::ZERO,
            world_aabb_min: Vec3::new(-1.0, -1.0, -4.0),
            world_aabb_max: Vec3::new(1.0, 1.0, -2.0),
            draw_source: DrawSource::CpuUploaded,
            index_count: Some(6),
        };
        let mut visible_slots = HashMap::new();
        visible_slots.insert(0, coord);
        let mut lod_selection = HashMap::new();
        lod_selection.insert(coord, ChunkLod::Near);
        let camera = Camera {
            pos: Vec3::ZERO,
            dir: Vec3::new(0.0, 0.0, -1.0),
            aspect: 1.0,
        };
        let visibility = DrawVisibilityInput {
            frustum_culling: true,
            vp_world: camera.view_proj(),
            world_camera_pos: camera_world_position(&camera),
            screen_h: 1080,
        };

        assert!(chunk_passes_draw_contract(
            coord,
            &draw,
            &visible_slots,
            &lod_selection,
            &HashSet::new(),
            &HashMap::new(),
            0,
            visibility,
        ));

        visible_slots.insert(0, ChunkCoord { x: 9, y: 0, z: 0 });
        assert!(!chunk_passes_draw_contract(
            coord,
            &draw,
            &visible_slots,
            &lod_selection,
            &HashSet::new(),
            &HashMap::new(),
            0,
            visibility,
        ));
    }
}
