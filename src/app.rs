use crate::chunk_store::ChunkStore;
use crate::floating_origin::{FloatingOriginConfig, FloatingOriginState};
use crate::input::{FpsController, InputState};
use crate::player::{camera_world_pos_from_blocks, grounded_eye_y_blocks};
use crate::procgen::{apply_generated_chunk, biome_hint_at_world, generate_chunk};
use crate::renderer::{
    Camera, LodMeshingBudgets, LodRadii, MeshRebuildStats, Renderer, RendererSettings,
    UnknownNeighborOcclusionPolicy, VOXEL_SIZE,
};
use crate::sim_world::Rng;
use crate::simulation::{
    SimulationMode, SimulationPhaseClass, SimulationRuntime, SimulationStepMetadata,
};
use crate::streaming::{
    is_urgent_chunk, ChunkStreaming, DesiredChunks, GenerateJobClass, VisibilityContext,
};
use crate::types::{voxel_to_chunk, ChunkCoord, VoxelCoord};
use crate::ui::{
    assign_hotbar_slot, draw, draw_fps_overlays, load_tool_textures, selected_material,
    ChunkDebugOverlayEntry, ToolKind, UiState, HOTBAR_SLOTS,
};
use crate::world::{AreaFootprintShape, BrushMode, BrushSettings, BrushShape, CHUNK_SIZE, EMPTY};
use glam::{Mat4, Vec3, Vec4};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

const RADIAL_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyE;
const RADIAL_MENU_TOGGLE_LABEL: &str = "E";
const TOOL_QUICK_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyQ;
const TOOL_TEXTURES_DIR: &str = "assets/tools";
const REMESH_JOB_BUDGET_PER_FRAME_BASE: usize = 32;
const REMESH_JOB_BUDGET_PER_FRAME_MIN: usize = 16;
const REMESH_JOB_BUDGET_PER_FRAME_MAX: usize = 160;
const MESH_UPLOAD_BYTES_MIN_PER_FRAME: usize = 512 * 1024;
const MESH_UPLOAD_BYTES_BASE_PER_FRAME: usize = 2 * 1024 * 1024;
const MESH_UPLOAD_BYTES_MAX_PER_FRAME: usize = 8 * 1024 * 1024;
const FRAME_TIME_TARGET_MS: f32 = 1000.0 / 60.0;
const MAINTENANCE_TICK_MIN_MS: f32 = 8.0;
const MAINTENANCE_TICK_MAX_MS: f32 = 48.0;
const MAINTENANCE_PRESSURE_PENDING_FINALIZE_START: usize = 32;
const MAINTENANCE_PRESSURE_PENDING_FINALIZE_HIGH: usize = 256;
const MAINTENANCE_PRESSURE_DISPATCH_QUEUE_START: usize = 24;
const MAINTENANCE_PRESSURE_DISPATCH_QUEUE_HIGH: usize = 192;
const MAINTENANCE_PRESSURE_COMPLETED_BACKLOG_START: usize = 12;
const MAINTENANCE_PRESSURE_COMPLETED_BACKLOG_HIGH: usize = 96;
const FINALIZE_BACKPRESSURE_PENDING_START: usize = 96;
const FINALIZE_BACKPRESSURE_PENDING_HIGH: usize = 320;
const FINALIZE_BACKPRESSURE_PROMOTION_LOW_WATERMARK: usize = 2;
const FINALIZE_BACKPRESSURE_LOW_PROGRESS_STREAK_MAX: u32 = 180;
const FIXED_SIM_STEP_SECONDS: f32 = 1.0 / 60.0;
const SIMULATION_RADIUS_CHUNKS: i32 = 1; // 3x3x3 = 27 chunks max
const SIM_REGION_RECOMPUTE_CHUNK_DELTA: i32 = 2;

const APPLY_BUDGET_MS: f32 = 1.5;
const APPLY_NEAR_PROTECTED_BUDGET_MS: f32 = 1.0;
const EVICT_BUDGET_MS: f32 = 1.0;
const RESIDENT_KEEP_MID_CAP: usize = 512;
const RESIDENT_KEEP_FAR_CAP: usize = 640;
const MESH_BACKPRESSURE_START: usize = 80;
const MESH_BACKPRESSURE_HIGH: usize = 180;
const DIRTY_BACKLOG_PRESSURE_START: usize = 96;
const DIRTY_BACKLOG_PRESSURE_HIGH: usize = 320;
const URGENT_GENERATION_BUDGET: usize = 4;
const PROTECTED_HIGH_PRIORITY_SLOTS: usize = 2;
const AUTO_TUNE_UPLOAD_LATENCY_START_MS: f32 = 200.0;
const AUTO_TUNE_UPLOAD_LATENCY_HIGH_MS: f32 = 450.0;
const AUTO_TUNE_RAMP_UP_PER_SEC: f32 = 3.5;
const AUTO_TUNE_RECOVER_PER_SEC: f32 = 0.6;
const DESIRED_VIEW_RECOMPUTE_DOT_DELTA: f32 = 0.01;
const DISPATCH_STARVATION_STREAK_FRAMES: u32 = 24;
const APPLY_STARVATION_STREAK_FRAMES: u32 = 24;
const CONVERGENCE_STALL_STREAK_FRAMES: u32 = 90;
const LOW_LOADED_HIGH_BACKLOG_THRESHOLD: usize = 64;
const LOW_LOADED_CHUNKS_THRESHOLD: usize = 24;

const COLLISION_SAFETY_RADIUS_VOXELS: i32 = 4;
const COLLISION_LOCAL_PRIORITY_REQUEST_BUDGET: usize = 12;
const COLLISION_URGENT_DISPATCH_BOOST: usize = 12;
const COLLISION_STRICT_FALLBACK_WINDOW_SECS: f32 = 0.35;
const COLLISION_TIMEOUT_DAMPING_FACTOR: f32 = 0.78;
const HITCH_CAPTURE_FRAME_MS: f32 = 120.0;
const HITCH_CAPTURE_RING_SIZE: usize = 64;
const SPAWN_SEARCH_RADIUS: i32 = 48;
const SPAWN_HEADROOM: i32 = 4;
const SPAWN_FALLBACK_EXTRA_HEIGHT: i32 = 12;
const SPAWN_CEILING_PROBE_HEIGHT: i32 = 20;
const SPAWN_MAX_SLOPE_DELTA: i32 = 3;
const SPAWN_RETRY_HEIGHT_ABOVE_SURFACE: i32 = 6;
const WORLD_SEED_SAVE_PATH: &str = ".world_seed";
const MAX_CACHED_MODIFIED: usize = 2048;

#[derive(Clone, Copy, Debug)]
enum SeedSource {
    Cli,
    Env,
    SaveState,
    Generated,
}

impl SeedSource {
    fn label(self) -> &'static str {
        match self {
            SeedSource::Cli => "cli",
            SeedSource::Env => "env",
            SeedSource::SaveState => "save-state",
            SeedSource::Generated => "generated",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SeedSelection {
    seed: u64,
    source: SeedSource,
}

#[derive(Clone, Debug)]
enum SpawnPendingReason {
    Searching,
    MissingNeighborhood,
    BlockedCapsule,
    NoValidColumn,
    UsingFallbackBand,
}

impl SpawnPendingReason {
    fn label(&self) -> &'static str {
        match self {
            SpawnPendingReason::Searching => "searching",
            SpawnPendingReason::MissingNeighborhood => "missing neighborhood",
            SpawnPendingReason::BlockedCapsule => "blocked capsule",
            SpawnPendingReason::NoValidColumn => "no valid column",
            SpawnPendingReason::UsingFallbackBand => "fallback above loaded surface",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SpawnCandidate {
    voxel: VoxelCoord,
    surface_y: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ModifiedChunkCacheMetrics {
    hits: u64,
    misses: u64,
    evictions: u64,
}

struct ModifiedChunkCache {
    max_entries: usize,
    entries: HashMap<ChunkCoord, crate::chunk_store::Chunk>,
    recency: VecDeque<ChunkCoord>,
    metrics: ModifiedChunkCacheMetrics,
}

impl ModifiedChunkCache {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            entries: HashMap::new(),
            recency: VecDeque::new(),
            metrics: ModifiedChunkCacheMetrics::default(),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
    }

    fn insert(&mut self, coord: ChunkCoord, chunk: crate::chunk_store::Chunk) {
        self.entries.insert(coord, chunk);
        self.touch(coord);
        self.enforce_limit();
    }

    fn take(&mut self, coord: ChunkCoord) -> Option<crate::chunk_store::Chunk> {
        let chunk = self.entries.remove(&coord);
        if chunk.is_some() {
            self.metrics.hits += 1;
            self.remove_from_recency(coord);
        } else {
            self.metrics.misses += 1;
        }
        chunk
    }

    fn metrics(&self) -> ModifiedChunkCacheMetrics {
        self.metrics
    }

    fn touch(&mut self, coord: ChunkCoord) {
        self.remove_from_recency(coord);
        self.recency.push_back(coord);
    }

    fn remove_from_recency(&mut self, coord: ChunkCoord) {
        if let Some(index) = self.recency.iter().position(|queued| *queued == coord) {
            self.recency.remove(index);
        }
    }

    fn enforce_limit(&mut self) {
        while self.entries.len() > self.max_entries {
            let Some(oldest) = self.recency.pop_front() else {
                break;
            };
            if self.entries.remove(&oldest).is_some() {
                self.metrics.evictions += 1;
            }
        }
    }
}

#[derive(Clone)]
struct GenJob {
    coord: ChunkCoord,
    requested_at: Instant,
    class: GenerateJobClass,
    version: u64,
}

struct GenResult {
    coord: ChunkCoord,
    chunk: crate::chunk_store::Chunk,
    requested_at: Instant,
    generated_at: Instant,
    class: GenerateJobClass,
    version: u64,
}

#[derive(Clone, Copy)]
struct GeneratorConfig {
    worker_count: usize,
    queue_bound: usize,
    dispatch_high: usize,
    dispatch_low: usize,
}

fn generator_config_from_hardware() -> GeneratorConfig {
    let cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let worker_count = cores.clamp(2, 12);
    let queue_bound = if cores <= 4 {
        128
    } else if cores <= 8 {
        256
    } else {
        384
    };
    GeneratorConfig {
        worker_count,
        queue_bound,
        dispatch_high: ((queue_bound as f32) * 0.9) as usize,
        dispatch_low: ((queue_bound as f32) * 0.5) as usize,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StreamingTuning {
    near_radius_xz: i32,
    mid_radius_xz: i32,
    far_radius_xz: i32,
    ultra_radius_xz: i32,
    vertical_radius: i32,
    lod_hysteresis: i32,
    base_apply_budget_items: usize,
    max_apply_budget_items: usize,
    base_generate_drain_items: usize,
    max_generate_drain_items: usize,
    lod_budget_near: usize,
    lod_budget_mid: usize,
    lod_budget_far: usize,
    lod_budget_ultra: usize,
}

impl Default for StreamingTuning {
    fn default() -> Self {
        Self {
            near_radius_xz: 6,
            mid_radius_xz: 12,
            far_radius_xz: 16,
            ultra_radius_xz: 24,
            vertical_radius: 4,
            lod_hysteresis: 1,
            base_apply_budget_items: 8,
            max_apply_budget_items: 48,
            base_generate_drain_items: 32,
            max_generate_drain_items: 160,
            lod_budget_near: 10,
            lod_budget_mid: 8,
            lod_budget_far: 12,
            lod_budget_ultra: 8,
        }
    }
}

impl StreamingTuning {
    fn normalized(mut self) -> Self {
        self.near_radius_xz = self.near_radius_xz.max(0);
        self.mid_radius_xz = self.mid_radius_xz.max(self.near_radius_xz);
        self.far_radius_xz = self.far_radius_xz.max(self.mid_radius_xz);
        self.ultra_radius_xz = self.ultra_radius_xz.max(self.far_radius_xz);
        self.vertical_radius = self.vertical_radius.max(0);
        self.lod_hysteresis = self.lod_hysteresis.max(0);
        self.base_apply_budget_items = self.base_apply_budget_items.max(1);
        self.max_apply_budget_items = self
            .max_apply_budget_items
            .max(self.base_apply_budget_items);
        self.base_generate_drain_items = self.base_generate_drain_items.max(1);
        self.max_generate_drain_items = self
            .max_generate_drain_items
            .max(self.base_generate_drain_items);
        self.lod_budget_near = self.lod_budget_near.max(1);
        self.lod_budget_mid = self.lod_budget_mid.max(1);
        self.lod_budget_far = self.lod_budget_far.max(1);
        self.lod_budget_ultra = self.lod_budget_ultra.max(1);
        self
    }
}

fn scaled_budget(base: usize, max: usize, backlog: usize, threshold: usize) -> usize {
    if backlog <= threshold {
        return base;
    }
    let pressure = ((backlog - threshold) / threshold.max(1)).min(8);
    (base + base * pressure).min(max)
}

#[derive(Clone, Copy, Debug)]
struct AutoTuneState {
    degrade_level: f32,
    latency_pressure: f32,
    queue_pressure: f32,
    dirty_pressure: f32,
}

impl Default for AutoTuneState {
    fn default() -> Self {
        Self {
            degrade_level: 0.0,
            latency_pressure: 0.0,
            queue_pressure: 0.0,
            dirty_pressure: 0.0,
        }
    }
}

impl AutoTuneState {
    fn update(
        &mut self,
        dt_seconds: f32,
        dirty_backlog: usize,
        meshing_queue_depth: usize,
        upload_latency_ms: f32,
    ) {
        self.latency_pressure = ((upload_latency_ms - AUTO_TUNE_UPLOAD_LATENCY_START_MS)
            / (AUTO_TUNE_UPLOAD_LATENCY_HIGH_MS - AUTO_TUNE_UPLOAD_LATENCY_START_MS))
            .clamp(0.0, 1.0);
        self.queue_pressure = ((meshing_queue_depth as f32 - MESH_BACKPRESSURE_START as f32)
            / (MESH_BACKPRESSURE_HIGH as f32 - MESH_BACKPRESSURE_START as f32))
            .clamp(0.0, 1.0);
        self.dirty_pressure = ((dirty_backlog as f32 - DIRTY_BACKLOG_PRESSURE_START as f32)
            / (DIRTY_BACKLOG_PRESSURE_HIGH as f32 - DIRTY_BACKLOG_PRESSURE_START as f32))
            .clamp(0.0, 1.0);

        let target = self
            .latency_pressure
            .max(self.queue_pressure * 0.8)
            .max(self.dirty_pressure * 0.65);
        if target > self.degrade_level {
            self.degrade_level =
                (self.degrade_level + AUTO_TUNE_RAMP_UP_PER_SEC * dt_seconds).min(target);
        } else {
            self.degrade_level =
                (self.degrade_level - AUTO_TUNE_RECOVER_PER_SEC * dt_seconds).max(target);
        }
    }

    fn is_active(self) -> bool {
        self.degrade_level > 0.01
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DesiredCapStats {
    near_kept: usize,
    near_dropped: usize,
    mid_kept: usize,
    mid_dropped: usize,
    far_kept: usize,
    far_dropped: usize,
    ultra_kept: usize,
    ultra_dropped: usize,
    uncapped_kept: usize,
    budget_dropped: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GenerationPressureModel {
    near_cap: usize,
    mid_cap: usize,
    far_cap: usize,
    ultra_cap: usize,
    global_cap: usize,
    near_dispatch_budget: usize,
    mid_dispatch_budget: usize,
    far_dispatch_budget: usize,
}

fn shell_span(start: i32, end: i32) -> usize {
    (end - start).max(1) as usize
}

fn compute_generation_pressure_model(
    effective_stream_tuning: &StreamingTuning,
    pending_generate_count: usize,
    worker_inflight: usize,
    apply_budget_items: usize,
    prior_mesh_backlog: usize,
    prior_meshing_queue_depth: usize,
    base_generate_drain_budget: usize,
) -> GenerationPressureModel {
    let queue_depth = pending_generate_count + worker_inflight;
    let queue_pressure =
        (queue_depth as f32 / (apply_budget_items.max(1) * 6) as f32).clamp(0.0, 1.0);
    let mesh_pressure = ((prior_mesh_backlog as f32 - MESH_BACKPRESSURE_START as f32)
        / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
        .clamp(0.0, 1.0);
    let meshing_queue_pressure = ((prior_meshing_queue_depth as f32
        - MESH_BACKPRESSURE_START as f32)
        / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
        .clamp(0.0, 1.0);
    let pressure = queue_pressure
        .max(mesh_pressure * 0.9)
        .max(meshing_queue_pressure * 0.75);
    let cap_scale = (1.0 - pressure * 0.75).clamp(0.3, 1.0);

    let near_span = shell_span(0, effective_stream_tuning.near_radius_xz + 1);
    let mid_span = shell_span(
        effective_stream_tuning.near_radius_xz,
        effective_stream_tuning.mid_radius_xz,
    );
    let far_span = shell_span(
        effective_stream_tuning.mid_radius_xz,
        effective_stream_tuning.far_radius_xz,
    );
    let ultra_span = shell_span(
        effective_stream_tuning.far_radius_xz,
        effective_stream_tuning.ultra_radius_xz,
    );

    let near_floor = (apply_budget_items * 2).max(PROTECTED_HIGH_PRIORITY_SLOTS * 3);
    let base_global = apply_budget_items.max(1) * 10 + base_generate_drain_budget * 2;
    let global_cap = ((base_global as f32) * cap_scale) as usize + near_floor;

    let span_sum = near_span + mid_span + far_span + ultra_span;
    let mut near_cap = (global_cap * near_span / span_sum).max(near_floor);
    let mut mid_cap = (global_cap * mid_span / span_sum).max(1);
    let mut far_cap = (global_cap * far_span / span_sum).max(1);
    let mut ultra_cap = (global_cap * ultra_span / span_sum).max(1);

    let mut total = near_cap + mid_cap + far_cap + ultra_cap;
    while total > global_cap {
        if ultra_cap > 1 {
            ultra_cap -= 1;
        } else if far_cap > 1 {
            far_cap -= 1;
        } else if mid_cap > 1 {
            mid_cap -= 1;
        } else if near_cap > near_floor {
            near_cap -= 1;
        } else {
            break;
        }
        total -= 1;
    }

    let dispatch_budget = (((base_generate_drain_budget as f32) * (1.0 + pressure * 0.35)).round()
        as usize)
        .max(PROTECTED_HIGH_PRIORITY_SLOTS)
        .min(effective_stream_tuning.max_generate_drain_items);
    let stable_near = queue_pressure < 0.4 && dispatch_budget > 2;
    let near_dispatch_budget = if stable_near {
        (dispatch_budget as f32 * 0.6).round() as usize
    } else {
        (dispatch_budget as f32 * 0.82).round() as usize
    }
    .clamp(PROTECTED_HIGH_PRIORITY_SLOTS, dispatch_budget);

    let mut remaining_dispatch = dispatch_budget.saturating_sub(near_dispatch_budget);
    let mut mid_dispatch_budget = ((remaining_dispatch as f32) * 0.55).round() as usize;
    let mut far_dispatch_budget = remaining_dispatch.saturating_sub(mid_dispatch_budget);
    if stable_near && remaining_dispatch > 0 {
        mid_dispatch_budget = mid_dispatch_budget.max(1);
        far_dispatch_budget = far_dispatch_budget.max(1);
    }
    if mid_dispatch_budget + far_dispatch_budget > remaining_dispatch {
        let overflow = mid_dispatch_budget + far_dispatch_budget - remaining_dispatch;
        far_dispatch_budget = far_dispatch_budget.saturating_sub(overflow);
    }
    remaining_dispatch = dispatch_budget
        .saturating_sub(near_dispatch_budget)
        .saturating_sub(mid_dispatch_budget)
        .saturating_sub(far_dispatch_budget);
    far_dispatch_budget += remaining_dispatch;

    GenerationPressureModel {
        near_cap,
        mid_cap,
        far_cap,
        ultra_cap,
        global_cap,
        near_dispatch_budget,
        mid_dispatch_budget,
        far_dispatch_budget,
    }
}

fn chebyshev_from_player(player_chunk: ChunkCoord, coord: ChunkCoord) -> i32 {
    (coord.x - player_chunk.x)
        .abs()
        .max((coord.y - player_chunk.y).abs())
        .max((coord.z - player_chunk.z).abs())
}

fn extract_frustum_planes(vp: Mat4) -> [Vec4; 6] {
    let m = vp.transpose().to_cols_array();
    let row = |idx: usize| Vec4::new(m[idx], m[4 + idx], m[8 + idx], m[12 + idx]);
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
    for plane in &mut planes {
        let normal = plane.truncate();
        let inv_len = normal.length_recip();
        if inv_len.is_finite() {
            *plane *= inv_len;
        }
    }
    planes
}

fn cap_desired_generation_order(
    desired: &DesiredChunks,
    player_chunk: ChunkCoord,
    pressure_model: GenerationPressureModel,
) -> (Vec<ChunkCoord>, DesiredCapStats) {
    let mut prioritized = Vec::with_capacity(desired.generation_order.len());
    let mut stats = DesiredCapStats::default();

    let mut push_with_cap =
        |coords: &[ChunkCoord], cap: usize, kept: &mut usize, dropped: &mut usize| {
            for &coord in coords {
                if prioritized.len() >= pressure_model.global_cap || *kept >= cap {
                    *dropped += 1;
                    continue;
                }
                *kept += 1;
                prioritized.push(coord);
            }
        };

    let mut urgent_near = Vec::new();
    let mut regular_near = Vec::new();
    for &coord in &desired.near {
        if chebyshev_from_player(player_chunk, coord) <= 1 {
            stats.uncapped_kept += 1;
            urgent_near.push(coord);
        } else {
            regular_near.push(coord);
        }
    }

    push_with_cap(
        &urgent_near,
        pressure_model.near_cap,
        &mut stats.near_kept,
        &mut stats.near_dropped,
    );
    push_with_cap(
        &regular_near,
        pressure_model.near_cap,
        &mut stats.near_kept,
        &mut stats.near_dropped,
    );
    push_with_cap(
        &desired.mid,
        pressure_model.mid_cap,
        &mut stats.mid_kept,
        &mut stats.mid_dropped,
    );
    push_with_cap(
        &desired.far,
        pressure_model.far_cap,
        &mut stats.far_kept,
        &mut stats.far_dropped,
    );
    push_with_cap(
        &desired.ultra,
        pressure_model.ultra_cap,
        &mut stats.ultra_kept,
        &mut stats.ultra_dropped,
    );

    stats.budget_dropped =
        stats.near_dropped + stats.mid_dropped + stats.far_dropped + stats.ultra_dropped;

    (prioritized, stats)
}

fn blend_i32(max: i32, min: i32, t: f32) -> i32 {
    let min = min.min(max);
    ((max as f32) - ((max - min) as f32 * t.clamp(0.0, 1.0))).round() as i32
}

fn blend_usize(max: usize, min: usize, t: f32) -> usize {
    let min = min.min(max);
    ((max as f32) - ((max - min) as f32 * t.clamp(0.0, 1.0))).round() as usize
}

fn adaptive_mesh_upload_budget(last_frame_ms: f32, queue_pressure: usize) -> usize {
    let frame_headroom =
        ((FRAME_TIME_TARGET_MS - last_frame_ms) / FRAME_TIME_TARGET_MS).clamp(-1.0, 1.0);
    let headroom_scale = (1.0 + frame_headroom * 0.5).clamp(0.6, 1.5);

    let pressure = if queue_pressure <= MESH_BACKPRESSURE_START {
        0.0
    } else {
        ((queue_pressure - MESH_BACKPRESSURE_START) as f32
            / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
            .clamp(0.0, 1.0)
    };
    let pressure_scale = 1.0 + pressure * 2.0;

    ((MESH_UPLOAD_BYTES_BASE_PER_FRAME as f32 * headroom_scale * pressure_scale) as usize).clamp(
        MESH_UPLOAD_BYTES_MIN_PER_FRAME,
        MESH_UPLOAD_BYTES_MAX_PER_FRAME,
    )
}

fn adaptive_remesh_job_budget(
    last_frame_ms: f32,
    dirty_backlog: usize,
    meshing_queue_depth: usize,
    visible_chunks: usize,
) -> usize {
    let frame_headroom =
        ((FRAME_TIME_TARGET_MS - last_frame_ms) / FRAME_TIME_TARGET_MS).clamp(-1.0, 1.0);
    let headroom_scale = (1.0 + frame_headroom * 0.5).clamp(0.7, 1.65);
    let dirty_pressure = ((dirty_backlog as f32 - DIRTY_BACKLOG_PRESSURE_START as f32)
        / (DIRTY_BACKLOG_PRESSURE_HIGH - DIRTY_BACKLOG_PRESSURE_START) as f32)
        .clamp(0.0, 1.0);
    let queue_pressure = ((meshing_queue_depth as f32 - MESH_BACKPRESSURE_START as f32)
        / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
        .clamp(0.0, 1.0);
    let visible_floor = (visible_chunks / 3).max(REMESH_JOB_BUDGET_PER_FRAME_BASE / 2);
    let pressure_boost = 1.0 + dirty_pressure * 1.6 + queue_pressure * 1.35;

    ((REMESH_JOB_BUDGET_PER_FRAME_BASE as f32 * headroom_scale * pressure_boost) as usize)
        .max(visible_floor)
        .clamp(
            REMESH_JOB_BUDGET_PER_FRAME_MIN,
            REMESH_JOB_BUDGET_PER_FRAME_MAX,
        )
}

#[derive(Clone, Copy, Debug)]
struct MaintenanceThrottleState {
    active: bool,
    severity: f32,
    remesh_scale: f32,
    upload_scale: f32,
    interval_scale: f32,
    reason: &'static str,
}

impl Default for MaintenanceThrottleState {
    fn default() -> Self {
        Self {
            active: false,
            severity: 0.0,
            remesh_scale: 1.0,
            upload_scale: 1.0,
            interval_scale: 1.0,
            reason: "normal",
        }
    }
}

fn compute_maintenance_throttle_state(
    pending_finalize_depth: usize,
    promoted_to_drawable: usize,
    low_progress_streak: u32,
) -> MaintenanceThrottleState {
    let pending_pressure = normalized_queue_pressure(
        pending_finalize_depth,
        FINALIZE_BACKPRESSURE_PENDING_START,
        FINALIZE_BACKPRESSURE_PENDING_HIGH,
    );
    if pending_pressure <= 0.0 {
        return MaintenanceThrottleState::default();
    }

    let promotion_ratio = promoted_to_drawable as f32 / pending_finalize_depth.max(1) as f32;
    let streak_ratio = (low_progress_streak as f32
        / FINALIZE_BACKPRESSURE_LOW_PROGRESS_STREAK_MAX as f32)
        .clamp(0.0, 1.0);
    let promotion_pressure = if promoted_to_drawable
        <= FINALIZE_BACKPRESSURE_PROMOTION_LOW_WATERMARK
        || promotion_ratio <= 0.015
    {
        1.0
    } else {
        (1.0 - (promotion_ratio / 0.06)).clamp(0.0, 1.0)
    };
    let severity =
        (pending_pressure * (0.45 + 0.55 * streak_ratio) * promotion_pressure).clamp(0.0, 1.0);
    if severity <= 0.0 {
        return MaintenanceThrottleState::default();
    }

    MaintenanceThrottleState {
        active: true,
        severity,
        remesh_scale: (1.0 - 0.65 * severity).clamp(0.35, 1.0),
        upload_scale: (1.0 - 0.45 * severity).clamp(0.55, 1.0),
        interval_scale: (1.0 + 0.35 * severity).clamp(1.0, 1.35),
        reason: "pending_finalize_high_and_promotions_low",
    }
}

fn normalized_queue_pressure(depth: usize, start: usize, high: usize) -> f32 {
    if depth <= start {
        0.0
    } else {
        ((depth - start) as f32 / (high.saturating_sub(start).max(1)) as f32).clamp(0.0, 1.0)
    }
}

fn adaptive_maintenance_interval_ms(
    pending_finalize_depth: usize,
    dispatch_queue_depth: usize,
    completed_mesh_backlog: usize,
    last_frame_ms: f32,
    interval_scale: f32,
) -> f32 {
    let pending_finalize_pressure = normalized_queue_pressure(
        pending_finalize_depth,
        MAINTENANCE_PRESSURE_PENDING_FINALIZE_START,
        MAINTENANCE_PRESSURE_PENDING_FINALIZE_HIGH,
    );
    let dispatch_pressure = normalized_queue_pressure(
        dispatch_queue_depth,
        MAINTENANCE_PRESSURE_DISPATCH_QUEUE_START,
        MAINTENANCE_PRESSURE_DISPATCH_QUEUE_HIGH,
    );
    let completed_backlog_pressure = normalized_queue_pressure(
        completed_mesh_backlog,
        MAINTENANCE_PRESSURE_COMPLETED_BACKLOG_START,
        MAINTENANCE_PRESSURE_COMPLETED_BACKLOG_HIGH,
    );
    let work_pressure = pending_finalize_pressure
        .max(dispatch_pressure)
        .max(completed_backlog_pressure);

    let base_interval = MAINTENANCE_TICK_MAX_MS
        - (MAINTENANCE_TICK_MAX_MS - MAINTENANCE_TICK_MIN_MS) * work_pressure;
    let frame_pressure = (last_frame_ms / FRAME_TIME_TARGET_MS).clamp(0.6, 2.2);
    let frame_scale = if frame_pressure > 1.0 {
        1.0 + (frame_pressure - 1.0) * 0.7
    } else {
        1.0 - (1.0 - frame_pressure) * 0.2
    };

    (base_interval * frame_scale * interval_scale.max(1.0))
        .clamp(MAINTENANCE_TICK_MIN_MS, MAINTENANCE_TICK_MAX_MS)
}

fn should_force_maintenance_tick(
    pending_finalize_depth: usize,
    dispatch_queue_depth: usize,
    completed_mesh_backlog: usize,
) -> bool {
    pending_finalize_depth >= MAINTENANCE_PRESSURE_PENDING_FINALIZE_HIGH
        || dispatch_queue_depth >= MAINTENANCE_PRESSURE_DISPATCH_QUEUE_HIGH
        || completed_mesh_backlog >= MAINTENANCE_PRESSURE_COMPLETED_BACKLOG_HIGH
}

fn adaptive_redraw_interval_ms(maintenance_interval_ms: f32, last_frame_ms: f32) -> f32 {
    let budget_pressure = (last_frame_ms / FRAME_TIME_TARGET_MS).clamp(0.7, 2.4);
    let budget_scale = if budget_pressure > 1.0 {
        1.0 + (budget_pressure - 1.0) * 0.45
    } else {
        1.0
    };
    let blended =
        FRAME_TIME_TARGET_MS + (maintenance_interval_ms - FRAME_TIME_TARGET_MS).max(0.0) * 0.35;
    (blended * budget_scale).clamp(FRAME_TIME_TARGET_MS, MAINTENANCE_TICK_MAX_MS)
}

fn coord_in_frustum(coord: ChunkCoord, frustum_planes: &[Vec4; 6]) -> bool {
    let center = Vec3::new(
        coord.x as f32 + 0.5,
        coord.y as f32 + 0.5,
        coord.z as f32 + 0.5,
    );
    frustum_planes.iter().all(|plane| {
        plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w >= -0.25
    })
}

struct BackgroundGenerator {
    tx: SyncSender<GenJob>,
    rx: Receiver<GenResult>,
}

impl BackgroundGenerator {
    fn new(seed: u64, config: GeneratorConfig) -> Self {
        let (tx, job_rx) = sync_channel::<GenJob>(config.queue_bound);
        let (result_tx, rx) = sync_channel::<GenResult>(config.queue_bound);
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

        for i in 0..config.worker_count {
            let worker_rx = std::sync::Arc::clone(&job_rx);
            let worker_tx = result_tx.clone();
            thread::Builder::new()
                .name(format!("chunk-gen-{i}"))
                .spawn(move || loop {
                    let job = {
                        let lock = worker_rx.lock().expect("gen worker rx lock");
                        lock.recv()
                    };
                    let Ok(job) = job else {
                        break;
                    };
                    let chunk = generate_chunk(seed, job.coord);
                    if worker_tx
                        .send(GenResult {
                            coord: job.coord,
                            chunk,
                            requested_at: job.requested_at,
                            generated_at: Instant::now(),
                            class: job.class,
                            version: job.version,
                        })
                        .is_err()
                    {
                        break;
                    }
                })
                .expect("spawn chunk generation worker");
        }

        Self { tx, rx }
    }

    fn try_request(&self, coord: ChunkCoord, class: GenerateJobClass, version: u64) -> bool {
        match self.tx.try_send(GenJob {
            coord,
            requested_at: Instant::now(),
            class,
            version,
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    fn try_recv(&self) -> Result<GenResult, TryRecvError> {
        self.rx.try_recv()
    }
}

fn resolve_world_seed() -> SeedSelection {
    if let Some(seed) = seed_from_cli() {
        let _ = persist_seed(seed);
        return SeedSelection {
            seed,
            source: SeedSource::Cli,
        };
    }

    if let Ok(value) = std::env::var("FALLING_SAND_WORLD_SEED") {
        if let Ok(seed) = value.trim().parse::<u64>() {
            let _ = persist_seed(seed);
            return SeedSelection {
                seed,
                source: SeedSource::Env,
            };
        }
    }

    if let Some(seed) = seed_from_save_state() {
        return SeedSelection {
            seed,
            source: SeedSource::SaveState,
        };
    }

    let seed = generate_startup_seed();
    let _ = persist_seed(seed);
    SeedSelection {
        seed,
        source: SeedSource::Generated,
    }
}

fn seed_from_cli() -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        if let Some(value) = arg.strip_prefix("--seed=") {
            if let Ok(seed) = value.trim().parse::<u64>() {
                return Some(seed);
            }
        }
        if let Some(value) = arg.strip_prefix("--world-seed=") {
            if let Ok(seed) = value.trim().parse::<u64>() {
                return Some(seed);
            }
        }
        if (arg == "--seed" || arg == "--world-seed") && i + 1 < args.len() {
            if let Ok(seed) = args[i + 1].trim().parse::<u64>() {
                return Some(seed);
            }
        }
    }
    None
}

fn cli_has_flag(name: &str) -> bool {
    std::env::args().any(|arg| arg == name)
}

fn seed_from_save_state() -> Option<u64> {
    fs::read_to_string(Path::new(WORLD_SEED_SAVE_PATH))
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
}

fn persist_seed(seed: u64) -> std::io::Result<()> {
    fs::write(Path::new(WORLD_SEED_SAVE_PATH), seed.to_string())
}

fn generate_startup_seed() -> u64 {
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let pid = std::process::id() as u64;
    let mixed = now_ns ^ pid.rotate_left(17) ^ 0xA5A5_5A5A_D3C1_BEEF;
    let mut h = mixed ^ (mixed >> 33);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h.max(1)
}

#[derive(Default)]
struct EditRuntimeState {
    last_edit_at: Option<Instant>,
    last_edit_mode: Option<BrushMode>,
}

#[derive(Clone, Copy, Debug)]
struct RaycastResult {
    hit: Option<[i32; 3]>,
    place: [i32; 3],
}

pub async fn run() -> anyhow::Result<()> {
    let mut sim_running = false;
    let mut step_once = false;

    let event_loop = EventLoop::new()?;
    let window: &'static winit::window::Window = Box::leak(Box::new(
        WindowBuilder::new()
            .with_title("3D Falling Sand Prototype")
            .build(&event_loop)?,
    ));

    let require_gpu_meshing = cli_has_flag("--require-gpu-meshing");
    let force_disabled_meshing = cli_has_flag("--disable-meshing");
    let mut renderer = Renderer::new(window, require_gpu_meshing, force_disabled_meshing).await?;
    let egui_ctx = egui::Context::default();
    egui_ctx.set_visuals(egui::Visuals::dark());
    let mut egui_state =
        egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, window, None, None);
    let mut egui_rpass =
        egui_wgpu::Renderer::new(&renderer.device, renderer.config.format, None, 1);

    let mut store = ChunkStore::new();
    let seed_selection = resolve_world_seed();
    println!(
        "[world] using seed {} (source: {})",
        seed_selection.seed,
        seed_selection.source.label()
    );
    let mut streaming = ChunkStreaming::new(seed_selection.seed);
    let generator_config = generator_config_from_hardware();
    let mut chunk_generator = BackgroundGenerator::new(streaming.seed, generator_config);
    let mut generated_ready: VecDeque<GenResult> = VecDeque::new();
    let mut stream_tuning = StreamingTuning::default().normalized();
    let mut cached_stream_tuning = stream_tuning.clone();
    let mut auto_tune = AutoTuneState::default();
    let mut rng = Rng::new(0x1234_5678);
    let mut simulation_runtime = SimulationRuntime::default();
    let mut sim_acc = 0.0f32;

    let mut input = InputState::default();
    let mut ctrl = FpsController {
        flying: true,
        ..Default::default()
    };
    let mut ui = UiState::default();
    ui.startup_backend_label = match renderer.startup_diagnostics.backend_selected {
        crate::gpu_compute::MeshPipelineBackend::Disabled => "disabled",
        crate::gpu_compute::MeshPipelineBackend::Cpu => "cpu",
        #[cfg(feature = "gpu-compute")]
        crate::gpu_compute::MeshPipelineBackend::Gpu => "gpu",
    }
    .to_string();
    ui.startup_required_limits = renderer.startup_diagnostics.required_limits_summary.clone();
    ui.startup_adapter_limits = renderer.startup_diagnostics.adapter_limits_summary.clone();
    ui.startup_error_message = renderer.startup_diagnostics.startup_error.clone();
    if renderer.mesh_backend == crate::gpu_compute::MeshPipelineBackend::Disabled
        && ui.startup_error_message.is_none()
    {
        ui.startup_error_message = Some(
            "Meshing backend is disabled via --disable-meshing; world rendering is limited."
                .to_string(),
        );
    }
    let mut brush = BrushSettings::default();
    let tool_textures = load_tool_textures(&egui_ctx, TOOL_TEXTURES_DIR);
    let mut edit_runtime = EditRuntimeState::default();

    let mut last = Instant::now();
    let start = Instant::now();
    let mut cursor_is_unlocked = false;
    let mut cursor_position_known = false;

    let mut floating_origin_state = FloatingOriginState::new();
    let mut origin_voxel = floating_origin_state.origin_translation;
    let floating_origin_config = FloatingOriginConfig::default();
    let mut cached_modified_chunks = ModifiedChunkCache::new(MAX_CACHED_MODIFIED);
    let mut cached_modified_last_metrics = ModifiedChunkCacheMetrics::default();
    let mut preview_block_list: Vec<[i32; 3]> = Vec::new();

    // === streaming/perf caches (MUST live outside RedrawRequested) ===
    let mut last_player_chunk: Option<ChunkCoord> = None;
    let mut last_sim_region_player_anchor: Option<ChunkCoord> = None;
    let mut cached_sim_region_emitters: HashSet<ChunkCoord> = HashSet::new();
    let mut cached_sim_region: HashSet<ChunkCoord> =
        chunk_cube(ChunkCoord { x: 0, y: 0, z: 0 }, SIMULATION_RADIUS_CHUNKS);
    let mut cached_gas_sim_region = cached_sim_region.clone();
    let mut last_desired_look_dir: Option<Vec3> = None;
    let mut desired_recompute_reason = String::from("init");
    let mut cached_desired: DesiredChunks = ChunkStreaming::desired_set(
        ChunkCoord { x: 0, y: 0, z: 0 },
        Vec3::ZERO,
        Vec3::Z,
        None,
        stream_tuning.near_radius_xz,
        stream_tuning.mid_radius_xz,
        Some(stream_tuning.far_radius_xz),
        Some(stream_tuning.ultra_radius_xz),
        stream_tuning.vertical_radius,
        RESIDENT_KEEP_MID_CAP,
        RESIDENT_KEEP_FAR_CAP,
        &HashMap::new(),
        0,
    );
    let mut stream_debug = String::new();
    let mut frame_counter: u64 = 0;
    let mut total_generated_chunks: u64 = 0;
    let collision_freeze_active = false;
    let mut collision_used_unloaded_chunks = false;
    let mut collision_unknown_samples = 0usize;
    let mut collision_unknown_blocks_as_solid = 0usize;
    let mut collision_axis_reverts = 0usize;
    let mut collision_missing_since: HashMap<ChunkCoord, f32> = HashMap::new();
    let mut collision_local_urgent_boost = 0usize;
    let mut collision_soft_timeout_active = false;
    let mut collision_blocked_unloaded_count_total = 0u64;
    let mut collision_blocked_unloaded_ms_total = 0.0f32;
    let mut last_player_world_voxel = local_to_world_voxel(ctrl.position, origin_voxel);
    let mut prior_mesh_backlog = 0usize;
    let mut prior_dirty_backlog = 0usize;
    let mut prior_meshing_queue_depth = 0usize;
    let mut prior_upload_latency_ms = 0.0f32;
    let mut gen_dispatch_paused = false;
    let mut gen_worker_inflight = 0usize;
    let mut gen_request_count = 0usize;
    let mut last_desired_cap_stats = DesiredCapStats::default();
    let mut spawn_pending = true;
    let mut spawn_pending_reason = SpawnPendingReason::Searching;
    let mut spawn_fallback_cursor: Option<VoxelCoord> = None;
    let mut last_mesh_stats = MeshRebuildStats::default();
    let mut next_maintenance_tick_at = Instant::now();
    let mut next_redraw_at = Instant::now();
    let mut finalize_low_progress_streak = 0u32;
    let mut maintenance_throttle_state = MaintenanceThrottleState::default();
    let mut startup_burst_budget_frames = 180u32;
    let mut dispatch_starvation_streak = 0u32;
    let mut apply_starvation_streak = 0u32;
    let mut convergence_stall_streak = 0u32;

    let _ = set_cursor(window, false);

    event_loop
        .run(move |event, elwt| match &event {
            Event::WindowEvent { event, window_id } if *window_id == window.id() => {
                match event {
                    WindowEvent::CursorMoved { .. } => cursor_position_known = true,
                    WindowEvent::CursorLeft { .. } => cursor_position_known = false,
                    _ => {}
                }

                // Only let egui see Tab/pointer when UI intends to own them.
                let block_tab_for_egui = matches!(
                    event,
                    WindowEvent::KeyboardInput { event, .. }
                        if !ui.paused_menu
                            && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Tab))
                );
                let ui_allows_pointer =
                    ui.paused_menu || ui.tab_palette_open || ui.show_tool_quick_menu;
                let block_pointer_for_egui = !ui_allows_pointer
                    && matches!(
                        event,
                        WindowEvent::CursorMoved { .. }
                            | WindowEvent::MouseInput { .. }
                            | WindowEvent::MouseWheel { .. }
                            | WindowEvent::TouchpadPressure { .. }
                    );
                let pointer_button_without_position = !cursor_position_known
                    && matches!(
                        event,
                        WindowEvent::MouseInput { .. }
                            | WindowEvent::MouseWheel { .. }
                            | WindowEvent::TouchpadPressure { .. }
                    );
                let egui_c = if block_tab_for_egui
                    || block_pointer_for_egui
                    || pointer_button_without_position
                {
                    false
                } else {
                    egui_state.on_window_event(window, event).consumed
                };

                let apply_cursor_mode =
                    |window: &winit::window::Window, ui: &UiState, unlocked: &mut bool| {
                        let should_unlock = should_unlock_cursor(ui, ui.show_tool_quick_menu, ui.tab_palette_open);
                        if should_unlock != *unlocked {
                            let _ = set_cursor(window, should_unlock);
                            *unlocked = should_unlock;
                        }
                        window.request_redraw();
                    };

                input.on_window_event(event);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => renderer.resize(*size),
                    WindowEvent::Focused(focused) => {
                        let should_unlock = !focused
                            || ui.paused_menu
                            || ui.show_tool_quick_menu
                            || ui.tab_palette_open;
                        if should_unlock != cursor_is_unlocked {
                            let _ = set_cursor(window, should_unlock);
                            cursor_is_unlocked = should_unlock;
                        }
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(key) = event.physical_key {
                            let egui_has_keyboard_focus = egui_ctx.wants_keyboard_input();
                            let toggle_pause_key = key == KeyCode::Escape;
                            let toggle_pause_pressed =
                                event.state == ElementState::Pressed && !event.repeat;
                            if toggle_pause_key && toggle_pause_pressed && !egui_has_keyboard_focus {
                                ui.paused_menu = !ui.paused_menu;
                                apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                            }

                            let toggle_sim_key = key == KeyCode::KeyP;
                            let toggle_sim_pressed =
                                event.state == ElementState::Pressed && !event.repeat;
                            if toggle_sim_key && toggle_sim_pressed && !egui_has_keyboard_focus {
                                sim_running = !sim_running;
                                if !sim_running {
                                    step_once = false;
                                }
                            }

                            if event.state == ElementState::Pressed {
                                let tab_palette_open = ui.tab_palette_open;
                                let hotbar_slot = key_to_hotbar_slot(key);

                                match key {
                                    KeyCode::Escape | KeyCode::KeyP => {}
                                    _ if ui.paused_menu => {}
                                    KeyCode::KeyB => {
                                        ui.show_brush = !ui.show_brush;
                                        apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                                    }
                                    KeyCode::Tab if !event.repeat => {
                                        ui.tab_palette_open = !ui.tab_palette_open;
                                        apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                                    }
                                    RADIAL_MENU_TOGGLE_KEY => {
                                        ui.show_radial_menu = !ui.show_radial_menu;
                                        apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                                    }
                                    TOOL_QUICK_MENU_TOGGLE_KEY if !ui.paused_menu => {
                                        ui.show_tool_quick_menu = true;
                                        apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                                    }
                                    KeyCode::KeyF => {
                                        brush.shape = if brush.shape == BrushShape::Sphere {
                                            BrushShape::Cube
                                        } else {
                                            BrushShape::Sphere
                                        }
                                    }
                                    KeyCode::F1 => {
                                        stream_tuning.base_generate_drain_items = stream_tuning.base_generate_drain_items.saturating_sub(4);
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F2 => {
                                        stream_tuning.base_generate_drain_items += 4;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F3 => {
                                        stream_tuning.base_apply_budget_items = stream_tuning.base_apply_budget_items.saturating_sub(2);
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F4 => {
                                        stream_tuning.base_apply_budget_items += 2;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F5 => {
                                        stream_tuning.far_radius_xz += 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::BracketLeft => ui.adjust_sim_speed(-1),
                                    KeyCode::BracketRight => ui.adjust_sim_speed(1),
                                    KeyCode::Backslash => ui.set_sim_speed(1.0),
                                    KeyCode::F6 => {
                                        stream_tuning.near_radius_xz -= 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F7 => {
                                        stream_tuning.near_radius_xz += 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F8 => {
                                        stream_tuning.mid_radius_xz -= 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F9 => {
                                        stream_tuning.mid_radius_xz += 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F10 => {
                                        stream_tuning.vertical_radius -= 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F11 => {
                                        stream_tuning.vertical_radius += 1;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    KeyCode::F12 => {
                                        stream_tuning.lod_hysteresis = (stream_tuning.lod_hysteresis + 1) % 4;
                                        stream_tuning = stream_tuning.clone().normalized();
                                    }
                                    _ if hotbar_slot.is_some() => {
                                        assign_or_select_hotbar(
                                            &mut ui,
                                            hotbar_slot.unwrap_or(0),
                                            tab_palette_open,
                                        );
                                    }
                                    KeyCode::KeyZ => ui.active_tool = ToolKind::Brush,
                                    KeyCode::KeyX => ui.active_tool = ToolKind::BuildersWand,
                                    KeyCode::KeyC => ui.active_tool = ToolKind::DestructorWand,
                                    KeyCode::KeyV => ui.active_tool = ToolKind::AreaTool,
                                    KeyCode::KeyN => ui.active_tool = ToolKind::MaterialJet,
                                    _ => {}
                                }
                            }

                            if key == TOOL_QUICK_MENU_TOGGLE_KEY
                                && event.state == ElementState::Released
                            {
                                if ui.show_tool_quick_menu && !input.lmb && !input.rmb {
                                    apply_quick_menu_hover_selection(&mut ui, &mut brush);
                                }
                                ui.show_tool_quick_menu = false;
                                apply_cursor_mode(window, &ui, &mut cursor_is_unlocked);
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - last).as_secs_f32().min(0.05);
                        last = now;

                        // Quick-menu state derived from held key
                        let quick_menu_held =
                            input.key(TOOL_QUICK_MENU_TOGGLE_KEY) && !ui.paused_menu;
                        let tab_palette_held = ui.tab_palette_open && !ui.paused_menu;
                        ui.show_tool_quick_menu = quick_menu_held;

                        if !quick_menu_held {
                            ui.hovered_shape = None;
                            ui.hovered_area_shape = None;
                            ui.hovered_tool = None;
                        }

                        let cursor_should_unlock =
                            should_unlock_cursor(&ui, quick_menu_held, tab_palette_held);
                        let gameplay_blocked = cursor_should_unlock;
                        gen_request_count = 0;

                        if spawn_pending {
                            if let Some(candidate) = find_safe_spawn_in_loaded_chunks(
                                &store,
                                streaming.seed,
                                spawn_fallback_cursor,
                            ) {
                                let candidate_local =
                                    world_spawn_to_local_pos(candidate.voxel, origin_voxel);
                                if is_spawn_collision_free(&store, candidate_local, origin_voxel) {
                                    let neighborhood_loaded = collision_neighborhood_loaded(
                                        &store,
                                        candidate_local,
                                        origin_voxel,
                                        COLLISION_SAFETY_RADIUS_VOXELS,
                                    );
                                    spawn_pending = !neighborhood_loaded;
                                    spawn_pending_reason = if spawn_pending {
                                        SpawnPendingReason::MissingNeighborhood
                                    } else {
                                        SpawnPendingReason::Searching
                                    };
                                    if !spawn_pending {
                                        ctrl.position = candidate_local;
                                        spawn_fallback_cursor = None;
                                        last_player_chunk = None;
                                        cached_desired = DesiredChunks::default();
                                        cached_sim_region.clear();
                                    }
                                } else {
                                    spawn_pending_reason = SpawnPendingReason::BlockedCapsule;
                                }
                            } else {
                                spawn_pending_reason = SpawnPendingReason::NoValidColumn;
                                spawn_fallback_cursor = highest_loaded_surface(&store).map(|surface| VoxelCoord {
                                    x: 0,
                                    y: surface + SPAWN_RETRY_HEIGHT_ABOVE_SURFACE,
                                    z: 0,
                                });
                                if let Some(cursor) = spawn_fallback_cursor {
                                    let cursor_local = world_spawn_to_local_pos(cursor, origin_voxel);
                                    if is_spawn_collision_free(&store, cursor_local, origin_voxel) {
                                        let neighborhood_loaded = collision_neighborhood_loaded(
                                            &store,
                                            cursor_local,
                                            origin_voxel,
                                            COLLISION_SAFETY_RADIUS_VOXELS,
                                        );
                                        spawn_pending = !neighborhood_loaded;
                                        spawn_pending_reason = if spawn_pending {
                                            SpawnPendingReason::UsingFallbackBand
                                        } else {
                                            SpawnPendingReason::Searching
                                        };
                                        if !spawn_pending {
                                            ctrl.position = cursor_local;
                                            last_player_chunk = None;
                                            cached_desired = DesiredChunks::default();
                                            cached_sim_region.clear();
                                        }
                                    }
                                }
                            }
                        }

                        if cursor_should_unlock != cursor_is_unlocked {
                            let _ = set_cursor(window, cursor_should_unlock);
                            cursor_is_unlocked = cursor_should_unlock;
                        }

                        // Ctrl/Alt wheel controls when gameplay owns input
                        if !gameplay_blocked && input.wheel.abs() > 0.0 {
                            let ctrl_held = input.key(KeyCode::ControlLeft)
                                || input.key(KeyCode::ControlRight);
                            let alt_held =
                                input.key(KeyCode::AltLeft) || input.key(KeyCode::AltRight);

                            if ctrl_held {
                                let delta = input.wheel.signum() as i32;
                                let r = (brush.radius + delta).clamp(0, 8);
                                brush.radius = r;
                            } else if alt_held {
                                brush.max_distance =
                                    (brush.max_distance + input.wheel).clamp(2.0, 48.0);
                            } else {
                                let dir = if input.wheel > 0.0 { 1 } else { -1 };
                                ui.selected_slot = ((ui.selected_slot as i32 + dir)
                                    .rem_euclid(HOTBAR_SLOTS as i32))
                                    as usize;
                            }
                        }

                        let now_instant = Instant::now();

                        // Origin shifting to keep float precision stable.
                        let origin_update = floating_origin_state.update(
                            ctrl.position,
                            now_instant,
                            floating_origin_config,
                        );
                        origin_voxel = origin_update.origin_translation;
                        ctrl.position = origin_update.player_local_position;
                        renderer.set_origin_voxel(origin_voxel);
                        // Keep resident meshes across recenter events.
                        //
                        // Chunk meshes are authored in chunk-local voxel coordinates, and the
                        // renderer applies chunk origin + origin_voxel in a single render transform. Clearing the
                        // cache here drops all mesh rebuild/version state without re-dirtying
                        // already-resident chunks, which can leave the nearby world invisible.

                        // === Player movement/collision: query the actual ChunkStore (not dummy world) ===
                        let player_local_for_collision = ctrl.position;
                        let player_world_for_collision = local_to_world_voxel(player_local_for_collision, origin_voxel);
                        let (player_chunk_for_collision, _) = voxel_to_chunk(player_world_for_collision);
                        let missing_collision_chunks = collision_neighborhood_missing_chunks(
                            &store,
                            player_local_for_collision,
                            origin_voxel,
                            COLLISION_SAFETY_RADIUS_VOXELS,
                        );
                        let collision_neighborhood_loaded = missing_collision_chunks.is_empty();
                        let now_secs = start.elapsed().as_secs_f32();
                        let mut present_missing: HashSet<ChunkCoord> = HashSet::new();
                        for coord in &missing_collision_chunks {
                            present_missing.insert(*coord);
                            collision_missing_since.entry(*coord).or_insert(now_secs);
                        }
                        collision_missing_since.retain(|coord, _| present_missing.contains(coord));

                        // Soft-fallback collision policy: no hard movement freeze when chunks are missing.

                        if !collision_neighborhood_loaded {
                            let missing_coords = missing_collision_chunks.clone();
                            let mut forced_local_requests = 0usize;
                            let mut urgent_missing = Vec::new();
                            let mut non_urgent_missing = Vec::new();
                            for coord in missing_coords {
                                if is_urgent_chunk(player_chunk_for_collision, coord) {
                                    urgent_missing.push(coord);
                                } else {
                                    non_urgent_missing.push(coord);
                                }
                            }
                            for coord in urgent_missing
                                .into_iter()
                                .chain(non_urgent_missing.into_iter())
                                .take(COLLISION_LOCAL_PRIORITY_REQUEST_BUDGET)
                            {
                                if streaming.resident.contains(&coord)
                                    || streaming.dispatched_generate.contains(&coord)
                                    || streaming.scheduled_generate.contains(&coord)
                                {
                                    continue;
                                }
                                let class = if is_urgent_chunk(player_chunk_for_collision, coord) {
                                    GenerateJobClass::Urgent
                                } else {
                                    GenerateJobClass::Near
                                };
                                if let Some(chunk) = cached_modified_chunks.take(coord) {
                                    apply_generated_chunk(&mut store, coord, chunk);
                                    streaming.mark_generated(coord, frame_counter);
                                    forced_local_requests += 1;
                                } else if streaming.dispatch_generation_for_class(
                                    coord,
                                    class,
                                    |coord| {
                                        if chunk_generator.try_request(coord, class, 0) {
                                            gen_request_count += 1;
                                            gen_worker_inflight += 1;
                                            true
                                        } else {
                                            false
                                        }
                                    },
                                ) {
                                    forced_local_requests += 1;
                                }
                            }
                            if forced_local_requests > 0 {
                                ui.log_once_per_second("collision_force_local", start.elapsed().as_secs_f32(), || {
                                    format!(
                                        "collision_force_local requested={} player_chunk=({}, {}, {})",
                                        forced_local_requests,
                                        player_chunk_for_collision.x,
                                        player_chunk_for_collision.y,
                                        player_chunk_for_collision.z,
                                    )
                                });
                            }
                            collision_local_urgent_boost = collision_local_urgent_boost
                                .max(forced_local_requests)
                                .max(COLLISION_URGENT_DISPATCH_BOOST);
                        } else {
                            collision_local_urgent_boost = 0;
                        }


                        collision_used_unloaded_chunks = false;
                        collision_unknown_samples = 0;
                        collision_unknown_blocks_as_solid = 0;
                        collision_soft_timeout_active = false;
                        let look_active = !(gameplay_blocked || spawn_pending) && !egui_c;
                        let ctrl_input = input.clone();
                        let origin_for_ctrl = origin_voxel;
                        let collision_position_before_step = ctrl.position;
                        let collision_step = ctrl.step(
                            |x, y, z| {
                                let voxel = VoxelCoord { x, y, z };
                                if store.is_voxel_chunk_loaded(voxel) {
                                    return store.get_voxel(voxel);
                                }
                                collision_unknown_samples = collision_unknown_samples.saturating_add(1);
                                let near_player =
                                    within_collision_safety_radius(voxel, player_local_for_collision, origin_for_ctrl, COLLISION_SAFETY_RADIUS_VOXELS);
                                if near_player {
                                    collision_used_unloaded_chunks = true;
                                    let (missing_chunk, _) = voxel_to_chunk(voxel);
                                    let missing_for_secs = collision_missing_since
                                        .get(&missing_chunk)
                                        .map(|seen_secs| now_secs - *seen_secs)
                                        .unwrap_or(0.0);
                                    if missing_for_secs <= COLLISION_STRICT_FALLBACK_WINDOW_SECS {
                                        collision_unknown_blocks_as_solid = collision_unknown_blocks_as_solid.saturating_add(1);
                                        return 1u16;
                                    }

                                    collision_soft_timeout_active = true;
                                }

                                0u16
                            },
                            &ctrl_input,
                            dt,
                            look_active,
                            start.elapsed().as_secs_f32(),
                            origin_for_ctrl,
                        );
                        collision_axis_reverts = collision_step.axis_reverts;
                        if collision_soft_timeout_active {
                            let damped_delta = ctrl.position - collision_position_before_step;
                            ctrl.position = collision_position_before_step
                                + damped_delta * COLLISION_TIMEOUT_DAMPING_FACTOR;
                        }
                        if collision_unknown_blocks_as_solid > 0 {
                            collision_blocked_unloaded_count_total = collision_blocked_unloaded_count_total
                                .saturating_add(collision_unknown_blocks_as_solid as u64);
                            collision_blocked_unloaded_ms_total += dt * 1000.0;
                        }

                        // Streaming regions in WORLD voxel space
                        let player_world_voxel = local_to_world_voxel(ctrl.position, origin_voxel);
                        let (player_chunk, _) = voxel_to_chunk(player_world_voxel);

                        // Cache desired/sim_region to avoid rebuilding allocations every frame
                        let desired_t0 = Instant::now();
                        auto_tune.update(
                            dt,
                            prior_dirty_backlog,
                            prior_meshing_queue_depth,
                            prior_upload_latency_ms,
                        );
                        let effective_far_radius = blend_i32(
                            stream_tuning.far_radius_xz,
                            stream_tuning.mid_radius_xz,
                            auto_tune.degrade_level,
                        );
                        let effective_ultra_radius = blend_i32(
                            stream_tuning.ultra_radius_xz,
                            effective_far_radius,
                            auto_tune.degrade_level,
                        );
                        let effective_far_budget = blend_usize(
                            stream_tuning.lod_budget_far,
                            1,
                            auto_tune.degrade_level,
                        )
                        .max(1);
                        let effective_ultra_budget = blend_usize(
                            stream_tuning.lod_budget_ultra,
                            1,
                            auto_tune.degrade_level,
                        )
                        .max(1);
                        let mut effective_stream_tuning = stream_tuning.clone();
                        effective_stream_tuning.far_radius_xz = effective_far_radius;
                        effective_stream_tuning.ultra_radius_xz = effective_ultra_radius;
                        effective_stream_tuning.lod_budget_far = effective_far_budget;
                        effective_stream_tuning.lod_budget_ultra = effective_ultra_budget;

                        let player_chunk_changed = last_player_chunk != Some(player_chunk);
                        let stream_tuning_changed = cached_stream_tuning != effective_stream_tuning;
                        let look_dir = ctrl.look_dir().normalize_or_zero();
                        let view_angle_changed = last_desired_look_dir
                            .map(|prev| prev.dot(look_dir) < (1.0 - DESIRED_VIEW_RECOMPUTE_DOT_DELTA))
                            .unwrap_or(false);
                        let recompute_desired =
                            player_chunk_changed || stream_tuning_changed || view_angle_changed;
                        let player_velocity_chunks = {
                            let dv = Vec3::new(
                                (player_world_voxel.x - last_player_world_voxel.x) as f32,
                                (player_world_voxel.y - last_player_world_voxel.y) as f32,
                                (player_world_voxel.z - last_player_world_voxel.z) as f32,
                            );
                            dv / crate::types::CHUNK_SIZE_VOXELS as f32
                        };
                        let camera_pos_blocks_world = ctrl.position
                            + Vec3::new(
                                origin_voxel.x as f32,
                                origin_voxel.y as f32,
                                origin_voxel.z as f32,
                            );
                        let camera_pos_world =
                            camera_world_pos_from_blocks(camera_pos_blocks_world, VOXEL_SIZE);
                        let cam = Camera {
                            pos: camera_pos_world,
                            dir: look_dir,
                            aspect: renderer.config.width as f32
                                / renderer.config.height.max(1) as f32,
                        };
                        let visible_now = renderer.cull_visible_chunks(&cam);
                        let visible_chunk_count = visible_now.len();
                        streaming.mark_visible_batch(visible_now, frame_counter);

                        let visible_history = streaming.last_visible_frames();
                        if recompute_desired {
                            let mut reasons = Vec::with_capacity(3);
                            if player_chunk_changed {
                                reasons.push("chunk");
                            }
                            if stream_tuning_changed {
                                reasons.push("tuning");
                            }
                            if view_angle_changed {
                                reasons.push("view");
                            }
                            desired_recompute_reason = reasons.join("|");

                            let chunk_size_meters = crate::types::CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
                            let camera_pos_chunks = camera_pos_world / chunk_size_meters;
                            let chunk_space_vp = cam.view_proj()
                                * Mat4::from_scale(Vec3::splat(chunk_size_meters));
                            let visibility_context = VisibilityContext {
                                camera_pos_chunks,
                                cone_inner_cos: (35.0f32).to_radians().cos(),
                                cone_outer_cos: (75.0f32).to_radians().cos(),
                                frustum_planes: Some(extract_frustum_planes(chunk_space_vp)),
                            };
                            cached_desired = ChunkStreaming::desired_set(
                                player_chunk,
                                player_velocity_chunks,
                                look_dir,
                                Some(&visibility_context),
                                effective_stream_tuning.near_radius_xz,
                                effective_stream_tuning.mid_radius_xz,
                                Some(effective_stream_tuning.far_radius_xz),
                                Some(effective_stream_tuning.ultra_radius_xz),
                                effective_stream_tuning.vertical_radius,
                                RESIDENT_KEEP_MID_CAP,
                                RESIDENT_KEEP_FAR_CAP,
                                &visible_history,
                                frame_counter,
                            );
                            if player_chunk_changed {
                                if let Some(previous_chunk) = last_player_chunk {
                                    let shift = (player_chunk.x - previous_chunk.x)
                                        .abs()
                                        .max((player_chunk.y - previous_chunk.y).abs())
                                        .max((player_chunk.z - previous_chunk.z).abs());
                                    if shift >= 2 {
                                        streaming.invalidate_far_jobs();
                                    }
                                }
                            }
                            last_player_chunk = Some(player_chunk);
                            cached_stream_tuning = effective_stream_tuning.clone();
                            last_desired_look_dir = Some(look_dir);
                        } else {
                            desired_recompute_reason.clear();
                            desired_recompute_reason.push_str("none");
                        }

                        let active_emitters = simulation_runtime.active_emitter_chunks().clone();
                        let player_region_shift = last_sim_region_player_anchor
                            .map(|anchor| {
                                (player_chunk.x - anchor.x)
                                    .abs()
                                    .max((player_chunk.y - anchor.y).abs())
                                    .max((player_chunk.z - anchor.z).abs())
                            })
                            .unwrap_or(i32::MAX);
                        let emitter_set_changed = active_emitters != cached_sim_region_emitters;
                        if player_region_shift >= SIM_REGION_RECOMPUTE_CHUNK_DELTA || emitter_set_changed {
                            let (solid_region, gas_region) = build_adaptive_sim_regions(
                                player_chunk,
                                SIMULATION_RADIUS_CHUNKS,
                                ui.sim_gas_vertical_range_chunks,
                                &active_emitters,
                            );
                            cached_sim_region = solid_region;
                            cached_gas_sim_region = gas_region;
                            cached_sim_region_emitters = active_emitters;
                            last_sim_region_player_anchor = Some(player_chunk);
                        }

                        last_player_world_voxel = player_world_voxel;
                        let desired_ms = desired_t0.elapsed().as_secs_f32() * 1000.0;

                        let streaming_t0 = Instant::now();
                        let desired_backlog_count = cached_desired
                            .generation_order
                            .iter()
                            .filter(|coord| !streaming.resident.contains(coord))
                            .count();
                        let near_backlog_count = cached_desired
                            .near
                            .iter()
                            .filter(|coord| !streaming.resident.contains(coord))
                            .count();
                        let generated_backlog = generated_ready.len();
                        let apply_backlog = generated_backlog
                            .max(desired_backlog_count / 3)
                            .max(near_backlog_count.saturating_mul(2));
                        let apply_budget_items = scaled_budget(
                            stream_tuning.base_apply_budget_items,
                            stream_tuning.max_apply_budget_items,
                            apply_backlog,
                            stream_tuning.base_apply_budget_items,
                        );
                        let pending_generate_depth = streaming.pending_generate_count();
                        let near_generation_pressure = near_backlog_count
                            .max(desired_backlog_count / 4)
                            .max(generated_backlog / 2);
                        let generation_drain_backlog = pending_generate_depth
                            .max(generated_backlog)
                            .max(near_generation_pressure)
                            .max(gen_worker_inflight);
                        let base_generate_drain_budget = scaled_budget(
                            stream_tuning.base_generate_drain_items,
                            stream_tuning.max_generate_drain_items,
                            generation_drain_backlog,
                            stream_tuning.base_generate_drain_items,
                        );
                        let pressure_model = compute_generation_pressure_model(
                            &effective_stream_tuning,
                            pending_generate_depth.max(near_generation_pressure),
                            gen_worker_inflight,
                            apply_budget_items,
                            prior_mesh_backlog,
                            prior_meshing_queue_depth,
                            base_generate_drain_budget,
                        );
                        streaming.max_generate_schedule_per_update = base_generate_drain_budget.max(PROTECTED_HIGH_PRIORITY_SLOTS);
                        let backpressure = if prior_mesh_backlog <= MESH_BACKPRESSURE_START {
                            0.0
                        } else {
                            ((prior_mesh_backlog - MESH_BACKPRESSURE_START) as f32
                                / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
                                .clamp(0.0, 1.0)
                        };
                        let (mut generation_priority, desired_cap_stats) = cap_desired_generation_order(
                            &cached_desired,
                            player_chunk,
                            pressure_model,
                        );
                        last_desired_cap_stats = desired_cap_stats;
                        if backpressure > 0.0 {
                            let chunk_size_meters = crate::types::CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
                            let frustum_planes = extract_frustum_planes(
                                cam.view_proj() * Mat4::from_scale(Vec3::splat(chunk_size_meters)),
                            );
                            let far_radius_cutoff = (effective_stream_tuning.mid_radius_xz as f32
                                + (effective_stream_tuning.far_radius_xz
                                    - effective_stream_tuning.mid_radius_xz) as f32
                                    * (1.0 - backpressure))
                                .round() as i32;
                            generation_priority.retain(|coord| {
                                let ring_distance = chebyshev_from_player(player_chunk, *coord);
                                if ring_distance <= effective_stream_tuning.mid_radius_xz {
                                    return true;
                                }
                                let in_frustum = coord_in_frustum(*coord, &frustum_planes);
                                if in_frustum {
                                    return true;
                                }
                                let dx = (coord.x - player_chunk.x) as f32;
                                let dz = (coord.z - player_chunk.z) as f32;
                                (dx * dx + dz * dz).sqrt() <= far_radius_cutoff as f32
                            });
                        }
                        let stream_stats = streaming.update(
                            &generation_priority,
                            &cached_desired.resident_keep,
                            player_chunk,
                            frame_counter,
                        );
                        streaming.reprioritize_generate_queue(
                            player_chunk,
                            &cached_desired.generation_scores,
                        );
                        let queue_age = streaming.queue_age_telemetry();
                        if queue_age.urgent.p95 > 24 {
                            ui.log_once_per_second("urgent_queue_age_alert", start.elapsed().as_secs_f32(), || {
                                format!(
                                    "urgent queue age exceeded threshold p50={} p95={} near_p95={} mid_p95={} far_p95={}",
                                    queue_age.urgent.p50,
                                    queue_age.urgent.p95,
                                    queue_age.near.p95,
                                    queue_age.mid.p95,
                                    queue_age.far.p95,
                                )
                            });
                        }

                        let gen_worker_inflight_before_dispatch = gen_worker_inflight;
                        if gen_dispatch_paused {
                            if gen_worker_inflight_before_dispatch <= generator_config.dispatch_low {
                                gen_dispatch_paused = false;
                            }
                        } else if gen_worker_inflight_before_dispatch >= generator_config.dispatch_high {
                            gen_dispatch_paused = true;
                        }

                        let mut dispatch_urgent = Vec::new();
                        let mut dispatch_near = Vec::new();
                        let mut dispatch_mid = Vec::new();
                        let mut dispatch_far = Vec::new();
                        let mut dispatch_ultra = Vec::new();

                        for &coord in &generation_priority {
                            if streaming.resident.contains(&coord)
                                || streaming.dispatched_generate.contains(&coord)
                            {
                                continue;
                            }
                            if is_urgent_chunk(player_chunk, coord) {
                                dispatch_urgent.push(coord);
                            } else if cached_desired.near.contains(&coord) {
                                dispatch_near.push(coord);
                            } else if cached_desired.mid.contains(&coord) {
                                dispatch_mid.push(coord);
                            } else if cached_desired.far.contains(&coord) {
                                dispatch_far.push(coord);
                            } else {
                                dispatch_ultra.push(coord);
                            }
                        }

                        let has_near_backlog = dispatch_near.iter().any(|coord| {
                            !streaming.resident.contains(coord)
                                && !streaming.dispatched_generate.contains(coord)
                        });
                        let has_mid_backlog = !has_near_backlog
                            && dispatch_mid.iter().any(|coord| {
                                !streaming.resident.contains(coord)
                                    && !streaming.dispatched_generate.contains(coord)
                            });
                        let gen_pause_reason = if has_near_backlog {
                            "near_ring_active"
                        } else if has_mid_backlog {
                            "mid_ring_active"
                        } else if gen_dispatch_paused {
                            "worker_queue_high_watermark"
                        } else {
                            "none"
                        };

                        let dispatch_coord = |coord: ChunkCoord,
                                              class: GenerateJobClass,
                                              gen_request_count: &mut usize,
                                              gen_worker_inflight: &mut usize,
                                              store: &mut ChunkStore,
                                              streaming: &mut ChunkStreaming,
                                              cached_modified_chunks: &mut ModifiedChunkCache|
                         -> bool {
                            let far_version_snapshot = streaming.far_generation_version();
                            if let Some(chunk) = cached_modified_chunks.take(coord) {
                                apply_generated_chunk(store, coord, chunk);
                                streaming.mark_generated(coord, frame_counter);
                                return true;
                            }
                            streaming.dispatch_generation_for_class(coord, class, |coord| {
                                let version = if matches!(class, GenerateJobClass::Mid | GenerateJobClass::Far) {
                                    far_version_snapshot
                                } else {
                                    0
                                };
                                if chunk_generator.try_request(
                                    coord,
                                    class,
                                    version,
                                ) {
                                    *gen_request_count += 1;
                                    *gen_worker_inflight += 1;
                                    true
                                } else {
                                    false
                                }
                            })
                        };

                        let urgent_dispatch_budget =
                            URGENT_GENERATION_BUDGET + collision_local_urgent_boost;
                        for coord in dispatch_urgent.into_iter().take(urgent_dispatch_budget) {
                            if !dispatch_coord(
                                coord,
                                GenerateJobClass::Urgent,
                                &mut gen_request_count,
                                &mut gen_worker_inflight,
                                &mut store,
                                &mut streaming,
                                &mut cached_modified_chunks,
                            ) {
                                break;
                            }
                        }

                        let near_budget = pressure_model.near_dispatch_budget;
                        let near_total = dispatch_near.len();
                        let mut near_sent = 0usize;
                        for coord in dispatch_near.into_iter().take(near_budget) {
                            if !dispatch_coord(
                                coord,
                                GenerateJobClass::Near,
                                &mut gen_request_count,
                                &mut gen_worker_inflight,
                                &mut store,
                                &mut streaming,
                                &mut cached_modified_chunks,
                            ) {
                                break;
                            }
                            near_sent += 1;
                        }

                        let near_unsatisfied = near_total.saturating_sub(near_sent);
                        let dispatch_starvation_active = has_near_backlog && near_sent == 0;
                        let near_reserve_budget = if dispatch_starvation_active {
                            base_generate_drain_budget.max(2)
                        } else {
                            near_unsatisfied.min(base_generate_drain_budget / 2)
                        };
                        let generate_drain_budget = if gen_dispatch_paused {
                            0
                        } else {
                            base_generate_drain_budget
                                .saturating_sub(near_sent)
                                .saturating_sub(near_reserve_budget)
                        };

                        if generate_drain_budget > 0 {
                            let mut mid_budget = generate_drain_budget
                                .min(pressure_model.mid_dispatch_budget)
                                .max(usize::from(has_mid_backlog && !dispatch_mid.is_empty()));
                            let mut mid_sent = 0usize;
                            for coord in dispatch_mid {
                                if mid_budget == 0 {
                                    break;
                                }
                                if !dispatch_coord(
                                    coord,
                                    GenerateJobClass::Mid,
                                    &mut gen_request_count,
                                    &mut gen_worker_inflight,
                                    &mut store,
                                    &mut streaming,
                                    &mut cached_modified_chunks,
                                ) {
                                    break;
                                }
                                mid_sent += 1;
                                mid_budget = mid_budget.saturating_sub(1);
                            }

                            let mut far_budget = generate_drain_budget
                                .saturating_sub(mid_sent)
                                .min(pressure_model.far_dispatch_budget);
                            for coord in dispatch_far {
                                if far_budget == 0 {
                                    break;
                                }
                                if !dispatch_coord(
                                    coord,
                                    GenerateJobClass::Far,
                                    &mut gen_request_count,
                                    &mut gen_worker_inflight,
                                    &mut store,
                                    &mut streaming,
                                    &mut cached_modified_chunks,
                                ) {
                                    break;
                                }
                                far_budget = far_budget.saturating_sub(1);
                            }
                            for coord in dispatch_ultra {
                                if far_budget == 0 {
                                    break;
                                }
                                if !dispatch_coord(
                                    coord,
                                    GenerateJobClass::Far,
                                    &mut gen_request_count,
                                    &mut gen_worker_inflight,
                                    &mut store,
                                    &mut streaming,
                                    &mut cached_modified_chunks,
                                ) {
                                    break;
                                }
                                far_budget = far_budget.saturating_sub(1);
                            }
                        }
                        let gen_recv_t0 = Instant::now();
                        let mut gen_completed_count = 0usize;
                        let mut gen_recv_empty = false;
                        let mut gen_recv_disconnected = false;
                        loop {
                            match chunk_generator.try_recv() {
                                Ok(res) => {
                                    let stale_far = matches!(res.class, GenerateJobClass::Mid | GenerateJobClass::Far)
                                        && res.version != streaming.far_generation_version();
                                    if stale_far {
                                        streaming.mark_generation_dropped(res.coord);
                                    } else {
                                        generated_ready.push_back(res);
                                        gen_completed_count += 1;
                                    }
                                    gen_worker_inflight = gen_worker_inflight.saturating_sub(1);
                                }
                                Err(TryRecvError::Empty) => {
                                    gen_recv_empty = true;
                                    break;
                                }
                                Err(TryRecvError::Disconnected) => {
                                    gen_recv_disconnected = true;
                                    break;
                                }
                            }
                        }
                        let gen_recv_ms = gen_recv_t0.elapsed().as_secs_f32() * 1000.0;

                        total_generated_chunks = total_generated_chunks
                            .saturating_add(gen_completed_count as u64);
                        let scheduled_count = streaming.scheduled_generate.len();
                        let dispatched_count = streaming.dispatched_generate.len();
                        let pending_count = streaming.pending_generate_count();
                        let recv_progress = if gen_completed_count > 0 {
                            "completed"
                        } else if gen_recv_disconnected {
                            "disconnected"
                        } else if gen_recv_empty {
                            "empty"
                        } else {
                            "blocked"
                        };
                        if has_near_backlog && near_sent == 0 {
                            dispatch_starvation_streak = dispatch_starvation_streak.saturating_add(1);
                        } else {
                            dispatch_starvation_streak = 0;
                        }
                        if dispatched_count >= generator_config.dispatch_high && gen_completed_count == 0 {
                            let now_secs = start.elapsed().as_secs_f32();
                            ui.log_once_per_second("gen_starvation", now_secs, || {
                                format!(
                                    "gen starvation scheduled={} dispatched={} pending_generate={} gen_paused={} recv_progress={} recv_ms={:.3}",
                                    scheduled_count,
                                    dispatched_count,
                                    pending_count,
                                    gen_dispatch_paused,
                                    recv_progress,
                                    gen_recv_ms,
                                )
                            });
                        }
                        if dispatch_starvation_streak >= DISPATCH_STARVATION_STREAK_FRAMES {
                            let now_secs = start.elapsed().as_secs_f32();
                            ui.log_once_per_second("dispatch_starvation", now_secs, || {
                                format!(
                                    "dispatch starvation near_backlog={} near_budget={} near_sent={} gen_drain_budget={} pending_generate={} inflight={} paused={} streak={}",
                                    near_total,
                                    near_budget,
                                    near_sent,
                                    generate_drain_budget,
                                    pending_count,
                                    gen_worker_inflight,
                                    gen_dispatch_paused,
                                    dispatch_starvation_streak,
                                )
                            });
                        }

                        if generated_ready.len() > 1 {
                            let mut prioritized: Vec<_> = generated_ready.drain(..).collect();
                            prioritized.sort_by(|a, b| {
                                let urgent_a = is_urgent_chunk(player_chunk, a.coord);
                                let urgent_b = is_urgent_chunk(player_chunk, b.coord);
                                urgent_b
                                    .cmp(&urgent_a)
                                    .then_with(|| {
                                        cached_desired
                                            .generation_scores
                                            .get(&b.coord)
                                            .copied()
                                            .unwrap_or(0.0)
                                            .total_cmp(
                                                &cached_desired
                                                    .generation_scores
                                                    .get(&a.coord)
                                                    .copied()
                                                    .unwrap_or(0.0),
                                            )
                                    })
                                    .then_with(|| a.generated_at.cmp(&b.generated_at))
                            });
                            generated_ready.extend(prioritized);
                        }

                        let apply_t0 = Instant::now();
                        let mut apply_count = 0usize;
                        while apply_count < apply_budget_items
                            && apply_t0.elapsed()
                                < Duration::from_secs_f32(APPLY_NEAR_PROTECTED_BUDGET_MS / 1000.0)
                        {
                            let Some(index) = generated_ready.iter().position(|done| {
                                matches!(done.class, GenerateJobClass::Urgent | GenerateJobClass::Near)
                            }) else {
                                break;
                            };
                            let done = generated_ready.remove(index).expect("protected near result exists");
                            apply_generated_chunk(&mut store, done.coord, done.chunk);
                            streaming.mark_generated(done.coord, frame_counter);
                            apply_count += 1;
                        }
                        while apply_count < apply_budget_items
                            && apply_t0.elapsed() < Duration::from_secs_f32(APPLY_BUDGET_MS / 1000.0)
                        {
                            let Some(done) = generated_ready.pop_front() else { break; };
                            apply_generated_chunk(&mut store, done.coord, done.chunk);
                            streaming.mark_generated(done.coord, frame_counter);
                            apply_count += 1;
                        }
                        let apply_ms = apply_t0.elapsed().as_secs_f32() * 1000.0;
                        if generated_ready.len() >= apply_budget_items && apply_count == 0 {
                            apply_starvation_streak = apply_starvation_streak.saturating_add(1);
                        } else {
                            apply_starvation_streak = 0;
                        }

                        let evict_t0 = Instant::now();
                        let mut evict_count = 0usize;
                        while evict_count < streaming.max_evict_schedule_per_update
                            && evict_t0.elapsed() < Duration::from_secs_f32(EVICT_BUDGET_MS / 1000.0)
                        {
                            let mut one = streaming.drain_evict_requests(1);
                            let Some(coord) = one.pop() else {
                                break;
                            };
                            if let Some((chunk, was_modified)) = store.remove_chunk_with_data(coord) {
                                if was_modified {
                                    cached_modified_chunks.insert(coord, chunk);
                                }
                            }
                            streaming.mark_evicted(coord, frame_counter);
                            evict_count += 1;
                        }
                        let evict_ms = evict_t0.elapsed().as_secs_f32() * 1000.0;

                        let streaming_ms = streaming_t0.elapsed().as_secs_f32() * 1000.0;
                        let now_secs = start.elapsed().as_secs_f32();
                        let cache_metrics = cached_modified_chunks.metrics();
                        let cache_hit_delta = cache_metrics.hits.saturating_sub(cached_modified_last_metrics.hits);
                        let cache_miss_delta = cache_metrics.misses.saturating_sub(cached_modified_last_metrics.misses);
                        let cache_evict_delta = cache_metrics
                            .evictions
                            .saturating_sub(cached_modified_last_metrics.evictions);
                        if cache_hit_delta > 0 || cache_miss_delta > 0 || cache_evict_delta > 0 {
                            ui.log_once_per_second("modified_cache", now_secs, || {
                                format!(
                                    "modified_cache hit={} miss={} evict={} totals[h={}, m={}, e={}]",
                                    cache_hit_delta,
                                    cache_miss_delta,
                                    cache_evict_delta,
                                    cache_metrics.hits,
                                    cache_metrics.misses,
                                    cache_metrics.evictions,
                                )
                            });
                            cached_modified_last_metrics = cache_metrics;
                        }
                        if player_chunk_changed {
                            ui.log_once_per_second("chunk_change", now_secs, || format!(
                                "chunk_change player=({}, {}, {}) desired={} newly_desired={} requested={} applied={} evicted={} pending_apply={} pending_evict={}",
                                player_chunk.x,
                                player_chunk.y,
                                player_chunk.z,
                                cached_desired.generation_order.len(),
                                stream_stats.newly_desired,
                                gen_request_count,
                                apply_count,
                                evict_count,
                                generated_ready.len(),
                                streaming.pending_evict_count()
                            ));
                        }
                        if recompute_desired {
                            let reason = desired_recompute_reason.clone();
                            ui.log_once_per_second("desired_recompute", now_secs, || {
                                format!(
                                    "desired_recompute reason={} player=({}, {}, {})",
                                    reason, player_chunk.x, player_chunk.y, player_chunk.z
                                )
                            });
                        }
                        if generated_ready.len() >= apply_budget_items {
                            ui.log_once_per_second("apply_budget", now_secs, || {
                                format!("apply budget hit: remaining queue={}", generated_ready.len())
                            });
                            ui.log_once_per_second("apply_pipeline_health", now_secs, || {
                                format!(
                                    "apply pipeline backlog queue={} apply_budget_items={} gen_pending={}",
                                    generated_ready.len(),
                                    apply_budget_items,
                                    streaming.pending_generate_count()
                                )
                            });
                        }
                        if apply_starvation_streak >= APPLY_STARVATION_STREAK_FRAMES {
                            ui.log_once_per_second("apply_starvation", now_secs, || {
                                format!(
                                    "apply starvation apply_queue={} apply_budget_items={} apply_count={} near_backlog={} pending_generate={} streak={}",
                                    generated_ready.len(),
                                    apply_budget_items,
                                    apply_count,
                                    near_backlog_count,
                                    pending_count,
                                    apply_starvation_streak,
                                )
                            });
                        }
                        let loaded_chunks = store.iter_loaded_chunks().count();
                        let loaded_or_inflight = loaded_chunks
                            .saturating_add(dispatched_count)
                            .saturating_add(scheduled_count);
                        let missing_loaded_to_desired = desired_backlog_count.saturating_sub(loaded_or_inflight);
                        let convergence_stall = ui.profiler.frame_ms <= FRAME_TIME_TARGET_MS * 1.1
                            && desired_backlog_count >= LOW_LOADED_HIGH_BACKLOG_THRESHOLD
                            && missing_loaded_to_desired >= LOW_LOADED_HIGH_BACKLOG_THRESHOLD / 2
                            && apply_count == 0
                            && gen_completed_count == 0;
                        if convergence_stall {
                            convergence_stall_streak = convergence_stall_streak.saturating_add(1);
                        } else {
                            convergence_stall_streak = 0;
                        }
                        if loaded_chunks <= LOW_LOADED_CHUNKS_THRESHOLD
                            && desired_backlog_count >= LOW_LOADED_HIGH_BACKLOG_THRESHOLD
                        {
                            ui.log_once_per_second("low_loaded_high_backlog", now_secs, || {
                                format!(
                                    "loaded low vs desired backlog loaded={} desired_backlog={} near_backlog={} scheduled={} generating={} apply_queue={}",
                                    loaded_chunks,
                                    desired_backlog_count,
                                    near_backlog_count,
                                    scheduled_count,
                                    dispatched_count,
                                    generated_ready.len(),
                                )
                            });
                        }
                        if convergence_stall_streak >= CONVERGENCE_STALL_STREAK_FRAMES {
                            let frame_ms = ui.profiler.frame_ms;
                            ui.log_once_per_second("convergence_failure", now_secs, || {
                                format!(
                                    "convergence stalled frame_ms={:.2} loaded={} desired_backlog={} missing_desired={} applied={} completed={} dispatch_streak={} apply_streak={}",
                                    frame_ms,
                                    loaded_chunks,
                                    desired_backlog_count,
                                    missing_loaded_to_desired,
                                    apply_count,
                                    gen_completed_count,
                                    dispatch_starvation_streak,
                                    apply_starvation_streak,
                                )
                            });
                        }
                        if streaming.pending_evict_count() > streaming.max_evict_schedule_per_update {
                            ui.log_once_per_second("evict_budget", now_secs, || {
                                format!("evict budget hit: remaining queue={}", streaming.pending_evict_count())
                            });
                        }
                        stream_debug = format!(
                            "Stream: resident={} near={} mid={} far={} scheduled={} dispatched={} gen_pending={} apply_queue={} gen_paused={} gen_pause_reason={} recv_progress={} player_chunk=({}, {}, {}) desired_recompute={} desired_cap[near/mid/far_drop]={}/{}/{} budget_drop={} radii[n/m/f/v]={}/{}/{}/{} lod_h={} budgets[gen/app]={}/{} lod_budgets[n/m/f]={}/{}/{} collision[freeze={} unknown_solid={} safety_voxels={}] world_origin_voxel=({}, {}, {}) player_world_voxel=({}, {}, {}) keys[F1/F2 gen, F3/F4 apply, F5 far+, F6/F7 near, F8/F9 mid, F10/F11 v, F12 lod-hyst]",
                            streaming.resident.len(),
                            cached_desired.near.len(),
                            cached_desired.mid.len(),
                            cached_desired.far.len(),
                            scheduled_count,
                            dispatched_count,
                            pending_count,
                            generated_ready.len(),
                            gen_dispatch_paused,
                            gen_pause_reason,
                            recv_progress,
                            player_chunk.x,
                            player_chunk.y,
                            player_chunk.z,
                            desired_recompute_reason,
                            last_desired_cap_stats.near_dropped,
                            last_desired_cap_stats.mid_dropped,
                            last_desired_cap_stats.far_dropped,
                            last_desired_cap_stats.budget_dropped,
                            stream_tuning.near_radius_xz,
                            stream_tuning.mid_radius_xz,
                            stream_tuning.far_radius_xz,
                            stream_tuning.vertical_radius,
                            stream_tuning.lod_hysteresis,
                            generate_drain_budget,
                            apply_budget_items,
                            stream_tuning.lod_budget_near,
                            stream_tuning.lod_budget_mid,
                            stream_tuning.lod_budget_far,
                            collision_freeze_active,
                            collision_used_unloaded_chunks,
                            COLLISION_SAFETY_RADIUS_VOXELS,
                            origin_voxel.x,
                            origin_voxel.y,
                            origin_voxel.z,
                            player_world_voxel.x,
                            player_world_voxel.y,
                            player_world_voxel.z,
                        );
                        ui.stream_debug = stream_debug.clone();

                        // Editing/raycast in WORLD voxel space
                        let raycast = target_for_edit(
                            &store,
                            origin_voxel,
                            ctrl.position,
                            ctrl.look_dir(),
                            &brush,
                        );
                        let preview_mode = current_action_mode(&input, raycast, ui.active_tool);

                        preview_block_list =
                            preview_blocks(&store, &brush, raycast, preview_mode, ui.active_tool);

                        if !gameplay_blocked {
                            if ui.active_tool == ToolKind::MaterialJet {
                                let _ = apply_material_jet(
                                    &mut store,
                                    &brush,
                                    selected_material(&ui, ui.selected_slot),
                                    &input,
                                    &mut edit_runtime,
                                    now,
                                    ctrl.look_dir(),
                                    raycast,
                                    &mut simulation_runtime,
                                    if ui.sim_use_gpu_pipeline { SimulationMode::GpuFluid } else { SimulationMode::CpuCellular },
                                );
                            } else if apply_mouse_edit(
                                &mut store,
                                &brush,
                                selected_material(&ui, ui.selected_slot),
                                &input,
                                &mut edit_runtime,
                                now,
                                raycast,
                                ui.active_tool,
                                &mut simulation_runtime,
                                    if ui.sim_use_gpu_pipeline { SimulationMode::GpuFluid } else { SimulationMode::CpuCellular },
                            ) {
                                // dirtied by set_voxel
                            }
                        }

                        let do_step = sim_running && !ui.paused_menu && ui.sim_speed > 0.0;
                        let sim_t0 = Instant::now();
                        let mut sim_chunk_steps = 0usize;
                        let mut sim_skipped_chunks_non_gas = 0usize;
                        let mut sim_skipped_chunks_gas = 0usize;
                        let mut sim_boundary_dissipated_particles = 0usize;
                        let mut sim_substeps_executed = 0usize;
                        let sim_substeps_budget = ui
                            .sim_max_substeps_per_frame
                            .clamp(UiState::SIM_MAX_SUBSTEPS_MIN, UiState::SIM_MAX_SUBSTEPS_MAX);
                        ui.sim_max_substeps_per_frame = sim_substeps_budget;
                        let sim_accumulator_cap_frames = ui.sim_accumulator_cap_frames.clamp(
                            UiState::SIM_ACC_CAP_FRAMES_MIN,
                            UiState::SIM_ACC_CAP_FRAMES_MAX,
                        );
                        ui.sim_accumulator_cap_frames = sim_accumulator_cap_frames;
                        let adaptive_trigger_ratio = ui.sim_adaptive_frame_time_ratio.clamp(
                            UiState::SIM_ADAPTIVE_THRESHOLD_MIN,
                            UiState::SIM_ADAPTIVE_THRESHOLD_MAX,
                        );
                        ui.sim_adaptive_frame_time_ratio = adaptive_trigger_ratio;
                        let sim_accumulator_cap_seconds =
                            sim_accumulator_cap_frames * FIXED_SIM_STEP_SECONDS;
                        let mut sim_accumulator_clamped = false;
                        let mut sim_substeps_budget_effective = sim_substeps_budget;
                        if do_step {
                            sim_acc += dt * ui.sim_speed;
                            if sim_acc > sim_accumulator_cap_seconds {
                                sim_acc = sim_accumulator_cap_seconds;
                                sim_accumulator_clamped = true;
                            }
                            if ui.sim_adaptive_substeps {
                                let frame_ms = dt * 1000.0;
                                let adaptive_trigger_ms = FRAME_TIME_TARGET_MS * adaptive_trigger_ratio;
                                if frame_ms > adaptive_trigger_ms {
                                    let overload = (frame_ms / adaptive_trigger_ms).floor() as usize;
                                    let reduction = overload.min(sim_substeps_budget.saturating_sub(1));
                                    sim_substeps_budget_effective =
                                        sim_substeps_budget.saturating_sub(reduction).max(1);
                                }
                            }
                            let ready_steps = (sim_acc / FIXED_SIM_STEP_SECONDS).floor() as usize;
                            let steps_to_run = ready_steps.min(sim_substeps_budget_effective);
                            let sim_mode = if ui.sim_use_gpu_pipeline {
                                SimulationMode::GpuFluid
                            } else {
                                SimulationMode::CpuCellular
                            };
                            for _ in 0..steps_to_run {
                                //sim_chunk_steps += sim_world.step_region( 
                                //Investigate this, is this function better or worse?
                                let non_gas = simulation_runtime.step(
                                    sim_mode,
                                    &mut store,
                                    player_chunk,
                                    &mut rng,
                                    SimulationStepMetadata {
                                        phase_class: Some(SimulationPhaseClass::SolidsLiquidsPowders),
                                        boundary_dissipation_strength: 0.0,
                                        core_radius_chunks: SIMULATION_RADIUS_CHUNKS,
                                        max_voxels_to_process: 8192,
                                    },
                                );
                                sim_chunk_steps += non_gas.stepped_chunks;
                                sim_skipped_chunks_non_gas += non_gas.skipped_chunks;

                                let gas = simulation_runtime.step(
                                    sim_mode,
                                    &mut store,
                                    player_chunk,
                                    &mut rng,
                                    SimulationStepMetadata {
                                        phase_class: Some(SimulationPhaseClass::Gas),
                                        boundary_dissipation_strength: ui.sim_gas_boundary_dissipation,
                                        core_radius_chunks: SIMULATION_RADIUS_CHUNKS,
                                        max_voxels_to_process: 8192,
                                    },
                                );
                                sim_chunk_steps += gas.stepped_chunks;
                                sim_skipped_chunks_gas += gas.skipped_chunks;
                                sim_boundary_dissipated_particles += gas.boundary_dissipated_particles;
                            }
                            sim_substeps_executed = steps_to_run;
                            sim_acc -= steps_to_run as f32 * FIXED_SIM_STEP_SECONDS;
                        } else if step_once && !ui.paused_menu {
                            let sim_mode = if ui.sim_use_gpu_pipeline {
                                SimulationMode::GpuFluid
                            } else {
                                SimulationMode::CpuCellular
                            };
                            //sim_chunk_steps += sim_world.step_region(
                            let non_gas = simulation_runtime.step(
                                sim_mode,
                                &mut store,
                                player_chunk,
                                &mut rng,
                                SimulationStepMetadata {
                                    phase_class: Some(SimulationPhaseClass::SolidsLiquidsPowders),
                                    boundary_dissipation_strength: 0.0,
                                    core_radius_chunks: SIMULATION_RADIUS_CHUNKS,
                                    max_voxels_to_process: 8192,
                                },
                            );
                            sim_chunk_steps += non_gas.stepped_chunks;
                            sim_skipped_chunks_non_gas += non_gas.skipped_chunks;

                            let gas = simulation_runtime.step(
                                sim_mode,
                                &mut store,
                                player_chunk,
                                &mut rng,
                                SimulationStepMetadata {
                                    phase_class: Some(SimulationPhaseClass::Gas),
                                    boundary_dissipation_strength: ui.sim_gas_boundary_dissipation,
                                    core_radius_chunks: SIMULATION_RADIUS_CHUNKS,
                                    max_voxels_to_process: 8192,
                                },
                            );
                            sim_chunk_steps += gas.stepped_chunks;
                            sim_skipped_chunks_gas += gas.skipped_chunks;
                            sim_boundary_dissipated_particles += gas.boundary_dissipated_particles;
                            sim_substeps_executed = 1;
                            step_once = false;
                        }
                        simulation_runtime.reset_active_emitters();
                        let sim_ms = sim_t0.elapsed().as_secs_f32() * 1000.0;

                        renderer.day = ui.day;
                        renderer.set_settings(RendererSettings {
                            frustum_culling: ui.renderer_frustum_culling,
                            greedy_meshing: ui.renderer_greedy_meshing,
                            unknown_neighbor_policy: if ui.renderer_conservative_neighbors {
                                UnknownNeighborOcclusionPolicy::Conservative
                            } else {
                                UnknownNeighborOcclusionPolicy::Aggressive
                            },
                        });
                        if last_mesh_stats.pending_finalize_total >= FINALIZE_BACKPRESSURE_PENDING_START
                            && last_mesh_stats.mesh_pending_promoted_to_drawable <= FINALIZE_BACKPRESSURE_PROMOTION_LOW_WATERMARK
                        {
                            finalize_low_progress_streak = finalize_low_progress_streak
                                .saturating_add(1)
                                .min(FINALIZE_BACKPRESSURE_LOW_PROGRESS_STREAK_MAX);
                        } else {
                            finalize_low_progress_streak = finalize_low_progress_streak.saturating_sub(1);
                        }
                        maintenance_throttle_state = compute_maintenance_throttle_state(
                            last_mesh_stats.pending_finalize_total,
                            last_mesh_stats.mesh_pending_promoted_to_drawable,
                            finalize_low_progress_streak,
                        );

                        let force_maintenance = should_force_maintenance_tick(
                            last_mesh_stats.pending_finalize_total,
                            last_mesh_stats.gpu_dispatch_queue_depth,
                            last_mesh_stats.meshing_completed_depth,
                        );
                        let startup_burst_active = startup_burst_budget_frames > 0
                            && ui.profiler.frame_ms > FRAME_TIME_TARGET_MS * 1.2
                            && last_mesh_stats.pending_finalize_total > MAINTENANCE_PRESSURE_PENDING_FINALIZE_START;
                        let convergence_guard_active = dispatch_starvation_streak
                            >= DISPATCH_STARVATION_STREAK_FRAMES / 2
                            || apply_starvation_streak >= APPLY_STARVATION_STREAK_FRAMES / 2
                            || near_backlog_count >= LOW_LOADED_HIGH_BACKLOG_THRESHOLD / 2;
                        let maintenance_deadline = now >= next_maintenance_tick_at;
                        let run_maintenance_tick = force_maintenance
                            || maintenance_deadline
                            || startup_burst_active
                            || convergence_guard_active;
                        let mut mesh_upload_budget = ui.profiler.mesh_upload_budget_bytes;
                        let mesh_stats = if run_maintenance_tick {
                            let base_upload_budget = adaptive_mesh_upload_budget(
                                ui.profiler.frame_ms,
                                prior_mesh_backlog,
                            );
                            mesh_upload_budget = ((base_upload_budget as f32)
                                * maintenance_throttle_state.upload_scale)
                                as usize;
                            mesh_upload_budget = mesh_upload_budget
                                .clamp(MESH_UPLOAD_BYTES_MIN_PER_FRAME, MESH_UPLOAD_BYTES_MAX_PER_FRAME);
                            let base_remesh_job_budget = adaptive_remesh_job_budget(
                                ui.profiler.frame_ms,
                                prior_dirty_backlog,
                                prior_meshing_queue_depth,
                                visible_chunk_count,
                            );
                            let remesh_job_budget = ((base_remesh_job_budget as f32)
                                * maintenance_throttle_state.remesh_scale)
                                as usize;
                            let remesh_job_budget = remesh_job_budget
                                .clamp(REMESH_JOB_BUDGET_PER_FRAME_MIN, REMESH_JOB_BUDGET_PER_FRAME_MAX);
                            let current_mesh_stats = renderer.rebuild_dirty_store_chunks(
                                &mut store,
                                player_chunk,
                                &cached_desired.generation_scores,
                                remesh_job_budget,
                                mesh_upload_budget,
                                LodRadii {
                                    near: effective_stream_tuning.near_radius_xz,
                                    mid: effective_stream_tuning.mid_radius_xz,
                                    far: effective_stream_tuning.far_radius_xz,
                                    ultra: effective_stream_tuning.ultra_radius_xz,
                                    hysteresis: effective_stream_tuning.lod_hysteresis,
                                },
                                LodMeshingBudgets {
                                    near: effective_stream_tuning.lod_budget_near,
                                    mid: effective_stream_tuning.lod_budget_mid,
                                    far: effective_stream_tuning.lod_budget_far,
                                    ultra: effective_stream_tuning.lod_budget_ultra,
                                },
                            );
                            ui.log_once_per_second("mesh_dispatch_pressure", now_secs, || {
                                format!(
                                    "mesh dispatch submitted/budget={}/{} queue={} headroom={:.2} adopt_latency_ms={:.2} dropped_before_drawable={} filtered_drawable={}",
                                    current_mesh_stats.gpu_dispatch_tasks_submitted,
                                    current_mesh_stats.gpu_dispatch_task_budget,
                                    current_mesh_stats.gpu_dispatch_queue_depth,
                                    current_mesh_stats.gpu_dispatch_headroom,
                                    current_mesh_stats.gpu_mesh_adoption_latency_ms,
                                    current_mesh_stats.mesh_dropped_before_drawable,
                                    current_mesh_stats.mesh_drawable_filtered_under_load,
                                )
                            });
                            let zero_waiting = current_mesh_stats.outcome_skipped_zero_geometry
                                .saturating_sub(current_mesh_stats.outcome_skipped_startup_zero_geometry);
                            let ownership_invalidations = current_mesh_stats.mesh_pending_superseded
                                + current_mesh_stats.mesh_pending_rejected
                                + current_mesh_stats.mesh_reject_invalid_page;
                            ui.log_once_per_second("mesh_finalize_telemetry", now_secs, || {
                                format!(
                                    "pending_finalize={} ready/promoted={}/{} ownership_invalidations={} zero_confirmed/meta_waiting={}/{} resident_visible_gpu={} throttle_active={} severity={:.2} remesh_scale={:.2} upload_scale={:.2} low_progress_streak={} reason={}",
                                    current_mesh_stats.pending_finalize_total,
                                    current_mesh_stats.mesh_pending_total,
                                    current_mesh_stats.mesh_pending_promoted_to_drawable,
                                    ownership_invalidations,
                                    current_mesh_stats.outcome_skipped_startup_zero_geometry,
                                    zero_waiting,
                                    current_mesh_stats.gpu_mesh_visible_count,
                                    maintenance_throttle_state.active,
                                    maintenance_throttle_state.severity,
                                    maintenance_throttle_state.remesh_scale,
                                    maintenance_throttle_state.upload_scale,
                                    finalize_low_progress_streak,
                                    maintenance_throttle_state.reason,
                                )
                            });
                            last_mesh_stats = current_mesh_stats;
                            last_mesh_stats
                        } else {
                            last_mesh_stats
                        };
                        let maintenance_interval_ms = adaptive_maintenance_interval_ms(
                            mesh_stats.pending_finalize_total,
                            mesh_stats.gpu_dispatch_queue_depth,
                            mesh_stats.meshing_completed_depth,
                            ui.profiler.frame_ms,
                            maintenance_throttle_state.interval_scale,
                        )
                        .min(if convergence_guard_active {
                            (MAINTENANCE_TICK_MIN_MS * 1.5).max(FRAME_TIME_TARGET_MS)
                        } else {
                            MAINTENANCE_TICK_MAX_MS
                        });
                        next_maintenance_tick_at = now
                            + Duration::from_secs_f32((maintenance_interval_ms.max(1.0)) / 1000.0);
                        let redraw_interval_ms = adaptive_redraw_interval_ms(
                            maintenance_interval_ms,
                            ui.profiler.frame_ms,
                        );
                        next_redraw_at = now + Duration::from_secs_f32((redraw_interval_ms.max(1.0)) / 1000.0);
                        if startup_burst_budget_frames > 0 {
                            startup_burst_budget_frames -= 1;
                        }
                        ui.set_mesh_timing(mesh_stats.max_ms);
                        ui.profiler.desired_ms = desired_ms;
                        ui.profiler.streaming_ms = streaming_ms;
                        ui.profiler.gen_request_count = gen_request_count;
                        ui.profiler.gen_inflight_count = gen_worker_inflight;
                        ui.profiler.gen_completed_count = gen_completed_count;
                        ui.profiler.gen_completed_total = total_generated_chunks;
                        ui.profiler.apply_ms = apply_ms;
                        ui.profiler.apply_count = apply_count;
                        ui.profiler.evict_ms = evict_ms;
                        ui.profiler.evict_count = evict_count;
                        ui.profiler.mesh_ms = mesh_stats.total_ms;
                        ui.profiler.mesh_count = mesh_stats.mesh_count;
                        let total_received = mesh_stats.mesh_artifacts_received.max(1) as f32;
                        ui.profiler.gpu_mesh_adoption_pct =
                            (mesh_stats.gpu_mesh_adopted_count as f32 / total_received) * 100.0;
                        ui.profiler.cpu_meshing_ms = (mesh_stats.total_ms - mesh_stats.gpu_dispatch_ms).max(0.0);
                        ui.profiler.gpu_meshing_dispatch_ms = mesh_stats.gpu_dispatch_ms;
                        ui.profiler.gpu_readback_bytes_frame = mesh_stats.gpu_readback_bytes;
                        ui.profiler.dirty_backlog = mesh_stats.dirty_backlog + store.dirty_count();
                        ui.profiler.mesh_queue_depth = mesh_stats.meshing_queue_depth;
                        ui.profiler.mesh_completed_depth = mesh_stats.meshing_completed_depth;
                        ui.profiler.mesh_upload_count = mesh_stats.upload_count;
                        ui.profiler.mesh_upload_bytes = mesh_stats.upload_bytes;
                        ui.profiler.mesh_upload_latency_ms = mesh_stats.upload_latency_ms;
                        ui.profiler.mesh_upload_budget_bytes = mesh_upload_budget;
                        ui.profiler.mesh_upload_budget_hit_count = mesh_stats.upload_budget_hit_count;
                        ui.profiler.mesh_upload_budget_deferred_chunks =
                            mesh_stats.upload_budget_deferred_chunks;
                        ui.profiler.mesh_gpu_adopt_count = mesh_stats.gpu_mesh_adopted_count;
                        ui.profiler.mesh_gpu_adopt_latency_ms = mesh_stats.gpu_mesh_adoption_latency_ms;
                        ui.profiler.mesh_gpu_dispatch_tasks_submitted = mesh_stats.gpu_dispatch_tasks_submitted;
                        ui.profiler.mesh_gpu_dispatch_task_budget = mesh_stats.gpu_dispatch_task_budget;
                        ui.profiler.mesh_gpu_dispatch_queue_depth = mesh_stats.gpu_dispatch_queue_depth;
                        ui.profiler.mesh_gpu_dispatch_headroom = mesh_stats.gpu_dispatch_headroom;
                        ui.profiler.mesh_gpu_visible_count = mesh_stats.gpu_mesh_visible_count;
                        ui.profiler.mesh_gpu_visible_slot_min = mesh_stats.gpu_mesh_visible_slot_min;
                        ui.profiler.mesh_gpu_visible_slot_max = mesh_stats.gpu_mesh_visible_slot_max;
                        ui.profiler.mesh_gpu_visible_slot_holes = mesh_stats.gpu_mesh_visible_slot_holes;
                        ui.profiler.mesh_flow_received = mesh_stats.flow_received;
                        ui.profiler.mesh_flow_adopted = mesh_stats.flow_adopted;
                        ui.profiler.mesh_flow_uploaded = mesh_stats.flow_uploaded;
                        ui.profiler.mesh_flow_rejected = mesh_stats.flow_rejected;
                        ui.profiler.mesh_pending_total = mesh_stats.mesh_pending_total;
                        ui.profiler.mesh_pending_promoted_to_drawable =
                            mesh_stats.mesh_pending_promoted_to_drawable;
                        ui.profiler.mesh_pending_waiting_on_fence =
                            mesh_stats.mesh_waiting_on_fence;
                        ui.profiler.mesh_pending_superseded = mesh_stats.mesh_pending_superseded;
                        ui.profiler.mesh_pending_rejected = mesh_stats.mesh_pending_rejected;
                        ui.profiler.mesh_completed_receive_budget_hits = mesh_stats.completed_receive_budget_hits;
                        ui.profiler.mesh_finalize_budget_hits = mesh_stats.finalize_budget_hits;
                        ui.profiler.mesh_adopt_budget_hits = mesh_stats.adopt_budget_hits;
                        ui.profiler.mesh_retry_budget_hits = mesh_stats.retry_budget_hits;
                        ui.profiler.mesh_completed_receive_deferred = mesh_stats.completed_receive_deferred;
                        ui.profiler.mesh_finalize_deferred = mesh_stats.finalize_deferred;
                        ui.profiler.mesh_adopt_deferred = mesh_stats.adopt_deferred;
                        ui.profiler.mesh_retry_deferred = mesh_stats.retry_deferred;
                        ui.profiler.mesh_pending_finalize_age_max = mesh_stats.pending_finalize_age_frames_max;
                        ui.profiler.mesh_pending_finalize_age_p50 = mesh_stats.pending_finalize_age_frames_p50;
                        ui.profiler.mesh_pending_finalize_age_p95 = mesh_stats.pending_finalize_age_frames_p95;
                        ui.profiler.mesh_drawable_filtered_under_load = mesh_stats.mesh_drawable_filtered_under_load;
                        ui.profiler.mesh_reject_stale = mesh_stats.mesh_reject_stale;
                        ui.profiler.mesh_reject_invalid_page = mesh_stats.mesh_reject_invalid_page;
                        ui.profiler.mesh_reject_zero_index = mesh_stats.mesh_reject_zero_index;
                        ui.profiler.mesh_reject_failed = mesh_stats.mesh_reject_failed;
                        ui.profiler.mesh_reject_unhandled = mesh_stats.mesh_reject_unhandled;
                        ui.profiler.mesh_zero_index_soft_retries =
                            mesh_stats.mesh_zero_index_soft_retries;
                        ui.profiler.mesh_resident_gpu_artifact = mesh_stats.resident_gpu_artifact;
                        ui.profiler.mesh_resident_cpu_uploaded = mesh_stats.resident_cpu_uploaded;
                        ui.profiler.mesh_resident_stale_cached = mesh_stats.resident_stale_cached;
                        ui.profiler.mesh_resident_fallback = mesh_stats.resident_fallback;
                        ui.profiler.mesh_resident_unknown = mesh_stats.resident_unknown;
                        ui.profiler.gpu_upload_bytes_frame = mesh_stats.upload_bytes;
                        // App frame does not expose independent GPU-compute telemetry yet; mirror meshing stats instead.
                        ui.profiler.gpu_compute_dispatch_ms = mesh_stats.gpu_dispatch_ms;
                        ui.profiler.gpu_compute_bytes_transferred = mesh_stats.gpu_readback_bytes;
                        ui.profiler.gpu_compute_chunks_completed = mesh_stats.gpu_mesh_adopted_count as u64;
                        ui.profiler.gpu_compute_chunks_per_sec = if ui.profiler.frame_ms > 0.0 {
                            (mesh_stats.gpu_mesh_adopted_count as f32)
                                / (ui.profiler.frame_ms / 1000.0)
                        } else {
                            0.0
                        };
                        ui.profiler.gpu_mesh_slots_used = mesh_stats.gpu_mesh_slots_used;
                        ui.profiler.gpu_mesh_slot_capacity = mesh_stats.gpu_mesh_slot_capacity;
                        ui.profiler.gpu_mesh_slot_in_flight_fences =
                            mesh_stats.gpu_mesh_slot_in_flight_fences;
                        ui.profiler.gpu_mesh_vertex_used = mesh_stats.gpu_mesh_vertex_used;
                        ui.profiler.gpu_mesh_vertex_capacity = mesh_stats.gpu_mesh_vertex_capacity;
                        ui.profiler.gpu_mesh_index_used = mesh_stats.gpu_mesh_index_used;
                        ui.profiler.gpu_mesh_index_capacity = mesh_stats.gpu_mesh_index_capacity;
                        ui.profiler.gpu_mesh_largest_free_vertex_span =
                            mesh_stats.gpu_mesh_largest_free_vertex_span;
                        ui.profiler.gpu_mesh_largest_free_index_span =
                            mesh_stats.gpu_mesh_largest_free_index_span;
                        ui.profiler.mesh_stale_drop_count = mesh_stats.stale_drop_count;
                        ui.profiler.mesh_stale_drop_retry_enqueued =
                            mesh_stats.stale_drop_retry_enqueued;
                        ui.profiler.dirty_urgent_depth = mesh_stats.dirty_urgent_depth;
                        ui.profiler.dirty_near_depth = mesh_stats.dirty_near_depth;
                        ui.profiler.dirty_normal_depth = mesh_stats.dirty_normal_depth;
                        ui.profiler.dirty_far_depth = mesh_stats.dirty_far_depth;
                        ui.profiler.effective_near_radius = effective_stream_tuning.near_radius_xz;
                        ui.profiler.effective_mid_radius = effective_stream_tuning.mid_radius_xz;
                        ui.profiler.effective_far_radius = effective_stream_tuning.far_radius_xz;
                        ui.profiler.effective_ultra_radius = effective_stream_tuning.ultra_radius_xz;
                        ui.profiler.effective_lod_hysteresis = effective_stream_tuning.lod_hysteresis;
                        ui.profiler.effective_lod_budget_near = effective_stream_tuning.lod_budget_near;
                        ui.profiler.effective_lod_budget_mid = effective_stream_tuning.lod_budget_mid;
                        ui.profiler.effective_lod_budget_far = effective_stream_tuning.lod_budget_far;
                        ui.profiler.effective_lod_budget_ultra = effective_stream_tuning.lod_budget_ultra;
                        ui.profiler.auto_tune_active = auto_tune.is_active();
                        ui.profiler.auto_tune_level = auto_tune.degrade_level;
                        ui.profiler.auto_tune_latency_pressure = auto_tune.latency_pressure;
                        ui.profiler.auto_tune_queue_pressure = auto_tune.queue_pressure;
                        ui.profiler.auto_tune_dirty_pressure = auto_tune.dirty_pressure;
                        ui.profiler.mesh_age_drop_count = mesh_stats.age_drop_count;
                        ui.profiler.mesh_pressure_drop_count = mesh_stats.pressure_drop_count;
                        ui.profiler.mesh_gpu_job_failure_count = mesh_stats.gpu_job_failures;
                        ui.profiler.mesh_gpu_job_timeout_count = mesh_stats.gpu_job_timeouts;
                        ui.profiler.mesh_gpu_job_skipped_count = mesh_stats.gpu_job_skipped;
                        ui.profiler.mesh_outcome_startup_zero_geometry_count =
                            mesh_stats.outcome_skipped_startup_zero_geometry;
                        ui.profiler.mesh_startup_seed_zero_count_seen =
                            mesh_stats.startup_seed_zero_count_seen;
                        ui.profiler.mesh_startup_seed_recovered_nonzero =
                            mesh_stats.startup_seed_recovered_nonzero;
                        ui.profiler.mesh_startup_zero_near_retry_enqueued =
                            mesh_stats.startup_zero_near_retry_enqueued;
                        ui.profiler.dirty_queue_drop_count = mesh_stats.dirty_queue_drop_count;
                        ui.profiler.gen_paused_by_worker_queue = gen_dispatch_paused;
                        ui.profiler.desired_budget_drop_count = last_desired_cap_stats.budget_dropped;
                        ui.profiler.configured_near_radius = stream_tuning.near_radius_xz;
                        ui.profiler.configured_mid_radius = stream_tuning.mid_radius_xz;
                        ui.profiler.configured_far_radius = stream_tuning.far_radius_xz;
                        ui.profiler.configured_ultra_radius = stream_tuning.ultra_radius_xz;
                        ui.profiler.configured_lod_hysteresis = stream_tuning.lod_hysteresis;
                        ui.profiler.configured_lod_budget_near = stream_tuning.lod_budget_near;
                        ui.profiler.configured_lod_budget_mid = stream_tuning.lod_budget_mid;
                        ui.profiler.configured_lod_budget_far = stream_tuning.lod_budget_far;
                        ui.profiler.configured_lod_budget_ultra = stream_tuning.lod_budget_ultra;
                        ui.profiler.mesh_near_count = mesh_stats.near_mesh_count;
                        ui.profiler.mesh_mid_count = mesh_stats.mid_mesh_count;
                        ui.profiler.mesh_far_count = mesh_stats.far_mesh_count;
                        prior_mesh_backlog = mesh_stats.meshing_queue_depth + mesh_stats.meshing_completed_depth;
                        prior_dirty_backlog = ui.profiler.dirty_backlog;
                        prior_meshing_queue_depth = mesh_stats.meshing_queue_depth;
                        prior_upload_latency_ms = mesh_stats.upload_latency_ms;
                        ui.profiler.mesh_ultra_count = mesh_stats.ultra_mesh_count;
                        ui.profiler.collision_blocked_unloaded_count = collision_blocked_unloaded_count_total;
                        ui.profiler.collision_blocked_unloaded_ms = collision_blocked_unloaded_ms_total;
                        ui.profiler.sim_ms = sim_ms;
                        ui.profiler.sim_chunk_steps = sim_chunk_steps;
                        ui.profiler.sim_skipped_chunks_non_gas = sim_skipped_chunks_non_gas;
                        ui.profiler.sim_skipped_chunks_gas = sim_skipped_chunks_gas;
                        ui.profiler.sim_boundary_dissipated_particles =
                            sim_boundary_dissipated_particles;
                        ui.profiler.sim_substeps_executed = sim_substeps_executed;
                        ui.profiler.sim_substeps_budget = sim_substeps_budget;
                        ui.profiler.sim_substeps_budget_effective = sim_substeps_budget_effective;
                        ui.profiler.sim_accumulator_steps = sim_acc / FIXED_SIM_STEP_SECONDS;
                        ui.profiler.sim_accumulator_cap_steps =
                            sim_accumulator_cap_seconds / FIXED_SIM_STEP_SECONDS;
                        ui.profiler.sim_accumulator_clamped = sim_accumulator_clamped;

                        let player_biome = biome_hint_at_world(
                            &crate::procgen::ProcGenConfig::for_size(CHUNK_SIZE, streaming.seed),
                            player_world_voxel.x,
                            player_world_voxel.z,
                        );
                        ui.set_biome_hint(
                            player_biome.label(),
                            player_world_voxel.x,
                            player_world_voxel.z,
                        );

                        // Egui
                        let egui_t0 = Instant::now();
                        let raw_input = egui_state.take_egui_input(window);
                        let out = egui_ctx.run(raw_input, |ctx| {
                            let actions =
                                draw(ctx, &mut ui, sim_running, &mut brush, &tool_textures);

                            if actions.toggle_run {
                                sim_running = !sim_running;
                            }
                            if actions.step_once {
                                step_once = true;
                            }
                            if actions.new_world || actions.new_procedural {
                                let next_seed = if actions.new_procedural {
                                    let generated = generate_startup_seed();
                                    let _ = persist_seed(generated);
                                    generated
                                } else {
                                    streaming.seed
                                };
                                store.clear();
                                streaming.reset_with_seed(next_seed);
                                chunk_generator = BackgroundGenerator::new(next_seed, generator_config);
                                last_player_chunk = None;
                                cached_desired = DesiredChunks::default();
                                cached_sim_region.clear();
                                generated_ready.clear();
                                cached_modified_chunks.clear();
                                total_generated_chunks = 0;
                                floating_origin_state.reset();
                                origin_voxel = floating_origin_state.origin_translation;
                                renderer.set_origin_voxel(origin_voxel);
                                ctrl.position = Vec3::new(8.0, 6.0, 8.0);
                                spawn_pending = true;
                                spawn_pending_reason = SpawnPendingReason::Searching;
                                spawn_fallback_cursor = None;
                                println!(
                                    "[world] {} world seed {}",
                                    if actions.new_procedural {
                                        "regenerated procedural"
                                    } else {
                                        "preserved"
                                    },
                                    next_seed
                                );
                            }
                        });
                        egui_state.handle_platform_output(window, out.platform_output);
                        ui.profiler.egui_ms = egui_t0.elapsed().as_secs_f32() * 1000.0;

                        for (id, delta) in &out.textures_delta.set {
                            egui_rpass.update_texture(
                                &renderer.device,
                                &renderer.queue,
                                *id,
                                delta,
                            );
                        }

                        let draw_stats = renderer.mesh_draw_stats(&cam);
                        ui.set_draw_stats(&draw_stats);
                        let cull_stats = renderer.cull_stats(&cam);
                        ui.profiler.culled_chunks = cull_stats.frustum_culled + cull_stats.screen_culled + cull_stats.lod_filtered;
                        ui.profiler.frustum_culled_chunks = cull_stats.frustum_culled;
                        let loaded_chunks = store.iter_loaded_chunks().count();
                        let desired_backlog_visible = cached_desired
                            .generation_order
                            .iter()
                            .filter(|coord| !streaming.resident.contains(coord))
                            .count();
                        ui.profiler.missing_in_radius = desired_backlog_visible;
                        ui.profiler.loaded_chunks = loaded_chunks;
                        ui.profiler.resident_chunks = streaming.resident.len();
                        ui.profiler.scheduled_chunks = streaming.scheduled_generate.len();
                        ui.profiler.generating_chunks = streaming.dispatched_generate.len();
                        ui.profiler.sim_region_chunks = cached_sim_region.len().max(cached_gas_sim_region.len());

                        let chunk_overlay_entries = if ui.show_chunk_overlay {
                            let mut entries = Vec::new();
                            let overlay_radius = 3;
                            for dz in -overlay_radius..=overlay_radius {
                                for dy in -1..=1 {
                                    for dx in -overlay_radius..=overlay_radius {
                                        let coord = ChunkCoord {
                                            x: player_chunk.x + dx,
                                            y: player_chunk.y + dy,
                                            z: player_chunk.z + dz,
                                        };
                                        let color = if store.is_dirty(coord) {
                                            [90, 170, 255, 180]
                                        } else if streaming.dispatched_generate.contains(&coord) {
                                            [255, 150, 60, 180]
                                        } else if streaming.scheduled_generate.contains(&coord) {
                                            [255, 230, 80, 180]
                                        } else if streaming.resident.contains(&coord) {
                                            [80, 235, 120, 180]
                                        } else {
                                            [130, 130, 130, 110]
                                        };
                                        let world_min = crate::types::chunk_to_world_min(coord);
                                        entries.push(ChunkDebugOverlayEntry {
                                            chunk_min: [world_min.x, world_min.y, world_min.z],
                                            color,
                                        });
                                    }
                                }
                            }
                            Some(entries)
                        } else {
                            None
                        };

                        let held_tool = {
                            let tool_texture = tool_textures.for_tool(ui.active_tool);
                            Some((tool_texture.texture.id(), tool_texture.size))
                        };

                        let spawn_debug = if spawn_pending {
                            format!(
                                "seed={} spawn pending: {}",
                                streaming.seed,
                                spawn_pending_reason.label()
                            )
                        } else {
                            format!("seed={}", streaming.seed)
                        };

                        draw_fps_overlays(
                            &egui_ctx,
                            ui.paused_menu,
                            ui.sim_speed,
                            cam.view_proj(),
                            [renderer.config.width, renderer.config.height],
                            &preview_block_list,
                            [0, 0, 0],
                            &brush,
                            preview_mode,
                            ui.show_radial_menu,
                            RADIAL_MENU_TOGGLE_LABEL,
                            VOXEL_SIZE,
                            held_tool,
                            start.elapsed().as_secs_f32(),
                            !gameplay_blocked && !egui_c && (input.lmb || input.rmb),
                            Some(&spawn_debug),
                            chunk_overlay_entries.as_deref(),
                        );

                        // Render
                        let render_submit_t0 = Instant::now();
                        let paint_jobs =
                            egui_ctx.tessellate(out.shapes, window.scale_factor() as f32);
                        let screen_desc = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [renderer.config.width, renderer.config.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        let frame = match renderer.surface.get_current_texture() {
                            Ok(f) => f,
                            Err(_) => {
                                renderer.resize(PhysicalSize::new(
                                    renderer.config.width,
                                    renderer.config.height,
                                ));
                                return;
                            }
                        };
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        let mut encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("main encoder"),
                            },
                        );

                        egui_rpass.update_buffers(
                            &renderer.device,
                            &renderer.queue,
                            &mut encoder,
                            &paint_jobs,
                            &screen_desc,
                        );

                        // World pass
                        {
                            let clear = if renderer.day {
                                wgpu::Color {
                                    r: 0.55,
                                    g: 0.72,
                                    b: 0.95,
                                    a: 1.0,
                                }
                            } else {
                                wgpu::Color {
                                    r: 0.03,
                                    g: 0.05,
                                    b: 0.1,
                                    a: 1.0,
                                }
                            };

                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("world pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(clear),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &renderer.depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            renderer.render_world(&mut pass, &cam);
                        }

                        // UI pass
                        {
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            let _ = egui_rpass.render(&mut pass, &paint_jobs, &screen_desc);
                        }

                        renderer.queue.submit(Some(encoder.finish()));
                        frame.present();
                        ui.profiler.render_submit_ms = render_submit_t0.elapsed().as_secs_f32() * 1000.0;
                        ui.profiler.frame_ms = now.elapsed().as_secs_f32() * 1000.0;
                        frame_counter = frame_counter.wrapping_add(1);

                        for id in &out.textures_delta.free {
                            egui_rpass.free_texture(id);
                        }

                        input.end_frame();
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent { event, .. } => {
                if let DeviceEvent::MouseMotion { .. } = event {
                    input.on_device_event(event);
                }
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let force_redraw = should_force_maintenance_tick(
                    last_mesh_stats.pending_finalize_total,
                    last_mesh_stats.gpu_dispatch_queue_depth,
                    last_mesh_stats.meshing_completed_depth,
                );
                if force_redraw || now >= next_redraw_at {
                    window.request_redraw();
                }
            }
            _ => {}
        })
        .map_err(anyhow::Error::from)
}

fn local_to_world_voxel(local_pos: Vec3, origin: VoxelCoord) -> VoxelCoord {
    VoxelCoord {
        x: local_pos.x.floor() as i32 + origin.x,
        y: local_pos.y.floor() as i32 + origin.y,
        z: local_pos.z.floor() as i32 + origin.z,
    }
}

fn world_spawn_to_local_pos(spawn_world: VoxelCoord, origin: VoxelCoord) -> Vec3 {
    let eye_height_above_surface = 1.0 + grounded_eye_y_blocks();
    Vec3::new(
        spawn_world.x as f32 + 0.5 - origin.x as f32,
        spawn_world.y as f32 + eye_height_above_surface - origin.y as f32,
        spawn_world.z as f32 + 0.5 - origin.z as f32,
    )
}

fn find_safe_spawn_in_loaded_chunks(
    store: &ChunkStore,
    seed: u64,
    fallback_cursor: Option<VoxelCoord>,
) -> Option<SpawnCandidate> {
    let center = VoxelCoord { x: 0, y: 0, z: 0 };
    let mut best_candidate: Option<SpawnCandidate> = None;

    for r in 0..=SPAWN_SEARCH_RADIUS {
        for dz in -r..=r {
            for dx in -r..=r {
                if r > 0 && dx.abs() < r && dz.abs() < r {
                    continue;
                }
                let x = center.x + dx;
                let z = center.z + dz;
                let bias = spawn_bias(seed, x, z);
                if bias < 0.08 {
                    continue;
                }
                if let Some(y) = valid_loaded_spawn_y(store, x, z) {
                    let candidate = SpawnCandidate {
                        voxel: VoxelCoord { x, y, z },
                        surface_y: y,
                    };
                    if best_candidate
                        .map(|best| candidate.surface_y > best.surface_y)
                        .unwrap_or(true)
                    {
                        best_candidate = Some(candidate);
                    }
                }
            }
        }
    }

    if best_candidate.is_some() {
        return best_candidate;
    }

    if let Some(cursor) = fallback_cursor {
        return Some(SpawnCandidate {
            voxel: cursor,
            surface_y: cursor.y,
        });
    }

    highest_loaded_surface(store).map(|surface| SpawnCandidate {
        voxel: VoxelCoord {
            x: center.x,
            y: surface + SPAWN_FALLBACK_EXTRA_HEIGHT,
            z: center.z,
        },
        surface_y: surface,
    })
}

fn valid_loaded_spawn_y(store: &ChunkStore, x: i32, z: i32) -> Option<i32> {
    let min_y = -64;
    let max_y = CHUNK_SIZE as i32 * 3;
    for y in (min_y..=max_y).rev() {
        let base = VoxelCoord { x, y, z };
        if !store.is_voxel_chunk_loaded(base) {
            continue;
        }
        let base_id = store.get_voxel(base);
        if !is_walkable_surface_material(base_id) {
            continue;
        }

        let above = VoxelCoord { x, y: y + 1, z };
        if !store.is_voxel_chunk_loaded(above) || store.get_voxel(above) != EMPTY {
            continue;
        }

        let mut has_headroom = true;
        for dy in 1..=SPAWN_HEADROOM {
            let head = VoxelCoord { x, y: y + dy, z };
            if !store.is_voxel_chunk_loaded(head) || store.get_voxel(head) != EMPTY {
                has_headroom = false;
                break;
            }
        }
        if !has_headroom {
            continue;
        }

        let mut blocked_above = false;
        for dy in 1..=SPAWN_CEILING_PROBE_HEIGHT {
            let probe = VoxelCoord { x, y: y + dy, z };
            if !store.is_voxel_chunk_loaded(probe) {
                break;
            }
            if store.get_voxel(probe) != EMPTY {
                blocked_above = true;
                break;
            }
        }
        if blocked_above {
            continue;
        }

        if !is_topmost_walkable_in_loaded_column(store, x, y, z) {
            continue;
        }

        if !has_safe_neighbors(store, x, y, z) {
            continue;
        }

        return Some(y);
    }
    None
}

fn is_topmost_walkable_in_loaded_column(store: &ChunkStore, x: i32, y: i32, z: i32) -> bool {
    let max_y = CHUNK_SIZE as i32 * 3;
    for probe_y in (y + 1)..=max_y {
        let probe = VoxelCoord { x, y: probe_y, z };
        if !store.is_voxel_chunk_loaded(probe) {
            return true;
        }
        if is_walkable_surface_material(store.get_voxel(probe)) {
            return false;
        }
    }
    true
}

fn has_safe_neighbors(store: &ChunkStore, x: i32, y: i32, z: i32) -> bool {
    for dz in -1..=1 {
        for dx in -1..=1 {
            let nx = x + dx;
            let nz = z + dz;
            let Some(neighbor_top_y) = highest_loaded_solid_y(store, nx, nz) else {
                return false;
            };
            if (neighbor_top_y - y).abs() > SPAWN_MAX_SLOPE_DELTA {
                return false;
            }
            let support = VoxelCoord {
                x: nx,
                y: neighbor_top_y,
                z: nz,
            };
            let stand = VoxelCoord {
                x: nx,
                y: neighbor_top_y + 1,
                z: nz,
            };
            if !store.is_voxel_chunk_loaded(support) || !store.is_voxel_chunk_loaded(stand) {
                return false;
            }
            let support_id = store.get_voxel(support);
            if !is_walkable_surface_material(support_id) {
                return false;
            }
            if store.get_voxel(stand) != EMPTY {
                return false;
            }
        }
    }
    true
}

fn is_walkable_surface_material(id: u16) -> bool {
    id != EMPTY && !matches!(id, 3 | 4 | 5 | 6 | 7 | 8 | 9 | 11 | 13 | 14 | 15)
}

fn highest_loaded_surface(store: &ChunkStore) -> Option<i32> {
    let mut best = None;
    for r in 0..=SPAWN_SEARCH_RADIUS {
        for dz in -r..=r {
            for dx in -r..=r {
                if r > 0 && dx.abs() < r && dz.abs() < r {
                    continue;
                }
                if let Some(y) = highest_loaded_solid_y(store, dx, dz) {
                    best = Some(best.map(|b: i32| b.max(y)).unwrap_or(y));
                }
            }
        }
    }
    best
}

fn highest_loaded_solid_y(store: &ChunkStore, x: i32, z: i32) -> Option<i32> {
    let min_y = -64;
    let max_y = CHUNK_SIZE as i32 * 3;
    for y in (min_y..=max_y).rev() {
        let voxel = VoxelCoord { x, y, z };
        if !store.is_voxel_chunk_loaded(voxel) {
            continue;
        }
        if store.get_voxel(voxel) != EMPTY {
            return Some(y);
        }
    }
    None
}

fn is_spawn_collision_free(
    store: &ChunkStore,
    player_local_pos: Vec3,
    origin_voxel: VoxelCoord,
) -> bool {
    let player_world_pos = player_local_pos
        + Vec3::new(
            origin_voxel.x as f32,
            origin_voxel.y as f32,
            origin_voxel.z as f32,
        );
    let (min, max) = FpsController::collision_sample_bounds_world(player_world_pos);

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let v = VoxelCoord { x, y, z };
                if !store.is_voxel_chunk_loaded(v) || store.get_voxel(v) != EMPTY {
                    return false;
                }
            }
        }
    }

    true
}

fn spawn_bias(seed: u64, x: i32, z: i32) -> f32 {
    let mut h = seed ^ 0xABCD_0001;
    h ^= (x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (z as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

fn within_collision_safety_radius(
    voxel: VoxelCoord,
    player_local_pos: Vec3,
    origin_voxel: VoxelCoord,
    radius_voxels: i32,
) -> bool {
    let player_world_voxel = local_to_world_voxel(player_local_pos, origin_voxel);
    (voxel.x - player_world_voxel.x).abs() <= radius_voxels
        && (voxel.y - player_world_voxel.y).abs() <= radius_voxels
        && (voxel.z - player_world_voxel.z).abs() <= radius_voxels
}

fn collision_neighborhood_loaded(
    store: &ChunkStore,
    player_local_pos: Vec3,
    origin_voxel: VoxelCoord,
    safety_radius_voxels: i32,
) -> bool {
    let player_world_pos = player_local_pos
        + Vec3::new(
            origin_voxel.x as f32,
            origin_voxel.y as f32,
            origin_voxel.z as f32,
        );
    let (min, max) = FpsController::collision_sample_bounds_world(player_world_pos);

    for z in (min.z - safety_radius_voxels)..=(max.z + safety_radius_voxels) {
        for y in (min.y - safety_radius_voxels)..=(max.y + safety_radius_voxels) {
            for x in (min.x - safety_radius_voxels)..=(max.x + safety_radius_voxels) {
                if !store.is_voxel_chunk_loaded(VoxelCoord { x, y, z }) {
                    return false;
                }
            }
        }
    }

    true
}
fn collision_neighborhood_missing_chunks(
    store: &ChunkStore,
    player_local_pos: Vec3,
    origin_voxel: VoxelCoord,
    safety_radius_voxels: i32,
) -> Vec<ChunkCoord> {
    let player_world_pos = player_local_pos
        + Vec3::new(
            origin_voxel.x as f32,
            origin_voxel.y as f32,
            origin_voxel.z as f32,
        );
    let (player_chunk, _) = voxel_to_chunk(VoxelCoord {
        x: player_world_pos.x.floor() as i32,
        y: player_world_pos.y.floor() as i32,
        z: player_world_pos.z.floor() as i32,
    });
    let (min, max) = FpsController::collision_sample_bounds_world(player_world_pos);

    let mut missing = HashSet::new();
    for z in (min.z - safety_radius_voxels)..=(max.z + safety_radius_voxels) {
        for y in (min.y - safety_radius_voxels)..=(max.y + safety_radius_voxels) {
            for x in (min.x - safety_radius_voxels)..=(max.x + safety_radius_voxels) {
                let voxel = VoxelCoord { x, y, z };
                if store.is_voxel_chunk_loaded(voxel) {
                    continue;
                }
                let (coord, _) = voxel_to_chunk(voxel);
                missing.insert(coord);
            }
        }
    }

    let mut ordered: Vec<ChunkCoord> = missing.into_iter().collect();
    ordered.sort_by_key(|&coord| ChunkStreaming::sort_key(player_chunk, coord));
    ordered
}

fn build_adaptive_sim_regions(
    player_chunk: ChunkCoord,
    solid_radius: i32,
    gas_vertical_range_chunks: i32,
    active_emitters: &HashSet<ChunkCoord>,
) -> (HashSet<ChunkCoord>, HashSet<ChunkCoord>) {
    let solid_region = chunk_cube(player_chunk, solid_radius);
    let mut gas_region = solid_region.clone();

    let gas_radius_xz = solid_radius + 1;
    let gas_down = solid_radius;
    for dz in -gas_radius_xz..=gas_radius_xz {
        for dy in -gas_down..=gas_vertical_range_chunks {
            for dx in -gas_radius_xz..=gas_radius_xz {
                gas_region.insert(ChunkCoord {
                    x: player_chunk.x + dx,
                    y: player_chunk.y + dy,
                    z: player_chunk.z + dz,
                });
            }
        }
    }

    for &emitter_chunk in active_emitters {
        for dz in -1..=1 {
            for dy in -1..=gas_vertical_range_chunks {
                for dx in -1..=1 {
                    gas_region.insert(ChunkCoord {
                        x: emitter_chunk.x + dx,
                        y: emitter_chunk.y + dy,
                        z: emitter_chunk.z + dz,
                    });
                }
            }
        }
    }

    (solid_region, gas_region)
}

fn chunk_cube(center: ChunkCoord, radius: i32) -> HashSet<ChunkCoord> {
    let mut out = HashSet::new();
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                out.insert(ChunkCoord {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                });
            }
        }
    }
    out
}

fn assign_or_select_hotbar(ui: &mut UiState, slot: usize, tab_palette_held: bool) {
    if tab_palette_held {
        if let Some(material_id) = ui.hovered_palette_material {
            assign_hotbar_slot(ui, slot, material_id);
            return;
        }
    }
    ui.selected_slot = slot.min(HOTBAR_SLOTS - 1);
}

fn key_to_hotbar_slot(key: KeyCode) -> Option<usize> {
    match key {
        KeyCode::Digit0 => Some(0),
        KeyCode::Digit1 => Some(1),
        KeyCode::Digit2 => Some(2),
        KeyCode::Digit3 => Some(3),
        KeyCode::Digit4 => Some(4),
        KeyCode::Digit5 => Some(5),
        KeyCode::Digit6 => Some(6),
        KeyCode::Digit7 => Some(7),
        KeyCode::Digit8 => Some(8),
        KeyCode::Digit9 => Some(9),
        _ => None,
    }
}

fn apply_quick_menu_hover_selection(ui: &mut UiState, brush: &mut BrushSettings) {
    if let Some(hovered) = ui.hovered_shape.take() {
        brush.shape = hovered;
    }
    if let Some(hovered) = ui.hovered_area_shape.take() {
        brush.area_tool.shape = hovered;
    }
    if let Some(hovered) = ui.hovered_tool.take() {
        ui.active_tool = hovered;
    }
}

fn should_unlock_cursor(ui: &UiState, quick_menu_held: bool, tab_palette_held: bool) -> bool {
    ui.paused_menu || quick_menu_held || tab_palette_held || ui.show_tool_quick_menu
}

fn set_cursor(window: &winit::window::Window, unlock: bool) -> anyhow::Result<()> {
    window.set_cursor_visible(unlock);
    if unlock {
        window.set_cursor_grab(CursorGrabMode::None)?;
    } else {
        let _ = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
    }
    Ok(())
}

fn current_action_mode(input: &InputState, raycast: RaycastResult, tool: ToolKind) -> BrushMode {
    if tool == ToolKind::BuildersWand {
        return BrushMode::Place;
    }
    if tool == ToolKind::DestructorWand {
        return BrushMode::Erase;
    }
    if tool == ToolKind::MaterialJet {
        return BrushMode::Place;
    }
    if input.rmb {
        BrushMode::Erase
    } else if input.lmb || raycast.hit.is_none() {
        BrushMode::Place
    } else {
        BrushMode::Erase
    }
}

fn held_action_mode(input: &InputState) -> Option<BrushMode> {
    if input.lmb {
        Some(BrushMode::Place)
    } else if input.rmb {
        Some(BrushMode::Erase)
    } else {
        None
    }
}

fn preview_blocks(
    store: &ChunkStore,
    brush: &BrushSettings,
    raycast: RaycastResult,
    mode: BrushMode,
    tool: ToolKind,
) -> Vec<[i32; 3]> {
    if tool == ToolKind::AreaTool {
        return preview_area_tool_blocks(brush, raycast, mode);
    }
    if tool == ToolKind::MaterialJet {
        return vec![raycast.place];
    }
    if (tool == ToolKind::BuildersWand || tool == ToolKind::DestructorWand) && brush.radius == 0 {
        return preview_wand_blocks(store, raycast, tool, 256);
    }
    brush_center(*brush, raycast, mode)
        .map(|center| preview_brush_volume(*brush, center))
        .unwrap_or_default()
}

fn preview_brush_volume(brush: BrushSettings, center: [i32; 3]) -> Vec<[i32; 3]> {
    let radius = brush.radius.max(0);
    let mut out = Vec::new();
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let include = match brush.shape {
                    BrushShape::Cube => true,
                    BrushShape::Sphere => dx * dx + dy * dy + dz * dz <= radius * radius,
                    BrushShape::Torus => {
                        let ring_radius = (radius as f32).max(1.0);
                        let tube_radius = (radius as f32 * 0.5).max(1.0);
                        let q = ((dx * dx + dz * dz) as f32).sqrt() - ring_radius;
                        (q * q + (dy as f32) * (dy as f32)) <= tube_radius * tube_radius
                    }
                    BrushShape::Hemisphere => {
                        dy >= 0 && (dx * dx + dy * dy + dz * dz) <= radius * radius
                    }
                    BrushShape::Bowl => {
                        if dy > 0 {
                            false
                        } else {
                            let r2 = dx * dx + dy * dy + dz * dz;
                            let outer = r2 <= radius * radius;
                            let inner_radius = (radius - 1).max(0);
                            let inner = r2 < inner_radius * inner_radius;
                            outer && !inner
                        }
                    }
                    BrushShape::InvertedBowl => {
                        if dy < 0 {
                            false
                        } else {
                            let r2 = dx * dx + dy * dy + dz * dz;
                            let outer = r2 <= radius * radius;
                            let inner_radius = (radius - 1).max(0);
                            let inner = r2 < inner_radius * inner_radius;
                            outer && !inner
                        }
                    }
                };
                if include {
                    out.push([center[0] + dx, center[1] + dy, center[2] + dz]);
                }
            }
        }
    }
    out
}

fn apply_mouse_edit(
    store: &mut ChunkStore,
    brush: &BrushSettings,
    mat: u16,
    input: &InputState,
    edit_runtime: &mut EditRuntimeState,
    now: Instant,
    raycast: RaycastResult,
    active_tool: ToolKind,
    simulation_runtime: &mut SimulationRuntime,
    sim_mode: SimulationMode,
) -> bool {
    let requested_mode = held_action_mode(input);
    let Some(mode) = requested_mode else {
        edit_runtime.last_edit_mode = None;
        return false;
    };

    let is_just_click = (mode == BrushMode::Place && input.just_lmb)
        || (mode == BrushMode::Erase && input.just_rmb);
    let repeat_interval_s = brush.repeat_interval_s.max(0.0);
    let repeat_ready = edit_runtime.last_edit_mode != Some(mode)
        || edit_runtime
            .last_edit_at
            .map(|last| (now - last).as_secs_f32() >= repeat_interval_s)
            .unwrap_or(true);

    if !is_just_click && !repeat_ready {
        return false;
    }

    let target = if mode == BrushMode::Place { mat } else { 0 };
    for p in preview_blocks(store, brush, raycast, mode, active_tool) {
        let coord = VoxelCoord {
            x: p[0],
            y: p[1],
            z: p[2],
        };
        store.set_voxel(coord, target);
        let (chunk_coord, _) = voxel_to_chunk(coord);
        store.mark_dirty_urgent(chunk_coord);
        simulation_runtime.queue_place_edit(sim_mode, coord, target);
    }
    edit_runtime.last_edit_at = Some(now);
    edit_runtime.last_edit_mode = Some(mode);
    true
}

fn apply_material_jet(
    store: &mut ChunkStore,
    brush: &BrushSettings,
    mat: u16,
    input: &InputState,
    edit_runtime: &mut EditRuntimeState,
    now: Instant,
    look_dir: Vec3,
    raycast: RaycastResult,
    simulation_runtime: &mut SimulationRuntime,
    sim_mode: SimulationMode,
) -> bool {
    if !input.lmb {
        edit_runtime.last_edit_mode = None;
        return false;
    }

    let repeat_interval_s = brush.repeat_interval_s.max(0.0);
    let repeat_ready = edit_runtime.last_edit_mode != Some(BrushMode::Place)
        || edit_runtime
            .last_edit_at
            .map(|last| (now - last).as_secs_f32() >= repeat_interval_s)
            .unwrap_or(true);
    if !repeat_ready {
        return false;
    }

    let cfg = brush.material_jet;
    let dir = look_dir.normalize_or_zero();
    if dir.length_squared() <= f32::EPSILON {
        return false;
    }
    let origin = Vec3::new(
        raycast.place[0] as f32,
        raycast.place[1] as f32,
        raycast.place[2] as f32,
    );
    let right = dir.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(dir).normalize_or_zero();
    let mut placed = false;
    let speed_scale = (cfg.launch_velocity / 20.0).clamp(0.25, 4.0);
    for i in 0..cfg.flow_rate {
        let fi = i as f32;
        let jitter =
            ((fi * 12.9898 + now.elapsed().as_secs_f32() * 17.13).sin() * 43758.5453).fract();
        let angle = fi * 2.3999632;
        let radius = cfg.spread * jitter;
        let spread = right * angle.cos() * radius + up * angle.sin() * radius;
        let travel = (1.0 + fi * 0.25) * speed_scale;
        let p = origin + (dir + spread).normalize_or_zero() * travel;
        let target = VoxelCoord {
            x: p.x.round() as i32,
            y: p.y.round() as i32,
            z: p.z.round() as i32,
        };
        let dist = Vec3::new(
            (target.x - raycast.place[0]) as f32,
            (target.y - raycast.place[1]) as f32,
            (target.z - raycast.place[2]) as f32,
        )
        .length();
        if dist > cfg.max_range {
            continue;
        }
        store.set_voxel(target, mat);
        let (chunk_coord, _) = voxel_to_chunk(target);
        store.mark_dirty_urgent(chunk_coord);
        simulation_runtime.queue_place_edit(sim_mode, target, mat);
        placed = true;
    }

    if placed {
        edit_runtime.last_edit_at = Some(now);
        edit_runtime.last_edit_mode = Some(BrushMode::Place);
    }
    placed
}

fn target_for_edit(
    store: &ChunkStore,
    origin_voxel: VoxelCoord,
    local_origin: Vec3,
    dir: Vec3,
    brush: &BrushSettings,
) -> RaycastResult {
    let world_origin = local_origin
        + Vec3::new(
            origin_voxel.x as f32,
            origin_voxel.y as f32,
            origin_voxel.z as f32,
        );
    if brush.fixed_distance && !brush.minecraft_style_placement {
        RaycastResult {
            hit: None,
            place: fixed_distance_target(world_origin, dir, brush.max_distance),
        }
    } else {
        raycast_target(store, world_origin, dir, brush.max_distance)
    }
}

fn brush_center(brush: BrushSettings, raycast: RaycastResult, mode: BrushMode) -> Option<[i32; 3]> {
    if brush.minecraft_style_placement {
        return match mode {
            BrushMode::Place => raycast.hit.map(|_| raycast.place),
            BrushMode::Erase => raycast.hit,
        };
    }
    if brush.fixed_distance {
        return Some(raycast.place);
    }
    match mode {
        BrushMode::Place => Some(raycast.place),
        BrushMode::Erase => raycast.hit,
    }
}

fn fixed_distance_target(origin: Vec3, dir: Vec3, max_dist: f32) -> [i32; 3] {
    let d = dir.normalize_or_zero();
    let point = origin + d * max_dist.max(0.0);
    [
        point.x.floor() as i32,
        point.y.floor() as i32,
        point.z.floor() as i32,
    ]
}

fn raycast_target(store: &ChunkStore, origin: Vec3, dir: Vec3, max_dist: f32) -> RaycastResult {
    let d = dir.normalize_or_zero();
    if d.length_squared() == 0.0 {
        return RaycastResult {
            hit: None,
            place: [
                origin.x.floor() as i32,
                origin.y.floor() as i32,
                origin.z.floor() as i32,
            ],
        };
    }

    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;
    let mut prev = [x, y, z];

    let step_x = if d.x >= 0.0 { 1 } else { -1 };
    let step_y = if d.y >= 0.0 { 1 } else { -1 };
    let step_z = if d.z >= 0.0 { 1 } else { -1 };

    let next_boundary = |cell: i32, step: i32| {
        if step > 0 {
            cell as f32 + 1.0
        } else {
            cell as f32
        }
    };

    let mut t_max_x = if d.x.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (next_boundary(x, step_x) - origin.x) / d.x
    };
    let mut t_max_y = if d.y.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (next_boundary(y, step_y) - origin.y) / d.y
    };
    let mut t_max_z = if d.z.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (next_boundary(z, step_z) - origin.z) / d.z
    };

    let t_delta_x = if d.x.abs() < 1e-6 {
        f32::INFINITY
    } else {
        1.0 / d.x.abs()
    };
    let t_delta_y = if d.y.abs() < 1e-6 {
        f32::INFINITY
    } else {
        1.0 / d.y.abs()
    };
    let t_delta_z = if d.z.abs() < 1e-6 {
        f32::INFINITY
    } else {
        1.0 / d.z.abs()
    };

    let mut t = 0.0;
    while t <= max_dist {
        if store.get_voxel(VoxelCoord { x, y, z }) != 0 {
            return RaycastResult {
                hit: Some([x, y, z]),
                place: prev,
            };
        }
        prev = [x, y, z];
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                x += step_x;
                t = t_max_x;
                t_max_x += t_delta_x;
            } else {
                z += step_z;
                t = t_max_z;
                t_max_z += t_delta_z;
            }
        } else if t_max_y < t_max_z {
            y += step_y;
            t = t_max_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
        }
    }

    let miss = origin + d * max_dist;
    RaycastResult {
        hit: None,
        place: [
            miss.x.floor() as i32,
            miss.y.floor() as i32,
            miss.z.floor() as i32,
        ],
    }
}

fn preview_area_tool_blocks(
    brush: &BrushSettings,
    raycast: RaycastResult,
    mode: BrushMode,
) -> Vec<[i32; 3]> {
    let Some(center) = area_tool_center(raycast, mode) else {
        return Vec::new();
    };
    let radius = brush.area_tool.radius.max(0);
    let thickness = brush.area_tool.thickness.max(1);

    let normal = [
        raycast.place[0] - raycast.hit.unwrap_or(raycast.place)[0],
        raycast.place[1] - raycast.hit.unwrap_or(raycast.place)[1],
        raycast.place[2] - raycast.hit.unwrap_or(raycast.place)[2],
    ];

    let (axis_u, axis_v) = if normal[1] != 0 {
        ([1, 0, 0], [0, 0, 1])
    } else if normal[0] != 0 {
        ([0, 1, 0], [0, 0, 1])
    } else {
        ([1, 0, 0], [0, 1, 0])
    };

    let mut out = Vec::new();
    for dv in -radius..=radius {
        for du in -radius..=radius {
            let include = match brush.area_tool.shape {
                AreaFootprintShape::Circle => du * du + dv * dv <= radius * radius,
                AreaFootprintShape::Square => true,
            };
            if !include {
                continue;
            }
            for depth in 0..thickness {
                out.push([
                    center[0] + axis_u[0] * du + axis_v[0] * dv + normal[0] * depth,
                    center[1] + axis_u[1] * du + axis_v[1] * dv + normal[1] * depth,
                    center[2] + axis_u[2] * du + axis_v[2] * dv + normal[2] * depth,
                ]);
            }
        }
    }
    out
}

fn area_tool_center(raycast: RaycastResult, mode: BrushMode) -> Option<[i32; 3]> {
    match mode {
        BrushMode::Place => Some(raycast.place),
        BrushMode::Erase => raycast.hit.or(Some(raycast.place)),
    }
}

fn preview_wand_blocks(
    store: &ChunkStore,
    raycast: RaycastResult,
    active_tool: ToolKind,
    max_blocks: usize,
) -> Vec<[i32; 3]> {
    if active_tool == ToolKind::Brush || active_tool == ToolKind::AreaTool {
        return Vec::new();
    }
    let Some(hit) = raycast.hit else {
        return Vec::new();
    };

    let normal = [
        raycast.place[0] - hit[0],
        raycast.place[1] - hit[1],
        raycast.place[2] - hit[2],
    ];

    let source_mat = store.get_voxel(VoxelCoord {
        x: hit[0],
        y: hit[1],
        z: hit[2],
    });
    if source_mat == 0 {
        return Vec::new();
    }

    let axis = if normal[0] != 0 {
        0
    } else if normal[1] != 0 {
        1
    } else {
        2
    };

    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut out = Vec::new();
    queue.push_back(hit);
    visited.insert(hit);

    while let Some(p) = queue.pop_front() {
        if out.len() >= max_blocks {
            break;
        }

        let target = if active_tool == ToolKind::BuildersWand {
            [p[0] + normal[0], p[1] + normal[1], p[2] + normal[2]]
        } else {
            p
        };

        let target_mat = store.get_voxel(VoxelCoord {
            x: target[0],
            y: target[1],
            z: target[2],
        });
        if target_mat == 0 || active_tool == ToolKind::DestructorWand {
            out.push(target);
        }

        for dir in [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ] {
            if dir[axis] != 0 {
                continue;
            }
            let np = [p[0] + dir[0], p[1] + dir[1], p[2] + dir[2]];
            if visited.contains(&np) {
                continue;
            }
            let np_mat = store.get_voxel(VoxelCoord {
                x: np[0],
                y: np[1],
                z: np[2],
            });
            if np_mat != source_mat {
                continue;
            }
            visited.insert(np);
            queue.push_back(np);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn ring_priority_orders_near_then_mid_then_far_without_capping() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = vec![ChunkCoord { x: 1, y: 0, z: 0 }];
        let mid = vec![ChunkCoord { x: 3, y: 0, z: 0 }];
        let far = vec![ChunkCoord { x: 6, y: 0, z: 0 }];

        let desired = DesiredChunks {
            near: near.clone(),
            mid: mid.clone(),
            far: far.clone(),
            ultra: Vec::new(),
            generation_order: Vec::new(),
            generation_scores: HashMap::new(),
            resident_keep: HashSet::new(),
        };

        let pressure_model = GenerationPressureModel {
            near_cap: 8,
            mid_cap: 8,
            far_cap: 8,
            ultra_cap: 8,
            global_cap: 32,
            near_dispatch_budget: 4,
            mid_dispatch_budget: 2,
            far_dispatch_budget: 2,
        };

        let (prioritized, stats) = cap_desired_generation_order(&desired, player, pressure_model);
        assert_eq!(prioritized, vec![near[0], mid[0], far[0]]);
        assert_eq!(stats.near_kept, 1);
        assert_eq!(stats.mid_kept, 1);
        assert_eq!(stats.far_kept, 1);
        assert_eq!(stats.budget_dropped, 0);
    }

    #[test]
    fn generation_order_hard_cap_enforced() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let near = vec![
            ChunkCoord { x: 2, y: 0, z: 0 },
            ChunkCoord { x: 3, y: 0, z: 0 },
        ];
        let mid = vec![
            ChunkCoord { x: 5, y: 0, z: 0 },
            ChunkCoord { x: 6, y: 0, z: 0 },
        ];
        let far = vec![
            ChunkCoord { x: 9, y: 0, z: 0 },
            ChunkCoord { x: 10, y: 0, z: 0 },
        ];

        let desired = DesiredChunks {
            near,
            mid,
            far,
            ultra: vec![ChunkCoord { x: 13, y: 0, z: 0 }],
            generation_order: Vec::new(),
            generation_scores: HashMap::new(),
            resident_keep: HashSet::new(),
        };

        let pressure_model = GenerationPressureModel {
            near_cap: 1,
            mid_cap: 1,
            far_cap: 1,
            ultra_cap: 1,
            global_cap: 3,
            near_dispatch_budget: 2,
            mid_dispatch_budget: 1,
            far_dispatch_budget: 1,
        };

        let (prioritized, stats) = cap_desired_generation_order(&desired, player, pressure_model);
        assert_eq!(prioritized.len(), 3);
        assert_eq!(stats.near_kept, 1);
        assert_eq!(stats.mid_kept, 1);
        assert_eq!(stats.far_kept, 1);
        assert_eq!(stats.ultra_kept, 0);
        assert_eq!(stats.near_dropped, 1);
        assert_eq!(stats.mid_dropped, 1);
        assert_eq!(stats.far_dropped, 1);
        assert_eq!(stats.ultra_dropped, 1);
        assert_eq!(stats.budget_dropped, 4);
    }

    #[test]
    fn generation_order_protects_urgent_and_near_before_mid_far_ultra() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let urgent = ChunkCoord { x: 1, y: 0, z: 0 };
        let near = ChunkCoord { x: 3, y: 0, z: 0 };
        let desired = DesiredChunks {
            near: vec![urgent, near],
            mid: vec![ChunkCoord { x: 6, y: 0, z: 0 }],
            far: vec![ChunkCoord { x: 9, y: 0, z: 0 }],
            ultra: vec![ChunkCoord { x: 12, y: 0, z: 0 }],
            generation_order: Vec::new(),
            generation_scores: HashMap::new(),
            resident_keep: HashSet::new(),
        };

        let pressure_model = GenerationPressureModel {
            near_cap: 2,
            mid_cap: 2,
            far_cap: 2,
            ultra_cap: 2,
            global_cap: 2,
            near_dispatch_budget: 2,
            mid_dispatch_budget: 1,
            far_dispatch_budget: 1,
        };

        let (prioritized, stats) = cap_desired_generation_order(&desired, player, pressure_model);
        assert_eq!(prioritized, vec![urgent, near]);
        assert_eq!(stats.uncapped_kept, 1);
        assert_eq!(stats.near_kept, 2);
        assert_eq!(stats.mid_dropped, 1);
        assert_eq!(stats.far_dropped, 1);
        assert_eq!(stats.ultra_dropped, 1);
    }

    #[test]
    fn pressure_model_avoids_mid_far_starvation_when_near_stable() {
        let tuning = StreamingTuning::default();
        let pressure = compute_generation_pressure_model(
            &tuning,
            8,
            4,
            12,
            16,
            12,
            tuning.base_generate_drain_items,
        );
        assert!(pressure.near_dispatch_budget < tuning.max_generate_drain_items);
        assert!(pressure.mid_dispatch_budget > 0);
        assert!(pressure.far_dispatch_budget > 0);
    }
}
