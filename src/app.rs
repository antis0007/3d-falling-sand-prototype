use crate::chunk_store::ChunkStore;
use crate::input::{FpsController, InputState};
use crate::player::camera_world_pos_from_blocks;
use crate::procgen::{apply_generated_chunk, generate_chunk};
use crate::renderer::{
    Camera, LodMeshingBudgets, LodRadii, Renderer, RendererSettings,
    UnknownNeighborOcclusionPolicy, VOXEL_SIZE,
};
use crate::sim_world::{step_region_profiled, Rng};
use crate::streaming::{ChunkStreaming, DesiredChunks};
use crate::types::{voxel_to_chunk, ChunkCoord, VoxelCoord};
use crate::ui::{
    assign_hotbar_slot, draw, draw_fps_overlays, load_tool_textures, selected_material, ToolKind,
    UiState, HOTBAR_SLOTS,
};
use crate::world::{AreaFootprintShape, BrushMode, BrushSettings, BrushShape, CHUNK_SIZE, EMPTY};
use glam::Vec3;
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

const RADIAL_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyE;
const RADIAL_MENU_TOGGLE_LABEL: &str = "E";
const TOOL_QUICK_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyQ;
const TOOL_TEXTURES_DIR: &str = "assets/tools";
const ORIGIN_SHIFT_THRESHOLD: f32 = 128.0;
const REMESH_JOB_BUDGET_PER_FRAME: usize = 24;
const MESH_UPLOAD_BYTES_MIN_PER_FRAME: usize = 512 * 1024;
const MESH_UPLOAD_BYTES_BASE_PER_FRAME: usize = 2 * 1024 * 1024;
const MESH_UPLOAD_BYTES_MAX_PER_FRAME: usize = 8 * 1024 * 1024;
const FRAME_TIME_TARGET_MS: f32 = 1000.0 / 60.0;
const SIMULATION_RADIUS_CHUNKS: i32 = 1; // 3x3x3 = 27 chunks max

const GENERATOR_THREADS: usize = 3;
const GENERATOR_QUEUE_BOUND: usize = 192;
const APPLY_BUDGET_MS: f32 = 1.5;
const EVICT_BUDGET_MS: f32 = 1.0;
const MESH_BACKPRESSURE_START: usize = 80;
const MESH_BACKPRESSURE_HIGH: usize = 180;
const DIRTY_BACKLOG_PRESSURE_START: usize = 96;
const DIRTY_BACKLOG_PRESSURE_HIGH: usize = 320;
const AUTO_TUNE_UPLOAD_LATENCY_START_MS: f32 = 200.0;
const AUTO_TUNE_UPLOAD_LATENCY_HIGH_MS: f32 = 450.0;
const AUTO_TUNE_RAMP_UP_PER_SEC: f32 = 3.5;
const AUTO_TUNE_RECOVER_PER_SEC: f32 = 0.6;
const DESIRED_VIEW_RECOMPUTE_DOT_DELTA: f32 = 0.01;

const COLLISION_SAFETY_RADIUS_VOXELS: i32 = 4;
const FREEZE_UNTIL_COLLISION_RESIDENT: bool = true;
const COLLISION_FREEZE_MAX_DURATION: Duration = Duration::from_millis(1500);
const COLLISION_FREEZE_BACKOFF_DURATION: Duration = Duration::from_millis(900);
const COLLISION_LOCAL_PRIORITY_REQUEST_BUDGET: usize = 12;
const SPAWN_SEARCH_RADIUS: i32 = 48;
const SPAWN_HEADROOM: i32 = 3;
const SPAWN_CLEARANCE: f32 = 3.4;
const SPAWN_FALLBACK_EXTRA_HEIGHT: i32 = 12;

#[derive(Clone)]
struct GenJob {
    coord: ChunkCoord,
}

struct GenResult {
    coord: ChunkCoord,
    chunk: crate::chunk_store::Chunk,
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
            base_generate_drain_items: 24,
            max_generate_drain_items: 96,
            lod_budget_near: 8,
            lod_budget_mid: 6,
            lod_budget_far: 10,
            lod_budget_ultra: 6,
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

struct BackgroundGenerator {
    tx: SyncSender<GenJob>,
    rx: Receiver<GenResult>,
}

impl BackgroundGenerator {
    fn new(seed: u64) -> Self {
        let (tx, job_rx) = sync_channel::<GenJob>(GENERATOR_QUEUE_BOUND);
        let (result_tx, rx) = sync_channel::<GenResult>(GENERATOR_QUEUE_BOUND);
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

        for i in 0..GENERATOR_THREADS {
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

    fn try_request(&self, coord: ChunkCoord) -> bool {
        match self.tx.try_send(GenJob { coord }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    fn try_recv(&self) -> Result<GenResult, TryRecvError> {
        self.rx.try_recv()
    }
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

    let mut renderer = Renderer::new(window).await?;
    let egui_ctx = egui::Context::default();
    egui_ctx.set_visuals(egui::Visuals::dark());
    let mut egui_state =
        egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, window, None, None);
    let mut egui_rpass =
        egui_wgpu::Renderer::new(&renderer.device, renderer.config.format, None, 1);

    let mut store = ChunkStore::new();
    let mut streaming = ChunkStreaming::new(1337);
    let chunk_generator = BackgroundGenerator::new(streaming.seed);
    let mut generated_ready: VecDeque<GenResult> = VecDeque::new();
    let mut stream_tuning = StreamingTuning::default().normalized();
    let mut cached_stream_tuning = stream_tuning.clone();
    let mut auto_tune = AutoTuneState::default();
    let mut rng = Rng::new(0x1234_5678);
    let mut sim_acc = 0.0f32;

    let mut input = InputState::default();
    let mut ctrl = FpsController {
        flying: true,
        ..Default::default()
    };
    let mut ui = UiState::default();
    let mut brush = BrushSettings::default();
    let tool_textures = load_tool_textures(&egui_ctx, TOOL_TEXTURES_DIR);
    let mut edit_runtime = EditRuntimeState::default();

    let mut last = Instant::now();
    let start = Instant::now();
    let mut cursor_is_unlocked = false;

    let mut origin_voxel = VoxelCoord { x: 0, y: 0, z: 0 };
    let mut preview_block_list: Vec<[i32; 3]> = Vec::new();

    // === streaming/perf caches (MUST live outside RedrawRequested) ===
    let mut last_player_chunk: Option<ChunkCoord> = None;
    let mut cached_sim_region: HashSet<ChunkCoord> =
        chunk_cube(ChunkCoord { x: 0, y: 0, z: 0 }, SIMULATION_RADIUS_CHUNKS);
    let mut last_desired_look_dir: Option<Vec3> = None;
    let mut desired_recompute_reason = String::from("init");
    let mut cached_desired: DesiredChunks = ChunkStreaming::desired_set(
        ChunkCoord { x: 0, y: 0, z: 0 },
        Vec3::ZERO,
        Vec3::Z,
        stream_tuning.near_radius_xz,
        stream_tuning.mid_radius_xz,
        Some(stream_tuning.far_radius_xz),
        stream_tuning.vertical_radius,
        &std::collections::HashMap::new(),
        0,
    );
    let mut stream_debug = String::new();
    let mut frame_counter: u64 = 0;
    let mut total_generated_chunks: u64 = 0;
    let mut collision_freeze_active = false;
    let mut collision_used_unloaded_chunks = false;
    let mut collision_freeze_started_at: Option<Instant> = None;
    let mut collision_freeze_backoff_until: Option<Instant> = None;
    let mut collision_freeze_last_duration_ms: f32 = 0.0;
    let mut previous_collision_freeze_active = false;
    let mut last_player_world_voxel = local_to_world_voxel(ctrl.position, origin_voxel);
    let mut prior_mesh_backlog = 0usize;
    let mut prior_dirty_backlog = 0usize;
    let mut prior_meshing_queue_depth = 0usize;
    let mut prior_upload_latency_ms = 0.0f32;
    let mut spawn_pending = true;

    let _ = set_cursor(window, false);

    event_loop
        .run(move |event, elwt| match &event {
            Event::WindowEvent { event, window_id } if *window_id == window.id() => {
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
                let egui_c = if block_tab_for_egui || block_pointer_for_egui {
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

                        if spawn_pending {
                            if let Some(spawn_world) =
                                find_safe_spawn_in_loaded_chunks(&store, streaming.seed)
                            {
                                ctrl.position = world_spawn_to_local_pos(spawn_world, origin_voxel);
                                let neighborhood_loaded = collision_neighborhood_loaded(
                                    &store,
                                    ctrl.position,
                                    origin_voxel,
                                    COLLISION_SAFETY_RADIUS_VOXELS,
                                );
                                spawn_pending = !neighborhood_loaded;
                                if !spawn_pending {
                                    collision_freeze_started_at = None;
                                    collision_freeze_backoff_until = None;
                                    previous_collision_freeze_active = false;
                                    last_player_chunk = None;
                                    cached_desired = DesiredChunks::default();
                                    cached_sim_region.clear();
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

                        // Origin shifting to keep float precision stable
                        let shift = origin_shift_for(ctrl.position);
                        if shift != [0, 0, 0] {
                            origin_voxel.x += shift[0];
                            origin_voxel.y += shift[1];
                            origin_voxel.z += shift[2];
                            ctrl.position -=
                                Vec3::new(shift[0] as f32, shift[1] as f32, shift[2] as f32);

                            // Conservative: mark all loaded chunks dirty (so meshes rebase)
                            let loaded: Vec<ChunkCoord> =
                                store.iter_loaded_chunks().copied().collect();
                            for coord in loaded {
                                store.mark_dirty(coord);
                            }
                        }

                        // === Player movement/collision: query the actual ChunkStore (not dummy world) ===
                        let now_instant = Instant::now();
                        let player_local_for_collision = ctrl.position;
                        let player_world_for_collision = local_to_world_voxel(player_local_for_collision, origin_voxel);
                        let (player_chunk_for_collision, _) = voxel_to_chunk(player_world_for_collision);
                        let collision_neighborhood_loaded = collision_neighborhood_loaded(
                            &store,
                            player_local_for_collision,
                            origin_voxel,
                            COLLISION_SAFETY_RADIUS_VOXELS,
                        );

                        let should_consider_freeze = FREEZE_UNTIL_COLLISION_RESIDENT
                            && !collision_neighborhood_loaded
                            && !(gameplay_blocked || spawn_pending)
                            && !egui_c;
                        let within_backoff = collision_freeze_backoff_until
                            .is_some_and(|until| now_instant < until);
                        collision_freeze_active = should_consider_freeze && !within_backoff;

                        if collision_freeze_active {
                            let started = collision_freeze_started_at.get_or_insert(now_instant);
                            if now_instant.duration_since(*started) > COLLISION_FREEZE_MAX_DURATION {
                                collision_freeze_active = false;
                                collision_freeze_last_duration_ms =
                                    now_instant.duration_since(*started).as_secs_f32() * 1000.0;
                                collision_freeze_started_at = None;
                                collision_freeze_backoff_until =
                                    Some(now_instant + COLLISION_FREEZE_BACKOFF_DURATION);
                            }
                        } else if let Some(started) = collision_freeze_started_at.take() {
                            collision_freeze_last_duration_ms =
                                now_instant.duration_since(started).as_secs_f32() * 1000.0;
                        }

                        if collision_neighborhood_loaded {
                            collision_freeze_backoff_until = None;
                        }

                        if !collision_neighborhood_loaded {
                            let missing_coords = collision_neighborhood_missing_chunks(
                                &store,
                                player_local_for_collision,
                                origin_voxel,
                                COLLISION_SAFETY_RADIUS_VOXELS,
                            );
                            let mut forced_local_requests = 0usize;
                            for coord in missing_coords
                                .into_iter()
                                .take(COLLISION_LOCAL_PRIORITY_REQUEST_BUDGET)
                            {
                                if streaming.resident.contains(&coord)
                                    || streaming.generating.contains(&coord)
                                {
                                    continue;
                                }
                                if chunk_generator.try_request(coord) {
                                    streaming.generating.insert(coord);
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
                        }

                        if collision_freeze_active != previous_collision_freeze_active {
                            if collision_freeze_active {
                                ui.log_once_per_second("collision_freeze_enter", start.elapsed().as_secs_f32(), || {
                                    format!(
                                        "collision_freeze enter player_chunk=({}, {}, {}) loaded_neighborhood={}",
                                        player_chunk_for_collision.x,
                                        player_chunk_for_collision.y,
                                        player_chunk_for_collision.z,
                                        collision_neighborhood_loaded,
                                    )
                                });
                            } else {
                                let backoff_remaining_ms = collision_freeze_backoff_until
                                    .map(|until| until.saturating_duration_since(now_instant).as_secs_f32() * 1000.0)
                                    .unwrap_or(0.0);
                                ui.log_once_per_second("collision_freeze_exit", start.elapsed().as_secs_f32(), || {
                                    format!(
                                        "collision_freeze exit duration_ms={:.1} neighborhood_loaded={} backoff_remaining_ms={:.1}",
                                        collision_freeze_last_duration_ms,
                                        collision_neighborhood_loaded,
                                        backoff_remaining_ms,
                                    )
                                });
                            }
                            previous_collision_freeze_active = collision_freeze_active;
                        }

                        if collision_freeze_active {
                            let freeze_ms = collision_freeze_started_at
                                .map(|started| now_instant.duration_since(started).as_secs_f32() * 1000.0)
                                .unwrap_or(0.0);
                            ui.log_once_per_second("collision_freeze_active", start.elapsed().as_secs_f32(), || {
                                format!(
                                    "collision_freeze active_ms={:.1} max_ms={:.1} backoff_ms={:.1} missing_neighborhood={}",
                                    freeze_ms,
                                    COLLISION_FREEZE_MAX_DURATION.as_secs_f32() * 1000.0,
                                    COLLISION_FREEZE_BACKOFF_DURATION.as_secs_f32() * 1000.0,
                                    !collision_neighborhood_loaded,
                                )
                            });
                        }

                        collision_used_unloaded_chunks = false;
                        let look_active = !(gameplay_blocked || spawn_pending) && !egui_c;
                        let mut ctrl_input = input.clone();
                        if collision_freeze_active {
                            ctrl_input.pressed.remove(&KeyCode::KeyW);
                            ctrl_input.pressed.remove(&KeyCode::KeyA);
                            ctrl_input.pressed.remove(&KeyCode::KeyS);
                            ctrl_input.pressed.remove(&KeyCode::KeyD);
                            ctrl_input.pressed.remove(&KeyCode::Space);
                            ctrl_input.pressed.remove(&KeyCode::ShiftLeft);
                        }
                        let origin_for_ctrl = origin_voxel;
                        ctrl.step(
                            |x, y, z| {
                                let voxel = VoxelCoord { x, y, z };
                                if store.is_voxel_chunk_loaded(voxel) {
                                    return store.get_voxel(voxel);
                                }

                                let near_player =
                                    within_collision_safety_radius(voxel, player_local_for_collision, origin_for_ctrl, COLLISION_SAFETY_RADIUS_VOXELS);
                                if near_player {
                                    collision_used_unloaded_chunks = true;
                                    return 1u16;
                                }

                                0u16
                            },
                            &ctrl_input,
                            dt,
                            look_active,
                            start.elapsed().as_secs_f32(),
                            origin_for_ctrl,
                        );

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

                            cached_desired = ChunkStreaming::desired_set(
                                player_chunk,
                                player_velocity_chunks,
                                look_dir,
                                effective_stream_tuning.near_radius_xz,
                                effective_stream_tuning.mid_radius_xz,
                                Some(effective_stream_tuning.far_radius_xz),
                                effective_stream_tuning.vertical_radius,
                                &visible_history,
                                frame_counter,
                            );
                            if player_chunk_changed {
                                cached_sim_region = chunk_cube(player_chunk, SIMULATION_RADIUS_CHUNKS);
                            }
                            last_player_chunk = Some(player_chunk);
                            cached_stream_tuning = effective_stream_tuning.clone();
                            last_desired_look_dir = Some(look_dir);
                        } else {
                            desired_recompute_reason.clear();
                            desired_recompute_reason.push_str("none");
                        }
                        last_player_world_voxel = player_world_voxel;
                        let desired_ms = desired_t0.elapsed().as_secs_f32() * 1000.0;

                        let streaming_t0 = Instant::now();
                        streaming.max_generate_schedule_per_update = scaled_budget(
                            stream_tuning.base_generate_drain_items,
                            stream_tuning.max_generate_drain_items,
                            streaming.pending_generate_count(),
                            stream_tuning.base_generate_drain_items,
                        );
                        let backpressure = if prior_mesh_backlog <= MESH_BACKPRESSURE_START {
                            0.0
                        } else {
                            ((prior_mesh_backlog - MESH_BACKPRESSURE_START) as f32
                                / (MESH_BACKPRESSURE_HIGH - MESH_BACKPRESSURE_START) as f32)
                                .clamp(0.0, 1.0)
                        };
                        let mut generation_priority = cached_desired.generation_order.clone();
                        if backpressure > 0.0 {
                            let far_radius_cutoff = (effective_stream_tuning.mid_radius_xz as f32
                                + (effective_stream_tuning.far_radius_xz - effective_stream_tuning.mid_radius_xz) as f32
                                    * (1.0 - backpressure))
                                .round() as i32;
                            generation_priority.retain(|coord| {
                                let cheb = (coord.x - player_chunk.x)
                                    .abs()
                                    .max((coord.y - player_chunk.y).abs())
                                    .max((coord.z - player_chunk.z).abs());
                                cheb <= far_radius_cutoff
                            });
                        }
                        let stream_stats = streaming.update(&generation_priority, &cached_desired.resident_keep, player_chunk, frame_counter);
                        let mut gen_request_count = 0usize;
                        let generate_drain_budget = scaled_budget(
                            stream_tuning.base_generate_drain_items,
                            stream_tuning.max_generate_drain_items,
                            streaming.pending_generate_count() + generated_ready.len(),
                            stream_tuning.base_generate_drain_items,
                        );
                        for coord in streaming.drain_generate_requests(generate_drain_budget) {
                            if chunk_generator.try_request(coord) {
                                gen_request_count += 1;
                            } else {
                                streaming.mark_generation_dropped(coord);
                            }
                        }

                        let mut gen_completed_count = 0usize;
                        loop {
                            match chunk_generator.try_recv() {
                                Ok(res) => {
                                    generated_ready.push_back(res);
                                    gen_completed_count += 1;
                                }
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Disconnected) => break,
                            }
                        }

                        total_generated_chunks = total_generated_chunks
                            .saturating_add(gen_completed_count as u64);

                        let apply_budget_items = scaled_budget(
                            stream_tuning.base_apply_budget_items,
                            stream_tuning.max_apply_budget_items,
                            generated_ready.len(),
                            stream_tuning.base_apply_budget_items,
                        );
                        let apply_t0 = Instant::now();
                        let mut apply_count = 0usize;
                        while apply_count < apply_budget_items
                            && apply_t0.elapsed() < Duration::from_secs_f32(APPLY_BUDGET_MS / 1000.0)
                        {
                            let Some(done) = generated_ready.pop_front() else { break; };
                            apply_generated_chunk(&mut store, done.coord, done.chunk);
                            streaming.mark_generated(done.coord, frame_counter);
                            apply_count += 1;
                        }
                        let apply_ms = apply_t0.elapsed().as_secs_f32() * 1000.0;

                        let evict_t0 = Instant::now();
                        let mut evict_count = 0usize;
                        while evict_count < streaming.max_evict_schedule_per_update
                            && evict_t0.elapsed() < Duration::from_secs_f32(EVICT_BUDGET_MS / 1000.0)
                        {
                            let mut one = streaming.drain_evict_requests(1);
                            let Some(coord) = one.pop() else {
                                break;
                            };
                            store.remove_chunk(coord);
                            streaming.mark_evicted(coord, frame_counter);
                            evict_count += 1;
                        }
                        let evict_ms = evict_t0.elapsed().as_secs_f32() * 1000.0;

                        let streaming_ms = streaming_t0.elapsed().as_secs_f32() * 1000.0;
                        let now_secs = start.elapsed().as_secs_f32();
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
                        }
                        if streaming.pending_evict_count() > streaming.max_evict_schedule_per_update {
                            ui.log_once_per_second("evict_budget", now_secs, || {
                                format!("evict budget hit: remaining queue={}", streaming.pending_evict_count())
                            });
                        }

                        stream_debug = format!(
                            "Stream: resident={} near={} mid={} far={} gen_pending={} apply_queue={} player_chunk=({}, {}, {}) desired_recompute={} radii[n/m/f/v]={}/{}/{}/{} lod_h={} budgets[gen/app]={}/{} lod_budgets[n/m/f]={}/{}/{} collision[freeze={} unknown_solid={} safety_voxels={}] origin=({}, {}, {}) player_world=({}, {}, {})",
                            streaming.resident.len(),
                            cached_desired.near.len(),
                            cached_desired.mid.len(),
                            cached_desired.far.len(),
                            streaming.pending_generate_count(),
                            generated_ready.len(),
                            player_chunk.x,
                            player_chunk.y,
                            player_chunk.z,
                            desired_recompute_reason,
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

                        if !gameplay_blocked
                            && apply_mouse_edit(
                                &mut store,
                                &brush,
                                selected_material(&ui, ui.selected_slot),
                                &input,
                                &mut edit_runtime,
                                now,
                                raycast,
                                ui.active_tool,
                            )
                        {
                            // dirtied by set_voxel
                        }

                        let do_step = sim_running && !ui.paused_menu && ui.sim_speed > 0.0;
                        let sim_t0 = Instant::now();
                        let mut sim_chunk_steps = 0usize;
                        if do_step {
                            sim_acc += dt * ui.sim_speed;
                            while sim_acc >= 1.0 / 60.0 {
                                sim_chunk_steps += step_region_profiled(
                                    &mut store,
                                    &cached_sim_region,
                                    player_chunk,
                                    &mut rng,
                                );
                                sim_acc -= 1.0 / 60.0;
                            }
                        } else if step_once && !ui.paused_menu {
                            sim_chunk_steps += step_region_profiled(
                                &mut store,
                                &cached_sim_region,
                                player_chunk,
                                &mut rng,
                            );
                            step_once = false;
                        }
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
                        let mesh_upload_budget = adaptive_mesh_upload_budget(
                            ui.profiler.frame_ms,
                            prior_mesh_backlog,
                        );
                        let mesh_stats = renderer.rebuild_dirty_store_chunks(
                            &mut store,
                            origin_voxel,
                            player_chunk,
                            &cached_desired.generation_scores,
                            REMESH_JOB_BUDGET_PER_FRAME,
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
                        ui.set_mesh_timing(mesh_stats.max_ms);
                        ui.profiler.desired_ms = desired_ms;
                        ui.profiler.streaming_ms = streaming_ms;
                        ui.profiler.gen_request_count = gen_request_count;
                        ui.profiler.gen_inflight_count = streaming.generating.len();
                        ui.profiler.gen_completed_count = gen_completed_count;
                        ui.profiler.gen_completed_total = total_generated_chunks;
                        ui.profiler.apply_ms = apply_ms;
                        ui.profiler.apply_count = apply_count;
                        ui.profiler.evict_ms = evict_ms;
                        ui.profiler.evict_count = evict_count;
                        ui.profiler.mesh_ms = mesh_stats.total_ms;
                        ui.profiler.mesh_count = mesh_stats.mesh_count;
                        ui.profiler.dirty_backlog = mesh_stats.dirty_backlog + store.dirty_count();
                        ui.profiler.mesh_queue_depth = mesh_stats.meshing_queue_depth;
                        ui.profiler.mesh_completed_depth = mesh_stats.meshing_completed_depth;
                        ui.profiler.mesh_upload_count = mesh_stats.upload_count;
                        ui.profiler.mesh_upload_bytes = mesh_stats.upload_bytes;
                        ui.profiler.mesh_upload_latency_ms = mesh_stats.upload_latency_ms;
                        ui.profiler.gpu_upload_bytes_frame = mesh_stats.upload_bytes;
                        ui.profiler.mesh_stale_drop_count = mesh_stats.stale_drop_count;
                        ui.profiler.near_radius = effective_stream_tuning.near_radius_xz;
                        ui.profiler.mid_radius = effective_stream_tuning.mid_radius_xz;
                        ui.profiler.far_radius = effective_stream_tuning.far_radius_xz;
                        ui.profiler.ultra_radius = effective_stream_tuning.ultra_radius_xz;
                        ui.profiler.lod_hysteresis = effective_stream_tuning.lod_hysteresis;
                        ui.profiler.lod_budget_near = effective_stream_tuning.lod_budget_near;
                        ui.profiler.lod_budget_mid = effective_stream_tuning.lod_budget_mid;
                        ui.profiler.lod_budget_far = effective_stream_tuning.lod_budget_far;
                        ui.profiler.lod_budget_ultra = effective_stream_tuning.lod_budget_ultra;
                        ui.profiler.auto_tune_active = auto_tune.is_active();
                        ui.profiler.auto_tune_level = auto_tune.degrade_level;
                        ui.profiler.auto_tune_latency_pressure = auto_tune.latency_pressure;
                        ui.profiler.auto_tune_queue_pressure = auto_tune.queue_pressure;
                        ui.profiler.auto_tune_dirty_pressure = auto_tune.dirty_pressure;
                        ui.profiler.mesh_age_drop_count = mesh_stats.age_drop_count;
                        ui.profiler.near_radius = stream_tuning.near_radius_xz;
                        ui.profiler.mid_radius = stream_tuning.mid_radius_xz;
                        ui.profiler.far_radius = stream_tuning.far_radius_xz;
                        ui.profiler.ultra_radius = stream_tuning.ultra_radius_xz;
                        ui.profiler.lod_hysteresis = stream_tuning.lod_hysteresis;
                        ui.profiler.lod_budget_near = stream_tuning.lod_budget_near;
                        ui.profiler.lod_budget_mid = stream_tuning.lod_budget_mid;
                        ui.profiler.lod_budget_far = stream_tuning.lod_budget_far;
                        ui.profiler.lod_budget_ultra = stream_tuning.lod_budget_ultra;
                        ui.profiler.mesh_near_count = mesh_stats.near_mesh_count;
                        ui.profiler.mesh_mid_count = mesh_stats.mid_mesh_count;
                        ui.profiler.mesh_far_count = mesh_stats.far_mesh_count;
                        prior_mesh_backlog = mesh_stats.meshing_queue_depth + mesh_stats.meshing_completed_depth;
                        prior_dirty_backlog = ui.profiler.dirty_backlog;
                        prior_meshing_queue_depth = mesh_stats.meshing_queue_depth;
                        prior_upload_latency_ms = mesh_stats.upload_latency_ms;
                        ui.profiler.mesh_ultra_count = mesh_stats.ultra_mesh_count;
                        ui.profiler.sim_ms = sim_ms;
                        ui.profiler.sim_chunk_steps = sim_chunk_steps;

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
                                store.clear();
                                streaming.clear();
                                last_player_chunk = None;
                                cached_desired = DesiredChunks::default();
                                cached_sim_region.clear();
                                generated_ready.clear();
                                total_generated_chunks = 0;
                                origin_voxel = VoxelCoord { x: 0, y: 0, z: 0 };
                                ctrl.position = Vec3::new(8.0, 6.0, 8.0);
                                spawn_pending = true;
                                collision_freeze_active = false;
                                collision_freeze_started_at = None;
                                collision_freeze_backoff_until = None;
                                previous_collision_freeze_active = false;
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

                        let cam = Camera {
                            pos: camera_world_pos_from_blocks(ctrl.position, VOXEL_SIZE),
                            dir: ctrl.look_dir(),
                            aspect: renderer.config.width as f32
                                / renderer.config.height.max(1) as f32,
                        };
                        let vp = cam.view_proj();
                        let (chunks_drawn, total_indices) = renderer.mesh_draw_stats(vp);
                        ui.set_draw_stats(chunks_drawn, total_indices);
                        let cull_stats = renderer.cull_stats(&cam);
                        ui.profiler.culled_chunks = cull_stats.frustum_culled + cull_stats.screen_culled + cull_stats.lod_filtered;
                        ui.profiler.frustum_culled_chunks = cull_stats.frustum_culled;
                        ui.profiler.missing_in_radius = cached_desired
                            .generation_order
                            .iter()
                            .filter(|coord| !streaming.resident.contains(coord) && !streaming.generating.contains(coord))
                            .count();

                        let preview_local: Vec<[i32; 3]> = preview_block_list
                            .iter()
                            .map(|p| {
                                [
                                    p[0] - origin_voxel.x,
                                    p[1] - origin_voxel.y,
                                    p[2] - origin_voxel.z,
                                ]
                            })
                            .collect();

                        draw_fps_overlays(
                            &egui_ctx,
                            ui.paused_menu,
                            ui.sim_speed,
                            cam.view_proj(),
                            [renderer.config.width, renderer.config.height],
                            &preview_local,
                            [0, 0, 0],
                            &brush,
                            preview_mode,
                            ui.show_radial_menu,
                            RADIAL_MENU_TOGGLE_LABEL,
                            VOXEL_SIZE,
                            None,
                            start.elapsed().as_secs_f32(),
                            !gameplay_blocked && !egui_c,
                            None,
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
                window.request_redraw();
            }
            _ => {}
        })
        .map_err(anyhow::Error::from)
}

fn origin_shift_for(pos: Vec3) -> [i32; 3] {
    let mut shift = [0, 0, 0];
    if pos.x.abs() > ORIGIN_SHIFT_THRESHOLD {
        shift[0] = pos.x.floor() as i32;
    }
    if pos.y.abs() > ORIGIN_SHIFT_THRESHOLD {
        shift[1] = pos.y.floor() as i32;
    }
    if pos.z.abs() > ORIGIN_SHIFT_THRESHOLD {
        shift[2] = pos.z.floor() as i32;
    }
    shift
}

fn local_to_world_voxel(local_pos: Vec3, origin: VoxelCoord) -> VoxelCoord {
    VoxelCoord {
        x: local_pos.x.floor() as i32 + origin.x,
        y: local_pos.y.floor() as i32 + origin.y,
        z: local_pos.z.floor() as i32 + origin.z,
    }
}

fn world_spawn_to_local_pos(spawn_world: VoxelCoord, origin: VoxelCoord) -> Vec3 {
    Vec3::new(
        spawn_world.x as f32 + 0.5 - origin.x as f32,
        spawn_world.y as f32 + SPAWN_CLEARANCE - origin.y as f32,
        spawn_world.z as f32 + 0.5 - origin.z as f32,
    )
}

fn find_safe_spawn_in_loaded_chunks(store: &ChunkStore, seed: u64) -> Option<VoxelCoord> {
    let center = VoxelCoord { x: 0, y: 0, z: 0 };
    let mut candidate: Option<VoxelCoord> = None;

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
                    return Some(VoxelCoord { x, y, z });
                }
            }
        }
    }

    if store.is_chunk_loaded(ChunkCoord { x: 0, y: 0, z: 0 }) {
        let sea_level = (18).min(CHUNK_SIZE as i32 - 10).max(10);
        candidate = Some(VoxelCoord {
            x: center.x,
            y: sea_level + SPAWN_FALLBACK_EXTRA_HEIGHT,
            z: center.z,
        });
    }

    candidate
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
        if base_id == EMPTY || base_id == 5 {
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

        let mut near_water = false;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let sample = VoxelCoord {
                    x: x + dx,
                    y: y + 1,
                    z: z + dz,
                };
                if !store.is_voxel_chunk_loaded(sample) {
                    continue;
                }
                if store.get_voxel(sample) == 5 {
                    near_water = true;
                    break;
                }
            }
            if near_water {
                break;
            }
        }
        if near_water {
            continue;
        }

        return Some(y);
    }
    None
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
        store.set_voxel(
            VoxelCoord {
                x: p[0],
                y: p[1],
                z: p[2],
            },
            target,
        );
    }
    edit_runtime.last_edit_at = Some(now);
    edit_runtime.last_edit_mode = Some(mode);
    true
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
