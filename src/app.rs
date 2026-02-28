use crate::input::{FpsController, InputState};
use crate::player::{
    camera_world_pos_from_blocks, eye_height_world_meters, PLAYER_EYE_HEIGHT_BLOCKS,
    PLAYER_HEIGHT_BLOCKS, PLAYER_WIDTH_BLOCKS,
};
use crate::procgen::{
    biome_hint_at_world, find_safe_spawn, generate_world_cancellable, ProcGenConfig,
};
use crate::renderer::{Camera, Renderer, VOXEL_SIZE};
use crate::sim::{prioritize_chunks_for_player, step, step_selected_chunks, SimState};
use crate::ui::{
    assign_hotbar_slot, draw, draw_fps_overlays, load_tool_textures, selected_material, ToolKind,
    ToolTextures, UiState, HOTBAR_SLOTS,
};
use crate::world::{
    default_save_path, load_world, save_world, AreaFootprintShape, BrushMode, BrushSettings,
    BrushShape, World,
};
use anyhow::Context;
use glam::Vec3;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

const RADIAL_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyE;
const RADIAL_MENU_TOGGLE_LABEL: &str = "E";
const TOOL_QUICK_MENU_TOGGLE_KEY: KeyCode = KeyCode::KeyQ;
const TOOL_TEXTURES_DIR: &str = "assets/tools";
const WAND_MAX_BLOCKS: usize = 512;
const BUSH_ID: u16 = 18;
const GRASS_ID: u16 = 19;
const LOW_PRIORITY_THROTTLE_TICKS: u64 = 6;
const PROCEDURAL_MACROCHUNK_SIZE: i32 = 64;
const PROCEDURAL_RENDER_DISTANCE_MACROS: i32 = 2;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcgenPriority {
    Prefetch,
    Urgent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcgenJobStatus {
    Queued,
    InFlight,
    Ready,
    Cancelled,
}

#[derive(Clone, Copy)]
struct ProcgenJobSpec {
    config: ProcGenConfig,
    prefer_safe_spawn: bool,
    target_global_pos: [i32; 3],
    priority: ProcgenPriority,
    epoch: u64,
}

struct ProcgenJobResult {
    key: [i32; 3],
    epoch: u64,
    config: ProcGenConfig,
    world: Option<World>,
    prefer_safe_spawn: bool,
    target_global_pos: [i32; 3],
    status: ProcgenJobStatus,
}

#[derive(Clone)]
struct ProcgenJobSystem {
    shared: Arc<(Mutex<ProcgenQueueState>, Condvar)>,
    epoch: Arc<AtomicU64>,
}

struct ProcgenQueueState {
    queue: BinaryHeap<QueuedProcgenJob>,
    queued_keys: HashSet<[i32; 3]>,
    in_flight: HashSet<[i32; 3]>,
    statuses: HashMap<[i32; 3], ProcgenJobStatus>,
    queued_limit: usize,
}

#[derive(Clone, Copy)]
struct QueuedProcgenJob {
    spec: ProcgenJobSpec,
    seq: u64,
}

impl PartialEq for QueuedProcgenJob {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}

impl Eq for QueuedProcgenJob {}

impl PartialOrd for QueuedProcgenJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedProcgenJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.spec
            .priority
            .cmp(&other.spec.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl Ord for ProcgenPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for ProcgenPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub async fn run() -> anyhow::Result<()> {
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

    let mut world = World::new([64, 64, 64]);
    let mut sim = SimState::default();
    let mut input = InputState::default();
    let mut ctrl = FpsController::default();
    let mut ui = UiState::default();
    let mut brush = BrushSettings::default();
    let tool_textures = load_tool_textures(&egui_ctx, TOOL_TEXTURES_DIR);
    let mut edit_runtime = EditRuntimeState::default();
    let mut last = Instant::now();
    let start = Instant::now();
    let mut sim_tick: u64 = 0;
    let mut active_procgen: Option<ProcGenConfig> = None;
    let mut active_procgen_origin: [i32; 3] = [0, 0, 0];
    let (procgen_tx, procgen_rx): (Sender<ProcgenJobResult>, Receiver<ProcgenJobResult>) =
        mpsc::channel();
    let procgen_jobs = ProcgenJobSystem::new(procgen_tx, 2, 12);
    let mut generated_regions: BTreeMap<[i32; 3], World> = BTreeMap::new();
    let mut residency_plan: BTreeSet<[i32; 3]> = BTreeSet::new();

    let _ = set_cursor(window, false);
    debug_assert!(PLAYER_HEIGHT_BLOCKS > 0.0 && PLAYER_WIDTH_BLOCKS > 0.0);
    debug_assert!(PLAYER_EYE_HEIGHT_BLOCKS <= PLAYER_HEIGHT_BLOCKS);
    debug_assert!((eye_height_world_meters(VOXEL_SIZE) - 1.6).abs() < f32::EPSILON);
    renderer.rebuild_dirty_chunks(&mut world);

    event_loop
        .run(move |event, elwt| match &event {
            Event::WindowEvent { event, window_id } if *window_id == window.id() => {
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
                input.on_window_event(event);
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => renderer.resize(*size),
                    WindowEvent::Focused(focused) => {
                        let should_unlock = !focused
                            || ui.paused_menu
                            || ui.show_tool_quick_menu
                            || ui.tab_palette_open;
                        let _ = set_cursor(window, should_unlock);
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(key) = event.physical_key {
                            if event.state == ElementState::Pressed {
                                let tab_palette_open = ui.tab_palette_open;
                                match key {
                                    KeyCode::Escape => ui.paused_menu = !ui.paused_menu,
                                    _ if ui.paused_menu => {}
                                    KeyCode::KeyP => sim.running = !sim.running,
                                    KeyCode::KeyB => ui.show_brush = !ui.show_brush,
                                    KeyCode::Tab if !event.repeat => {
                                        ui.tab_palette_open = !ui.tab_palette_open
                                    }
                                    RADIAL_MENU_TOGGLE_KEY => {
                                        ui.show_radial_menu = !ui.show_radial_menu
                                    }
                                    KeyCode::KeyF => {
                                        brush.shape = if brush.shape == BrushShape::Sphere {
                                            BrushShape::Cube
                                        } else {
                                            BrushShape::Sphere
                                        }
                                    }
                                    KeyCode::BracketLeft => ui.adjust_sim_speed(-1),
                                    KeyCode::BracketRight => ui.adjust_sim_speed(1),
                                    KeyCode::Backslash => ui.set_sim_speed(1.0),
                                    KeyCode::Digit0 => {
                                        assign_or_select_hotbar(&mut ui, 0, tab_palette_open)
                                    }
                                    KeyCode::Digit1 => {
                                        assign_or_select_hotbar(&mut ui, 1, tab_palette_open)
                                    }
                                    KeyCode::Digit2 => {
                                        assign_or_select_hotbar(&mut ui, 2, tab_palette_open)
                                    }
                                    KeyCode::Digit3 => {
                                        assign_or_select_hotbar(&mut ui, 3, tab_palette_open)
                                    }
                                    KeyCode::Digit4 => {
                                        assign_or_select_hotbar(&mut ui, 4, tab_palette_open)
                                    }
                                    KeyCode::Digit5 => {
                                        assign_or_select_hotbar(&mut ui, 5, tab_palette_open)
                                    }
                                    KeyCode::Digit6 => {
                                        assign_or_select_hotbar(&mut ui, 6, tab_palette_open)
                                    }
                                    KeyCode::Digit7 => {
                                        assign_or_select_hotbar(&mut ui, 7, tab_palette_open)
                                    }
                                    KeyCode::Digit8 => {
                                        assign_or_select_hotbar(&mut ui, 8, tab_palette_open)
                                    }
                                    KeyCode::Digit9 => {
                                        assign_or_select_hotbar(&mut ui, 9, tab_palette_open)
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
                                && ui.show_tool_quick_menu
                                && !input.lmb
                                && !input.rmb
                            {
                                if let Some(hovered) = ui.hovered_shape.take() {
                                    brush.shape = hovered;
                                }
                                if let Some(hovered) = ui.hovered_area_shape.take() {
                                    brush.area_tool.shape = hovered;
                                }
                                if let Some(hovered) = ui.hovered_tool.take() {
                                    ui.active_tool = hovered;
                                }
                                let _ = set_cursor(
                                    window,
                                    should_unlock_cursor(&ui, false, ui.tab_palette_open),
                                );
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - last).as_secs_f32().min(0.05);
                        last = now;

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
                        let _ = set_cursor(window, cursor_should_unlock);

                        if !gameplay_blocked {
                            ctrl.sensitivity = ui.mouse_sensitivity;
                            ctrl.step(
                                &world,
                                &input,
                                dt,
                                !cursor_should_unlock,
                                start.elapsed().as_secs_f32(),
                            );
                            if input.wheel.abs() > 0.0 {
                                if input.key(KeyCode::ControlLeft) {
                                    brush.radius =
                                        (brush.radius + input.wheel.signum() as i32).clamp(0, 8);
                                } else if input.key(KeyCode::AltLeft)
                                    || input.key(KeyCode::AltRight)
                                {
                                    brush.max_distance = (brush.max_distance
                                        + input.wheel.signum())
                                    .clamp(2.0, 48.0);
                                } else {
                                    let mut s =
                                        ui.selected_slot as i32 - input.wheel.signum() as i32;
                                    if s < 0 {
                                        s += HOTBAR_SLOTS as i32;
                                    }
                                    ui.selected_slot = (s as usize) % HOTBAR_SLOTS;
                                }
                            }
                        }

                        let modifier_hint = if input.key(KeyCode::ControlLeft)
                            || input.key(KeyCode::ControlRight)
                        {
                            Some(format!("Brush Size: {}", brush.radius))
                        } else if input.key(KeyCode::AltLeft) || input.key(KeyCode::AltRight) {
                            Some(format!("Brush Distance: {:.1}", brush.max_distance))
                        } else {
                            None
                        };

                        let raycast =
                            target_for_edit(&world, ctrl.position, ctrl.look_dir(), &brush);
                        let preview_mode = current_action_mode(&input, raycast, ui.active_tool);
                        let preview_blocks = preview_blocks(
                            &world,
                            &brush,
                            raycast,
                            preview_mode,
                            ui.active_tool,
                            !cursor_should_unlock || !egui_c,
                        );

                        if !gameplay_blocked {
                            apply_mouse_edit(
                                &mut world,
                                &brush,
                                selected_material(&ui, ui.selected_slot),
                                &input,
                                &mut edit_runtime,
                                now,
                                raycast,
                                ui.active_tool,
                            );
                        }

                        if sim.running && !ui.paused_menu {
                            let step_dt = (sim.fixed_dt / ui.sim_speed).max(1e-4);
                            sim.accumulator += dt;
                            while sim.accumulator >= step_dt {
                                let (high_priority, low_priority) =
                                    prioritize_chunks_for_player(&world, ctrl.position);
                                step_selected_chunks(&mut world, &mut sim.rng, &high_priority);
                                if sim_tick % LOW_PRIORITY_THROTTLE_TICKS == 0 {
                                    step_selected_chunks(&mut world, &mut sim.rng, &low_priority);
                                }
                                sim_tick = sim_tick.wrapping_add(1);
                                sim.accumulator -= step_dt;
                            }
                        }
                        renderer.day = ui.day;

                        while let Ok(result) = procgen_rx.try_recv() {
                            procgen_jobs.record_result_status(&result);
                            if result.status != ProcgenJobStatus::Ready {
                                continue;
                            }
                            if result.epoch != procgen_jobs.current_epoch() {
                                continue;
                            }
                            if !residency_plan.contains(&result.key) {
                                continue;
                            }
                            let Some(incoming_world) = result.world else {
                                continue;
                            };

                            let previous_origin = active_procgen_origin;
                            let frac_x = ctrl.position.x - ctrl.position.x.floor();
                            let frac_z = ctrl.position.z - ctrl.position.z.floor();
                            let tracked_global = if active_procgen.is_some() {
                                [
                                    previous_origin[0] + ctrl.position.x.floor() as i32,
                                    0,
                                    previous_origin[2] + ctrl.position.z.floor() as i32,
                                ]
                            } else {
                                result.target_global_pos
                            };

                            let chosen_world = if let Some(existing) =
                                generated_regions.get(&result.config.world_origin)
                            {
                                existing.clone()
                            } else {
                                incoming_world.clone()
                            };
                            generated_regions
                                .entry(result.config.world_origin)
                                .or_insert_with(|| incoming_world.clone());
                            if result.key == active_procgen_origin || active_procgen.is_none() {
                                active_procgen = Some(result.config);
                                active_procgen_origin = result.config.world_origin;
                                world = chosen_world;
                                if result.prefer_safe_spawn {
                                    let spawn = find_safe_spawn(&world, result.config.seed);
                                    ctrl.position = Vec3::new(spawn[0], spawn[1], spawn[2]);
                                } else {
                                    ctrl.position = Vec3::new(
                                        (tracked_global[0] - active_procgen_origin[0]) as f32
                                            + frac_x,
                                        ctrl.position.y,
                                        (tracked_global[2] - active_procgen_origin[2]) as f32
                                            + frac_z,
                                    );
                                }
                            }
                        }

                        if let Some(cfg) = active_procgen {
                            let global = [
                                active_procgen_origin[0] + ctrl.position.x.floor() as i32,
                                0,
                                active_procgen_origin[2] + ctrl.position.z.floor() as i32,
                            ];
                            let player_macro = [
                                floor_div(global[0], PROCEDURAL_MACROCHUNK_SIZE),
                                0,
                                floor_div(global[2], PROCEDURAL_MACROCHUNK_SIZE),
                            ];
                            let desired_origin = desired_active_origin(
                                active_procgen_origin,
                                player_macro,
                                PROCEDURAL_MACROCHUNK_SIZE,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            );
                            residency_plan = desired_residency_origins(
                                player_macro,
                                PROCEDURAL_MACROCHUNK_SIZE,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            );
                            procgen_jobs.cancel_unplanned(&residency_plan);

                            if desired_origin != active_procgen_origin {
                                generated_regions.insert(active_procgen_origin, world.clone());
                                if let Some(cached) = generated_regions.get(&desired_origin) {
                                    active_procgen_origin = desired_origin;
                                    active_procgen = Some(cfg.with_origin(desired_origin));
                                    world = cached.clone();
                                } else {
                                    procgen_jobs.enqueue(ProcgenJobSpec {
                                        config: cfg.with_origin(desired_origin),
                                        prefer_safe_spawn: false,
                                        target_global_pos: global,
                                        priority: ProcgenPriority::Urgent,
                                        epoch: procgen_jobs.current_epoch(),
                                    });
                                }
                            }

                            if let Some(prefetch_origin) = next_missing_region_origin(
                                &generated_regions,
                                active_procgen_origin,
                                player_macro,
                                PROCEDURAL_MACROCHUNK_SIZE,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            ) {
                                procgen_jobs.enqueue(ProcgenJobSpec {
                                    config: cfg.with_origin(prefetch_origin),
                                    prefer_safe_spawn: false,
                                    target_global_pos: global,
                                    priority: ProcgenPriority::Prefetch,
                                    epoch: procgen_jobs.current_epoch(),
                                });
                            }
                            prune_generated_regions(
                                &mut generated_regions,
                                active_procgen_origin,
                                PROCEDURAL_MACROCHUNK_SIZE,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            );
                        }

                        renderer.rebuild_dirty_chunks(&mut world);

                        ui.biome_hint = if let Some(cfg) = active_procgen {
                            let biome = biome_hint_at_world(
                                &cfg,
                                active_procgen_origin[0] + ctrl.position.x.floor() as i32,
                                active_procgen_origin[2] + ctrl.position.z.floor() as i32,
                            );
                            let status = procgen_jobs.status_for(active_procgen_origin);
                            if matches!(
                                status,
                                Some(ProcgenJobStatus::Queued | ProcgenJobStatus::InFlight)
                            ) {
                                format!("Biome: {} (generating...)", biome.label())
                            } else {
                                format!("Biome: {}", biome.label())
                            }
                        } else {
                            "Biome: Flat Test World".to_string()
                        };

                        let raw = egui_state.take_egui_input(window);
                        let out = egui_ctx.run(raw, |ctx| {
                            let actions =
                                draw(ctx, &mut ui, sim.running, &mut brush, &tool_textures);
                            let cam = Camera {
                                pos: camera_world_pos_from_blocks(ctrl.position, VOXEL_SIZE),
                                dir: ctrl.look_dir(),
                                aspect: renderer.config.width as f32
                                    / renderer.config.height.max(1) as f32,
                            };
                            let held_tool = texture_for_active_tool(&tool_textures, ui.active_tool);
                            draw_fps_overlays(
                                ctx,
                                ui.paused_menu,
                                ui.sim_speed,
                                cam.view_proj(),
                                [renderer.config.width, renderer.config.height],
                                &preview_blocks,
                                &brush,
                                preview_mode,
                                ui.show_radial_menu,
                                RADIAL_MENU_TOGGLE_LABEL,
                                VOXEL_SIZE,
                                held_tool,
                                start.elapsed().as_secs_f32(),
                                !cursor_should_unlock && (input.lmb || input.rmb),
                                modifier_hint.as_deref(),
                            );
                            if actions.new_world || actions.new_procedural {
                                let n = ui.new_world_size.max(16) / 16 * 16;
                                if actions.new_procedural {
                                    let seed = start.elapsed().as_nanos() as u64;
                                    let macro_span = PROCEDURAL_RENDER_DISTANCE_MACROS * 2 + 1;
                                    let proc_size =
                                        (PROCEDURAL_MACROCHUNK_SIZE * macro_span) as usize;
                                    generated_regions.clear();
                                    active_procgen_origin = [
                                        -PROCEDURAL_MACROCHUNK_SIZE,
                                        0,
                                        -PROCEDURAL_MACROCHUNK_SIZE,
                                    ];
                                    let config = ProcGenConfig::for_size(proc_size, seed)
                                        .with_origin(active_procgen_origin);
                                    procgen_jobs.bump_epoch();
                                    active_procgen = Some(config);
                                    residency_plan.clear();
                                    residency_plan.insert(active_procgen_origin);
                                    procgen_jobs.enqueue(ProcgenJobSpec {
                                        config,
                                        prefer_safe_spawn: true,
                                        target_global_pos: [0, 0, 0],
                                        priority: ProcgenPriority::Urgent,
                                        epoch: procgen_jobs.current_epoch(),
                                    });
                                } else {
                                    procgen_jobs.bump_epoch();
                                    generated_regions.clear();
                                    residency_plan.clear();
                                    active_procgen = None;
                                    active_procgen_origin = [0, 0, 0];
                                    world = World::new([n, n, n]);
                                    let spawn = find_safe_spawn(&world, 1);
                                    ctrl.position = Vec3::new(spawn[0], spawn[1], spawn[2]);
                                }
                                sim.running = false;
                            }
                            if actions.save {
                                if let Ok(path) = default_save_path() {
                                    let _ = save_world(&path, &world);
                                }
                            }
                            if actions.load {
                                if let Ok(path) = default_save_path() {
                                    if let Ok(w) = load_world(&path) {
                                        world = w;
                                        procgen_jobs.bump_epoch();
                                        generated_regions.clear();
                                        residency_plan.clear();
                                        active_procgen = None;
                                        active_procgen_origin = [0, 0, 0];
                                        let spawn = find_safe_spawn(&world, 1);
                                        ctrl.position = Vec3::new(spawn[0], spawn[1], spawn[2]);
                                        sim.running = false;
                                    }
                                }
                            }
                            if actions.toggle_run {
                                sim.running = !sim.running;
                            }
                            if actions.step_once {
                                step(&mut world, &mut sim.rng);
                            }
                        });
                        egui_state.handle_platform_output(window, out.platform_output);

                        for (id, delta) in &out.textures_delta.set {
                            egui_rpass.update_texture(
                                &renderer.device,
                                &renderer.queue,
                                *id,
                                delta,
                            );
                        }
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
                            let cam = Camera {
                                pos: camera_world_pos_from_blocks(ctrl.position, VOXEL_SIZE),
                                dir: ctrl.look_dir(),
                                aspect: renderer.config.width as f32
                                    / renderer.config.height.max(1) as f32,
                            };
                            renderer.render_world(&mut pass, &cam);
                        }
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
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .context("event loop")
}

fn texture_for_active_tool(
    tool_textures: &ToolTextures,
    active_tool: ToolKind,
) -> Option<(egui::TextureId, [usize; 2])> {
    let texture = tool_textures.for_tool(active_tool);
    Some((texture.texture.id(), texture.size))
}

impl ProcgenJobSystem {
    fn new(result_tx: Sender<ProcgenJobResult>, workers: usize, queued_limit: usize) -> Self {
        let shared = Arc::new((
            Mutex::new(ProcgenQueueState {
                queue: BinaryHeap::new(),
                queued_keys: HashSet::new(),
                in_flight: HashSet::new(),
                statuses: HashMap::new(),
                queued_limit,
            }),
            Condvar::new(),
        ));
        let epoch = Arc::new(AtomicU64::new(1));
        let seq = Arc::new(AtomicU64::new(1));
        for _ in 0..workers.max(1) {
            let shared_c = shared.clone();
            let tx_c = result_tx.clone();
            let epoch_c = epoch.clone();
            let seq_c = seq.clone();
            std::thread::spawn(move || loop {
                let spec = {
                    let (lock, cv) = &*shared_c;
                    let mut st = lock.lock().expect("procgen queue lock poisoned");
                    while st.queue.is_empty() {
                        st = cv.wait(st).expect("procgen queue wait poisoned");
                    }
                    let job = st.queue.pop().expect("queue not empty");
                    let key = job.spec.config.world_origin;
                    st.queued_keys.remove(&key);
                    st.in_flight.insert(key);
                    st.statuses.insert(key, ProcgenJobStatus::InFlight);
                    job.spec
                };
                let key = spec.config.world_origin;
                let should_cancel = || epoch_c.load(AtomicOrdering::Relaxed) != spec.epoch;
                let world = generate_world_cancellable(spec.config, &should_cancel);
                let status = if world.is_some() {
                    ProcgenJobStatus::Ready
                } else {
                    ProcgenJobStatus::Cancelled
                };
                let _ = tx_c.send(ProcgenJobResult {
                    key,
                    epoch: spec.epoch,
                    config: spec.config,
                    world,
                    prefer_safe_spawn: spec.prefer_safe_spawn,
                    target_global_pos: spec.target_global_pos,
                    status,
                });
                let (lock, _) = &*shared_c;
                let mut st = lock.lock().expect("procgen queue lock poisoned");
                st.in_flight.remove(&key);
                st.statuses.insert(key, status);
                let _ = seq_c.fetch_add(1, AtomicOrdering::Relaxed);
            });
        }
        Self { shared, epoch }
    }

    fn current_epoch(&self) -> u64 {
        self.epoch.load(AtomicOrdering::Relaxed)
    }

    fn bump_epoch(&self) {
        self.epoch.fetch_add(1, AtomicOrdering::Relaxed);
        let (lock, _) = &*self.shared;
        let mut st = lock.lock().expect("procgen queue lock poisoned");
        st.queue.clear();
        st.queued_keys.clear();
        for status in st.statuses.values_mut() {
            if matches!(
                *status,
                ProcgenJobStatus::Queued | ProcgenJobStatus::InFlight
            ) {
                *status = ProcgenJobStatus::Cancelled;
            }
        }
    }

    fn enqueue(&self, spec: ProcgenJobSpec) -> ProcgenJobStatus {
        let key = spec.config.world_origin;
        let (lock, cv) = &*self.shared;
        let mut st = lock.lock().expect("procgen queue lock poisoned");
        if st.queued_keys.contains(&key) || st.in_flight.contains(&key) {
            return *st.statuses.get(&key).unwrap_or(&ProcgenJobStatus::Queued);
        }
        if st.queue.len() >= st.queued_limit {
            if spec.priority == ProcgenPriority::Prefetch {
                st.statuses.insert(key, ProcgenJobStatus::Cancelled);
                return ProcgenJobStatus::Cancelled;
            }
            let mut drained = Vec::new();
            while let Some(job) = st.queue.pop() {
                if job.spec.priority == ProcgenPriority::Prefetch {
                    st.queued_keys.remove(&job.spec.config.world_origin);
                    st.statuses
                        .insert(job.spec.config.world_origin, ProcgenJobStatus::Cancelled);
                    break;
                }
                drained.push(job);
            }
            for job in drained {
                st.queue.push(job);
            }
            if st.queue.len() >= st.queued_limit {
                st.statuses.insert(key, ProcgenJobStatus::Cancelled);
                return ProcgenJobStatus::Cancelled;
            }
        }
        let seq = st.statuses.len() as u64 + st.queue.len() as u64 + 1;
        st.queue.push(QueuedProcgenJob { spec, seq });
        st.queued_keys.insert(key);
        st.statuses.insert(key, ProcgenJobStatus::Queued);
        cv.notify_one();
        ProcgenJobStatus::Queued
    }

    fn cancel_unplanned(&self, plan: &BTreeSet<[i32; 3]>) {
        let (lock, _) = &*self.shared;
        let mut st = lock.lock().expect("procgen queue lock poisoned");
        st.queue
            .retain(|job| plan.contains(&job.spec.config.world_origin));
        st.queued_keys.retain(|key| plan.contains(key));
        for (key, status) in st.statuses.iter_mut() {
            if !plan.contains(key) && matches!(*status, ProcgenJobStatus::Queued) {
                *status = ProcgenJobStatus::Cancelled;
            }
        }
    }

    fn status_for(&self, key: [i32; 3]) -> Option<ProcgenJobStatus> {
        let (lock, _) = &*self.shared;
        let st = lock.lock().expect("procgen queue lock poisoned");
        st.statuses.get(&key).copied()
    }

    fn record_result_status(&self, result: &ProcgenJobResult) {
        let (lock, _) = &*self.shared;
        let mut st = lock.lock().expect("procgen queue lock poisoned");
        st.statuses.insert(result.key, result.status);
    }
}

fn desired_active_origin(
    active_procgen_origin: [i32; 3],
    player_macro: [i32; 3],
    macro_size: i32,
    render_distance_macros: i32,
) -> [i32; 3] {
    let origin_macro = [
        floor_div(active_procgen_origin[0], macro_size),
        floor_div(active_procgen_origin[2], macro_size),
    ];
    let center_macro = [
        origin_macro[0] + render_distance_macros,
        origin_macro[1] + render_distance_macros,
    ];
    let outside_inner_band = (player_macro[0] - center_macro[0]).abs() > 1
        || (player_macro[2] - center_macro[1]).abs() > 1;
    if outside_inner_band {
        [
            (player_macro[0] - render_distance_macros) * macro_size,
            0,
            (player_macro[2] - render_distance_macros) * macro_size,
        ]
    } else {
        active_procgen_origin
    }
}

fn desired_residency_origins(
    player_macro: [i32; 3],
    macro_size: i32,
    render_distance_macros: i32,
) -> BTreeSet<[i32; 3]> {
    let mut out = BTreeSet::new();
    let center_x = player_macro[0] - render_distance_macros;
    let center_z = player_macro[2] - render_distance_macros;
    for dz in -1..=1 {
        for dx in -1..=1 {
            out.insert([
                (center_x + dx) * macro_size,
                0,
                (center_z + dz) * macro_size,
            ]);
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

fn should_unlock_cursor(ui: &UiState, quick_menu_held: bool, tab_palette_held: bool) -> bool {
    ui.paused_menu || quick_menu_held || tab_palette_held || ui.show_tool_quick_menu
}

fn next_missing_region_origin(
    generated_regions: &BTreeMap<[i32; 3], World>,
    active_origin: [i32; 3],
    player_macro: [i32; 3],
    macro_size: i32,
    render_distance_macros: i32,
) -> Option<[i32; 3]> {
    let center_x = player_macro[0] - render_distance_macros;
    let center_z = player_macro[2] - render_distance_macros;
    let mut candidates = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            let origin = [
                (center_x + dx) * macro_size,
                0,
                (center_z + dz) * macro_size,
            ];
            if origin == active_origin || generated_regions.contains_key(&origin) {
                continue;
            }
            let manhattan = dx.abs() + dz.abs();
            candidates.push((manhattan, origin));
        }
    }
    candidates.sort_by_key(|(dist, _)| *dist);
    candidates.into_iter().map(|(_, origin)| origin).next()
}

fn prune_generated_regions(
    generated_regions: &mut BTreeMap<[i32; 3], World>,
    center_origin: [i32; 3],
    macro_size: i32,
    keep_radius: i32,
) {
    let center_mx = floor_div(center_origin[0], macro_size);
    let center_mz = floor_div(center_origin[2], macro_size);
    generated_regions.retain(|origin, _| {
        let dx = (floor_div(origin[0], macro_size) - center_mx).abs();
        let dz = (floor_div(origin[2], macro_size) - center_mz).abs();
        dx <= keep_radius && dz <= keep_radius
    });
}

fn floor_div(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn apply_mouse_edit(
    world: &mut World,
    brush: &BrushSettings,
    mat: u16,
    input: &InputState,
    edit_runtime: &mut EditRuntimeState,
    now: Instant,
    raycast: RaycastResult,
    active_tool: ToolKind,
) {
    let requested_mode = held_action_mode(input);

    let Some(mode) = requested_mode else {
        edit_runtime.last_edit_mode = None;
        return;
    };

    let is_just_click = matches!(mode, BrushMode::Place) && input.just_lmb
        || matches!(mode, BrushMode::Erase) && input.just_rmb;

    let repeat_interval_s = brush.repeat_interval_s.max(0.0);
    let repeat_ready = edit_runtime.last_edit_mode != Some(mode)
        || edit_runtime
            .last_edit_at
            .map(|last| (now - last).as_secs_f32() >= repeat_interval_s)
            .unwrap_or(true);

    if !is_just_click && !repeat_ready {
        return;
    }

    if active_tool == ToolKind::AreaTool {
        let target = if mode == BrushMode::Place { mat } else { 0 };
        for p in preview_area_tool_blocks(world, brush, raycast, mode) {
            let resolved = resolve_place_target(world, p, target, mode);
            world.set(p[0], p[1], p[2], resolved);
        }
        edit_runtime.last_edit_at = Some(now);
        edit_runtime.last_edit_mode = Some(mode);
        return;
    }

    if brush.radius == 0
        && (active_tool == ToolKind::BuildersWand || active_tool == ToolKind::DestructorWand)
    {
        if active_tool == ToolKind::BuildersWand && mode != BrushMode::Place {
            return;
        }
        if active_tool == ToolKind::DestructorWand && mode != BrushMode::Erase {
            return;
        }
        let wand_blocks = preview_wand_blocks(world, raycast, active_tool, WAND_MAX_BLOCKS);
        let target = if active_tool == ToolKind::BuildersWand {
            mat
        } else {
            0
        };
        for p in wand_blocks {
            let resolved = resolve_place_target(world, p, target, mode);
            world.set(p[0], p[1], p[2], resolved);
        }
        edit_runtime.last_edit_at = Some(now);
        edit_runtime.last_edit_mode = Some(mode);
        return;
    }

    let Some(center) = brush_center(*brush, raycast, mode) else {
        return;
    };
    apply_brush_with_placement_rules(world, center, *brush, mat, mode);
    edit_runtime.last_edit_at = Some(now);
    edit_runtime.last_edit_mode = Some(mode);
}

fn apply_brush_with_placement_rules(
    world: &mut World,
    center: [i32; 3],
    brush: BrushSettings,
    mat: u16,
    mode: BrushMode,
) {
    let rad = brush.radius.max(0);
    for dz in -rad..=rad {
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let include = brush_shape_includes(brush.shape, dx, dy, dz, rad);
                if !include {
                    continue;
                }
                let p = [center[0] + dx, center[1] + dy, center[2] + dz];
                let target = resolve_place_target(world, p, mat, mode);
                world.set(p[0], p[1], p[2], target);
            }
        }
    }
}

fn resolve_place_target(world: &World, p: [i32; 3], mat: u16, mode: BrushMode) -> u16 {
    if mode == BrushMode::Erase {
        return 0;
    }
    if !matches!(mat, GRASS_ID | BUSH_ID) {
        return mat;
    }
    let below = world.get(p[0], p[1] - 1, p[2]);
    if below == 0 {
        return 0;
    }
    mat
}

fn held_action_mode(input: &InputState) -> Option<BrushMode> {
    if input.just_rmb || input.rmb {
        Some(BrushMode::Erase)
    } else if input.just_lmb || input.lmb {
        Some(BrushMode::Place)
    } else {
        None
    }
}

fn current_action_mode(
    input: &InputState,
    raycast: RaycastResult,
    active_tool: ToolKind,
) -> BrushMode {
    if let Some(mode) = held_action_mode(input) {
        return mode;
    }
    if active_tool == ToolKind::AreaTool && raycast.hit.is_some() {
        BrushMode::Erase
    } else {
        BrushMode::Place
    }
}

fn preview_blocks(
    world: &World,
    brush: &BrushSettings,
    raycast: RaycastResult,
    mode: BrushMode,
    active_tool: ToolKind,
    show_hover_hints: bool,
) -> Vec<[i32; 3]> {
    if active_tool == ToolKind::AreaTool {
        return preview_area_tool_blocks(world, brush, raycast, mode);
    }

    let Some(center) = brush_center(*brush, raycast, mode) else {
        if !show_hover_hints {
            return Vec::new();
        }
        return Vec::new();
    };
    if brush.radius == 0 {
        if active_tool != ToolKind::Brush {
            if active_tool == ToolKind::BuildersWand && mode != BrushMode::Place {
                return Vec::new();
            }
            if active_tool == ToolKind::DestructorWand && mode != BrushMode::Erase {
                return Vec::new();
            }
            return preview_wand_blocks(world, raycast, active_tool, WAND_MAX_BLOCKS);
        }
        if let Some(hit) = raycast.hit {
            return vec![hit];
        }
        if mode == BrushMode::Place || brush.fixed_distance {
            return vec![raycast.place];
        }
        return Vec::new();
    }

    let mut out = Vec::new();
    let rad = brush.radius.max(0);
    for dz in -rad..=rad {
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let include = brush_shape_includes(brush.shape, dx, dy, dz, rad);
                if !include {
                    continue;
                }
                let p = [center[0] + dx, center[1] + dy, center[2] + dz];
                if p[0] < 0
                    || p[1] < 0
                    || p[2] < 0
                    || p[0] >= world.dims[0] as i32
                    || p[1] >= world.dims[1] as i32
                    || p[2] >= world.dims[2] as i32
                {
                    continue;
                }
                out.push(p);
            }
        }
    }
    out
}

fn preview_area_tool_blocks(
    world: &World,
    brush: &BrushSettings,
    raycast: RaycastResult,
    mode: BrushMode,
) -> Vec<[i32; 3]> {
    let Some(center) = area_tool_center(raycast, mode) else {
        return Vec::new();
    };
    let normal = raycast
        .hit
        .map(|hit| {
            [
                raycast.place[0] - hit[0],
                raycast.place[1] - hit[1],
                raycast.place[2] - hit[2],
            ]
        })
        .filter(|n| *n != [0, 0, 0])
        .unwrap_or([0, 1, 0]);

    let (axis_u, axis_v) = match normal {
        [1, _, _] | [-1, _, _] => ([0, 1, 0], [0, 0, 1]),
        [_, 1, _] | [_, -1, _] => ([1, 0, 0], [0, 0, 1]),
        _ => ([1, 0, 0], [0, 1, 0]),
    };

    let mut out = Vec::new();
    let radius = brush.area_tool.radius.max(0);
    let thickness = brush.area_tool.thickness.max(1);

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
                let p = [
                    center[0] + axis_u[0] * du + axis_v[0] * dv + normal[0] * depth,
                    center[1] + axis_u[1] * du + axis_v[1] * dv + normal[1] * depth,
                    center[2] + axis_u[2] * du + axis_v[2] * dv + normal[2] * depth,
                ];
                if p[0] < 0
                    || p[1] < 0
                    || p[2] < 0
                    || p[0] >= world.dims[0] as i32
                    || p[1] >= world.dims[1] as i32
                    || p[2] >= world.dims[2] as i32
                {
                    continue;
                }
                out.push(p);
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
    if brush.radius == 0 {
        return match mode {
            BrushMode::Place => Some(raycast.place),
            BrushMode::Erase => raycast.hit,
        };
    }
    match mode {
        BrushMode::Place => Some(raycast.place),
        BrushMode::Erase => raycast.hit,
    }
}

fn target_for_edit(world: &World, origin: Vec3, dir: Vec3, brush: &BrushSettings) -> RaycastResult {
    if brush.fixed_distance && !brush.minecraft_style_placement {
        RaycastResult {
            hit: None,
            place: fixed_distance_target(origin, dir, brush.max_distance),
        }
    } else {
        raycast_target(world, origin, dir, brush.max_distance)
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

fn brush_shape_includes(shape: BrushShape, dx: i32, dy: i32, dz: i32, rad: i32) -> bool {
    match shape {
        BrushShape::Sphere => (dx * dx + dy * dy + dz * dz) <= rad * rad,
        BrushShape::Cube => dx.abs().max(dy.abs()).max(dz.abs()) <= rad,
        BrushShape::Torus => {
            let ring_radius = (rad as f32).max(1.0);
            let tube_radius = (rad as f32 * 0.5).max(1.0);
            let q = ((dx * dx + dz * dz) as f32).sqrt() - ring_radius;
            (q * q + (dy as f32) * (dy as f32)) <= tube_radius * tube_radius
        }
        BrushShape::Hemisphere => dy >= 0 && (dx * dx + dy * dy + dz * dz) <= rad * rad,
        BrushShape::Bowl => {
            if dy > 0 {
                return false;
            }
            let r2 = dx * dx + dy * dy + dz * dz;
            let inner = (rad - 1).max(0);
            r2 <= rad * rad && r2 >= inner * inner
        }
        BrushShape::InvertedBowl => {
            if dy < 0 {
                return false;
            }
            let r2 = dx * dx + dy * dy + dz * dz;
            let inner = (rad - 1).max(0);
            r2 <= rad * rad && r2 >= inner * inner
        }
    }
}

fn preview_wand_blocks(
    world: &World,
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
    if normal == [0, 0, 0] {
        return Vec::new();
    }

    let source_mat = world.get(hit[0], hit[1], hit[2]);
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

        if world.get(target[0], target[1], target[2]) == 0
            || active_tool == ToolKind::DestructorWand
        {
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
            if world.get(np[0], np[1], np[2]) != source_mat {
                continue;
            }
            visited.insert(np);
            queue.push_back(np);
        }
    }

    out
}

fn raycast_target(world: &World, origin: Vec3, dir: Vec3, max_dist: f32) -> RaycastResult {
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

    if t_max_x < 0.0 {
        t_max_x = 0.0;
    }
    if t_max_y < 0.0 {
        t_max_y = 0.0;
    }
    if t_max_z < 0.0 {
        t_max_z = 0.0;
    }

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
        if world.get(x, y, z) != 0 {
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
