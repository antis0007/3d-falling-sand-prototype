use crate::input::{FpsController, InputState};
use crate::player::{
    camera_world_pos_from_blocks, eye_height_world_meters, PLAYER_EYE_HEIGHT_BLOCKS,
    PLAYER_HEIGHT_BLOCKS, PLAYER_WIDTH_BLOCKS,
};
use crate::procgen::{
    biome_hint_at_world, find_safe_spawn, generate_world_with_control, ProcGenConfig,
    ProcGenControl,
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
use crate::world_stream::{floor_div, ChunkResidency, WorldStream};
use anyhow::Context;
use glam::Vec3;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
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
const PROCEDURAL_SIM_DISTANCE_MACROS: i32 = 1;
const PROCEDURAL_RENDER_DISTANCE_MACROS: i32 = 4;
const PROCGEN_URGENT_BUDGET_PER_FRAME: usize = 6;
const PROCGEN_PREFETCH_BUDGET_PER_FRAME: usize = 8;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProcgenJobKey([i32; 3]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcgenPriority {
    Urgent,
    Prefetch,
}

#[derive(Clone, Copy)]
struct ProcgenJobSpec {
    generation_id: u64,
    key: ProcgenJobKey,
    coord: [i32; 3],
    config: ProcGenConfig,
    prefer_safe_spawn: bool,
    target_global_pos: [i32; 3],
    priority: ProcgenPriority,
    epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcgenJobStatus {
    Queued,
    InFlight,
    Ready,
    Cancelled,
}

struct ProcgenJobResult {
    spec: ProcgenJobSpec,
    status: ProcgenJobStatus,
    world: Option<World>,
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
    let mut active_procgen: Option<ProcGenConfig> = None;
    let mut stream: Option<WorldStream> = None;
    let mut active_stream_coord: [i32; 3] = [0, 0, 0];
    let (procgen_tx, procgen_rx): (Sender<ProcgenJobResult>, Receiver<ProcgenJobResult>) =
        mpsc::channel();
    let mut next_procgen_id: u64 = 1;
    let mut next_epoch: u64 = 1;
    let mut job_status: HashMap<ProcgenJobKey, ProcgenJobStatus> = HashMap::new();
    let mut requested_procgen_id: Option<u64> = None;
    let mut requested_procgen_coord: Option<[i32; 3]> = None;
    let procgen_workers = ProcgenWorkerPool::new(procgen_tx.clone(), 3, 3, 64, 6);

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
                                let hotbar_slot = key_to_hotbar_slot(key);
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
                                    _ if hotbar_slot.is_some() => {
                                        if let Some(slot) = hotbar_slot {
                                            assign_or_select_hotbar(
                                                &mut ui,
                                                slot,
                                                tab_palette_open,
                                            );
                                        }
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
                                apply_quick_menu_hover_selection(&mut ui, &mut brush);
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

                        let mut procedural_simulation_allowed = true;
                        renderer.day = ui.day;

                        while let Ok(result) = procgen_rx.try_recv() {
                            job_status.insert(result.spec.key, result.status);
                            if let Some(stream_ref) = stream.as_mut() {
                                match result.status {
                                    ProcgenJobStatus::Ready => {
                                        if let Some(generated) = result.world {
                                            stream_ref
                                                .apply_generated(result.spec.coord, generated);
                                            if result.spec.coord == active_stream_coord {
                                                let _ = stream_ref.load_coord_into_world(
                                                    active_stream_coord,
                                                    &mut world,
                                                );
                                                if result.spec.prefer_safe_spawn {
                                                    let spawn = find_safe_spawn(
                                                        &world,
                                                        result.spec.config.seed,
                                                    );
                                                    ctrl.position =
                                                        Vec3::new(spawn[0], spawn[1], spawn[2]);
                                                }
                                            }
                                        }
                                    }
                                    ProcgenJobStatus::Cancelled => {
                                        job_status.remove(&result.spec.key);
                                    }
                                    _ => {}
                                }
                            }
                        }

                        if let (Some(cfg), Some(stream_ref)) = (active_procgen, stream.as_mut()) {
                            let chunk_size = cfg.dims[0] as i32;
                            stream_ref.persist_coord_state(active_stream_coord, &world);
                            let active_origin = stream_ref.chunk_origin(active_stream_coord);
                            let global = [
                                active_origin[0] + ctrl.position.x.floor() as i32,
                                0,
                                active_origin[2] + ctrl.position.z.floor() as i32,
                            ];
                            let desired_coord = [
                                floor_div(global[0], chunk_size),
                                0,
                                floor_div(global[2], chunk_size),
                            ];

                            let sim_residency =
                                macro_residency_set(desired_coord, PROCEDURAL_SIM_DISTANCE_MACROS);
                            let render_residency = macro_residency_set(
                                desired_coord,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            );
                            procedural_simulation_allowed =
                                sim_residency.contains(&active_stream_coord);

                            if desired_coord != active_stream_coord {
                                let frac_x = ctrl.position.x - ctrl.position.x.floor();
                                let frac_z = ctrl.position.z - ctrl.position.z.floor();
                                if stream_ref.load_coord_into_world(desired_coord, &mut world) {
                                    active_stream_coord = desired_coord;
                                    let new_origin = stream_ref.chunk_origin(active_stream_coord);
                                    ctrl.position = Vec3::new(
                                        (global[0] - new_origin[0]) as f32 + frac_x,
                                        ctrl.position.y,
                                        (global[2] - new_origin[2]) as f32 + frac_z,
                                    );
                                }
                            }

                            for (&key, status) in job_status.clone().iter() {
                                if !render_residency.contains(&key.0)
                                    && matches!(
                                        status,
                                        ProcgenJobStatus::Queued | ProcgenJobStatus::InFlight
                                    )
                                {
                                    next_epoch = next_epoch.wrapping_add(1);
                                    procgen_workers.cancel_key(key, next_epoch);
                                }
                            }

                            let mut unloaded_render: Vec<[i32; 3]> = render_residency
                                .iter()
                                .copied()
                                .filter(|&coord| {
                                    stream_ref.state(coord) == ChunkResidency::Unloaded
                                })
                                .collect();
                            unloaded_render
                                .sort_by_key(|coord| macro_distance_sq(*coord, desired_coord));

                            let urgent_coords: Vec<[i32; 3]> = unloaded_render
                                .iter()
                                .copied()
                                .filter(|coord| sim_residency.contains(coord))
                                .collect();
                            schedule_procgen_batch(
                                &urgent_coords,
                                PROCGEN_URGENT_BUDGET_PER_FRAME,
                                ProcgenPriority::Urgent,
                                stream_ref,
                                &procgen_workers,
                                &mut next_procgen_id,
                                &mut next_epoch,
                                &mut job_status,
                                global,
                            );

                            let prefetch_coords: Vec<[i32; 3]> = unloaded_render
                                .into_iter()
                                .filter(|coord| !sim_residency.contains(coord))
                                .collect();
                            schedule_procgen_batch(
                                &prefetch_coords,
                                PROCGEN_PREFETCH_BUDGET_PER_FRAME,
                                ProcgenPriority::Prefetch,
                                stream_ref,
                                &procgen_workers,
                                &mut next_procgen_id,
                                &mut next_epoch,
                                &mut job_status,
                                global,
                            );
                        }

                        if sim.running && !ui.paused_menu && procedural_simulation_allowed {
                            let step_dt = (sim.fixed_dt / ui.sim_speed).max(1e-4);
                            sim.accumulator += dt;
                            while sim.accumulator >= step_dt {
                                let (high_priority, _low_priority) =
                                    prioritize_chunks_for_player(&world, ctrl.position);
                                step_selected_chunks(&mut world, &mut sim.rng, &high_priority);
                                sim.accumulator -= step_dt;
                            }
                        }

                        renderer.rebuild_dirty_chunks(&mut world);

                        ui.biome_hint = if let (Some(cfg), Some(stream_ref)) =
                            (active_procgen, stream.as_ref())
                        {
                            let origin = stream_ref.chunk_origin(active_stream_coord);
                            let biome = biome_hint_at_world(
                                &cfg,
                                origin[0] + ctrl.position.x.floor() as i32,
                                origin[2] + ctrl.position.z.floor() as i32,
                            );
                            if job_status.values().any(|s| {
                                matches!(s, ProcgenJobStatus::Queued | ProcgenJobStatus::InFlight)
                            }) {
                                format!("Biome: {} (generating...)", biome.label())
                            } else {
                                format!("Biome: {}", biome.label())
                            }
                        } else {
                            "Biome: Flat Test World".to_string()
                        };

                        let queued_count = job_status
                            .values()
                            .filter(|status| matches!(status, ProcgenJobStatus::Queued))
                            .count();
                        let in_flight_count = job_status
                            .values()
                            .filter(|status| matches!(status, ProcgenJobStatus::InFlight))
                            .count();
                        let ready_count = job_status
                            .values()
                            .filter(|status| matches!(status, ProcgenJobStatus::Ready))
                            .count();
                        ui.stream_debug = if active_procgen.is_some() {
                            format!(
                                "Stream {:?} | q:{} in-flight:{} ready:{} | sim_r={} render_r={}",
                                active_stream_coord,
                                queued_count,
                                in_flight_count,
                                ready_count,
                                PROCEDURAL_SIM_DISTANCE_MACROS,
                                PROCEDURAL_RENDER_DISTANCE_MACROS,
                            )
                        } else {
                            "Stream: n/a".to_string()
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
                                    renderer.clear_mesh_cache();
                                    job_status.clear();
                                    let seed = start.elapsed().as_nanos() as u64;
                                    let config = ProcGenConfig::for_size(64, seed);
                                    let mut new_stream = WorldStream::new(config);
                                    let start_coord = [0, 0, 0];
                                    let generated_start = generate_world_with_control(
                                        new_stream.make_config(start_coord),
                                        ProcGenControl {
                                            epoch: 0,
                                            should_cancel: &|_| false,
                                        },
                                    )
                                    .unwrap_or_else(|| World::new(config.dims));
                                    new_stream.apply_generated(start_coord, generated_start);
                                    let _ =
                                        new_stream.load_coord_into_world(start_coord, &mut world);
                                    let spawn = find_safe_spawn(&world, seed);
                                    ctrl.position = Vec3::new(spawn[0], spawn[1], spawn[2]);

                                    active_procgen = Some(config);
                                    active_stream_coord = start_coord;
                                    stream = Some(new_stream);

                                    if let Some(stream_ref) = stream.as_mut() {
                                        let render_coords = sorted_residency_coords(
                                            start_coord,
                                            PROCEDURAL_RENDER_DISTANCE_MACROS,
                                        );
                                        let urgent_coords: Vec<[i32; 3]> = render_coords
                                            .iter()
                                            .copied()
                                            .filter(|coord| {
                                                *coord != start_coord
                                                    && macro_distance_sq(*coord, start_coord)
                                                        <= (PROCEDURAL_SIM_DISTANCE_MACROS
                                                            * PROCEDURAL_SIM_DISTANCE_MACROS)
                                            })
                                            .collect();
                                        schedule_procgen_batch(
                                            &urgent_coords,
                                            PROCGEN_URGENT_BUDGET_PER_FRAME,
                                            ProcgenPriority::Urgent,
                                            stream_ref,
                                            &procgen_workers,
                                            &mut next_procgen_id,
                                            &mut next_epoch,
                                            &mut job_status,
                                            [0, 0, 0],
                                        );
                                        let prefetch_coords: Vec<[i32; 3]> = render_coords
                                            .into_iter()
                                            .filter(|coord| {
                                                *coord != start_coord
                                                    && macro_distance_sq(*coord, start_coord)
                                                        > (PROCEDURAL_SIM_DISTANCE_MACROS
                                                            * PROCEDURAL_SIM_DISTANCE_MACROS)
                                            })
                                            .collect();
                                        schedule_procgen_batch(
                                            &prefetch_coords,
                                            PROCGEN_PREFETCH_BUDGET_PER_FRAME,
                                            ProcgenPriority::Prefetch,
                                            stream_ref,
                                            &procgen_workers,
                                            &mut next_procgen_id,
                                            &mut next_epoch,
                                            &mut job_status,
                                            [0, 0, 0],
                                        );
                                    }
                                } else {
                                    job_status.clear();
                                    active_procgen = None;
                                    stream = None;
                                    active_stream_coord = [0, 0, 0];
                                    clear_procedural_streaming_state(
                                        &mut requested_procgen_id,
                                        &mut requested_procgen_coord,
                                        &mut active_procgen,
                                        &mut stream,
                                        &mut active_stream_coord,
                                    );
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
                                        job_status.clear();
                                        active_procgen = None;
                                        stream = None;
                                        active_stream_coord = [0, 0, 0];
                                        clear_procedural_streaming_state(
                                            &mut requested_procgen_id,
                                            &mut requested_procgen_coord,
                                            &mut active_procgen,
                                            &mut stream,
                                            &mut active_stream_coord,
                                        );
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

#[derive(Clone)]
struct ProcgenQueueEntry {
    spec: ProcgenJobSpec,
    sequence: u64,
}

impl PartialEq for ProcgenQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.spec.key == other.spec.key && self.sequence == other.sequence
    }
}
impl Eq for ProcgenQueueEntry {}
impl Ord for ProcgenQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = |p: ProcgenPriority| match p {
            ProcgenPriority::Urgent => 1u8,
            ProcgenPriority::Prefetch => 0u8,
        };
        rank(self.spec.priority)
            .cmp(&rank(other.spec.priority))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}
impl PartialOrd for ProcgenQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ProcgenQueueState {
    queue: BinaryHeap<ProcgenQueueEntry>,
    queued_keys: HashSet<ProcgenJobKey>,
    in_flight: HashSet<ProcgenJobKey>,
    max_queue: usize,
    max_in_flight: usize,
    sequence: u64,
}

struct ProcgenWorkerPool {
    state: Arc<(Mutex<ProcgenQueueState>, Condvar)>,
    cancel_epochs: Arc<Mutex<HashMap<ProcgenJobKey, u64>>>,
    shutdown: Arc<AtomicBool>,
}

impl ProcgenWorkerPool {
    fn new(
        tx: Sender<ProcgenJobResult>,
        workers: usize,
        max_in_flight: usize,
        max_queue: usize,
        _max_pending_per_key: usize,
    ) -> Self {
        let state = Arc::new((
            Mutex::new(ProcgenQueueState {
                queue: BinaryHeap::new(),
                queued_keys: HashSet::new(),
                in_flight: HashSet::new(),
                max_queue,
                max_in_flight,
                sequence: 0,
            }),
            Condvar::new(),
        ));
        let cancel_epochs = Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        for _ in 0..workers {
            let tx = tx.clone();
            let state_c = state.clone();
            let cancel_c = cancel_epochs.clone();
            let shutdown_c = shutdown.clone();
            std::thread::spawn(move || worker_loop(tx, state_c, cancel_c, shutdown_c));
        }
        Self {
            state,
            cancel_epochs,
            shutdown,
        }
    }

    fn enqueue_or_coalesce(&self, spec: ProcgenJobSpec) -> ProcgenJobStatus {
        self.cancel_epochs
            .lock()
            .expect("cancel lock")
            .insert(spec.key, spec.epoch);
        let (lock, cv) = &*self.state;
        let mut st = lock.lock().expect("procgen queue lock");
        if st.queued_keys.contains(&spec.key) || st.in_flight.contains(&spec.key) {
            return ProcgenJobStatus::Queued;
        }
        if st.queue.len() >= st.max_queue {
            if let Some(dropped) = st.queue.pop() {
                st.queued_keys.remove(&dropped.spec.key);
            }
        }
        st.sequence = st.sequence.wrapping_add(1);
        let seq = st.sequence;
        st.queued_keys.insert(spec.key);
        st.queue.push(ProcgenQueueEntry {
            spec,
            sequence: seq,
        });
        cv.notify_one();
        ProcgenJobStatus::Queued
    }

    fn cancel_key(&self, key: ProcgenJobKey, epoch: u64) {
        self.cancel_epochs
            .lock()
            .expect("cancel lock")
            .insert(key, epoch);
    }
}

fn worker_loop(
    tx: Sender<ProcgenJobResult>,
    state: Arc<(Mutex<ProcgenQueueState>, Condvar)>,
    cancel_epochs: Arc<Mutex<HashMap<ProcgenJobKey, u64>>>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(AtomicOrdering::Relaxed) {
            break;
        }
        let spec = {
            let (lock, cv) = &*state;
            let mut st = lock.lock().expect("procgen queue lock");
            while (st.queue.is_empty() || st.in_flight.len() >= st.max_in_flight)
                && !shutdown.load(AtomicOrdering::Relaxed)
            {
                st = cv.wait(st).expect("condvar wait");
            }
            if shutdown.load(AtomicOrdering::Relaxed) {
                return;
            }
            let Some(entry) = st.queue.pop() else {
                continue;
            };
            st.queued_keys.remove(&entry.spec.key);
            st.in_flight.insert(entry.spec.key);
            let _ = tx.send(ProcgenJobResult {
                spec: entry.spec,
                status: ProcgenJobStatus::InFlight,
                world: None,
            });
            entry.spec
        };

        let cancel_probe = |epoch: u64| {
            let map = cancel_epochs.lock().expect("cancel lock");
            map.get(&spec.key).copied().unwrap_or(epoch) != epoch
        };
        let world = generate_world_with_control(
            spec.config,
            ProcGenControl {
                epoch: spec.epoch,
                should_cancel: &cancel_probe,
            },
        );
        let status = if world.is_some() {
            ProcgenJobStatus::Ready
        } else {
            ProcgenJobStatus::Cancelled
        };
        let _ = tx.send(ProcgenJobResult {
            spec,
            status,
            world,
        });

        let (lock, cv) = &*state;
        let mut st = lock.lock().expect("procgen queue lock");
        st.in_flight.remove(&spec.key);
        cv.notify_one();
    }
}

fn macro_distance_sq(a: [i32; 3], b: [i32; 3]) -> i32 {
    let dx = a[0] - b[0];
    let dz = a[2] - b[2];
    dx * dx + dz * dz
}

fn macro_residency_set(center: [i32; 3], radius: i32) -> HashSet<[i32; 3]> {
    let mut set = HashSet::new();
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            set.insert([center[0] + dx, center[1], center[2] + dz]);
        }
    }
    set
}

fn sorted_residency_coords(center: [i32; 3], radius: i32) -> Vec<[i32; 3]> {
    let mut coords: Vec<[i32; 3]> = macro_residency_set(center, radius).into_iter().collect();
    coords.sort_by_key(|coord| macro_distance_sq(*coord, center));
    coords
}

fn schedule_procgen_batch(
    coords: &[[i32; 3]],
    budget: usize,
    priority: ProcgenPriority,
    stream: &mut WorldStream,
    procgen_workers: &ProcgenWorkerPool,
    next_procgen_id: &mut u64,
    next_epoch: &mut u64,
    job_status: &mut HashMap<ProcgenJobKey, ProcgenJobStatus>,
    target_global_pos: [i32; 3],
) {
    for coord in coords.iter().take(budget) {
        if stream.state(*coord) != ChunkResidency::Unloaded {
            continue;
        }
        let spec = ProcgenJobSpec {
            generation_id: *next_procgen_id,
            key: ProcgenJobKey(*coord),
            coord: *coord,
            config: stream.make_config(*coord),
            prefer_safe_spawn: false,
            target_global_pos,
            priority,
            epoch: *next_epoch,
        };
        *next_procgen_id = next_procgen_id.wrapping_add(1);
        *next_epoch = next_epoch.wrapping_add(1);
        stream.mark_generating(spec.coord);
        let st = procgen_workers.enqueue_or_coalesce(spec);
        job_status.insert(spec.key, st);
    }
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

fn clear_procedural_streaming_state(
    requested_procgen_id: &mut Option<u64>,
    requested_procgen_coord: &mut Option<[i32; 3]>,
    active_procgen: &mut Option<ProcGenConfig>,
    stream: &mut Option<WorldStream>,
    active_stream_coord: &mut [i32; 3],
) {
    *requested_procgen_id = None;
    *requested_procgen_coord = None;
    *active_procgen = None;
    *stream = None;
    *active_stream_coord = [0, 0, 0];
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
    if (active_tool == ToolKind::AreaTool || active_tool == ToolKind::DestructorWand)
        && raycast.hit.is_some()
    {
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
