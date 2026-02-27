use crate::input::{FpsController, InputState};
use crate::player::{
    camera_world_pos_from_blocks, eye_height_world_meters, PLAYER_EYE_HEIGHT_BLOCKS,
    PLAYER_HEIGHT_BLOCKS, PLAYER_WIDTH_BLOCKS,
};
use crate::renderer::{Camera, Renderer, VOXEL_SIZE};
use crate::sim::{step, SimState};
use crate::ui::{draw, draw_fps_overlays, selected_material, UiState};
use crate::world::{default_save_path, load_world, save_world, BrushMode, BrushSettings, World};
use anyhow::Context;
use glam::Vec3;
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

#[derive(Default)]
struct EditRuntimeState {
    last_edit_at: Option<Instant>,
    last_edit_mode: Option<BrushMode>,
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
    let mut edit_runtime = EditRuntimeState::default();
    let mut last = Instant::now();
    let start = Instant::now();

    let _ = set_cursor(window, false);
    debug_assert!(PLAYER_HEIGHT_BLOCKS > 0.0 && PLAYER_WIDTH_BLOCKS > 0.0);
    debug_assert!(PLAYER_EYE_HEIGHT_BLOCKS <= PLAYER_HEIGHT_BLOCKS);
    debug_assert!((eye_height_world_meters(VOXEL_SIZE) - 1.6).abs() < f32::EPSILON);
    renderer.rebuild_dirty_chunks(&mut world);

    event_loop
        .run(move |event, elwt| match &event {
            Event::WindowEvent { event, window_id } if *window_id == window.id() => {
                let egui_c = egui_state.on_window_event(window, event).consumed;
                input.on_window_event(event);
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => renderer.resize(*size),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let PhysicalKey::Code(key) = event.physical_key {
                                match key {
                                    KeyCode::Escape => {
                                        ui.paused_menu = !ui.paused_menu;
                                        let _ = set_cursor(window, ui.paused_menu);
                                    }
                                    KeyCode::KeyP => sim.running = !sim.running,
                                    KeyCode::KeyB => ui.show_brush = !ui.show_brush,
                                    KeyCode::BracketLeft => ui.adjust_sim_speed(-1),
                                    KeyCode::BracketRight => ui.adjust_sim_speed(1),
                                    KeyCode::Backslash => ui.set_sim_speed(1.0),
                                    KeyCode::Digit0 => ui.selected_slot = 0,
                                    KeyCode::Digit1 => ui.selected_slot = 1,
                                    KeyCode::Digit2 => ui.selected_slot = 2,
                                    KeyCode::Digit3 => ui.selected_slot = 3,
                                    KeyCode::Digit4 => ui.selected_slot = 4,
                                    KeyCode::Digit5 => ui.selected_slot = 5,
                                    KeyCode::Digit6 => ui.selected_slot = 6,
                                    KeyCode::Digit7 => ui.selected_slot = 7,
                                    KeyCode::Digit8 => ui.selected_slot = 8,
                                    KeyCode::Digit9 => ui.selected_slot = 9,
                                    _ => {}
                                }
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - last).as_secs_f32().min(0.05);
                        last = now;

                        if !ui.paused_menu && !egui_c {
                            ctrl.sensitivity = ui.mouse_sensitivity;
                            ctrl.step(&input, dt, true, start.elapsed().as_secs_f32());
                            if input.wheel.abs() > 0.0 {
                                let mut s = ui.selected_slot as i32 - input.wheel.signum() as i32;
                                if s < 0 {
                                    s += 10;
                                }
                                ui.selected_slot = (s as usize) % 10;
                            }
                            apply_mouse_edit(
                                &mut world,
                                &ctrl,
                                &brush,
                                selected_material(ui.selected_slot),
                                &input,
                                &mut edit_runtime,
                                now,
                            );
                        }

                        let ray_hit =
                            raycast_hit(&world, ctrl.position, ctrl.look_dir(), brush.max_distance);

                        if sim.running {
                            let step_dt = (sim.fixed_dt / ui.sim_speed).max(1e-4);
                            sim.accumulator += dt;
                            while sim.accumulator >= step_dt {
                                step(&mut world, &mut sim.rng);
                                sim.accumulator -= step_dt;
                            }
                        }
                        renderer.day = ui.day;
                        renderer.rebuild_dirty_chunks(&mut world);

                        let raw = egui_state.take_egui_input(window);
                        let out = egui_ctx.run(raw, |ctx| {
                            let actions = draw(ctx, &mut ui, sim.running, &mut brush);
                            let cam = Camera {
                                pos: camera_world_pos_from_blocks(ctrl.position, VOXEL_SIZE),
                                dir: ctrl.look_dir(),
                                aspect: renderer.config.width as f32
                                    / renderer.config.height.max(1) as f32,
                            };
                            draw_fps_overlays(
                                ctx,
                                ui.paused_menu,
                                ui.sim_speed,
                                cam.view_proj(),
                                [renderer.config.width, renderer.config.height],
                                ray_hit,
                                VOXEL_SIZE,
                            );
                            if actions.new_world {
                                let n = ui.new_world_size.max(16) / 16 * 16;
                                world = World::new([n, n, n]);
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

fn apply_mouse_edit(
    world: &mut World,
    ctrl: &FpsController,
    brush: &BrushSettings,
    mat: u16,
    input: &InputState,
    edit_runtime: &mut EditRuntimeState,
    now: Instant,
) {
    let requested_mode = if input.just_rmb || input.rmb {
        Some(BrushMode::Erase)
    } else if input.just_lmb || input.lmb {
        Some(BrushMode::Place)
    } else {
        None
    };

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

    if let Some(hit) = raycast_hit(world, ctrl.position, ctrl.look_dir(), brush.max_distance) {
        world.apply_brush(hit, *brush, mat, Some(mode));
        edit_runtime.last_edit_at = Some(now);
        edit_runtime.last_edit_mode = Some(mode);
    }
}

fn raycast_hit(world: &World, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<[i32; 3]> {
    let d = dir.normalize_or_zero();
    if d.length_squared() == 0.0 {
        return None;
    }

    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;

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
            return Some([x, y, z]);
        }
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

    None
}
