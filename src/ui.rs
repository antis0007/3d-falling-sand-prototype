use crate::sim::{material, MATERIALS};
use crate::world::{BrushMode, BrushSettings, BrushShape, MaterialId};
use glam::{Mat4, Vec2, Vec3};

#[derive(Clone)]
pub struct UiState {
    pub paused_menu: bool,
    pub show_brush: bool,
    pub selected_slot: usize,
    pub new_world_size: usize,
    pub day: bool,
    pub mouse_sensitivity: f32,
    pub sim_speed: f32,
}

impl UiState {
    pub const SIM_SPEED_MIN: f32 = 0.1;
    pub const SIM_SPEED_MAX: f32 = 4.0;
    pub const SIM_SPEED_STEP: f32 = 0.1;

    pub fn clamp_quantize_sim_speed(value: f32) -> f32 {
        let steps = (value / Self::SIM_SPEED_STEP).round();
        (steps * Self::SIM_SPEED_STEP).clamp(Self::SIM_SPEED_MIN, Self::SIM_SPEED_MAX)
    }

    pub fn set_sim_speed(&mut self, value: f32) {
        self.sim_speed = Self::clamp_quantize_sim_speed(value);
    }

    pub fn adjust_sim_speed(&mut self, delta_steps: i32) {
        self.set_sim_speed(self.sim_speed + delta_steps as f32 * Self::SIM_SPEED_STEP);
    }
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            paused_menu: false,
            show_brush: false,
            selected_slot: 3,
            new_world_size: 64,
            day: true,
            mouse_sensitivity: 0.001,
            sim_speed: 1.0,
        }
    }
}

#[derive(Default)]
pub struct UiActions {
    pub new_world: bool,
    pub save: bool,
    pub load: bool,
    pub toggle_run: bool,
    pub step_once: bool,
}

pub fn draw(
    ctx: &egui::Context,
    ui_state: &mut UiState,
    sim_running: bool,
    brush: &mut BrushSettings,
) -> UiActions {
    let mut actions = UiActions::default();
    egui::TopBottomPanel::top("top").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("New World").clicked() {
                actions.new_world = true;
            }
            if ui.button("Save").clicked() {
                actions.save = true;
            }
            if ui.button("Load").clicked() {
                actions.load = true;
            }
            if ui
                .button(if sim_running { "Pause" } else { "Run" })
                .clicked()
            {
                actions.toggle_run = true;
            }
            if ui.button("Step Once").clicked() {
                actions.step_once = true;
            }
            if ui
                .button(if ui_state.day { "Day" } else { "Night" })
                .clicked()
            {
                ui_state.day = !ui_state.day;
            }
            ui.add(
                egui::Slider::new(&mut ui_state.new_world_size, 32..=128)
                    .text("New world size")
                    .step_by(16.0),
            );
            ui.add(
                egui::Slider::new(&mut ui_state.mouse_sensitivity, 0.0002..=0.003)
                    .text("Mouse sensitivity"),
            );
            let changed = ui
                .add(
                    egui::Slider::new(
                        &mut ui_state.sim_speed,
                        UiState::SIM_SPEED_MIN..=UiState::SIM_SPEED_MAX,
                    )
                    .text("Sim speed")
                    .step_by(UiState::SIM_SPEED_STEP as f64),
                )
                .changed();
            if changed {
                ui_state.set_sim_speed(ui_state.sim_speed);
            }
            ui.label(format!(
                "Mat: {}",
                material(selected_material(ui_state.selected_slot)).name
            ));
            ui.label(format!("Brush r={}", brush.radius));
        });
    });

    egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            for i in 0..10 {
                let id = selected_material(i);
                let m = material(id);
                let mut b = egui::Button::new(format!("{}\n{}", i, m.name));
                b = b.fill(egui::Color32::from_rgba_premultiplied(
                    m.color[0], m.color[1], m.color[2], m.color[3],
                ));
                if i == ui_state.selected_slot {
                    b = b.stroke(egui::Stroke::new(2.0, egui::Color32::YELLOW));
                }
                if ui.add_sized([72.0, 44.0], b).clicked() {
                    ui_state.selected_slot = i;
                }
            }
        });
    });

    if ui_state.show_brush {
        egui::Window::new("Brush").show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut brush.radius, 0..=8).text("Radius"));
            ui.add(egui::Slider::new(&mut brush.max_distance, 2.0..=48.0).text("Distance"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut brush.shape, BrushShape::Sphere, "Sphere");
                ui.selectable_value(&mut brush.shape, BrushShape::Cube, "Cube");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut brush.mode, BrushMode::Place, "Place");
                ui.selectable_value(&mut brush.mode, BrushMode::Erase, "Erase");
            });
        });
    }

    if ui_state.paused_menu {
        egui::Window::new("Paused")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label("ESC to return to game");
                if ui.button("Resume").clicked() {
                    ui_state.paused_menu = false;
                }
            });
    }

    actions
}

pub fn selected_material(slot: usize) -> MaterialId {
    MATERIALS[slot.min(9)].id
}

pub fn draw_fps_overlays(
    ctx: &egui::Context,
    paused: bool,
    sim_speed: f32,
    vp: Mat4,
    viewport: [u32; 2],
    hit: Option<[i32; 3]>,
    voxel_size: f32,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("fps_overlays"),
    ));

    if !paused {
        let c = ctx.screen_rect().center();
        let s = 6.0;
        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        painter.line_segment([egui::pos2(c.x - s, c.y), egui::pos2(c.x + s, c.y)], stroke);
        painter.line_segment([egui::pos2(c.x, c.y - s), egui::pos2(c.x, c.y + s)], stroke);
    }

    let speed_pos = egui::pos2(10.0, 10.0);
    painter.text(
        speed_pos,
        egui::Align2::LEFT_TOP,
        format!("{sim_speed:.1}x"),
        egui::FontId::proportional(14.0),
        egui::Color32::WHITE,
    );

    if let Some(block) = hit {
        draw_block_outline(&painter, vp, viewport, block, voxel_size);
    }
}

fn draw_block_outline(
    painter: &egui::Painter,
    vp: Mat4,
    viewport: [u32; 2],
    block: [i32; 3],
    voxel_size: f32,
) {
    let b = Vec3::new(block[0] as f32, block[1] as f32, block[2] as f32) * voxel_size;
    let s = voxel_size;
    let corners = [
        b,
        b + Vec3::new(s, 0.0, 0.0),
        b + Vec3::new(s, s, 0.0),
        b + Vec3::new(0.0, s, 0.0),
        b + Vec3::new(0.0, 0.0, s),
        b + Vec3::new(s, 0.0, s),
        b + Vec3::new(s, s, s),
        b + Vec3::new(0.0, s, s),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let projected: Vec<Option<egui::Pos2>> = corners
        .into_iter()
        .map(|c| project(vp, viewport, c))
        .collect();
    let stroke = egui::Stroke::new(2.0, egui::Color32::YELLOW);
    for (a, b) in edges {
        if let (Some(pa), Some(pb)) = (projected[a], projected[b]) {
            painter.line_segment([pa, pb], stroke);
        }
    }
}

fn project(vp: Mat4, viewport: [u32; 2], p: Vec3) -> Option<egui::Pos2> {
    let clip = vp * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    let size = Vec2::new(viewport[0] as f32, viewport[1] as f32);
    let x = (ndc.x * 0.5 + 0.5) * size.x;
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * size.y;
    Some(egui::pos2(x, y))
}
