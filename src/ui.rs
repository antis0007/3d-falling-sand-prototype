use crate::sim::{material, MATERIALS};
use crate::world::{BrushMode, BrushSettings, BrushShape, MaterialId};

#[derive(Clone)]
pub struct UiState {
    pub paused_menu: bool,
    pub show_brush: bool,
    pub selected_slot: usize,
    pub new_world_size: usize,
    pub day: bool,
    pub mouse_sensitivity: f32,
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
