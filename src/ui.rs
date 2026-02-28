use crate::sim::{material, MATERIALS};
use crate::world::{AreaFootprintShape, BrushMode, BrushSettings, BrushShape, MaterialId};
use glam::{Mat4, Vec2, Vec3};
use std::path::Path;

pub const HOTBAR_SLOTS: usize = 10;

#[derive(Clone, Copy, Debug)]
enum DragSource {
    Hotbar(usize),
    Palette(MaterialId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolKind {
    Brush,
    BuildersWand,
    DestructorWand,
    AreaTool,
}

impl ToolKind {
    pub const ALL: [ToolKind; 4] = [
        ToolKind::Brush,
        ToolKind::BuildersWand,
        ToolKind::DestructorWand,
        ToolKind::AreaTool,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ToolKind::Brush => "Brush",
            ToolKind::BuildersWand => "Builder's Wand",
            ToolKind::DestructorWand => "Destructor Wand",
            ToolKind::AreaTool => "Area Tool",
        }
    }
}

#[derive(Clone)]
pub struct UiState {
    pub paused_menu: bool,
    pub show_brush: bool,
    pub selected_slot: usize,
    pub show_radial_menu: bool,
    pub show_tool_quick_menu: bool,
    pub active_tool: ToolKind,
    pub hovered_shape: Option<BrushShape>,
    pub hovered_tool: Option<ToolKind>,
    pub new_world_size: usize,
    pub day: bool,
    pub mouse_sensitivity: f32,
    pub sim_speed: f32,
    pub hotbar: [MaterialId; HOTBAR_SLOTS],
    pub hovered_palette_material: Option<MaterialId>,
    drag_source: Option<DragSource>,
    drag_target_slot: Option<usize>,
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
            show_radial_menu: false,
            show_tool_quick_menu: false,
            active_tool: ToolKind::Brush,
            hovered_shape: None,
            hovered_tool: None,
            new_world_size: 64,
            day: true,
            mouse_sensitivity: 0.001,
            sim_speed: 1.0,
            hotbar: [1, 2, 3, 4, 5, 6, 7, 11, 12, 16],
            hovered_palette_material: None,
            drag_source: None,
            drag_target_slot: None,
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
    tool_textures: &ToolTextures,
    tab_palette_held: bool,
) -> UiActions {
    let mut actions = UiActions::default();
    let opaque_panel = egui::Frame::none().fill(egui::Color32::from_rgb(18, 20, 26));
    egui::TopBottomPanel::top("top")
        .frame(opaque_panel)
        .show(ctx, |ui| {
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
                    material(selected_material(ui_state, ui_state.selected_slot)).name
                ));
                ui.label(format!("Brush r={}", brush.radius));
                ui.label(format!("Tool: {}", ui_state.active_tool.label()));
            });
        });

    egui::TopBottomPanel::bottom("toolbar")
        .frame(egui::Frame::none().fill(egui::Color32::from_rgb(16, 18, 22)))
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui_state.drag_target_slot = None;
                for i in 0..HOTBAR_SLOTS {
                    let id = selected_material(ui_state, i);
                    let m = material(id);
                    let response = draw_material_button(
                        ui,
                        [72.0, 44.0],
                        i,
                        m.name,
                        m.color,
                        i == ui_state.selected_slot,
                    );
                    if response.clicked() {
                        ui_state.selected_slot = i;
                    }

                    if response.drag_started() {
                        ui_state.drag_source = Some(DragSource::Hotbar(i));
                    }
                    if ui_state.drag_source.is_some() && response.hovered() {
                        ui_state.drag_target_slot = Some(i);
                    }
                }
            });
        });

    if tab_palette_held && !ui_state.paused_menu {
        egui::Window::new("Materials (TAB)")
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -64.0])
            .collapsible(false)
            .resizable(false)
            .title_bar(false)
            .frame(egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgb(20, 22, 28)))
            .show(ctx, |ui| {
                ui_state.hovered_palette_material = None;
                ui.horizontal_wrapped(|ui| {
                    for i in 0..MATERIALS.len() {
                        let id = MATERIALS[i].id;
                        let m = material(id);
                        let response = draw_material_button(
                            ui,
                            [120.0, 64.0],
                            i,
                            m.name,
                            m.color,
                            id == selected_material(ui_state, ui_state.selected_slot),
                        );
                        if response.clicked() {
                            if let Some(slot) =
                                ui_state.hotbar.iter().position(|&slot_id| slot_id == id)
                            {
                                ui_state.selected_slot = slot;
                            }
                        }
                        if response.hovered() {
                            ui_state.hovered_palette_material = Some(id);
                        }
                        if response.drag_started() {
                            ui_state.drag_source = Some(DragSource::Palette(id));
                        }
                    }
                });
                ui.separator();
                ui.label("TAB Palette Options");
                ui.add(egui::Slider::new(&mut brush.radius, 0..=8).text("Brush radius"));
                ui.add(
                    egui::Slider::new(&mut brush.area_tool.radius, 0..=12).text("Area tool radius"),
                );
                ui.add(
                    egui::Slider::new(&mut brush.max_distance, 2.0..=48.0)
                        .text("Placement distance"),
                );
                ui.label("Hold TAB to configure quick brush options");
                ui.label(
                    "Drag materials onto hotbar slots, or press 0-9 while hovering a material.",
                );
            });
    }

    if ui_state.drag_source.is_some() && ctx.input(|i| i.pointer.any_released()) {
        if let (Some(source), Some(target)) =
            (ui_state.drag_source.take(), ui_state.drag_target_slot)
        {
            match source {
                DragSource::Palette(material_id) => {
                    ui_state.hotbar[target] = material_id;
                    ui_state.selected_slot = target;
                }
                DragSource::Hotbar(from) if from != target => {
                    ui_state.hotbar.swap(from, target);
                    if ui_state.selected_slot == from {
                        ui_state.selected_slot = target;
                    } else if ui_state.selected_slot == target {
                        ui_state.selected_slot = from;
                    }
                }
                _ => {}
            }
        }
        ui_state.drag_target_slot = None;
    }

    if ui_state.show_brush {
        egui::Window::new("Brush").show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut brush.radius, 0..=8).text("Radius"));
            ui.add(egui::Slider::new(&mut brush.max_distance, 2.0..=48.0).text("Distance"));
            ui.add(
                egui::Slider::new(&mut brush.repeat_interval_s, 0.01..=0.5).text("Hold repeat (s)"),
            );
            ui.checkbox(
                &mut brush.minecraft_style_placement,
                "Minecraft style placement (no midair placement/hints)",
            );
            ui.add_enabled_ui(!brush.minecraft_style_placement, |ui| {
                ui.checkbox(
                    &mut brush.fixed_distance,
                    "Fixed placement distance (always, no raycast collision)",
                );
            });
            if brush.minecraft_style_placement {
                brush.fixed_distance = false;
            }
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut brush.shape, BrushShape::Sphere, "Sphere");
                ui.selectable_value(&mut brush.shape, BrushShape::Cube, "Cube");
                ui.selectable_value(&mut brush.shape, BrushShape::Torus, "Torus");
                ui.selectable_value(&mut brush.shape, BrushShape::Hemisphere, "Hemisphere");
                ui.selectable_value(&mut brush.shape, BrushShape::Bowl, "Bowl");
                ui.selectable_value(&mut brush.shape, BrushShape::InvertedBowl, "Inverted Bowl");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut brush.mode, BrushMode::Place, "Place");
                ui.selectable_value(&mut brush.mode, BrushMode::Erase, "Erase");
            });
            ui.separator();
            ui.label("Area tool");
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut brush.area_tool.shape,
                    AreaFootprintShape::Circle,
                    "Circle",
                );
                ui.selectable_value(
                    &mut brush.area_tool.shape,
                    AreaFootprintShape::Square,
                    "Square",
                );
            });
            ui.add(egui::Slider::new(&mut brush.area_tool.radius, 0..=12).text("Size"));
            ui.add(egui::Slider::new(&mut brush.area_tool.thickness, 1..=4).text("Thickness"));
        });
    }

    if ui_state.show_tool_quick_menu && !ui_state.paused_menu {
        egui::Window::new("Tools")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .title_bar(false)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Shape Quick Select");
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        for (shape, label) in [
                            (BrushShape::Sphere, "Sphere"),
                            (BrushShape::Cube, "Cube"),
                            (BrushShape::Torus, "Torus"),
                            (BrushShape::Hemisphere, "Hemisphere"),
                            (BrushShape::Bowl, "Bowl"),
                            (BrushShape::InvertedBowl, "Inverted Bowl"),
                        ] {
                            let response = ui.add_sized(
                                egui::vec2(184.0, 32.0),
                                egui::SelectableLabel::new(brush.shape == shape, label),
                            );
                            if response.hovered() {
                                ui_state.hovered_shape = Some(shape);
                            }
                            if response.clicked() {
                                brush.shape = shape;
                            }
                        }
                    });
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Tool Quick Select");
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        for tool in ToolKind::ALL {
                            let texture = tool_textures.for_tool(tool);
                            let mut button = egui::Button::image_and_text(
                                egui::Image::new((texture.texture.id(), egui::vec2(32.0, 32.0)))
                                    .texture_options(egui::TextureOptions::NEAREST),
                                tool.label(),
                            )
                            .min_size(egui::vec2(184.0, 52.0));
                            if tool == ui_state.active_tool {
                                button = button.fill(egui::Color32::from_rgb(60, 110, 160));
                            }
                            let response = ui.add(button);
                            if response.hovered() {
                                ui_state.hovered_tool = Some(tool);
                            }
                            if response.clicked() {
                                ui_state.active_tool = tool;
                            }
                        }
                    });
                    ui.add_space(6.0);
                    if ui_state.active_tool == ToolKind::AreaTool {
                        ui.separator();
                        ui.label("Area tool settings");
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut brush.area_tool.shape,
                                AreaFootprintShape::Circle,
                                "Circle",
                            );
                            ui.selectable_value(
                                &mut brush.area_tool.shape,
                                AreaFootprintShape::Square,
                                "Square",
                            );
                        });
                        ui.add(egui::Slider::new(&mut brush.area_tool.radius, 0..=12).text("Size"));
                    }
                    ui.label("Hold Q and release to select hovered tool");
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

fn draw_material_button(
    ui: &mut egui::Ui,
    size: [f32; 2],
    slot: usize,
    name: &str,
    color: [u8; 4],
    selected: bool,
) -> egui::Response {
    let desired = egui::vec2(size[0], size[1]);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
    if !ui.is_rect_visible(rect) {
        return response;
    }

    let fill = egui::Color32::from_rgba_premultiplied(color[0], color[1], color[2], color[3]);
    let painter = ui.painter();
    let rounding = egui::Rounding::same(6.0);
    painter.rect_filled(rect, rounding, fill);

    let label_bg = egui::Color32::from_rgba_premultiplied(10, 10, 10, 205);
    let text_color = egui::Color32::from_rgb(235, 235, 235);
    let text_outline = egui::Color32::from_rgba_premultiplied(0, 0, 0, 225);

    let label_height = (rect.height() * 0.58).clamp(24.0, 38.0);
    let label_rect = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + 4.0, rect.max.y - label_height - 4.0),
        egui::pos2(rect.max.x - 4.0, rect.max.y - 4.0),
    );
    painter.rect_filled(label_rect, egui::Rounding::same(4.0), label_bg);

    let text_pos = label_rect.center();
    let text = format!("{} {}", slot, name);
    let font = egui::FontId::proportional(12.0);
    for offset in [
        egui::vec2(-1.0, 0.0),
        egui::vec2(1.0, 0.0),
        egui::vec2(0.0, -1.0),
        egui::vec2(0.0, 1.0),
    ] {
        painter.text(
            text_pos + offset,
            egui::Align2::CENTER_CENTER,
            &text,
            font.clone(),
            text_outline,
        );
    }
    painter.text(
        text_pos,
        egui::Align2::CENTER_CENTER,
        &text,
        font,
        text_color,
    );

    let border_color = if selected {
        egui::Color32::from_rgb(0, 215, 255)
    } else {
        egui::Color32::from_rgba_premultiplied(255, 255, 255, 80)
    };
    let border_width = if selected { 3.0 } else { 1.0 };
    painter.rect_stroke(
        rect,
        rounding,
        egui::Stroke::new(border_width + 1.5, egui::Color32::BLACK),
    );
    painter.rect_stroke(
        rect,
        rounding,
        egui::Stroke::new(border_width, border_color),
    );

    response
}

pub fn selected_material(ui_state: &UiState, slot: usize) -> MaterialId {
    ui_state.hotbar[slot.min(HOTBAR_SLOTS - 1)]
}

pub fn assign_hotbar_slot(ui_state: &mut UiState, slot: usize, material_id: MaterialId) {
    if slot >= HOTBAR_SLOTS {
        return;
    }
    ui_state.hotbar[slot] = material_id;
    ui_state.selected_slot = slot;
}

pub fn draw_fps_overlays(
    ctx: &egui::Context,
    paused: bool,
    sim_speed: f32,
    vp: Mat4,
    viewport: [u32; 2],
    preview_blocks: &[[i32; 3]],
    brush: &BrushSettings,
    action_mode: BrushMode,
    show_radial_menu: bool,
    radial_toggle_key_label: &str,
    voxel_size: f32,
    held_tool: Option<(egui::TextureId, [usize; 2])>,
    now_s: f32,
    using_tool: bool,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("fps_overlays"),
    ));

    if !paused {
        let c = ctx.screen_rect().center();
        let s = 6.0;
        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        painter.line_segment([egui::pos2(c.x - s, c.y), egui::pos2(c.x + s, c.y)], stroke);
        painter.line_segment([egui::pos2(c.x, c.y - s), egui::pos2(c.x, c.y + s)], stroke);
    }

    let rect = ctx.screen_rect();
    let speed_pos = egui::pos2(rect.max.x - 12.0, 12.0);
    painter.text(
        speed_pos,
        egui::Align2::RIGHT_TOP,
        format!("{sim_speed:.1}x"),
        egui::FontId::proportional(14.0),
        egui::Color32::WHITE,
    );

    let outline_color = if action_mode == BrushMode::Place {
        egui::Color32::from_rgb(120, 220, 120)
    } else {
        egui::Color32::from_rgb(255, 120, 120)
    };

    for block in preview_blocks {
        draw_block_outline(
            &painter,
            vp,
            viewport,
            ctx.pixels_per_point(),
            *block,
            voxel_size,
            outline_color,
        );
    }

    draw_brush_radial_hint(
        ctx,
        &painter,
        ctx.screen_rect(),
        brush,
        action_mode,
        !paused && show_radial_menu,
        radial_toggle_key_label,
    );

    if let Some((texture, size)) = held_tool {
        draw_held_tool_sprite(
            &painter,
            ctx.screen_rect(),
            texture,
            size,
            now_s,
            using_tool,
        );
    }
}

fn draw_held_tool_sprite(
    painter: &egui::Painter,
    rect: egui::Rect,
    texture: egui::TextureId,
    size: [usize; 2],
    now_s: f32,
    using_tool: bool,
) {
    let base_w = (size[0] as f32 * 5.0).clamp(96.0, 260.0);
    let aspect = (size[1] as f32 / size[0].max(1) as f32).clamp(0.35, 1.8);
    let height = base_w * aspect;
    let bob = (now_s * 4.0).sin() * 5.0;
    let swing = if using_tool {
        (now_s * 18.0).sin().abs() * 22.0
    } else {
        0.0
    };

    let pivot = egui::pos2(
        rect.max.x - 84.0 - swing,
        rect.max.y - 64.0 + bob + swing * 0.2,
    );
    let top_left = egui::pos2(pivot.x - base_w * 0.66, pivot.y - height - 10.0);
    let top_right = egui::pos2(pivot.x - base_w * 0.08, pivot.y - height * 1.02);
    let bottom_right = egui::pos2(pivot.x + base_w * 0.10, pivot.y + 7.0);
    let bottom_left = egui::pos2(pivot.x - base_w * 0.46, pivot.y + 20.0);

    let mut mesh = egui::Mesh::with_texture(texture);
    let tint = if using_tool {
        egui::Color32::from_white_alpha(255)
    } else {
        egui::Color32::from_white_alpha(235)
    };
    let idx = mesh.vertices.len() as u32;
    mesh.vertices.push(egui::epaint::Vertex {
        pos: top_left,
        uv: egui::pos2(0.0, 0.0),
        color: tint,
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: top_right,
        uv: egui::pos2(1.0, 0.0),
        color: tint,
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: bottom_right,
        uv: egui::pos2(1.0, 1.0),
        color: tint,
    });
    mesh.vertices.push(egui::epaint::Vertex {
        pos: bottom_left,
        uv: egui::pos2(0.0, 1.0),
        color: tint,
    });
    mesh.indices
        .extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
    painter.add(egui::Shape::mesh(mesh));
}

fn draw_block_outline(
    painter: &egui::Painter,
    vp: Mat4,
    viewport: [u32; 2],
    pixels_per_point: f32,
    block: [i32; 3],
    voxel_size: f32,
    color: egui::Color32,
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
        .map(|c| project(vp, viewport, pixels_per_point, c))
        .collect();
    let stroke = egui::Stroke::new(2.0, color);
    for (a, b) in edges {
        if let (Some(pa), Some(pb)) = (projected[a], projected[b]) {
            painter.line_segment([pa, pb], stroke);
        }
    }
}

fn draw_brush_radial_hint(
    ctx: &egui::Context,
    painter: &egui::Painter,
    rect: egui::Rect,
    brush: &BrushSettings,
    action_mode: BrushMode,
    show_radial_menu: bool,
    radial_toggle_key_label: &str,
) {
    let anim = ctx.animate_bool(egui::Id::new("brush_radial_menu_anim"), show_radial_menu);
    if anim <= 0.0 {
        return;
    }

    let center = rect.center();
    let radius = 52.0 * anim;
    let primary_alpha = (110.0 * anim) as u8;
    let secondary_alpha = (70.0 * anim) as u8;
    let text_alpha = (255.0 * anim) as u8;
    let hint_alpha = (210.0 * anim) as u8;

    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.5, egui::Color32::from_white_alpha(primary_alpha)),
    );
    painter.circle_stroke(
        center,
        (radius - 12.0 * anim).max(0.0),
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(secondary_alpha)),
    );

    painter.text(
        center + egui::vec2(0.0, -radius - 18.0 * anim),
        egui::Align2::CENTER_CENTER,
        format!("Radius {}", brush.radius),
        egui::FontId::proportional(13.0),
        egui::Color32::from_white_alpha(text_alpha),
    );
    painter.text(
        center + egui::vec2(radius + 22.0 * anim, 0.0),
        egui::Align2::CENTER_CENTER,
        if action_mode == BrushMode::Place {
            "Place"
        } else {
            "Erase"
        },
        egui::FontId::proportional(12.0),
        egui::Color32::from_white_alpha(text_alpha),
    );
    painter.text(
        center + egui::vec2(-radius - 26.0 * anim, 0.0),
        egui::Align2::CENTER_CENTER,
        format!("{:?}", brush.shape),
        egui::FontId::proportional(12.0),
        egui::Color32::from_white_alpha(text_alpha),
    );
    painter.text(
        center + egui::vec2(0.0, radius + 18.0 * anim),
        egui::Align2::CENTER_CENTER,
        format!(
            "LMB: add | RMB: erase | Scroll: material | Ctrl+Scroll: radius | Alt+Scroll: range | {}: radial",
            radial_toggle_key_label
        ),
        egui::FontId::proportional(11.0),
        egui::Color32::from_white_alpha(hint_alpha),
    );
}

fn project(vp: Mat4, viewport: [u32; 2], pixels_per_point: f32, p: Vec3) -> Option<egui::Pos2> {
    let clip = vp * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    let size = Vec2::new(viewport[0] as f32, viewport[1] as f32);
    let x = ((ndc.x * 0.5 + 0.5) * size.x) / pixels_per_point;
    let y = ((1.0 - (ndc.y * 0.5 + 0.5)) * size.y) / pixels_per_point;
    Some(egui::pos2(x, y))
}

#[derive(Clone)]
pub struct ToolTexture {
    pub texture: egui::TextureHandle,
    pub size: [usize; 2],
}

#[derive(Clone)]
pub struct ToolTextures {
    pub brush: ToolTexture,
    pub builders_wand: ToolTexture,
    pub destructor_wand: ToolTexture,
    pub area_tool: ToolTexture,
}

impl ToolTextures {
    pub fn for_tool(&self, tool: ToolKind) -> &ToolTexture {
        match tool {
            ToolKind::Brush => &self.brush,
            ToolKind::BuildersWand => &self.builders_wand,
            ToolKind::DestructorWand => &self.destructor_wand,
            ToolKind::AreaTool => &self.area_tool,
        }
    }
}

pub fn load_tool_textures(ctx: &egui::Context, dir: impl AsRef<Path>) -> ToolTextures {
    let dir = dir.as_ref().to_path_buf();
    let brush = load_single_tool_texture(
        ctx,
        &dir,
        "brush",
        "tool_brush",
        fallback_tool_texture([130, 180, 240, 255]),
    );
    let builders_wand = load_single_tool_texture(
        ctx,
        &dir,
        "builders_wand",
        "tool_builders_wand",
        fallback_tool_texture([140, 220, 120, 255]),
    );
    let destructor_wand = load_single_tool_texture(
        ctx,
        &dir,
        "destructor_wand",
        "tool_destructor_wand",
        fallback_tool_texture([240, 140, 140, 255]),
    );
    let area_tool = load_single_tool_texture(
        ctx,
        &dir,
        "area_tool",
        "tool_area_tool",
        fallback_tool_texture([240, 220, 120, 255]),
    );
    ToolTextures {
        brush,
        builders_wand,
        destructor_wand,
        area_tool,
    }
}

fn load_single_tool_texture(
    ctx: &egui::Context,
    dir: &Path,
    file_stem: &str,
    egui_name: &str,
    fallback: egui::ColorImage,
) -> ToolTexture {
    let image = ["png", "ppm"]
        .iter()
        .find_map(|ext| {
            let path = dir.join(format!("{file_stem}.{ext}"));
            std::fs::read(path)
                .ok()
                .and_then(|bytes| parse_tool_image(&bytes))
        })
        .unwrap_or(fallback);
    let size = image.size;
    let texture = ctx.load_texture(egui_name.to_string(), image, egui::TextureOptions::NEAREST);
    ToolTexture { texture, size }
}

fn parse_tool_image(bytes: &[u8]) -> Option<egui::ColorImage> {
    parse_standard_image(bytes).or_else(|| parse_ppm_image(bytes))
}

fn parse_standard_image(bytes: &[u8]) -> Option<egui::ColorImage> {
    let dyn_img = image::load_from_memory(bytes).ok()?;
    let rgba = dyn_img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        rgba.as_raw(),
    ))
}

fn parse_ppm_image(bytes: &[u8]) -> Option<egui::ColorImage> {
    fn next_token<'a>(bytes: &'a [u8], i: &mut usize) -> Option<&'a [u8]> {
        while *i < bytes.len() {
            let b = bytes[*i];
            if b == b'#' {
                while *i < bytes.len() && bytes[*i] != b'\n' {
                    *i += 1;
                }
            }
            if *i >= bytes.len() || !bytes[*i].is_ascii_whitespace() {
                break;
            }
            *i += 1;
        }
        let start = *i;
        while *i < bytes.len() && !bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
        (start < *i).then_some(&bytes[start..*i])
    }

    let mut i = 0;
    let magic = next_token(bytes, &mut i)?;
    if magic != b"P6" {
        return None;
    }
    let w: usize = std::str::from_utf8(next_token(bytes, &mut i)?)
        .ok()?
        .parse()
        .ok()?;
    let h: usize = std::str::from_utf8(next_token(bytes, &mut i)?)
        .ok()?
        .parse()
        .ok()?;
    let max_val: usize = std::str::from_utf8(next_token(bytes, &mut i)?)
        .ok()?
        .parse()
        .ok()?;
    if max_val != 255 {
        return None;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let expected = w.checked_mul(h)?.checked_mul(3)?;
    let data = bytes.get(i..i + expected)?;
    let mut rgba = vec![0u8; w * h * 4];
    for px in 0..(w * h) {
        let r = data[px * 3];
        let g = data[px * 3 + 1];
        let b = data[px * 3 + 2];
        let alpha = if r >= 250 && g >= 250 && b >= 250 {
            0
        } else {
            255
        };
        rgba[px * 4] = r;
        rgba[px * 4 + 1] = g;
        rgba[px * 4 + 2] = b;
        rgba[px * 4 + 3] = alpha;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

fn fallback_tool_texture(color: [u8; 4]) -> egui::ColorImage {
    let size = [16, 16];
    let mut pixels = vec![egui::Color32::TRANSPARENT; size[0] * size[1]];
    for y in 0..size[1] {
        for x in 0..size[0] {
            let idx = y * size[0] + x;
            let border = x == 0 || y == 0 || x == size[0] - 1 || y == size[1] - 1;
            if border {
                pixels[idx] = egui::Color32::from_rgba_premultiplied(16, 16, 16, 230);
            } else {
                pixels[idx] =
                    egui::Color32::from_rgba_premultiplied(color[0], color[1], color[2], color[3]);
            }
        }
    }
    egui::ColorImage { size, pixels }
}
