use crate::sim::{material, MATERIALS};
use crate::world::{BrushMode, BrushSettings, BrushShape, MaterialId};
use glam::{Mat4, Vec2, Vec3};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolKind {
    Brush,
    BuildersWand,
    DestructorWand,
}

impl ToolKind {
    pub const ALL: [ToolKind; 3] = [
        ToolKind::Brush,
        ToolKind::BuildersWand,
        ToolKind::DestructorWand,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ToolKind::Brush => "Brush",
            ToolKind::BuildersWand => "Builder's Wand",
            ToolKind::DestructorWand => "Destructor Wand",
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
    pub hovered_tool: Option<ToolKind>,
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
            show_radial_menu: false,
            show_tool_quick_menu: false,
            active_tool: ToolKind::Brush,
            hovered_tool: None,
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
            ui.label(format!("Tool: {}", ui_state.active_tool.label()));
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
        });
    }

    if ui_state.show_tool_quick_menu && !ui_state.paused_menu {
        egui::Window::new("Tools")
            .anchor(egui::Align2::CENTER_TOP, [0.0, 56.0])
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for tool in ToolKind::ALL {
                        let mut button = egui::Button::new(tool.label());
                        if tool == ui_state.active_tool {
                            button = button.fill(egui::Color32::from_rgb(60, 110, 160));
                        }
                        let response = ui.add_sized([140.0, 34.0], button);
                        if response.hovered() {
                            ui_state.hovered_tool = Some(tool);
                        }
                        if response.clicked() {
                            ui_state.active_tool = tool;
                        }
                    }
                });
                ui.label("Hold Q and release to select hovered tool");
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
    preview_blocks: &[[i32; 3]],
    brush: &BrushSettings,
    show_radial_menu: bool,
    radial_toggle_key_label: &str,
    voxel_size: f32,
    held_tool: Option<(egui::TextureId, [usize; 2])>,
    now_s: f32,
    using_tool: bool,
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

    let rect = ctx.screen_rect();
    let speed_pos = egui::pos2(rect.max.x - 12.0, 12.0);
    painter.text(
        speed_pos,
        egui::Align2::RIGHT_TOP,
        format!("{sim_speed:.1}x"),
        egui::FontId::proportional(14.0),
        egui::Color32::WHITE,
    );

    let outline_color = if brush.mode == BrushMode::Place {
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
    let base_w = (size[0] as f32).clamp(48.0, 220.0);
    let aspect = size[1] as f32 / size[0] as f32;
    let sprite_size = egui::vec2(base_w, base_w * aspect.max(0.3));
    let bob = (now_s * 4.5).sin() * 4.0;
    let use_kick = if using_tool {
        (now_s * 18.0).sin().abs() * 16.0
    } else {
        0.0
    };
    let min = egui::pos2(
        rect.max.x - sprite_size.x - 24.0,
        rect.max.y - sprite_size.y - 14.0 + bob + use_kick,
    );
    let max = min + sprite_size;
    let tint = if using_tool {
        egui::Color32::from_white_alpha(255)
    } else {
        egui::Color32::from_white_alpha(220)
    };
    painter.image(
        texture,
        egui::Rect::from_min_max(min, max),
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        tint,
    );
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
        if brush.mode == BrushMode::Place {
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
            "Scroll: material | Ctrl+Scroll: radius | Alt+Scroll: range | {}: toggle radial",
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
}

impl ToolTextures {
    pub fn for_tool(&self, tool: ToolKind) -> &ToolTexture {
        match tool {
            ToolKind::Brush => &self.brush,
            ToolKind::BuildersWand => &self.builders_wand,
            ToolKind::DestructorWand => &self.destructor_wand,
        }
    }
}

pub fn load_tool_textures(ctx: &egui::Context, dir: impl AsRef<Path>) -> ToolTextures {
    let dir = dir.as_ref().to_path_buf();
    let brush = load_single_tool_texture(
        ctx,
        &dir,
        "brush.ppm",
        "tool_brush",
        fallback_tool_texture([130, 180, 240, 255]),
    );
    let builders_wand = load_single_tool_texture(
        ctx,
        &dir,
        "builders_wand.ppm",
        "tool_builders_wand",
        fallback_tool_texture([140, 220, 120, 255]),
    );
    let destructor_wand = load_single_tool_texture(
        ctx,
        &dir,
        "destructor_wand.ppm",
        "tool_destructor_wand",
        fallback_tool_texture([240, 140, 140, 255]),
    );
    ToolTextures {
        brush,
        builders_wand,
        destructor_wand,
    }
}

fn load_single_tool_texture(
    ctx: &egui::Context,
    dir: &Path,
    file_name: &str,
    egui_name: &str,
    fallback: egui::ColorImage,
) -> ToolTexture {
    let path = dir.join(file_name);
    let image = std::fs::read(&path)
        .ok()
        .and_then(|bytes| parse_ppm_image(&bytes))
        .unwrap_or(fallback);
    let size = image.size;
    let texture = ctx.load_texture(egui_name.to_string(), image, egui::TextureOptions::LINEAR);
    ToolTexture { texture, size }
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
        rgba[px * 4] = data[px * 3];
        rgba[px * 4 + 1] = data[px * 3 + 1];
        rgba[px * 4 + 2] = data[px * 3 + 2];
        rgba[px * 4 + 3] = 255;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

fn fallback_tool_texture(color: [u8; 4]) -> egui::ColorImage {
    let size = [64, 64];
    let mut pixels = vec![egui::Color32::TRANSPARENT; size[0] * size[1]];
    for y in 0..size[1] {
        for x in 0..size[0] {
            let idx = y * size[0] + x;
            let border = x < 4 || y < 4 || x >= size[0] - 4 || y >= size[1] - 4;
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            pixels[idx] = if border {
                egui::Color32::from_rgba_premultiplied(35, 35, 35, 255)
            } else if checker {
                egui::Color32::from_rgba_premultiplied(color[0], color[1], color[2], color[3])
            } else {
                egui::Color32::from_rgba_premultiplied(
                    color[0] / 2,
                    color[1] / 2,
                    color[2] / 2,
                    255,
                )
            };
        }
    }
    egui::ColorImage { size, pixels }
}
