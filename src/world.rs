pub type MaterialId = u16;
pub const EMPTY: MaterialId = 0;
pub const CHUNK_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushShape {
    Sphere,
    Cube,
    Torus,
    Hemisphere,
    Bowl,
    InvertedBowl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushMode {
    Place,
    Erase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaFootprintShape {
    Circle,
    Square,
}

#[derive(Clone, Copy, Debug)]
pub struct AreaToolSettings {
    pub radius: i32,
    pub shape: AreaFootprintShape,
    pub thickness: i32,
}

impl Default for AreaToolSettings {
    fn default() -> Self {
        Self {
            radius: 1,
            shape: AreaFootprintShape::Circle,
            thickness: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialJetSettings {
    pub launch_velocity: f32,
    pub flow_rate: usize,
    pub spread: f32,
    pub max_range: f32,
}

impl Default for MaterialJetSettings {
    fn default() -> Self {
        Self {
            launch_velocity: 28.0,
            flow_rate: 10,
            spread: 0.15,
            max_range: 18.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BrushSettings {
    pub radius: i32,
    pub shape: BrushShape,
    pub mode: BrushMode,
    pub max_distance: f32,
    pub fixed_distance: bool,
    pub repeat_interval_s: f32,
    pub minecraft_style_placement: bool,
    pub area_tool: AreaToolSettings,
    pub material_jet: MaterialJetSettings,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 1,
            shape: BrushShape::Sphere,
            mode: BrushMode::Place,
            max_distance: 16.0,
            fixed_distance: false,
            repeat_interval_s: 0.02,
            minecraft_style_placement: false,
            area_tool: AreaToolSettings::default(),
            material_jet: MaterialJetSettings::default(),
        }
    }
}
