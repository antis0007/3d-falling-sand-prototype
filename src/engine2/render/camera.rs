//! Camera state used by extraction.

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraState {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
}
