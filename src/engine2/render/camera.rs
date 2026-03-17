//! Camera state used by extraction.

#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub view_distance: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            world_x: 0.0,
            world_y: 0.0,
            world_z: 0.0,
            view_distance: 256.0,
        }
    }
}

impl CameraState {
    pub fn from_world_position(position: [f32; 3], view_distance: f32) -> Self {
        Self {
            world_x: position[0],
            world_y: position[1],
            world_z: position[2],
            view_distance,
        }
    }
}
