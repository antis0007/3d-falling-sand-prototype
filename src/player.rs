use glam::Vec3;

pub const PLAYER_HEIGHT_BLOCKS: f32 = 4.0;
pub const PLAYER_WIDTH_BLOCKS: f32 = 2.0;
pub const PLAYER_EYE_HEIGHT_BLOCKS: f32 = 3.2;
pub const GROUND_CONTACT_EPSILON_BLOCKS: f32 = 0.01;

pub fn grounded_eye_y_blocks() -> f32 {
    PLAYER_EYE_HEIGHT_BLOCKS
}

pub fn jump_eligibility_y_blocks() -> f32 {
    grounded_eye_y_blocks() + GROUND_CONTACT_EPSILON_BLOCKS
}

pub fn eye_height_world_meters(voxel_size: f32) -> f32 {
    PLAYER_EYE_HEIGHT_BLOCKS * voxel_size
}

pub fn camera_world_pos_from_blocks(player_position_blocks: Vec3, voxel_size: f32) -> Vec3 {
    player_position_blocks * voxel_size
}
