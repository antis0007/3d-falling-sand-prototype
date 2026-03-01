use crate::types::VoxelCoord;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginConfig {
    pub recenter_threshold_voxels: i32,
    pub recenter_hysteresis_voxels: i32,
    pub recenter_cooldown_frames: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OriginShiftResult {
    pub voxel_shift_delta: VoxelCoord,
    pub updated_local_player_offset: Vec3,
    pub recentered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginState {
    pub origin_translation: VoxelCoord,
    recenter_armed: [bool; 3],
    cooldown_frames_remaining: u32,
}

impl FloatingOriginState {
    pub fn new() -> Self {
        Self {
            origin_translation: VoxelCoord { x: 0, y: 0, z: 0 },
            recenter_armed: [true, true, true],
            cooldown_frames_remaining: 0,
        }
    }

    pub fn update(
        &mut self,
        player_local_pos: Vec3,
        config: FloatingOriginConfig,
    ) -> OriginShiftResult {
        if self.cooldown_frames_remaining > 0 {
            self.cooldown_frames_remaining -= 1;
        }

        let threshold = config.recenter_threshold_voxels.max(0) as f32;
        let rearm_threshold =
            (config.recenter_threshold_voxels - config.recenter_hysteresis_voxels).max(0) as f32;
        let axis_positions = [player_local_pos.x, player_local_pos.y, player_local_pos.z];
        for axis in 0..3 {
            if !self.recenter_armed[axis] && axis_positions[axis].abs() <= rearm_threshold {
                self.recenter_armed[axis] = true;
            }
        }

        let mut shift = VoxelCoord { x: 0, y: 0, z: 0 };
        if self.cooldown_frames_remaining == 0 {
            if self.recenter_armed[0] && player_local_pos.x.abs() > threshold {
                shift.x = player_local_pos.x.floor() as i32;
                self.recenter_armed[0] = false;
            }
            if self.recenter_armed[1] && player_local_pos.y.abs() > threshold {
                shift.y = player_local_pos.y.floor() as i32;
                self.recenter_armed[1] = false;
            }
            if self.recenter_armed[2] && player_local_pos.z.abs() > threshold {
                shift.z = player_local_pos.z.floor() as i32;
                self.recenter_armed[2] = false;
            }
        }

        let recentered = shift != VoxelCoord { x: 0, y: 0, z: 0 };
        if recentered {
            self.origin_translation.x += shift.x;
            self.origin_translation.y += shift.y;
            self.origin_translation.z += shift.z;
            self.cooldown_frames_remaining = config.recenter_cooldown_frames.unwrap_or(0);
        }

        OriginShiftResult {
            voxel_shift_delta: shift,
            updated_local_player_offset: player_local_pos
                - Vec3::new(shift.x as f32, shift.y as f32, shift.z as f32),
            recentered,
        }
    }
}
