use crate::types::VoxelCoord;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginConfig {
    pub recenter_threshold_voxels: i32,
    pub hysteresis_voxels: i32,
    pub allow_vertical_recentering: bool,
}

impl Default for FloatingOriginConfig {
    fn default() -> Self {
        Self {
            recenter_threshold_voxels: 128,
            hysteresis_voxels: 16,
            allow_vertical_recentering: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginState {
    pub origin_translation: VoxelCoord,
}

impl FloatingOriginState {
    pub fn new() -> Self {
        Self {
            origin_translation: VoxelCoord { x: 0, y: 0, z: 0 },
        }
    }

    pub fn recenter_shift_for(
        &self,
        local_position: Vec3,
        config: FloatingOriginConfig,
    ) -> VoxelCoord {
        let threshold = config.recenter_threshold_voxels.max(1) as f32;
        let hysteresis = config.hysteresis_voxels.max(0) as f32;
        let trigger = threshold + hysteresis;
        let shift_x = if local_position.x.abs() >= trigger {
            local_position.x.floor() as i32
        } else {
            0
        };
        let shift_y = if config.allow_vertical_recentering && local_position.y.abs() >= trigger {
            local_position.y.floor() as i32
        } else {
            0
        };
        let shift_z = if local_position.z.abs() >= trigger {
            local_position.z.floor() as i32
        } else {
            0
        };
        VoxelCoord {
            x: shift_x,
            y: shift_y,
            z: shift_z,
        }
    }
}
