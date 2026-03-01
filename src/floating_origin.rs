use crate::types::VoxelCoord;
use glam::Vec3;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginConfig {
    pub recenter_threshold_voxels: i32,
    pub hysteresis_voxels: i32,
    pub allow_vertical_recentering: bool,
    pub recenter_cooldown: Option<Duration>,
}

impl Default for FloatingOriginConfig {
    fn default() -> Self {
        Self {
            recenter_threshold_voxels: 128,
            hysteresis_voxels: 16,
            allow_vertical_recentering: true,
            recenter_cooldown: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OriginShiftResult {
    pub shift_voxels: VoxelCoord,
    pub player_local_position: Vec3,
    pub origin_translation: VoxelCoord,
    pub recentered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginState {
    pub origin_translation: VoxelCoord,
    next_recenter_allowed_at: Option<Instant>,
}

impl FloatingOriginState {
    pub fn new() -> Self {
        Self {
            origin_translation: VoxelCoord { x: 0, y: 0, z: 0 },
            next_recenter_allowed_at: None,
        }
    }

    pub fn reset(&mut self) {
        self.origin_translation = VoxelCoord { x: 0, y: 0, z: 0 };
        self.next_recenter_allowed_at = None;
    }

    fn recenter_shift_for(local_position: Vec3, config: FloatingOriginConfig) -> VoxelCoord {
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

    pub fn update(
        &mut self,
        player_local_position: Vec3,
        now: Instant,
        config: FloatingOriginConfig,
    ) -> OriginShiftResult {
        if self
            .next_recenter_allowed_at
            .is_some_and(|allowed_at| now < allowed_at)
        {
            return OriginShiftResult {
                shift_voxels: VoxelCoord { x: 0, y: 0, z: 0 },
                player_local_position,
                origin_translation: self.origin_translation,
                recentered: false,
            };
        }

        let shift_voxels = Self::recenter_shift_for(player_local_position, config);
        let recentered = shift_voxels != (VoxelCoord { x: 0, y: 0, z: 0 });
        let mut updated_local_position = player_local_position;

        if recentered {
            self.origin_translation.x += shift_voxels.x;
            self.origin_translation.y += shift_voxels.y;
            self.origin_translation.z += shift_voxels.z;
            updated_local_position -= Vec3::new(
                shift_voxels.x as f32,
                shift_voxels.y as f32,
                shift_voxels.z as f32,
            );
            self.next_recenter_allowed_at = config.recenter_cooldown.map(|cooldown| now + cooldown);
        }

        OriginShiftResult {
            shift_voxels,
            player_local_position: updated_local_position,
            origin_translation: self.origin_translation,
            recentered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_recenters_and_preserves_world_position() {
        let mut state = FloatingOriginState::new();
        let config = FloatingOriginConfig::default();
        let before_local = Vec3::new(145.2, 0.5, -144.1);
        let before_world = before_local
            + Vec3::new(
                state.origin_translation.x as f32,
                state.origin_translation.y as f32,
                state.origin_translation.z as f32,
            );

        let result = state.update(before_local, Instant::now(), config);
        let after_world = result.player_local_position
            + Vec3::new(
                result.origin_translation.x as f32,
                result.origin_translation.y as f32,
                result.origin_translation.z as f32,
            );

        assert!(result.recentered);
        assert_eq!(before_world, after_world);
    }

    #[test]
    fn cooldown_prevents_immediate_second_recentering() {
        let mut state = FloatingOriginState::new();
        let config = FloatingOriginConfig {
            recenter_cooldown: Some(Duration::from_millis(500)),
            ..FloatingOriginConfig::default()
        };
        let now = Instant::now();
        let first = state.update(Vec3::new(160.0, 0.0, 0.0), now, config);
        let second = state.update(Vec3::new(160.0, 0.0, 0.0), now, config);

        assert!(first.recentered);
        assert!(!second.recentered);
    }
}
