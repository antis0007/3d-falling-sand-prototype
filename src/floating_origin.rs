use crate::types::VoxelCoord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOriginConfig {
    pub recenter_threshold_voxels: i32,
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
}
