//! Core engine2 value types.

/// Marker for the fixed brick dimensions used by engine2 world residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickDim;

impl BrickDim {
    pub const EDGE: i32 = 16;
    pub const EDGE_U32: u32 = 16;
    pub const VOLUME_U32: u32 = Self::EDGE_U32 * Self::EDGE_U32 * Self::EDGE_U32;
}

pub const BRICK_EDGE_VOXELS: u32 = BrickDim::EDGE_U32;
pub const BRICK_VOLUME_VOXELS: u32 = BrickDim::VOLUME_U32;

/// Signed world voxel coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Signed world brick-space coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Sparse map key for a brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl From<BrickCoord> for BrickKey {
    fn from(value: BrickCoord) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<BrickKey> for BrickCoord {
    fn from(value: BrickKey) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl BrickKey {
    pub fn from_voxel(voxel: VoxelCoord) -> Self {
        voxel_to_brick_coord(voxel).into()
    }
}

/// Local voxel coordinate within a 16^3 brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalVoxelCoord {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

/// Future GPU brick page handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPage(pub u32);

pub type GpuPageHandle = GpuPage;

pub fn voxel_to_brick_coord(voxel: VoxelCoord) -> BrickCoord {
    BrickCoord {
        x: voxel.x.div_euclid(BrickDim::EDGE),
        y: voxel.y.div_euclid(BrickDim::EDGE),
        z: voxel.z.div_euclid(BrickDim::EDGE),
    }
}

pub fn voxel_to_brick_key(voxel: VoxelCoord) -> BrickKey {
    voxel_to_brick_coord(voxel).into()
}

pub fn voxel_to_local_coord(voxel: VoxelCoord) -> LocalVoxelCoord {
    LocalVoxelCoord {
        x: voxel.x.rem_euclid(BrickDim::EDGE) as u8,
        y: voxel.y.rem_euclid(BrickDim::EDGE) as u8,
        z: voxel.z.rem_euclid(BrickDim::EDGE) as u8,
    }
}

pub fn voxel_to_brick_and_local(voxel: VoxelCoord) -> (BrickKey, LocalVoxelCoord) {
    (voxel_to_brick_key(voxel), voxel_to_local_coord(voxel))
}

pub fn brick_and_local_to_voxel(brick: BrickCoord, local: LocalVoxelCoord) -> VoxelCoord {
    VoxelCoord {
        x: brick.x * BrickDim::EDGE + i32::from(local.x),
        y: brick.y * BrickDim::EDGE + i32::from(local.y),
        z: brick.z * BrickDim::EDGE + i32::from(local.z),
    }
}
