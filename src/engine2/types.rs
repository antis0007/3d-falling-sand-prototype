//! Core engine2 value types.

pub const BRICK_EDGE_VOXELS: u32 = 16;
pub const BRICK_VOLUME_VOXELS: u32 = BRICK_EDGE_VOXELS * BRICK_EDGE_VOXELS * BRICK_EDGE_VOXELS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPage(pub u32);
