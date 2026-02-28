pub use crate::world::MaterialId;

pub const CHUNK_SIZE_VOXELS: i32 = crate::world::CHUNK_SIZE as i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[inline]
fn floor_div(value: i32, divisor: i32) -> i32 {
    value.div_euclid(divisor)
}

#[inline]
fn floor_mod(value: i32, divisor: i32) -> i32 {
    value.rem_euclid(divisor)
}

pub fn voxel_to_chunk(voxel: VoxelCoord) -> (ChunkCoord, [u32; 3]) {
    let chunk = ChunkCoord {
        x: floor_div(voxel.x, CHUNK_SIZE_VOXELS),
        y: floor_div(voxel.y, CHUNK_SIZE_VOXELS),
        z: floor_div(voxel.z, CHUNK_SIZE_VOXELS),
    };

    let local = [
        floor_mod(voxel.x, CHUNK_SIZE_VOXELS) as u32,
        floor_mod(voxel.y, CHUNK_SIZE_VOXELS) as u32,
        floor_mod(voxel.z, CHUNK_SIZE_VOXELS) as u32,
    ];

    (chunk, local)
}

pub fn chunk_to_world_min(chunk: ChunkCoord) -> VoxelCoord {
    VoxelCoord {
        x: chunk.x * CHUNK_SIZE_VOXELS,
        y: chunk.y * CHUNK_SIZE_VOXELS,
        z: chunk.z * CHUNK_SIZE_VOXELS,
    }
}
