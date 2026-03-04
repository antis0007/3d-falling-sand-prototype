pub use crate::world::MaterialId;

pub const CHUNK_SIZE_VOXELS: i32 = crate::world::CHUNK_SIZE as i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldVoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

pub type VoxelCoord = WorldVoxelCoord;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalVoxelCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl LocalVoxelCoord {
    #[inline]
    pub fn to_array(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkOriginWorld {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct GpuPageIndex(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MeshHandle(pub u32);

#[inline]
fn floor_div(value: i32, divisor: i32) -> i32 {
    value.div_euclid(divisor)
}

#[inline]
fn floor_mod(value: i32, divisor: i32) -> i32 {
    value.rem_euclid(divisor)
}

pub fn voxel_to_chunk_local(voxel: WorldVoxelCoord) -> (ChunkCoord, LocalVoxelCoord) {
    let chunk = ChunkCoord {
        x: floor_div(voxel.x, CHUNK_SIZE_VOXELS),
        y: floor_div(voxel.y, CHUNK_SIZE_VOXELS),
        z: floor_div(voxel.z, CHUNK_SIZE_VOXELS),
    };

    let local = LocalVoxelCoord {
        x: floor_mod(voxel.x, CHUNK_SIZE_VOXELS) as u32,
        y: floor_mod(voxel.y, CHUNK_SIZE_VOXELS) as u32,
        z: floor_mod(voxel.z, CHUNK_SIZE_VOXELS) as u32,
    };

    (chunk, local)
}

pub fn voxel_to_chunk(voxel: WorldVoxelCoord) -> (ChunkCoord, [u32; 3]) {
    let (chunk, local) = voxel_to_chunk_local(voxel);
    (chunk, local.to_array())
}

pub fn chunk_to_world_min(chunk: ChunkCoord) -> WorldVoxelCoord {
    WorldVoxelCoord {
        x: chunk.x * CHUNK_SIZE_VOXELS,
        y: chunk.y * CHUNK_SIZE_VOXELS,
        z: chunk.z * CHUNK_SIZE_VOXELS,
    }
}
