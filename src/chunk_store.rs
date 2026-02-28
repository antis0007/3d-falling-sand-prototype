use crate::types::{ChunkCoord, MaterialId, VoxelCoord};
use crate::world::Chunk;

pub struct ChunkStore;

impl ChunkStore {
    pub fn new() -> Self {
        Self
    }

    pub fn get_voxel(&self, _coord: VoxelCoord) -> Option<MaterialId> {
        todo!("ChunkStore will become the authoritative sparse chunk map")
    }

    pub fn set_voxel(&mut self, _coord: VoxelCoord, _material: MaterialId) {
        todo!("ChunkStore write path is not implemented yet")
    }

    pub fn get_chunk(&self, _coord: ChunkCoord) -> Option<&Chunk> {
        todo!("Chunk accessors are not implemented yet")
    }

    pub fn mark_dirty(&mut self, _coord: ChunkCoord) {
        todo!("Dirty tracking is not implemented yet")
    }
}

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}
