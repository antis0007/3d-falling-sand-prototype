use crate::types::{voxel_to_chunk, ChunkCoord, MaterialId, VoxelCoord};
use crate::world::Chunk;
use std::collections::{HashMap, HashSet};

pub struct ChunkStore {
    chunks: HashMap<ChunkCoord, Chunk>,
    dirty: HashSet<ChunkCoord>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            dirty: HashSet::new(),
        }
    }

    pub fn get_voxel(&self, coord: VoxelCoord) -> Option<MaterialId> {
        let (chunk_coord, local) = voxel_to_chunk(coord);
        self.chunks
            .get(&chunk_coord)
            .map(|chunk| chunk.get(local[0] as usize, local[1] as usize, local[2] as usize))
    }

    pub fn set_voxel(&mut self, coord: VoxelCoord, material: MaterialId) {
        let (chunk_coord, local) = voxel_to_chunk(coord);
        let chunk = self.chunks.entry(chunk_coord).or_insert_with(Chunk::new);
        chunk.set(
            local[0] as usize,
            local[1] as usize,
            local[2] as usize,
            material,
        );
        self.mark_dirty(chunk_coord);
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        self.dirty.insert(coord);
    }

    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkCoord> {
        self.dirty.drain().collect()
    }
}

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}
