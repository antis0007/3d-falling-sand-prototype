use std::collections::{HashMap, HashSet};

use crate::types::voxel_to_chunk;
use crate::types::{ChunkCoord, MaterialId, VoxelCoord};
use crate::world::Chunk;

pub struct ChunkStore {
    chunks: HashMap<ChunkCoord, Chunk>,
    dirty_chunks: HashSet<ChunkCoord>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            dirty_chunks: HashSet::new(),
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

    pub fn insert_chunk(&mut self, coord: ChunkCoord, chunk: Chunk) {
        self.chunks.insert(coord, chunk);
        self.mark_dirty(coord);
    }

    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        self.dirty_chunks.insert(coord);
    }
}

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}
