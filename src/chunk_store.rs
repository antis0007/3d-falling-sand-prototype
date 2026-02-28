use std::collections::{HashMap, HashSet};

use crate::types::{voxel_to_chunk, ChunkCoord, MaterialId, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{Chunk, EMPTY};

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

        if material == EMPTY && !self.chunks.contains_key(&chunk_coord) {
            return;
        }

        let chunk = self.chunks.entry(chunk_coord).or_insert_with(Chunk::new);
        if chunk.get(local[0] as usize, local[1] as usize, local[2] as usize) == material {
            return;
        }

        chunk.set(
            local[0] as usize,
            local[1] as usize,
            local[2] as usize,
            material,
        );
        self.mark_dirty(chunk_coord);

        if local[0] == 0 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x - 1,
                y: chunk_coord.y,
                z: chunk_coord.z,
            });
        }
        if local[0] as i32 == CHUNK_SIZE_VOXELS - 1 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x + 1,
                y: chunk_coord.y,
                z: chunk_coord.z,
            });
        }
        if local[1] == 0 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y - 1,
                z: chunk_coord.z,
            });
        }
        if local[1] as i32 == CHUNK_SIZE_VOXELS - 1 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y + 1,
                z: chunk_coord.z,
            });
        }
        if local[2] == 0 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y,
                z: chunk_coord.z - 1,
            });
        }
        if local[2] as i32 == CHUNK_SIZE_VOXELS - 1 {
            self.mark_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y,
                z: chunk_coord.z + 1,
            });
        }
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        self.dirty_chunks.insert(coord);
    }

    pub fn is_dirty(&self, coord: ChunkCoord) -> bool {
        self.dirty_chunks.contains(&coord)
    }
}

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}
