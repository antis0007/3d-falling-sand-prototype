use std::collections::{HashMap, HashSet};

use crate::types::{voxel_to_chunk, ChunkCoord, MaterialId, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

const CHUNK_VOLUME: usize =
    CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize;

#[derive(Clone)]
pub struct Chunk {
    voxels: Box<[MaterialId]>,
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            voxels: vec![EMPTY; CHUNK_VOLUME].into_boxed_slice(),
        }
    }

    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        let sz = CHUNK_SIZE_VOXELS as usize;
        (z * sz + y) * sz + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> MaterialId {
        self.voxels[Self::index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: MaterialId) {
        let idx = Self::index(x, y, z);
        self.voxels[idx] = id;
    }
}

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

    pub fn get_voxel(&self, coord: VoxelCoord) -> MaterialId {
        let (chunk_coord, local) = voxel_to_chunk(coord);
        self.chunks
            .get(&chunk_coord)
            .map(|chunk| chunk.get(local[0] as usize, local[1] as usize, local[2] as usize))
            .unwrap_or(EMPTY)
    }

    pub fn set_voxel(&mut self, coord: VoxelCoord, material: MaterialId) {
        let (chunk_coord, local) = voxel_to_chunk(coord);
        let (x, y, z) = (local[0] as usize, local[1] as usize, local[2] as usize);

        let chunk = self.ensure_chunk(chunk_coord);
        if chunk.get(x, y, z) == material {
            return;
        }
        chunk.set(x, y, z, material);
        self.dirty_chunks.insert(chunk_coord);

        let last = (CHUNK_SIZE_VOXELS - 1) as usize;
        if x == 0 {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x - 1,
                y: chunk_coord.y,
                z: chunk_coord.z,
            });
        }
        if x == last {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x + 1,
                y: chunk_coord.y,
                z: chunk_coord.z,
            });
        }
        if y == 0 {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y - 1,
                z: chunk_coord.z,
            });
        }
        if y == last {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y + 1,
                z: chunk_coord.z,
            });
        }
        if z == 0 {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y,
                z: chunk_coord.z - 1,
            });
        }
        if z == last {
            self.mark_neighbor_dirty(ChunkCoord {
                x: chunk_coord.x,
                y: chunk_coord.y,
                z: chunk_coord.z + 1,
            });
        }
    }

    pub fn ensure_chunk(&mut self, coord: ChunkCoord) -> &mut Chunk {
        self.chunks.entry(coord).or_insert_with(Chunk::new_empty)
    }

    pub fn chunk_exists(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains_key(&coord)
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.dirty_chunks.insert(coord);
        }
    }

    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkCoord> {
        self.dirty_chunks.drain().collect()
    }

    pub fn iter_loaded_chunks(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.chunks.keys()
    }

    fn mark_neighbor_dirty(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.dirty_chunks.insert(coord);
        }
    }
}

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}
