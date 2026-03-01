use std::collections::{HashMap, HashSet};

use crate::types::{voxel_to_chunk, ChunkCoord, MaterialId, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::Chunk as LegacyChunk;
use crate::world::EMPTY;

const CHUNK_VOLUME: usize =
    CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize;
const CHUNK_SIDE: usize = CHUNK_SIZE_VOXELS as usize;
const CHUNK_BORDER_AREA: usize = CHUNK_SIDE * CHUNK_SIDE;

#[derive(Clone)]
pub struct ChunkMeshingInput<'a> {
    pub voxels: &'a [MaterialId],
    pub neg_x: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_x: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_y: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_y: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_z: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_z: [MaterialId; CHUNK_BORDER_AREA],
}

impl ChunkMeshingInput<'_> {
    #[inline]
    pub fn border_index(u: usize, v: usize) -> usize {
        u * CHUNK_SIDE + v
    }
}

fn chunk_neighbors_6(coord: ChunkCoord) -> [ChunkCoord; 6] {
    [
        ChunkCoord {
            x: coord.x - 1,
            y: coord.y,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x + 1,
            y: coord.y,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y - 1,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y + 1,
            z: coord.z,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z - 1,
        },
        ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z + 1,
        },
    ]
}

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

    pub fn iter_raw(&self) -> &[MaterialId] {
        &self.voxels
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

    pub fn is_chunk_loaded(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains_key(&coord)
    }

    pub fn is_voxel_chunk_loaded(&self, coord: VoxelCoord) -> bool {
        let (chunk_coord, _) = voxel_to_chunk(coord);
        self.is_chunk_loaded(chunk_coord)
    }

    pub fn set_voxel(&mut self, coord: VoxelCoord, material: MaterialId) {
        let (chunk_coord, local) = voxel_to_chunk(coord);
        let (x, y, z) = (local[0] as usize, local[1] as usize, local[2] as usize);

        // Critical for streaming worlds: don't allocate empty chunks when erasing.
        if material == EMPTY && !self.chunk_exists(chunk_coord) {
            return;
        }

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
        self.is_chunk_loaded(coord)
    }

    pub fn get_chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn build_meshing_input(&self, coord: ChunkCoord) -> Option<ChunkMeshingInput<'_>> {
        let current = self.get_chunk(coord)?;

        let mut neg_x = [EMPTY; CHUNK_BORDER_AREA];
        let mut pos_x = [EMPTY; CHUNK_BORDER_AREA];
        let mut neg_y = [EMPTY; CHUNK_BORDER_AREA];
        let mut pos_y = [EMPTY; CHUNK_BORDER_AREA];
        let mut neg_z = [EMPTY; CHUNK_BORDER_AREA];
        let mut pos_z = [EMPTY; CHUNK_BORDER_AREA];

        let last = CHUNK_SIDE - 1;
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x - 1,
            y: coord.y,
            z: coord.z,
        }) {
            for z in 0..CHUNK_SIDE {
                for y in 0..CHUNK_SIDE {
                    neg_x[ChunkMeshingInput::border_index(y, z)] = chunk.get(last, y, z);
                }
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x + 1,
            y: coord.y,
            z: coord.z,
        }) {
            for z in 0..CHUNK_SIDE {
                for y in 0..CHUNK_SIDE {
                    pos_x[ChunkMeshingInput::border_index(y, z)] = chunk.get(0, y, z);
                }
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y - 1,
            z: coord.z,
        }) {
            for z in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    neg_y[ChunkMeshingInput::border_index(x, z)] = chunk.get(x, last, z);
                }
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y + 1,
            z: coord.z,
        }) {
            for z in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    pos_y[ChunkMeshingInput::border_index(x, z)] = chunk.get(x, 0, z);
                }
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z - 1,
        }) {
            for y in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    neg_z[ChunkMeshingInput::border_index(x, y)] = chunk.get(x, y, last);
                }
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z + 1,
        }) {
            for y in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    pos_z[ChunkMeshingInput::border_index(x, y)] = chunk.get(x, y, 0);
                }
            }
        }

        Some(ChunkMeshingInput {
            voxels: current.iter_raw(),
            neg_x,
            pos_x,
            neg_y,
            pos_y,
            neg_z,
            pos_z,
        })
    }

    pub fn insert_chunk(&mut self, coord: ChunkCoord, chunk: LegacyChunk) {
        let mut dst = Chunk::new_empty();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    dst.set(x, y, z, chunk.get(x, y, z));
                }
            }
        }
        self.chunks.insert(coord, dst);
        self.dirty_chunks.insert(coord);
        self.mark_existing_neighbors_dirty(coord);
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.mark_existing_neighbors_dirty(coord);
            self.chunks.remove(&coord);
            self.dirty_chunks.remove(&coord);
            self.mark_existing_neighbors_dirty(coord);
        }
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();
    }

    pub fn mark_dirty(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.dirty_chunks.insert(coord);
        }
    }

    pub fn is_dirty(&self, coord: ChunkCoord) -> bool {
        self.dirty_chunks.contains(&coord)
    }

    pub fn take_dirty_chunks(&mut self) -> Vec<ChunkCoord> {
        self.dirty_chunks.drain().collect()
    }

    pub fn iter_loaded_chunks(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.chunks.keys()
    }

    pub fn dirty_count(&self) -> usize {
        self.dirty_chunks.len()
    }

    fn mark_neighbor_dirty(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.dirty_chunks.insert(coord);
        }
    }

    fn mark_existing_neighbors_dirty(&mut self, coord: ChunkCoord) {
        for neighbor_coord in chunk_neighbors_6(coord) {
            self.mark_neighbor_dirty(neighbor_coord);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_chunk_with_fill(fill: MaterialId) -> LegacyChunk {
        let mut chunk = LegacyChunk::new();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    chunk.set(x, y, z, fill);
                }
            }
        }
        chunk
    }

    #[test]
    fn insert_chunk_marks_existing_neighbors_dirty() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };

        store.insert_chunk(east, legacy_chunk_with_fill(1));
        store.take_dirty_chunks();

        store.insert_chunk(center, legacy_chunk_with_fill(2));

        assert!(store.is_dirty(center));
        assert!(store.is_dirty(east));
    }

    #[test]
    fn remove_chunk_marks_existing_neighbors_dirty() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };

        store.insert_chunk(center, legacy_chunk_with_fill(1));
        store.insert_chunk(east, legacy_chunk_with_fill(2));
        store.take_dirty_chunks();

        store.remove_chunk(center);

        assert!(!store.chunk_exists(center));
        assert!(store.is_dirty(east));
    }

    #[test]
    fn exposes_chunk_residency_for_world_voxels() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };

        assert!(!store.is_chunk_loaded(center));
        assert!(!store.is_voxel_chunk_loaded(VoxelCoord { x: 0, y: 0, z: 0 }));

        store.insert_chunk(center, legacy_chunk_with_fill(1));

        assert!(store.is_chunk_loaded(center));
        assert!(store.is_voxel_chunk_loaded(VoxelCoord { x: 15, y: 1, z: 15 }));
        assert!(!store.is_voxel_chunk_loaded(VoxelCoord { x: 16, y: 1, z: 15 }));
    }
}
