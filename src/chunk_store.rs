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

#[derive(Clone, Copy)]
struct NeighborFace {
    coord: ChunkCoord,
    axis: usize,
    edge: usize,
}

fn chunk_neighbor_faces(coord: ChunkCoord) -> [NeighborFace; 6] {
    let last = CHUNK_SIZE_VOXELS as usize - 1;
    [
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x - 1,
                y: coord.y,
                z: coord.z,
            },
            axis: 0,
            edge: 0,
        },
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x + 1,
                y: coord.y,
                z: coord.z,
            },
            axis: 0,
            edge: last,
        },
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x,
                y: coord.y - 1,
                z: coord.z,
            },
            axis: 1,
            edge: 0,
        },
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x,
                y: coord.y + 1,
                z: coord.z,
            },
            axis: 1,
            edge: last,
        },
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x,
                y: coord.y,
                z: coord.z - 1,
            },
            axis: 2,
            edge: 0,
        },
        NeighborFace {
            coord: ChunkCoord {
                x: coord.x,
                y: coord.y,
                z: coord.z + 1,
            },
            axis: 2,
            edge: last,
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

    fn face_non_empty_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (idx, face) in chunk_neighbor_faces(ChunkCoord { x: 0, y: 0, z: 0 })
            .into_iter()
            .enumerate()
        {
            if self.face_has_non_empty(face.axis, face.edge) {
                mask |= 1 << idx;
            }
        }
        mask
    }

    fn face_has_non_empty(&self, axis: usize, edge: usize) -> bool {
        let last = CHUNK_SIZE_VOXELS as usize;
        match axis {
            0 => {
                for z in 0..last {
                    for y in 0..last {
                        if self.get(edge, y, z) != EMPTY {
                            return true;
                        }
                    }
                }
            }
            1 => {
                for z in 0..last {
                    for x in 0..last {
                        if self.get(x, edge, z) != EMPTY {
                            return true;
                        }
                    }
                }
            }
            _ => {
                for y in 0..last {
                    for x in 0..last {
                        if self.get(x, y, edge) != EMPTY {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

impl From<LegacyChunk> for Chunk {
    fn from(value: LegacyChunk) -> Self {
        Self {
            voxels: value.iter_raw().to_vec().into_boxed_slice(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborDirtyPolicy {
    MarkExisting,
    GeneratedConditional,
    None,
}

pub struct ChunkStore {
    chunks: HashMap<ChunkCoord, Chunk>,
    dirty_chunks: HashSet<ChunkCoord>,
    unmeshed_chunks: HashSet<ChunkCoord>,
    deferred_dirty_on_load: HashSet<ChunkCoord>,
    deferred_neighbor_dirty_on_mesh: HashSet<ChunkCoord>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            dirty_chunks: HashSet::new(),
            unmeshed_chunks: HashSet::new(),
            deferred_dirty_on_load: HashSet::new(),
            deferred_neighbor_dirty_on_mesh: HashSet::new(),
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
        self.insert_chunk_with_policy(coord, chunk.into(), true, NeighborDirtyPolicy::MarkExisting);
    }

    pub fn insert_chunk_with_policy(
        &mut self,
        coord: ChunkCoord,
        chunk: Chunk,
        mark_self_dirty: bool,
        neighbor_dirty: NeighborDirtyPolicy,
    ) {
        let new_face_mask = chunk.face_non_empty_mask();
        let old_face_mask = self.chunks.get(&coord).map(Chunk::face_non_empty_mask);
        self.chunks.insert(coord, chunk);
        self.unmeshed_chunks.insert(coord);

        if mark_self_dirty || self.deferred_dirty_on_load.remove(&coord) {
            self.dirty_chunks.insert(coord);
        }

        let should_defer_neighbors =
            matches!(neighbor_dirty, NeighborDirtyPolicy::GeneratedConditional);
        if should_defer_neighbors {
            self.deferred_neighbor_dirty_on_mesh.insert(coord);
            return;
        }

        self.apply_neighbor_dirty_policy(coord, neighbor_dirty, new_face_mask, old_face_mask);
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        if self.chunk_exists(coord) {
            self.mark_existing_neighbors_dirty(coord);
            self.chunks.remove(&coord);
            self.dirty_chunks.remove(&coord);
            self.unmeshed_chunks.remove(&coord);
            self.deferred_neighbor_dirty_on_mesh.remove(&coord);
            self.mark_existing_neighbors_dirty(coord);
        }
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();
        self.unmeshed_chunks.clear();
        self.deferred_dirty_on_load.clear();
        self.deferred_neighbor_dirty_on_mesh.clear();
    }

    pub fn mark_chunk_meshed(&mut self, coord: ChunkCoord) {
        if !self.unmeshed_chunks.remove(&coord) {
            return;
        }
        if !self.deferred_neighbor_dirty_on_mesh.remove(&coord) {
            return;
        }

        if let Some(chunk) = self.chunks.get(&coord) {
            self.apply_neighbor_dirty_policy(
                coord,
                NeighborDirtyPolicy::GeneratedConditional,
                chunk.face_non_empty_mask(),
                None,
            );
        }
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
        } else {
            self.deferred_dirty_on_load.insert(coord);
        }
    }

    fn mark_existing_neighbors_dirty(&mut self, coord: ChunkCoord) {
        for neighbor_coord in chunk_neighbors_6(coord) {
            self.mark_neighbor_dirty(neighbor_coord);
        }
    }

    fn apply_neighbor_dirty_policy(
        &mut self,
        coord: ChunkCoord,
        policy: NeighborDirtyPolicy,
        new_face_mask: u8,
        old_face_mask: Option<u8>,
    ) {
        match policy {
            NeighborDirtyPolicy::MarkExisting => self.mark_existing_neighbors_dirty(coord),
            NeighborDirtyPolicy::GeneratedConditional => {
                let old_mask = old_face_mask.unwrap_or(0);
                let topology_changed = old_face_mask.is_some() && old_mask != new_face_mask;
                for (idx, face) in chunk_neighbor_faces(coord).into_iter().enumerate() {
                    let face_bit = 1u8 << idx;
                    let boundary_non_empty = (new_face_mask & face_bit) != 0;
                    let face_changed = (old_mask & face_bit) != (new_face_mask & face_bit);
                    if boundary_non_empty || (topology_changed && face_changed) {
                        self.mark_neighbor_dirty(face.coord);
                    }
                }
            }
            NeighborDirtyPolicy::None => {}
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

    fn store_chunk_with_voxel(x: usize, y: usize, z: usize, id: MaterialId) -> Chunk {
        let mut chunk = Chunk::new_empty();
        chunk.set(x, y, z, id);
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

    #[test]
    fn generated_insert_defers_neighbor_remesh_until_meshed() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };
        let last = CHUNK_SIZE_VOXELS as usize - 1;

        store.insert_chunk(east, legacy_chunk_with_fill(1));
        store.take_dirty_chunks();

        store.insert_chunk_with_policy(
            center,
            store_chunk_with_voxel(last, 0, 0, 2),
            true,
            NeighborDirtyPolicy::GeneratedConditional,
        );

        assert!(store.is_dirty(center));
        assert!(!store.is_dirty(east));

        store.mark_chunk_meshed(center);
        assert!(store.is_dirty(east));
    }

    #[test]
    fn deferred_neighbor_dirty_applies_when_chunk_loads() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let west = ChunkCoord { x: -1, y: 0, z: 0 };

        store.insert_chunk_with_policy(
            center,
            store_chunk_with_voxel(0, 0, 0, 2),
            true,
            NeighborDirtyPolicy::GeneratedConditional,
        );
        store.mark_chunk_meshed(center);
        store.take_dirty_chunks();

        store.insert_chunk_with_policy(west, Chunk::new_empty(), false, NeighborDirtyPolicy::None);
        assert!(store.is_dirty(west));
    }
}
