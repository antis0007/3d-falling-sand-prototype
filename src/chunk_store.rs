use std::collections::{HashMap, HashSet};

use crate::types::{voxel_to_chunk, ChunkCoord, MaterialId, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::Chunk as LegacyChunk;
use crate::world::EMPTY;

const CHUNK_VOLUME: usize =
    CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize;
const CHUNK_SIDE: usize = CHUNK_SIZE_VOXELS as usize;
const CHUNK_BORDER_AREA: usize = CHUNK_SIDE * CHUNK_SIDE;

#[derive(Clone)]
pub struct ChunkBorderStrips {
    pub neg_x: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_x: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_y: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_y: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_z: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_z: [MaterialId; CHUNK_BORDER_AREA],
    pub known_neighbor_mask: u8,
}

impl Default for ChunkBorderStrips {
    fn default() -> Self {
        Self {
            neg_x: [EMPTY; CHUNK_BORDER_AREA],
            pos_x: [EMPTY; CHUNK_BORDER_AREA],
            neg_y: [EMPTY; CHUNK_BORDER_AREA],
            pos_y: [EMPTY; CHUNK_BORDER_AREA],
            neg_z: [EMPTY; CHUNK_BORDER_AREA],
            pos_z: [EMPTY; CHUNK_BORDER_AREA],
            known_neighbor_mask: 0,
        }
    }
}

#[derive(Clone)]
pub struct ChunkMeshingInput<'a> {
    pub voxels: &'a [MaterialId],
    pub neg_x: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_x: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_y: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_y: [MaterialId; CHUNK_BORDER_AREA],
    pub neg_z: [MaterialId; CHUNK_BORDER_AREA],
    pub pos_z: [MaterialId; CHUNK_BORDER_AREA],
    pub known_neighbor_mask: u8,
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
    face_non_empty_counts: [u16; 6],
    face_non_empty_mask: u8,
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            voxels: vec![EMPTY; CHUNK_VOLUME].into_boxed_slice(),
            face_non_empty_counts: [0; 6],
            face_non_empty_mask: 0,
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
        let prev = self.voxels[idx];
        if prev == id {
            return;
        }
        self.voxels[idx] = id;
        self.update_border_face_cache(x, y, z, prev, id);
    }

    pub fn iter_raw(&self) -> &[MaterialId] {
        &self.voxels
    }

    pub fn face_non_empty_mask(&self) -> u8 {
        self.face_non_empty_mask
    }

    fn update_border_face_cache(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        prev: MaterialId,
        next: MaterialId,
    ) {
        let was_non_empty = prev != EMPTY;
        let is_non_empty = next != EMPTY;
        if was_non_empty == is_non_empty {
            return;
        }

        let last = CHUNK_SIDE - 1;
        if x == 0 {
            self.bump_face_count(0, is_non_empty);
        }
        if x == last {
            self.bump_face_count(1, is_non_empty);
        }
        if y == 0 {
            self.bump_face_count(2, is_non_empty);
        }
        if y == last {
            self.bump_face_count(3, is_non_empty);
        }
        if z == 0 {
            self.bump_face_count(4, is_non_empty);
        }
        if z == last {
            self.bump_face_count(5, is_non_empty);
        }
    }

    fn bump_face_count(&mut self, face_idx: usize, became_non_empty: bool) {
        if became_non_empty {
            self.face_non_empty_counts[face_idx] =
                self.face_non_empty_counts[face_idx].saturating_add(1);
        } else {
            self.face_non_empty_counts[face_idx] =
                self.face_non_empty_counts[face_idx].saturating_sub(1);
        }
        if self.face_non_empty_counts[face_idx] == 0 {
            self.face_non_empty_mask &= !(1 << face_idx);
        } else {
            self.face_non_empty_mask |= 1 << face_idx;
        }
    }

    fn rebuild_face_cache(&mut self) {
        self.face_non_empty_counts = [0; 6];
        self.face_non_empty_mask = 0;
        let last = CHUNK_SIDE - 1;
        for z in 0..CHUNK_SIDE {
            for y in 0..CHUNK_SIDE {
                if self.get(0, y, z) != EMPTY {
                    self.bump_face_count(0, true);
                }
                if self.get(last, y, z) != EMPTY {
                    self.bump_face_count(1, true);
                }
            }
        }
        for z in 0..CHUNK_SIDE {
            for x in 0..CHUNK_SIDE {
                if self.get(x, 0, z) != EMPTY {
                    self.bump_face_count(2, true);
                }
                if self.get(x, last, z) != EMPTY {
                    self.bump_face_count(3, true);
                }
            }
        }
        for y in 0..CHUNK_SIDE {
            for x in 0..CHUNK_SIDE {
                if self.get(x, y, 0) != EMPTY {
                    self.bump_face_count(4, true);
                }
                if self.get(x, y, last) != EMPTY {
                    self.bump_face_count(5, true);
                }
            }
        }
    }

    pub fn fill_face_border(
        &self,
        axis: usize,
        edge: usize,
        out: &mut [MaterialId; CHUNK_BORDER_AREA],
    ) {
        let last = CHUNK_SIZE_VOXELS as usize;
        match axis {
            0 => {
                for z in 0..last {
                    for y in 0..last {
                        out[ChunkMeshingInput::border_index(y, z)] = self.get(edge, y, z);
                    }
                }
            }
            1 => {
                for z in 0..last {
                    for x in 0..last {
                        out[ChunkMeshingInput::border_index(x, z)] = self.get(x, edge, z);
                    }
                }
            }
            _ => {
                for y in 0..last {
                    for x in 0..last {
                        out[ChunkMeshingInput::border_index(x, y)] = self.get(x, y, edge);
                    }
                }
            }
        }
    }
}

impl From<LegacyChunk> for Chunk {
    fn from(value: LegacyChunk) -> Self {
        let mut chunk = Self {
            voxels: value.iter_raw().to_vec().into_boxed_slice(),
            face_non_empty_counts: [0; 6],
            face_non_empty_mask: 0,
        };
        chunk.rebuild_face_cache();
        chunk
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
    modified_chunks: HashSet<ChunkCoord>,
    unmeshed_chunks: HashSet<ChunkCoord>,
    deferred_dirty_on_load: HashSet<ChunkCoord>,
    deferred_neighbor_dirty_on_mesh: HashMap<ChunkCoord, Option<u8>>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            dirty_chunks: HashSet::new(),
            modified_chunks: HashSet::new(),
            unmeshed_chunks: HashSet::new(),
            deferred_dirty_on_load: HashSet::new(),
            deferred_neighbor_dirty_on_mesh: HashMap::new(),
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
        self.modified_chunks.insert(chunk_coord);

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
        let borders = self.chunk_border_strips(coord);

        Some(ChunkMeshingInput {
            voxels: current.iter_raw(),
            neg_x: borders.neg_x,
            pos_x: borders.pos_x,
            neg_y: borders.neg_y,
            pos_y: borders.pos_y,
            neg_z: borders.neg_z,
            pos_z: borders.pos_z,
            known_neighbor_mask: borders.known_neighbor_mask,
        })
    }

    pub fn chunk_face_non_empty_mask(&self, coord: ChunkCoord) -> u8 {
        self.get_chunk(coord)
            .map(Chunk::face_non_empty_mask)
            .unwrap_or(0)
    }

    pub fn chunk_border_strips(&self, coord: ChunkCoord) -> ChunkBorderStrips {
        let mut borders = ChunkBorderStrips::default();
        let last = CHUNK_SIDE - 1;

        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x - 1,
            y: coord.y,
            z: coord.z,
        }) {
            borders.known_neighbor_mask |= 1 << 0;
            if (chunk.face_non_empty_mask() & (1 << 1)) != 0 {
                chunk.fill_face_border(0, last, &mut borders.neg_x);
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x + 1,
            y: coord.y,
            z: coord.z,
        }) {
            borders.known_neighbor_mask |= 1 << 1;
            if (chunk.face_non_empty_mask() & (1 << 0)) != 0 {
                chunk.fill_face_border(0, 0, &mut borders.pos_x);
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y - 1,
            z: coord.z,
        }) {
            borders.known_neighbor_mask |= 1 << 2;
            if (chunk.face_non_empty_mask() & (1 << 3)) != 0 {
                chunk.fill_face_border(1, last, &mut borders.neg_y);
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y + 1,
            z: coord.z,
        }) {
            borders.known_neighbor_mask |= 1 << 3;
            if (chunk.face_non_empty_mask() & (1 << 2)) != 0 {
                chunk.fill_face_border(1, 0, &mut borders.pos_y);
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z - 1,
        }) {
            borders.known_neighbor_mask |= 1 << 4;
            if (chunk.face_non_empty_mask() & (1 << 5)) != 0 {
                chunk.fill_face_border(2, last, &mut borders.neg_z);
            }
        }
        if let Some(chunk) = self.get_chunk(ChunkCoord {
            x: coord.x,
            y: coord.y,
            z: coord.z + 1,
        }) {
            borders.known_neighbor_mask |= 1 << 5;
            if (chunk.face_non_empty_mask() & (1 << 4)) != 0 {
                chunk.fill_face_border(2, 0, &mut borders.pos_z);
            }
        }

        borders
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
        self.modified_chunks.remove(&coord);

        if mark_self_dirty || self.deferred_dirty_on_load.remove(&coord) {
            self.dirty_chunks.insert(coord);
        }

        let should_defer_neighbors =
            matches!(neighbor_dirty, NeighborDirtyPolicy::GeneratedConditional);
        if should_defer_neighbors {
            self.deferred_neighbor_dirty_on_mesh
                .insert(coord, old_face_mask);
            return;
        }

        self.apply_neighbor_dirty_policy(coord, neighbor_dirty, new_face_mask, old_face_mask);
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        let _ = self.remove_chunk_with_data(coord);
    }

    pub fn remove_chunk_with_data(&mut self, coord: ChunkCoord) -> Option<(Chunk, bool)> {
        if self.chunk_exists(coord) {
            self.mark_existing_neighbors_dirty(coord);
            let removed = self.chunks.remove(&coord);
            let was_modified = self.modified_chunks.remove(&coord);
            self.dirty_chunks.remove(&coord);
            self.unmeshed_chunks.remove(&coord);
            self.deferred_neighbor_dirty_on_mesh.remove(&coord);
            return removed.map(|chunk| (chunk, was_modified));
        }
        None
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();
        self.modified_chunks.clear();
        self.unmeshed_chunks.clear();
        self.deferred_dirty_on_load.clear();
        self.deferred_neighbor_dirty_on_mesh.clear();
    }

    pub fn mark_chunk_meshed(&mut self, coord: ChunkCoord) {
        if !self.unmeshed_chunks.remove(&coord) {
            return;
        }
        let Some(old_face_mask) = self.deferred_neighbor_dirty_on_mesh.remove(&coord) else {
            return;
        };

        if let Some(chunk) = self.chunks.get(&coord) {
            self.apply_neighbor_dirty_policy(
                coord,
                NeighborDirtyPolicy::GeneratedConditional,
                chunk.face_non_empty_mask(),
                old_face_mask,
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
                for (idx, face) in chunk_neighbor_faces(coord).into_iter().enumerate() {
                    let face_bit = 1u8 << idx;
                    let face_changed = (old_mask & face_bit) != (new_face_mask & face_bit);
                    let should_dirty = if old_face_mask.is_some() {
                        face_changed
                    } else {
                        (new_face_mask & face_bit) != 0
                    };
                    if should_dirty {
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
        let edge = CHUNK_SIZE_VOXELS - 1;
        let outside = CHUNK_SIZE_VOXELS;
        assert!(store.is_voxel_chunk_loaded(VoxelCoord {
            x: edge,
            y: 1,
            z: edge,
        }));
        assert!(!store.is_voxel_chunk_loaded(VoxelCoord {
            x: outside,
            y: 1,
            z: edge,
        }));
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

        let dirty_before_meshing = store.take_dirty_chunks();
        assert!(dirty_before_meshing.contains(&center));
        assert!(!dirty_before_meshing.contains(&east));

        store.mark_chunk_meshed(center);
        assert!(store.is_dirty(east));

        store.take_dirty_chunks();
        store.mark_chunk_meshed(center);
        assert!(!store.is_dirty(east));
    }

    #[test]
    fn chunk_border_strips_capture_neighbor_boundary_voxels() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };
        let west = ChunkCoord { x: -1, y: 0, z: 0 };

        store.insert_chunk_with_policy(
            center,
            Chunk::new_empty(),
            false,
            NeighborDirtyPolicy::None,
        );
        store.insert_chunk_with_policy(
            east,
            store_chunk_with_voxel(0, 3, 7, 7),
            false,
            NeighborDirtyPolicy::None,
        );
        let last = CHUNK_SIZE_VOXELS as usize - 1;
        store.insert_chunk_with_policy(
            west,
            store_chunk_with_voxel(last, 5, 9, 9),
            false,
            NeighborDirtyPolicy::None,
        );

        let borders = store.chunk_border_strips(center);
        assert_eq!(borders.pos_x[ChunkMeshingInput::border_index(3, 7)], 7);
        assert_eq!(borders.neg_x[ChunkMeshingInput::border_index(5, 9)], 9);
        assert_eq!(borders.pos_x[ChunkMeshingInput::border_index(2, 2)], EMPTY);
    }

    #[test]
    fn chunk_face_non_empty_mask_returns_zero_for_unloaded_chunk() {
        let store = ChunkStore::new();
        assert_eq!(
            store.chunk_face_non_empty_mask(ChunkCoord { x: 4, y: 5, z: 6 }),
            0
        );
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

    #[test]
    fn meshing_input_marks_missing_neighbors_as_unknown() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        store.insert_chunk_with_policy(
            center,
            Chunk::new_empty(),
            false,
            NeighborDirtyPolicy::None,
        );

        let input = store.build_meshing_input(center).expect("center input");
        assert_eq!(input.known_neighbor_mask, 0);

        let east = ChunkCoord { x: 1, y: 0, z: 0 };
        store.insert_chunk_with_policy(east, Chunk::new_empty(), false, NeighborDirtyPolicy::None);
        let input = store
            .build_meshing_input(center)
            .expect("center input with east");
        assert_ne!(input.known_neighbor_mask & (1 << 1), 0);
    }

    #[test]
    fn cached_face_mask_updates_after_border_edits() {
        let mut chunk = Chunk::new_empty();
        let last = CHUNK_SIDE - 1;
        assert_eq!(chunk.face_non_empty_mask(), 0);

        chunk.set(0, 0, 0, 4);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 0), 0);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 2), 0);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 4), 0);

        chunk.set(0, 0, 0, EMPTY);
        assert_eq!(chunk.face_non_empty_mask(), 0);

        chunk.set(last, last, last, 7);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 1), 0);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 3), 0);
        assert_ne!(chunk.face_non_empty_mask() & (1 << 5), 0);
    }

    #[test]
    fn generated_conditional_only_marks_neighbors_when_boundary_changes() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };
        let last = CHUNK_SIDE - 1;

        store.insert_chunk(east, legacy_chunk_with_fill(1));
        store.take_dirty_chunks();

        let mut first = Chunk::new_empty();
        first.set(last, 2, 2, 5);
        store.insert_chunk_with_policy(
            center,
            first,
            false,
            NeighborDirtyPolicy::GeneratedConditional,
        );
        store.mark_chunk_meshed(center);
        assert!(store.is_dirty(east));

        store.take_dirty_chunks();
        let mut second = Chunk::new_empty();
        second.set(last, 3, 3, 6);
        store.insert_chunk_with_policy(
            center,
            second,
            false,
            NeighborDirtyPolicy::GeneratedConditional,
        );
        store.mark_chunk_meshed(center);
        assert!(!store.is_dirty(east));
    }

    #[test]
    fn generated_conditional_uses_previous_mesh_mask_when_replaced_before_meshed() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };
        let last = CHUNK_SIDE - 1;

        store.insert_chunk(east, legacy_chunk_with_fill(1));
        store.take_dirty_chunks();

        let mut first = Chunk::new_empty();
        first.set(last, 1, 1, 4);
        store.insert_chunk_with_policy(
            center,
            first,
            false,
            NeighborDirtyPolicy::GeneratedConditional,
        );

        let mut second = Chunk::new_empty();
        second.set(last, 2, 2, 7);
        store.insert_chunk_with_policy(
            center,
            second,
            false,
            NeighborDirtyPolicy::GeneratedConditional,
        );
        store.mark_chunk_meshed(center);

        assert!(!store.is_dirty(east));
    }

    #[test]
    fn remove_chunk_only_marks_neighbors_once() {
        let mut store = ChunkStore::new();
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let east = ChunkCoord { x: 1, y: 0, z: 0 };

        store.insert_chunk(center, legacy_chunk_with_fill(1));
        store.insert_chunk(east, legacy_chunk_with_fill(2));
        store.take_dirty_chunks();

        store.remove_chunk(center);
        let dirty = store.take_dirty_chunks();
        assert_eq!(dirty.iter().filter(|&&c| c == east).count(), 1);
    }
}
