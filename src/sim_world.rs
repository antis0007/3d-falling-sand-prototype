use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim::{material, Phase, XorShift32};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

pub type Rng = XorShift32;

const CHUNK_COOLDOWN_TICKS: u8 = 8;
const MAX_CHUNKS_PER_STEP: usize = 96;

#[derive(Default)]
struct ChunkFrontierState {
    active_bits: Vec<u64>,
    frontier: Vec<u16>,
    cooldown_ticks: u8,
    recent_activity: u16,
}

impl ChunkFrontierState {
    fn new() -> Self {
        let chunk_volume = (CHUNK_SIZE_VOXELS as usize).pow(3);
        let words = chunk_volume.div_ceil(64);
        Self {
            active_bits: vec![0; words],
            frontier: Vec::with_capacity(512),
            cooldown_ticks: 0,
            recent_activity: 0,
        }
    }

    fn take_frontier(&mut self) -> Vec<u16> {
        std::mem::take(&mut self.frontier)
    }

    fn return_frontier_buffer(&mut self, mut buffer: Vec<u16>) {
        buffer.clear();
        self.frontier = buffer;
    }

    fn enqueue_local(&mut self, idx: u16) {
        let idx_usize = idx as usize;
        let word = idx_usize / 64;
        let bit = 1u64 << (idx_usize % 64);
        if (self.active_bits[word] & bit) != 0 {
            return;
        }
        self.active_bits[word] |= bit;
        self.frontier.push(idx);
    }

    fn clear_active(&mut self, idx: u16) {
        let idx_usize = idx as usize;
        let word = idx_usize / 64;
        let bit = 1u64 << (idx_usize % 64);
        self.active_bits[word] &= !bit;
    }

    fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimSoAChunkBuffer {
    pub material_ids: Vec<u16>,
    pub phases: Vec<u8>,
    pub settled: Vec<u8>,
}

#[derive(Default)]
pub struct SimWorld {
    chunks: HashMap<ChunkCoord, ChunkFrontierState>,
    known_seeded_chunks: HashSet<ChunkCoord>,
}

impl SimWorld {
    pub fn notify_voxel_edit(&mut self, coord: VoxelCoord) {
        self.enqueue_with_neighbors(coord);
    }

    pub fn step_region(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
    ) -> usize {
        self.seed_region_frontier_if_needed(store, region);
        self.enqueue_dirty_voxels(store, region);

        let mut candidates = self.prioritized_chunks(center, region);
        if candidates.len() > MAX_CHUNKS_PER_STEP {
            candidates.truncate(MAX_CHUNKS_PER_STEP);
        }

        let mut stepped_chunks = 0usize;
        for chunk_coord in candidates {
            if !store.is_chunk_loaded(chunk_coord) {
                self.chunks.remove(&chunk_coord);
                continue;
            }

            let mut activation_centers = Vec::new();
            {
                let Some(state) = self.chunks.get_mut(&chunk_coord) else {
                    continue;
                };
                if state.is_empty() {
                    continue;
                }

                stepped_chunks += 1;
                let mut frontier = state.take_frontier();
                rng.shuffle(&mut frontier);

                let mut pending_writes: Vec<(VoxelCoord, u16)> = Vec::new();
                let mut moved_sources: HashSet<u16> = HashSet::new();
                let mut claimed_destinations: HashSet<VoxelCoord> = HashSet::new();
                let mut moved_any = false;

                for &idx in &frontier {
                    state.clear_active(idx);
                }

                for idx in frontier.iter().copied() {
                    if moved_sources.contains(&idx) {
                        continue;
                    }
                    let source = local_index_to_world(chunk_coord, idx);
                    let (source_chunk, _) = voxel_to_chunk(source);
                    if source_chunk != chunk_coord || !store.is_chunk_loaded(source_chunk) {
                        continue;
                    }

                    let mat_id = store.get_voxel(source);
                    if mat_id == EMPTY {
                        continue;
                    }
                    let mat = material(mat_id);
                    let candidates = movement_candidates(source, mat.phase, rng);

                    for destination in candidates {
                        if claimed_destinations.contains(&destination) {
                            continue;
                        }
                        let (destination_chunk, _) = voxel_to_chunk(destination);
                        if !store.is_chunk_loaded(destination_chunk) {
                            continue;
                        }

                        let target_id = store.get_voxel(destination);
                        if target_id == mat_id {
                            continue;
                        }
                        let target_mat = material(target_id);
                        if !can_displace(
                            mat.phase,
                            mat.density,
                            target_mat.phase,
                            target_mat.density,
                            target_id,
                        ) {
                            continue;
                        }

                        pending_writes.push((source, EMPTY));
                        pending_writes.push((destination, mat_id));
                        moved_sources.insert(idx);
                        claimed_destinations.insert(destination);
                        moved_any = true;
                        activation_centers.push(source);
                        activation_centers.push(destination);
                        break;
                    }
                }

                for (coord, mat_id) in pending_writes {
                    store.set_voxel(coord, mat_id);
                }

                if moved_any {
                    state.cooldown_ticks = 0;
                    state.recent_activity = state.recent_activity.saturating_add(4);
                } else {
                    state.cooldown_ticks = CHUNK_COOLDOWN_TICKS;
                    state.recent_activity = state.recent_activity.saturating_sub(1);
                }

                state.return_frontier_buffer(frontier);
            }

            for center in activation_centers {
                enqueue_world_neighbors_to_chunks(&mut self.chunks, center, region);
            }
        }

        stepped_chunks
    }

    pub fn build_soa_for_chunk(
        &self,
        store: &ChunkStore,
        chunk_coord: ChunkCoord,
    ) -> Option<SimSoAChunkBuffer> {
        let chunk = store.get_chunk(chunk_coord)?;
        let mut material_ids = Vec::with_capacity(chunk.iter_raw().len());
        let mut phases = Vec::with_capacity(chunk.iter_raw().len());
        let mut settled = Vec::with_capacity(chunk.iter_raw().len());

        let active_state = self.chunks.get(&chunk_coord);
        for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
            material_ids.push(mat_id);
            phases.push(material(mat_id).phase as u8);
            let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
            let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
            let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
            let is_active = active_state
                .map(|state| {
                    let idx = local_to_index(x as usize, y as usize, z as usize);
                    let word = idx / 64;
                    let bit = 1u64 << (idx % 64);
                    (state.active_bits[word] & bit) != 0
                })
                .unwrap_or(false);
            settled.push((!is_active) as u8);
        }

        Some(SimSoAChunkBuffer {
            material_ids,
            phases,
            settled,
        })
    }

    fn enqueue_dirty_voxels(&mut self, store: &mut ChunkStore, region: &HashSet<ChunkCoord>) {
        for coord in store.take_sim_dirty_voxels() {
            self.enqueue_with_neighbors_if_in_region(coord, region);
        }
    }

    fn seed_region_frontier_if_needed(&mut self, store: &ChunkStore, region: &HashSet<ChunkCoord>) {
        for &chunk_coord in region {
            if !store.is_chunk_loaded(chunk_coord)
                || self.known_seeded_chunks.contains(&chunk_coord)
            {
                continue;
            }
            self.known_seeded_chunks.insert(chunk_coord);
            let Some(chunk) = store.get_chunk(chunk_coord) else {
                continue;
            };
            let base = chunk_to_world_min(chunk_coord);
            for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
                if mat_id == EMPTY {
                    continue;
                }
                let x = idx % CHUNK_SIZE_VOXELS as usize;
                let y = (idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize;
                let z = idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize);
                let world = VoxelCoord {
                    x: base.x + x as i32,
                    y: base.y + y as i32,
                    z: base.z + z as i32,
                };
                self.enqueue_with_neighbors_if_in_region(world, region);
            }
        }
    }

    fn enqueue_with_neighbors_if_in_region(
        &mut self,
        center: VoxelCoord,
        region: &HashSet<ChunkCoord>,
    ) {
        enqueue_world_neighbors_to_chunks(&mut self.chunks, center, region);
    }

    fn enqueue_with_neighbors(&mut self, center: VoxelCoord) {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let world = VoxelCoord {
                        x: center.x + dx,
                        y: center.y + dy,
                        z: center.z + dz,
                    };
                    let (chunk_coord, local) = voxel_to_chunk(world);
                    let idx =
                        local_to_index(local[0] as usize, local[1] as usize, local[2] as usize);
                    let state = self
                        .chunks
                        .entry(chunk_coord)
                        .or_insert_with(ChunkFrontierState::new);
                    state.enqueue_local(idx as u16);
                    state.cooldown_ticks = 0;
                }
            }
        }
    }

    fn prioritized_chunks(
        &mut self,
        center: ChunkCoord,
        region: &HashSet<ChunkCoord>,
    ) -> Vec<ChunkCoord> {
        let mut out = Vec::new();
        for (&coord, state) in &mut self.chunks {
            if !region.contains(&coord) {
                continue;
            }
            if state.is_empty() {
                if state.cooldown_ticks > 0 {
                    state.cooldown_ticks -= 1;
                }
                continue;
            }
            if state.cooldown_ticks > 0 {
                state.cooldown_ticks -= 1;
                continue;
            }
            out.push((coord, state.recent_activity));
        }

        out.sort_by_key(|(coord, activity)| {
            let dx = i64::from(coord.x - center.x);
            let dy = i64::from(coord.y - center.y);
            let dz = i64::from(coord.z - center.z);
            let dist2 = dx * dx + dy * dy + dz * dz;
            (dist2, std::cmp::Reverse(*activity))
        });
        out.into_iter().map(|(coord, _)| coord).collect()
    }
}

pub fn step_region_profiled(
    store: &mut ChunkStore,
    region: &HashSet<ChunkCoord>,
    center: ChunkCoord,
    rng: &mut Rng,
) -> usize {
    let mut sim = SimWorld::default();
    sim.step_region(store, region, center, rng)
}

pub fn step_region(store: &mut ChunkStore, region: &HashSet<ChunkCoord>, rng: &mut Rng) {
    let mut sim = SimWorld::default();
    let _ = sim.step_region(store, region, ChunkCoord { x: 0, y: 0, z: 0 }, rng);
}

fn enqueue_world_neighbors_to_chunks(
    chunks: &mut HashMap<ChunkCoord, ChunkFrontierState>,
    center: VoxelCoord,
    region: &HashSet<ChunkCoord>,
) {
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let world = VoxelCoord {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                };
                let (chunk_coord, local) = voxel_to_chunk(world);
                if !region.contains(&chunk_coord) {
                    continue;
                }
                let idx = local_to_index(local[0] as usize, local[1] as usize, local[2] as usize);
                let state = chunks
                    .entry(chunk_coord)
                    .or_insert_with(ChunkFrontierState::new);
                state.enqueue_local(idx as u16);
                state.cooldown_ticks = 0;
            }
        }
    }
}

fn local_to_index(x: usize, y: usize, z: usize) -> usize {
    x + y * CHUNK_SIZE_VOXELS as usize
        + z * (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)
}

fn local_index_to_world(chunk_coord: ChunkCoord, idx: u16) -> VoxelCoord {
    let base = chunk_to_world_min(chunk_coord);
    let idx = idx as usize;
    let sz = CHUNK_SIZE_VOXELS as usize;
    let x = idx % sz;
    let y = (idx / sz) % sz;
    let z = idx / (sz * sz);
    VoxelCoord {
        x: base.x + x as i32,
        y: base.y + y as i32,
        z: base.z + z as i32,
    }
}

fn movement_candidates(source: VoxelCoord, phase: Phase, rng: &mut Rng) -> Vec<VoxelCoord> {
    let mut lateral = vec![
        VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z,
        },
        VoxelCoord {
            x: source.x + 1,
            y: source.y,
            z: source.z,
        },
        VoxelCoord {
            x: source.x,
            y: source.y,
            z: source.z - 1,
        },
        VoxelCoord {
            x: source.x,
            y: source.y,
            z: source.z + 1,
        },
    ];
    rng.shuffle(&mut lateral);

    match phase {
        Phase::Powder => {
            let mut downward_diagonal = vec![
                VoxelCoord {
                    x: source.x - 1,
                    y: source.y - 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x + 1,
                    y: source.y - 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y - 1,
                    z: source.z - 1,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y - 1,
                    z: source.z + 1,
                },
            ];
            rng.shuffle(&mut downward_diagonal);
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y - 1,
                z: source.z,
            }];
            candidates.extend(downward_diagonal);
            candidates
        }
        Phase::Liquid => {
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y - 1,
                z: source.z,
            }];
            candidates.extend(lateral);
            candidates
        }
        Phase::Gas => {
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y + 1,
                z: source.z,
            }];
            candidates.extend(lateral);
            candidates
        }
        Phase::Solid => Vec::new(),
    }
}

fn can_displace(
    mover_phase: Phase,
    mover_density: i16,
    target_phase: Phase,
    target_density: i16,
    target_id: u16,
) -> bool {
    if target_id == EMPTY {
        return true;
    }

    match mover_phase {
        Phase::Powder | Phase::Liquid => {
            target_phase == Phase::Gas || mover_density > target_density
        }
        Phase::Gas => target_phase == Phase::Gas && mover_density < target_density,
        Phase::Solid => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::XorShift32;
    use crate::world::{Chunk as LegacyChunk, EMPTY};

    #[test]
    fn skips_cross_chunk_moves_into_unloaded_destination_chunks() {
        let mut store = ChunkStore::new();
        let source_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let source_world = chunk_to_world_min(source_chunk);
        let source = VoxelCoord {
            x: source_world.x,
            y: source_world.y,
            z: source_world.z,
        };

        let mut chunk = LegacyChunk::new();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    chunk.set(x, y, z, 1);
                }
            }
        }
        chunk.set(0, 0, 0, 5);
        store.insert_chunk(source_chunk, chunk);

        let region = HashSet::from([source_chunk]);
        let mut rng = XorShift32::new(1);
        let mut sim = SimWorld::default();

        sim.step_region(&mut store, &region, source_chunk, &mut rng);

        assert_eq!(store.get_voxel(source), 5);
        let destination = VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z,
        };
        assert_eq!(store.get_voxel(destination), EMPTY);
        assert!(!store.is_voxel_chunk_loaded(destination));
    }

    #[test]
    fn deterministic_frontier_matches_reference_for_simple_fall() {
        let mut store = ChunkStore::new();
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(c);
        let top = VoxelCoord {
            x: base.x + 3,
            y: base.y + 5,
            z: base.z + 3,
        };
        store.set_voxel(top, 3);

        let region = HashSet::from([c]);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(top);

        let mut rng_a = XorShift32::new(42);
        let mut rng_b = XorShift32::new(42);
        let mut store_ref = ChunkStore::new();
        store_ref.set_voxel(top, 3);
        let mut sim_ref = SimWorld::default();
        sim_ref.notify_voxel_edit(top);

        for _ in 0..4 {
            sim.step_region(&mut store, &region, c, &mut rng_a);
            sim_ref.step_region(&mut store_ref, &region, c, &mut rng_b);
        }

        assert_eq!(
            store.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            }),
            3
        );
        assert_eq!(
            store.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            }),
            store_ref.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            })
        );
    }

    #[test]
    fn soa_layout_captures_settled_bit() {
        let mut store = ChunkStore::new();
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(c);
        let v = VoxelCoord {
            x: base.x + 1,
            y: base.y + 1,
            z: base.z + 1,
        };
        store.set_voxel(v, 5);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(v);
        let region = HashSet::from([c]);
        let mut rng = XorShift32::new(7);
        sim.step_region(&mut store, &region, c, &mut rng);

        let soa = sim
            .build_soa_for_chunk(&store, c)
            .expect("chunk should exist");
        assert_eq!(soa.material_ids.len(), (CHUNK_SIZE_VOXELS as usize).pow(3));
        assert_eq!(soa.phases.len(), soa.material_ids.len());
        assert_eq!(soa.settled.len(), soa.material_ids.len());
    }
}
