use std::collections::{HashMap, HashSet, VecDeque};

use crate::chunk_store::ChunkStore;
use crate::sim::{material, Phase, XorShift32};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

pub type Rng = XorShift32;

const CHUNK_COOLDOWN_TICKS: u8 = 8;
const MAX_CHUNKS_PER_STEP: usize = 96;

#[derive(Default)]
struct ChunkFrontierState {
    active_voxels: HashSet<VoxelCoord>,
    cooldown_ticks: u8,
    recent_activity: u16,
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
    pending_enqueue: VecDeque<VoxelCoord>,
    known_seeded_chunks: HashSet<ChunkCoord>,
}

impl SimWorld {
    pub fn notify_voxel_edit(&mut self, coord: VoxelCoord) {
        enqueue_with_neighbors(&mut self.pending_enqueue, coord);
    }

    pub fn step_region(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
    ) -> usize {
        self.seed_region_frontier_if_needed(store, region);
        self.enqueue_dirty_chunks(store, region);
        self.drain_enqueue_queue(region);

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

            let Some(state) = self.chunks.get_mut(&chunk_coord) else {
                continue;
            };
            if state.active_voxels.is_empty() {
                continue;
            }

            stepped_chunks += 1;
            let mut frontier: Vec<VoxelCoord> = state.active_voxels.drain().collect();
            rng.shuffle(&mut frontier);

            let mut pending_writes: Vec<(VoxelCoord, u16)> = Vec::new();
            let mut moved_sources: HashSet<VoxelCoord> = HashSet::new();
            let mut claimed_destinations: HashSet<VoxelCoord> = HashSet::new();
            let mut moved_any = false;

            for source in frontier {
                if moved_sources.contains(&source) {
                    continue;
                }
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
                    moved_sources.insert(source);
                    claimed_destinations.insert(destination);
                    moved_any = true;
                    enqueue_with_neighbors(&mut self.pending_enqueue, source);
                    enqueue_with_neighbors(&mut self.pending_enqueue, destination);
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
        }

        self.drain_enqueue_queue(region);
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

        let active = self
            .chunks
            .get(&chunk_coord)
            .map(|st| &st.active_voxels)
            .cloned()
            .unwrap_or_default();
        let base = chunk_to_world_min(chunk_coord);

        for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
            material_ids.push(mat_id);
            phases.push(material(mat_id).phase as u8);
            let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
            let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
            let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
            let world = VoxelCoord {
                x: base.x + x,
                y: base.y + y,
                z: base.z + z,
            };
            settled.push((!active.contains(&world)) as u8);
        }

        Some(SimSoAChunkBuffer {
            material_ids,
            phases,
            settled,
        })
    }

    fn enqueue_dirty_chunks(&mut self, store: &ChunkStore, region: &HashSet<ChunkCoord>) {
        for chunk_coord in store.dirty_chunks_snapshot() {
            if !region.contains(&chunk_coord) {
                continue;
            }
            let Some(chunk) = store.get_chunk(chunk_coord) else {
                continue;
            };
            let base = chunk_to_world_min(chunk_coord);
            for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
                if mat_id == EMPTY {
                    continue;
                }
                let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
                let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
                let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
                self.pending_enqueue.push_back(VoxelCoord {
                    x: base.x + x,
                    y: base.y + y,
                    z: base.z + z,
                });
            }
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
            let state = self.chunks.entry(chunk_coord).or_default();
            for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
                if mat_id == EMPTY {
                    continue;
                }
                let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
                let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
                let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
                state.active_voxels.insert(VoxelCoord {
                    x: base.x + x,
                    y: base.y + y,
                    z: base.z + z,
                });
            }
        }
    }

    fn drain_enqueue_queue(&mut self, region: &HashSet<ChunkCoord>) {
        while let Some(coord) = self.pending_enqueue.pop_front() {
            let (chunk_coord, _) = voxel_to_chunk(coord);
            if !region.contains(&chunk_coord) {
                continue;
            }
            let state = self.chunks.entry(chunk_coord).or_default();
            state.active_voxels.insert(coord);
            state.cooldown_ticks = 0;
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
            if state.active_voxels.is_empty() {
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

fn enqueue_with_neighbors(queue: &mut VecDeque<VoxelCoord>, center: VoxelCoord) {
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                queue.push_back(VoxelCoord {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                });
            }
        }
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
