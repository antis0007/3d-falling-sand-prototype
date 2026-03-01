use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim::{material, Phase, XorShift32};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

pub type Rng = XorShift32;

pub struct SimWorld;

impl SimWorld {
    pub fn step_region(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        rng: &mut Rng,
    ) {
        step_region(store, region, rng);
    }
}

pub fn region_sorted_by_distance(
    center: ChunkCoord,
    region: &HashSet<ChunkCoord>,
) -> Vec<ChunkCoord> {
    let mut chunks: Vec<ChunkCoord> = region.iter().copied().collect();
    chunks.sort_by_key(|coord| {
        let dx = i64::from(coord.x - center.x);
        let dy = i64::from(coord.y - center.y);
        let dz = i64::from(coord.z - center.z);
        dx * dx + dy * dy + dz * dz
    });
    chunks
}

pub fn step_region_profiled(
    store: &mut ChunkStore,
    region: &HashSet<ChunkCoord>,
    center: ChunkCoord,
    rng: &mut Rng,
) -> usize {
    let mut snapshot: HashMap<VoxelCoord, u16> = HashMap::new();
    let mut active_voxels = Vec::new();

    let sorted = region_sorted_by_distance(center, region);
    let mut stepped_chunks = 0usize;

    for chunk_coord in sorted {
        let Some(chunk) = store.get_chunk(chunk_coord) else {
            continue;
        };
        stepped_chunks += 1;

        let base = chunk_to_world_min(chunk_coord);
        for (idx, &mat) in chunk.iter_raw().iter().enumerate() {
            if mat == EMPTY {
                continue;
            }
            let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
            let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
            let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
            let coord = VoxelCoord {
                x: base.x + x,
                y: base.y + y,
                z: base.z + z,
            };
            snapshot.insert(coord, mat);
            active_voxels.push((coord, mat));
        }
    }

    rng.shuffle(&mut active_voxels);

    let mut pending: HashMap<VoxelCoord, u16> = HashMap::new();
    let mut moved_sources: HashSet<VoxelCoord> = HashSet::new();
    let mut claimed_destinations: HashSet<VoxelCoord> = HashSet::new();

    for (source, mat_id) in active_voxels {
        if moved_sources.contains(&source) {
            continue;
        }

        let mat = material(mat_id);
        let candidates = movement_candidates(source, mat.phase, rng);
        let (source_chunk, _) = voxel_to_chunk(source);

        if !store.is_chunk_loaded(source_chunk) {
            continue;
        }

        for destination in candidates {
            if claimed_destinations.contains(&destination) {
                continue;
            }

            let (destination_chunk, _) = voxel_to_chunk(destination);
            if !store.is_chunk_loaded(destination_chunk) {
                continue;
            }

            let target_id = snapshot
                .get(&destination)
                .copied()
                .or_else(|| Some(store.get_voxel(destination)))
                .unwrap_or(EMPTY);
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

            pending.insert(source, EMPTY);
            pending.insert(destination, mat_id);
            moved_sources.insert(source);
            claimed_destinations.insert(destination);
            break;
        }
    }

    for (coord, mat_id) in pending {
        store.set_voxel(coord, mat_id);
    }

    stepped_chunks
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

        step_region_profiled(&mut store, &region, source_chunk, &mut rng);

        assert_eq!(store.get_voxel(source), 5);
        let destination = VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z,
        };
        assert_eq!(store.get_voxel(destination), EMPTY);
        assert!(!store.is_voxel_chunk_loaded(destination));
    }
}

pub fn step_region(store: &mut ChunkStore, region: &HashSet<ChunkCoord>, rng: &mut Rng) {
    let _ = step_region_profiled(store, region, ChunkCoord { x: 0, y: 0, z: 0 }, rng);
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
