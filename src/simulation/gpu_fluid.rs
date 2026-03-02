use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::chunk_store::ChunkStore;
use crate::renderer::{ChunkLod, ChunkSnapshot, MeshJob};
use crate::sim::material;
use crate::sim::Phase;
use crate::sim_world::Rng;
use crate::simulation::{SimulationBackend, SimulationStepMetadata, SimulationStepStats};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

const CHUNK_EDGE: i32 = CHUNK_SIZE_VOXELS as i32;
const SUBSTEPS: usize = 3;

#[derive(Clone, Copy, Debug)]
struct SimCommand {
    coord: VoxelCoord,
    material_id: u16,
}

#[derive(Clone, Copy, Debug)]
struct MoveIntent {
    from: VoxelCoord,
    to: VoxelCoord,
    priority: u64,
}

#[derive(Default)]
pub struct GpuFluidBackend {
    command_buffer: Vec<SimCommand>,
    frame_index: u64,
}

impl SimulationBackend for GpuFluidBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        _center: ChunkCoord,
        _rng: &mut Rng,
        _metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        #[cfg(feature = "gpu-compute")]
        {
            return self.step_gpu_native(store, region);
        }

        #[cfg(not(feature = "gpu-compute"))]
        {
            let mut stepped_chunks: HashSet<ChunkCoord> = HashSet::new();

            let edited_chunks = self.stage_edit_commands(store, region);
            stepped_chunks.extend(edited_chunks);

            for substep in 0..SUBSTEPS {
                let touched = self.dispatch_substep(store, region, substep as u64);
                stepped_chunks.extend(touched);
            }

            self.frame_index = self.frame_index.wrapping_add(1);
            SimulationStepStats {
                stepped_chunks: stepped_chunks.len(),
                ..SimulationStepStats::default()
            }
        }
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, material_id: u16) {
        self.command_buffer.push(SimCommand { coord, material_id });
    }
}

impl GpuFluidBackend {
    #[cfg(feature = "gpu-compute")]
    fn step_gpu_native(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
    ) -> SimulationStepStats {
        let mut stepped_chunks: HashSet<ChunkCoord> = self.stage_edit_commands(store, region);

        for &chunk_coord in region {
            let Some(chunk) = store.get_chunk(chunk_coord) else {
                continue;
            };
            let input_materials: Vec<u16> = chunk.iter_raw().to_vec();
            let job = MeshJob {
                coord: chunk_coord,
                lod: ChunkLod::Near,
                version: self.frame_index,
                queued_at: Instant::now(),
                snapshot: ChunkSnapshot {
                    world_min: chunk_to_world_min(chunk_coord),
                    center_voxels: Arc::from(input_materials.clone()),
                    border_strips: Arc::new(crate::chunk_store::ChunkBorderStrips::default()),
                },
                greedy: true,
            };

            let Ok(output) = crate::gpu_compute::run_chunk_job_on_worker(&job) else {
                continue;
            };
            for (idx, &next) in output.generated_materials.iter().enumerate() {
                if input_materials.get(idx).copied() == Some(next) {
                    continue;
                }
                let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
                let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
                let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
                store.set_voxel(
                    VoxelCoord {
                        x: chunk_to_world_min(chunk_coord).x + x,
                        y: chunk_to_world_min(chunk_coord).y + y,
                        z: chunk_to_world_min(chunk_coord).z + z,
                    },
                    next,
                );
                stepped_chunks.insert(chunk_coord);
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);
        SimulationStepStats {
            stepped_chunks: stepped_chunks.len(),
            ..SimulationStepStats::default()
        }
    }

    fn stage_edit_commands(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
    ) -> HashSet<ChunkCoord> {
        let mut touched = HashSet::new();
        let staged: Vec<_> = self.command_buffer.drain(..).collect();
        for cmd in staged {
            store.set_voxel(cmd.coord, cmd.material_id);
            let (chunk_coord, local) = voxel_to_chunk(cmd.coord);
            if region.contains(&chunk_coord) {
                touched.insert(chunk_coord);
            }
            for neighbor in voxel_neighbors_26(cmd.coord) {
                let (neighbor_chunk, _) = voxel_to_chunk(neighbor);
                if region.contains(&neighbor_chunk) {
                    touched.insert(neighbor_chunk);
                }
            }
            let last_local = (CHUNK_SIZE_VOXELS - 1) as u32;
            if local[0] == 0 || local[0] == last_local {
                touched.insert(ChunkCoord {
                    x: chunk_coord.x - 1,
                    y: chunk_coord.y,
                    z: chunk_coord.z,
                });
                touched.insert(ChunkCoord {
                    x: chunk_coord.x + 1,
                    y: chunk_coord.y,
                    z: chunk_coord.z,
                });
            }
            if local[1] == 0 || local[1] == last_local {
                touched.insert(ChunkCoord {
                    x: chunk_coord.x,
                    y: chunk_coord.y - 1,
                    z: chunk_coord.z,
                });
                touched.insert(ChunkCoord {
                    x: chunk_coord.x,
                    y: chunk_coord.y + 1,
                    z: chunk_coord.z,
                });
            }
            if local[2] == 0 || local[2] == last_local {
                touched.insert(ChunkCoord {
                    x: chunk_coord.x,
                    y: chunk_coord.y,
                    z: chunk_coord.z - 1,
                });
                touched.insert(ChunkCoord {
                    x: chunk_coord.x,
                    y: chunk_coord.y,
                    z: chunk_coord.z + 1,
                });
            }
        }
        touched.retain(|coord| region.contains(coord));
        touched
    }

    fn dispatch_substep(
        &self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        substep: u64,
    ) -> HashSet<ChunkCoord> {
        let snapshot = self.build_region_snapshot(store, region);
        let intents = self.advect_material_states(region, &snapshot, substep);
        self.apply_intents(store, region, &snapshot, intents)
    }

    fn build_region_snapshot(
        &self,
        store: &ChunkStore,
        region: &HashSet<ChunkCoord>,
    ) -> HashMap<VoxelCoord, u16> {
        let mut snapshot = HashMap::new();
        for &chunk in region {
            let origin = chunk_to_world_min(chunk);
            for z in -1..=CHUNK_EDGE {
                for y in -1..=CHUNK_EDGE {
                    for x in -1..=CHUNK_EDGE {
                        let coord = VoxelCoord {
                            x: origin.x + x,
                            y: origin.y + y,
                            z: origin.z + z,
                        };
                        snapshot
                            .entry(coord)
                            .or_insert_with(|| store.get_voxel(coord));
                    }
                }
            }
        }
        snapshot
    }

    fn advect_material_states(
        &self,
        region: &HashSet<ChunkCoord>,
        snapshot: &HashMap<VoxelCoord, u16>,
        substep: u64,
    ) -> Vec<MoveIntent> {
        let mut intents = Vec::new();
        for &chunk_coord in region {
            let origin = chunk_to_world_min(chunk_coord);
            for z in (0..CHUNK_EDGE).rev() {
                for y in (0..CHUNK_EDGE).rev() {
                    for x in 0..CHUNK_EDGE {
                        let from = VoxelCoord {
                            x: origin.x + x,
                            y: origin.y + y,
                            z: origin.z + z,
                        };
                        let mat_id = *snapshot.get(&from).unwrap_or(&EMPTY);
                        if mat_id == EMPTY {
                            continue;
                        }
                        let phase = material(mat_id).phase;
                        let candidates = movement_candidates(from, phase);
                        for to in candidates {
                            if *snapshot.get(&to).unwrap_or(&EMPTY) != EMPTY {
                                continue;
                            }
                            intents.push(MoveIntent {
                                from,
                                to,
                                priority: intent_priority(
                                    self.frame_index,
                                    substep,
                                    chunk_coord,
                                    from,
                                    to,
                                ),
                            });
                            break;
                        }
                    }
                }
            }
        }
        intents
    }

    fn apply_intents(
        &self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        snapshot: &HashMap<VoxelCoord, u16>,
        mut intents: Vec<MoveIntent>,
    ) -> HashSet<ChunkCoord> {
        intents.sort_by_key(|intent| intent.priority);

        let mut claimed_src = HashSet::new();
        let mut claimed_dst = HashSet::new();
        let mut touched_chunks = HashSet::new();

        for intent in intents {
            if claimed_src.contains(&intent.from) || claimed_dst.contains(&intent.to) {
                continue;
            }
            let src_material = *snapshot.get(&intent.from).unwrap_or(&EMPTY);
            if src_material == EMPTY {
                continue;
            }
            if *snapshot.get(&intent.to).unwrap_or(&EMPTY) != EMPTY {
                continue;
            }
            claimed_src.insert(intent.from);
            claimed_dst.insert(intent.to);

            store.set_voxel(intent.to, src_material);
            store.set_voxel(intent.from, EMPTY);

            let (src_chunk, _) = voxel_to_chunk(intent.from);
            let (dst_chunk, _) = voxel_to_chunk(intent.to);
            if region.contains(&src_chunk) {
                touched_chunks.insert(src_chunk);
            }
            if region.contains(&dst_chunk) {
                touched_chunks.insert(dst_chunk);
            }
        }

        touched_chunks
    }
}

fn movement_candidates(origin: VoxelCoord, phase: Phase) -> Vec<VoxelCoord> {
    let dirs: &[(i32, i32, i32)] = match phase {
        Phase::Solid => &[],
        Phase::Powder => &[
            (0, -1, 0),
            (-1, -1, 0),
            (1, -1, 0),
            (0, -1, -1),
            (0, -1, 1),
            (-1, 0, 0),
            (1, 0, 0),
        ],
        Phase::Liquid => &[
            (0, -1, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (0, 0, -1),
            (0, 0, 1),
            (-1, -1, 0),
            (1, -1, 0),
        ],
        Phase::Gas => &[
            (0, 1, 0),
            (-1, 1, 0),
            (1, 1, 0),
            (0, 1, -1),
            (0, 1, 1),
            (-1, 0, 0),
            (1, 0, 0),
        ],
    };

    dirs.iter()
        .map(|(dx, dy, dz)| VoxelCoord {
            x: origin.x + dx,
            y: origin.y + dy,
            z: origin.z + dz,
        })
        .collect()
}

fn voxel_neighbors_26(origin: VoxelCoord) -> impl Iterator<Item = VoxelCoord> {
    (-1..=1).flat_map(move |dz| {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dx| {
                if dx == 0 && dy == 0 && dz == 0 {
                    None
                } else {
                    Some(VoxelCoord {
                        x: origin.x + dx,
                        y: origin.y + dy,
                        z: origin.z + dz,
                    })
                }
            })
        })
    })
}

fn intent_priority(
    frame_index: u64,
    substep: u64,
    chunk_coord: ChunkCoord,
    from: VoxelCoord,
    to: VoxelCoord,
) -> u64 {
    let mut h = frame_index ^ (substep.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    h ^= mix_i32(chunk_coord.x).rotate_left(7);
    h ^= mix_i32(chunk_coord.y).rotate_left(19);
    h ^= mix_i32(chunk_coord.z).rotate_left(31);
    h ^= mix_i32(from.x).rotate_left(11);
    h ^= mix_i32(from.y).rotate_left(23);
    h ^= mix_i32(from.z).rotate_left(37);
    h ^= mix_i32(to.x).rotate_left(13);
    h ^= mix_i32(to.y).rotate_left(29);
    h ^= mix_i32(to.z).rotate_left(43);
    h ^ (h >> 33).wrapping_mul(0xff51_afd7_ed55_8ccd)
}

fn mix_i32(v: i32) -> u64 {
    let x = v as i64 as u64;
    x.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powder_prefers_downward_candidates() {
        let origin = VoxelCoord {
            x: 10,
            y: 10,
            z: 10,
        };
        let candidates = movement_candidates(origin, Phase::Powder);
        assert_eq!(
            candidates.first().copied(),
            Some(VoxelCoord { x: 10, y: 9, z: 10 })
        );
    }

    #[test]
    fn gas_candidates_include_upward_motion() {
        let origin = VoxelCoord {
            x: 12,
            y: 12,
            z: 12,
        };
        let candidates = movement_candidates(origin, Phase::Gas);
        assert!(candidates.contains(&VoxelCoord {
            x: 12,
            y: 13,
            z: 12
        }));
    }
}
