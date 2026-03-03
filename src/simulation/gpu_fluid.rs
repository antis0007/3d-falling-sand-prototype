use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim::material;
use crate::sim::Phase;
use crate::sim_world::Rng;
use crate::simulation::{SimulationBackend, SimulationStepMetadata, SimulationStepStats};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

const CHUNK_EDGE: i32 = CHUNK_SIZE_VOXELS as i32;
const SUBSTEPS: usize = 3;
const CHUNK_VOLUME: usize =
    CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize;

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
    resident_pages_current: HashMap<ChunkCoord, Vec<u16>>,
    resident_pages_next: HashMap<ChunkCoord, Vec<u16>>,
    last_dirty_metadata: DirtySimulationMetadata,
}

#[derive(Default)]
struct DirtySimulationMetadata {
    dirty_chunks: HashSet<ChunkCoord>,
    dirty_regions: HashMap<ChunkCoord, DirtyChunkRegion>,
    dirty_bounds: Option<(VoxelCoord, VoxelCoord)>,
}

#[derive(Clone, Copy)]
struct DirtyChunkRegion {
    min: [u32; 3],
    max: [u32; 3],
}

struct HaloFaces {
    neg_x: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    pos_x: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    neg_y: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    pos_y: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    neg_z: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    pos_z: [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
}

impl Default for HaloFaces {
    fn default() -> Self {
        Self {
            neg_x: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
            pos_x: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
            neg_y: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
            pos_y: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
            neg_z: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
            pos_z: [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
        }
    }
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

        self.ensure_resident_pages(store, region);
        self.last_dirty_metadata = DirtySimulationMetadata::default();

        for substep in 0..SUBSTEPS {
            let halos = self.exchange_halos(region);
            let intents = self.advect_material_states_gpu(region, &halos, substep as u64);
            self.run_deterministic_passes(region, intents);
        }
        let dirty_chunks = self.flush_targeted_updates_to_cpu(store, region);
        stepped_chunks.extend(dirty_chunks.iter().copied());
        self.last_dirty_metadata.dirty_chunks = dirty_chunks;

        self.frame_index = self.frame_index.wrapping_add(1);
        SimulationStepStats {
            stepped_chunks: stepped_chunks.len(),
            ..SimulationStepStats::default()
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn ensure_resident_pages(&mut self, store: &ChunkStore, region: &HashSet<ChunkCoord>) {
        self.resident_pages_current
            .retain(|coord, _| region.contains(coord));
        self.resident_pages_next
            .retain(|coord, _| region.contains(coord));
        for &chunk in region {
            self.resident_pages_current.entry(chunk).or_insert_with(|| {
                store
                    .get_chunk(chunk)
                    .map(|c| c.iter_raw().to_vec())
                    .unwrap_or_else(|| vec![EMPTY; CHUNK_VOLUME])
            });
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn exchange_halos(&self, region: &HashSet<ChunkCoord>) -> HashMap<ChunkCoord, HaloFaces> {
        let mut halos = HashMap::with_capacity(region.len());
        for &chunk in region {
            let mut faces = HaloFaces::default();
            self.fill_halo_face(chunk, [-1, 0, 0], &mut faces.neg_x);
            self.fill_halo_face(chunk, [1, 0, 0], &mut faces.pos_x);
            self.fill_halo_face(chunk, [0, -1, 0], &mut faces.neg_y);
            self.fill_halo_face(chunk, [0, 1, 0], &mut faces.pos_y);
            self.fill_halo_face(chunk, [0, 0, -1], &mut faces.neg_z);
            self.fill_halo_face(chunk, [0, 0, 1], &mut faces.pos_z);
            halos.insert(chunk, faces);
        }
        halos
    }

    #[cfg(feature = "gpu-compute")]
    fn fill_halo_face(
        &self,
        chunk: ChunkCoord,
        delta: [i32; 3],
        out: &mut [u16; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize],
    ) {
        let neighbor = ChunkCoord {
            x: chunk.x + delta[0],
            y: chunk.y + delta[1],
            z: chunk.z + delta[2],
        };
        let Some(voxels) = self.resident_pages_current.get(&neighbor) else {
            *out = [EMPTY; CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize];
            return;
        };
        let last = CHUNK_SIZE_VOXELS as usize - 1;
        for v in 0..CHUNK_SIZE_VOXELS as usize {
            for u in 0..CHUNK_SIZE_VOXELS as usize {
                let (x, y, z) = if delta[0] != 0 {
                    (if delta[0] < 0 { last } else { 0 }, u, v)
                } else if delta[1] != 0 {
                    (u, if delta[1] < 0 { last } else { 0 }, v)
                } else {
                    (u, v, if delta[2] < 0 { last } else { 0 })
                };
                out[u + v * CHUNK_SIZE_VOXELS as usize] =
                    voxels[crate::chunk_store::Chunk::index(x, y, z)];
            }
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn advect_material_states_gpu(
        &self,
        region: &HashSet<ChunkCoord>,
        halos: &HashMap<ChunkCoord, HaloFaces>,
        substep: u64,
    ) -> Vec<MoveIntent> {
        let mut chunk_coords: Vec<_> = region.iter().copied().collect();
        chunk_coords.sort_by_key(|c| (c.z, c.y, c.x));
        let mut intents = Vec::new();
        for chunk_coord in chunk_coords {
            let Some(voxels) = self.resident_pages_current.get(&chunk_coord) else {
                continue;
            };
            let origin = chunk_to_world_min(chunk_coord);
            for z in (0..CHUNK_EDGE).rev() {
                for y in (0..CHUNK_EDGE).rev() {
                    for x in 0..CHUNK_EDGE {
                        let local = [x as usize, y as usize, z as usize];
                        let from = VoxelCoord {
                            x: origin.x + x,
                            y: origin.y + y,
                            z: origin.z + z,
                        };
                        let mat_id =
                            voxels[crate::chunk_store::Chunk::index(local[0], local[1], local[2])];
                        if mat_id == EMPTY {
                            continue;
                        }
                        let phase = material(mat_id).phase;
                        for to in movement_candidates(from, phase) {
                            if self.read_material_with_halo(halos, to) != EMPTY {
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

    #[cfg(feature = "gpu-compute")]
    fn read_material_with_halo(
        &self,
        _halos: &HashMap<ChunkCoord, HaloFaces>,
        coord: VoxelCoord,
    ) -> u16 {
        let (chunk, local) = voxel_to_chunk(coord);
        self.resident_pages_current
            .get(&chunk)
            .map(|page| {
                page[crate::chunk_store::Chunk::index(
                    local[0] as usize,
                    local[1] as usize,
                    local[2] as usize,
                )]
            })
            .unwrap_or(EMPTY)
    }

    #[cfg(feature = "gpu-compute")]
    fn run_deterministic_passes(&mut self, region: &HashSet<ChunkCoord>, intents: Vec<MoveIntent>) {
        self.resident_pages_next = self.resident_pages_current.clone();
        self.apply_intents_to_pages(region, intents);
        // deterministic pass ordering placeholders
        self.compute_divergence_pass();
        self.pressure_solve_pass();
        self.projection_pass();
        self.material_update_pass();
        std::mem::swap(
            &mut self.resident_pages_current,
            &mut self.resident_pages_next,
        );
    }

    #[cfg(feature = "gpu-compute")]
    fn apply_intents_to_pages(
        &mut self,
        region: &HashSet<ChunkCoord>,
        mut intents: Vec<MoveIntent>,
    ) {
        intents.sort_by_key(|intent| intent.priority);
        let mut claimed_src = HashSet::new();
        let mut claimed_dst = HashSet::new();
        for intent in intents {
            if claimed_src.contains(&intent.from) || claimed_dst.contains(&intent.to) {
                continue;
            }
            let (src_chunk, src_local) = voxel_to_chunk(intent.from);
            let (dst_chunk, dst_local) = voxel_to_chunk(intent.to);
            if !region.contains(&src_chunk) || !region.contains(&dst_chunk) {
                continue;
            }
            let Some(src_page_read) = self.resident_pages_current.get(&src_chunk) else {
                continue;
            };
            let src_idx = crate::chunk_store::Chunk::index(
                src_local[0] as usize,
                src_local[1] as usize,
                src_local[2] as usize,
            );
            let dst_idx = crate::chunk_store::Chunk::index(
                dst_local[0] as usize,
                dst_local[1] as usize,
                dst_local[2] as usize,
            );
            let src_material = src_page_read[src_idx];
            if src_material == EMPTY {
                continue;
            }
            let dst_material = self
                .resident_pages_current
                .get(&dst_chunk)
                .map(|p| p[dst_idx])
                .unwrap_or(EMPTY);
            if dst_material != EMPTY {
                continue;
            }
            if let Some(src_page_write) = self.resident_pages_next.get_mut(&src_chunk) {
                src_page_write[src_idx] = EMPTY;
            }
            if let Some(dst_page_write) = self.resident_pages_next.get_mut(&dst_chunk) {
                dst_page_write[dst_idx] = src_material;
            }
            claimed_src.insert(intent.from);
            claimed_dst.insert(intent.to);
        }
    }

    #[cfg(feature = "gpu-compute")]
    fn compute_divergence_pass(&self) {}
    #[cfg(feature = "gpu-compute")]
    fn pressure_solve_pass(&self) {}
    #[cfg(feature = "gpu-compute")]
    fn projection_pass(&self) {}
    #[cfg(feature = "gpu-compute")]
    fn material_update_pass(&self) {}

    #[cfg(feature = "gpu-compute")]
    fn flush_targeted_updates_to_cpu(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
    ) -> HashSet<ChunkCoord> {
        let mut touched_chunks = HashSet::new();
        for &chunk in region {
            let Some(gpu_page) = self.resident_pages_current.get(&chunk).cloned() else {
                continue;
            };
            let origin = chunk_to_world_min(chunk);
            for z in 0..CHUNK_SIZE_VOXELS as usize {
                for y in 0..CHUNK_SIZE_VOXELS as usize {
                    for x in 0..CHUNK_SIZE_VOXELS as usize {
                        let idx = crate::chunk_store::Chunk::index(x, y, z);
                        let gpu_value = gpu_page[idx];
                        let world = VoxelCoord {
                            x: origin.x + x as i32,
                            y: origin.y + y as i32,
                            z: origin.z + z as i32,
                        };
                        if store.get_voxel(world) == gpu_value {
                            continue;
                        }
                        store.set_voxel(world, gpu_value);
                        touched_chunks.insert(chunk);
                        self.record_dirty_voxel(chunk, [x as u32, y as u32, z as u32], world);
                    }
                }
            }
        }
        touched_chunks
    }

    #[cfg(feature = "gpu-compute")]
    fn record_dirty_voxel(&mut self, chunk: ChunkCoord, local: [u32; 3], world: VoxelCoord) {
        let entry = self
            .last_dirty_metadata
            .dirty_regions
            .entry(chunk)
            .or_insert(DirtyChunkRegion {
                min: local,
                max: local,
            });
        for axis in 0..3 {
            entry.min[axis] = entry.min[axis].min(local[axis]);
            entry.max[axis] = entry.max[axis].max(local[axis]);
        }
        self.last_dirty_metadata.dirty_bounds = match self.last_dirty_metadata.dirty_bounds {
            None => Some((world, world)),
            Some((mut min, mut max)) => {
                min.x = min.x.min(world.x);
                min.y = min.y.min(world.y);
                min.z = min.z.min(world.z);
                max.x = max.x.max(world.x);
                max.y = max.y.max(world.y);
                max.z = max.z.max(world.z);
                Some((min, max))
            }
        };
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
    fn liquid_crosses_two_chunk_boundary() {
        let mut backend = GpuFluidBackend::default();
        let mut store = ChunkStore::new();
        let a = ChunkCoord { x: 0, y: 0, z: 0 };
        let b = ChunkCoord { x: 1, y: 0, z: 0 };
        let region = HashSet::from([a, b]);

        // water at +X edge of chunk A
        store.set_voxel(VoxelCoord { x: 31, y: 4, z: 4 }, 5);
        // block downward and in-chunk lateral options so +X neighbor is selected
        store.set_voxel(VoxelCoord { x: 31, y: 3, z: 4 }, 1);
        store.set_voxel(VoxelCoord { x: 30, y: 4, z: 4 }, 1);
        store.set_voxel(VoxelCoord { x: 31, y: 4, z: 3 }, 1);
        store.set_voxel(VoxelCoord { x: 31, y: 4, z: 5 }, 1);

        let touched = backend.dispatch_substep(&mut store, &region, 0);
        assert!(touched.contains(&a));
        assert!(touched.contains(&b));
        assert_eq!(store.get_voxel(VoxelCoord { x: 31, y: 4, z: 4 }), EMPTY);
        assert_eq!(store.get_voxel(VoxelCoord { x: 32, y: 4, z: 4 }), 5);
    }

    #[test]
    fn liquid_settles_continuously_across_boundary() {
        let mut backend = GpuFluidBackend::default();
        let mut store = ChunkStore::new();
        let a = ChunkCoord { x: 0, y: 0, z: 0 };
        let b = ChunkCoord { x: 1, y: 0, z: 0 };
        let region = HashSet::from([a, b]);

        store.set_voxel(VoxelCoord { x: 31, y: 8, z: 4 }, 5);
        store.set_voxel(VoxelCoord { x: 32, y: 8, z: 4 }, 5);
        for x in 30..=33 {
            store.set_voxel(VoxelCoord { x, y: 0, z: 4 }, 1);
        }

        for substep in 0..16 {
            backend.dispatch_substep(&mut store, &region, substep);
        }

        assert_eq!(store.get_voxel(VoxelCoord { x: 31, y: 1, z: 4 }), 5);
        assert_eq!(store.get_voxel(VoxelCoord { x: 32, y: 1, z: 4 }), 5);
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
