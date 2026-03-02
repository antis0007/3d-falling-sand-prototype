use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim::material;
use crate::sim::Phase;
use crate::sim_world::Rng;
use crate::simulation::SimulationBackend;
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

const CHUNK_EDGE: usize = CHUNK_SIZE_VOXELS as usize;
const CHUNK_VOLUME: usize = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;
const SUBSTEPS: usize = 3;

#[derive(Clone, Copy, Debug)]
struct SimCommand {
    coord: VoxelCoord,
    material_id: u16,
}

#[derive(Default)]
struct DirtyRegionMetadata {
    dirty_chunks: HashSet<ChunkCoord>,
    dirty_voxel_indices: HashMap<ChunkCoord, Vec<usize>>,
}

#[derive(Default)]
struct ChunkGpuBuffers {
    occupancy_material: Vec<u16>,
}

impl ChunkGpuBuffers {
    fn ensure_allocated(&mut self) {
        if self.occupancy_material.len() == CHUNK_VOLUME {
            return;
        }
        self.occupancy_material = vec![EMPTY; CHUNK_VOLUME];
    }
}

#[derive(Default)]
pub struct GpuFluidBackend {
    chunk_buffers: HashMap<ChunkCoord, ChunkGpuBuffers>,
    command_buffer: Vec<SimCommand>,
    dirty_regions: DirtyRegionMetadata,
}

impl SimulationBackend for GpuFluidBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        _center: ChunkCoord,
        _rng: &mut Rng,
    ) -> usize {
        self.stage_edit_commands();

        for _ in 0..SUBSTEPS {
            self.dispatch_substep(region);
        }

        self.read_back_modified_chunks(store);
        region.len()
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, material_id: u16) {
        self.command_buffer.push(SimCommand { coord, material_id });
    }
}

impl GpuFluidBackend {
    fn stage_edit_commands(&mut self) {
        let staged: Vec<_> = self.command_buffer.drain(..).collect();
        for cmd in staged {
            let (chunk_coord, local) = voxel_to_chunk(cmd.coord);
            let index = linear_idx(local);
            let buffers = self.chunk_buffers.entry(chunk_coord).or_default();
            buffers.ensure_allocated();
            buffers.occupancy_material[index] = cmd.material_id;
            self.mark_dirty(chunk_coord, index);
        }
    }

    fn dispatch_substep(&mut self, region: &HashSet<ChunkCoord>) {
        for &chunk_coord in region {
            let Some(buffers) = self.chunk_buffers.get_mut(&chunk_coord) else {
                continue;
            };
            Self::advect_material_states(buffers, &mut self.dirty_regions, chunk_coord);
        }
    }

    fn advect_material_states(
        buffers: &mut ChunkGpuBuffers,
        dirty_regions: &mut DirtyRegionMetadata,
        chunk_coord: ChunkCoord,
    ) {
        for i in (0..CHUNK_VOLUME).rev() {
            let mat_id = buffers.occupancy_material[i];
            if mat_id == EMPTY {
                continue;
            }
            let phase = material(mat_id).phase;
            let candidates = movement_candidates(i, phase);
            if candidates.is_empty() {
                continue;
            }
            for next in candidates {
                if buffers.occupancy_material[next] != EMPTY {
                    continue;
                }
                buffers.occupancy_material[next] = buffers.occupancy_material[i];
                buffers.occupancy_material[i] = EMPTY;
                dirty_regions.dirty_chunks.insert(chunk_coord);
                dirty_regions
                    .dirty_voxel_indices
                    .entry(chunk_coord)
                    .or_default()
                    .extend_from_slice(&[i, next]);
                break;
            }
        }
    }

    fn mark_dirty(&mut self, chunk_coord: ChunkCoord, voxel_index: usize) {
        self.dirty_regions.dirty_chunks.insert(chunk_coord);
        self.dirty_regions
            .dirty_voxel_indices
            .entry(chunk_coord)
            .or_default()
            .push(voxel_index);
    }

    fn read_back_modified_chunks(&mut self, store: &mut ChunkStore) {
        let dirty_chunks: Vec<_> = self.dirty_regions.dirty_chunks.drain().collect();
        for chunk_coord in dirty_chunks {
            let Some(buffers) = self.chunk_buffers.get(&chunk_coord) else {
                continue;
            };
            let Some(indices) = self.dirty_regions.dirty_voxel_indices.remove(&chunk_coord) else {
                continue;
            };
            let world_origin = chunk_to_world_min(chunk_coord);
            for index in indices {
                let local = idx_to_local(index);
                let world = VoxelCoord {
                    x: world_origin.x + local[0] as i32,
                    y: world_origin.y + local[1] as i32,
                    z: world_origin.z + local[2] as i32,
                };
                store.set_voxel(world, buffers.occupancy_material[index]);
            }
        }
    }
}

fn movement_candidates(index: usize, phase: Phase) -> Vec<usize> {
    let [x, y, z] = idx_to_local(index);
    let (x, y, z) = (x as i32, y as i32, z as i32);
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
        .filter_map(|(dx, dy, dz)| {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= CHUNK_EDGE as i32
                || ny >= CHUNK_EDGE as i32
                || nz >= CHUNK_EDGE as i32
            {
                return None;
            }
            Some(linear_idx([nx as u32, ny as u32, nz as u32]))
        })
        .collect()
}

fn linear_idx(local: [u32; 3]) -> usize {
    local[0] as usize + local[1] as usize * CHUNK_EDGE + local[2] as usize * CHUNK_EDGE * CHUNK_EDGE
}

fn idx_to_local(index: usize) -> [u32; 3] {
    let plane = CHUNK_EDGE * CHUNK_EDGE;
    let z = index / plane;
    let rem = index % plane;
    let y = rem / CHUNK_EDGE;
    let x = rem % CHUNK_EDGE;
    [x as u32, y as u32, z as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powder_prefers_downward_candidates() {
        let idx = linear_idx([10, 10, 10]);
        let candidates = movement_candidates(idx, Phase::Powder);
        assert_eq!(candidates.first().copied(), Some(linear_idx([10, 9, 10])));
    }

    #[test]
    fn gas_candidates_include_upward_motion() {
        let idx = linear_idx([12, 12, 12]);
        let candidates = movement_candidates(idx, Phase::Gas);
        assert!(candidates.contains(&linear_idx([12, 13, 12])));
    }
}
