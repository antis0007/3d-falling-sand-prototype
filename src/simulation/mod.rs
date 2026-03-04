use std::collections::HashSet;

use crate::chunk_store::ChunkStore;
use crate::sim_world::{Rng, SimWorld};
use crate::types::{ChunkCoord, VoxelCoord};

mod gpu_fluid;

pub use gpu_fluid::GpuFluidBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationPhaseClass {
    SolidsLiquidsPowders,
    Gas,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimulationStepMetadata {
    pub phase_class: Option<SimulationPhaseClass>,
    pub boundary_dissipation_strength: f32,
    pub core_radius_chunks: i32,
    pub max_voxels_to_process: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimulationStepStats {
    pub stepped_chunks: usize,
    pub skipped_chunks: usize,
    pub boundary_dissipated_particles: usize,
    pub processed_frontier_voxels: usize,
    pub skipped_active_chunks: usize,
    pub avg_chunk_wait_ticks: f32,
}

pub trait SimulationBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        active_chunks: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats;

    fn queue_place_edit(&mut self, _coord: VoxelCoord, _material_id: u16) {}
}

#[derive(Default)]
pub struct CpuCellularBackend {
    sim_world: SimWorld,
}

impl SimulationBackend for CpuCellularBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        _active_chunks: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        self.sim_world.step_active(store, center, rng, metadata)
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, _material_id: u16) {
        self.sim_world.notify_voxel_edit(coord);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationMode {
    CpuCellular,
    GpuFluid,
}

#[derive(Default)]
pub struct SimulationRuntime {
    cpu: CpuCellularBackend,
    gpu: GpuFluidBackend,
    active_emitter_chunks: HashSet<ChunkCoord>,
    active_chunk_set: HashSet<ChunkCoord>,
}

impl SimulationRuntime {
    pub fn queue_place_edit(&mut self, mode: SimulationMode, coord: VoxelCoord, material_id: u16) {
        let edited_chunk = crate::types::voxel_to_chunk(coord).0;
        self.active_emitter_chunks.insert(edited_chunk);
        self.active_chunk_set.insert(edited_chunk);
        // Keep all backends aware of user edits so mode switches do not strand queued writes
        // in a single backend and appear as frozen particles.
        self.cpu.queue_place_edit(coord, material_id);
        self.gpu.queue_place_edit(coord, material_id);
        let _ = mode;
    }

    pub fn active_emitter_chunks(&self) -> &HashSet<ChunkCoord> {
        &self.active_emitter_chunks
    }

    pub fn reset_active_emitters(&mut self) {
        self.active_emitter_chunks.clear();
    }

    pub fn step(
        &mut self,
        mode: SimulationMode,
        store: &mut ChunkStore,
        center: ChunkCoord,
        rng: &mut Rng,
        mut metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        if metadata.max_voxels_to_process == 0 {
            metadata.max_voxels_to_process = 8_192;
        }
        let stats = match mode {
            SimulationMode::CpuCellular => {
                self.cpu
                    .step(store, &self.active_chunk_set, center, rng, metadata)
            }
            SimulationMode::GpuFluid => {
                self.gpu
                    .step(store, &self.active_chunk_set, center, rng, metadata)
            }
        };
        self.active_chunk_set = self.cpu.sim_world.active_chunks_snapshot();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::ChunkStore;
    use crate::sim::XorShift32;
    use crate::types::{ChunkCoord, VoxelCoord};
    use crate::world::EMPTY;
    #[test]
    fn gpu_mode_consumes_queued_edits_and_moves_material() {
        let mut runtime = SimulationRuntime::default();
        let mut store = ChunkStore::new();
        let mut rng = XorShift32::new(7);
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        runtime.queue_place_edit(SimulationMode::GpuFluid, VoxelCoord { x: 4, y: 4, z: 4 }, 3);

        let _stats = runtime.step(
            SimulationMode::GpuFluid,
            &mut store,
            center,
            &mut rng,
            SimulationStepMetadata {
                max_voxels_to_process: 8192,
                ..SimulationStepMetadata::default()
            },
        );

        assert_eq!(store.get_voxel(VoxelCoord { x: 4, y: 4, z: 4 }), EMPTY);
        let mut found_sand = false;
        for y in 0..=4 {
            for z in 3..=5 {
                for x in 3..=5 {
                    if store.get_voxel(VoxelCoord { x, y, z }) == 3 {
                        found_sand = true;
                    }
                }
            }
        }
        assert!(
            found_sand,
            "queued GPU edit was not simulated into the store"
        );
    }
}
