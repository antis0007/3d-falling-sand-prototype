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
        region: &HashSet<ChunkCoord>,
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
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        self.sim_world
            .step_region(store, region, center, rng, metadata)
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, _material_id: u16) {
        self.sim_world.notify_voxel_edit(coord);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationMode {
    CpuCellular,
    GpuFluid,
    CpuFallback,
}

#[derive(Default)]
pub struct CpuFallbackBackend {
    cpu: CpuCellularBackend,
}

impl SimulationBackend for CpuFallbackBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        self.cpu.step(store, region, center, rng, metadata)
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, material_id: u16) {
        self.cpu.queue_place_edit(coord, material_id);
    }
}

#[derive(Default)]
pub struct SimulationRuntime {
    cpu: CpuCellularBackend,
    gpu: GpuFluidBackend,
    fallback: CpuFallbackBackend,
    warned_gpu_emulation: bool,
    active_emitter_chunks: HashSet<ChunkCoord>,
}

impl SimulationRuntime {
    pub fn queue_place_edit(&mut self, mode: SimulationMode, coord: VoxelCoord, material_id: u16) {
        self.active_emitter_chunks
            .insert(crate::types::voxel_to_chunk(coord).0);
        match mode {
            SimulationMode::CpuCellular => self.cpu.queue_place_edit(coord, material_id),
            SimulationMode::GpuFluid => self.gpu.queue_place_edit(coord, material_id),
            SimulationMode::CpuFallback => self.fallback.queue_place_edit(coord, material_id),
        }
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
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        match mode {
            SimulationMode::CpuCellular => self.cpu.step(store, region, center, rng, metadata),
            SimulationMode::GpuFluid => {
                if !self.warned_gpu_emulation {
                    log::warn!(
                        "GPU simulation mode is currently running deterministic CPU emulation; enable the `gpu-compute` feature for WGSL compute pipelines"
                    );
                    self.warned_gpu_emulation = true;
                }
                self.gpu.step(store, region, center, rng, metadata)
            }
            SimulationMode::CpuFallback => self.fallback.step(store, region, center, rng, metadata),
        }
    }
}
