//! Engine2 is the new GPU-first voxel engine path.
//! Phase 1 provides compile-safe module boundaries and stubs.

pub mod app_bridge;
pub mod commands;
pub mod gpu;
pub mod render;
pub mod sim;
pub mod types;
pub mod world;

use std::sync::Arc;

use crate::engine2::commands::CommandQueue;
use crate::engine2::gpu::buffers::BufferPoolConfig;
use crate::engine2::gpu::Engine2Gpu;
use crate::engine2::sim::edit_apply::EditApplier;
use crate::engine2::sim::scheduler::SimScheduler;
use crate::engine2::world::procgen::ProcgenInterface;
use crate::engine2::world::residency::{ResidencyDecision, ResidencyStateMap};
use crate::engine2::world::storage::WorldStorage;

/// Root object for the engine2 world path.
///
/// Ownership boundary:
/// - CPU residency/storage remain the cold-state authority for what should be loaded.
/// - Engine2 GPU owns hot resident page layout, upload staging, and active/dirty/remesh queues.
#[derive(Debug, Default)]
pub struct Engine2State {
    pub residency: ResidencyStateMap,
    pub commands: CommandQueue,
    pub storage: WorldStorage,
    pub procgen: ProcgenInterface,
    pub gpu: Engine2Gpu,
    pub edit_applier: EditApplier,
    pub scheduler: SimScheduler,
}

impl Engine2State {
    pub fn initialize_gpu(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: BufferPoolConfig,
    ) {
        self.gpu.initialize(device, queue, config);
    }

    /// Applies CPU residency transitions into engine2 GPU hot-state.
    ///
    /// In phase 3 this is intentionally minimal and upload/queue focused.
    pub fn sync_gpu_hot_state(&mut self) {
        if !self.gpu.is_initialized() {
            return;
        }

        let decisions = self.residency.collect_active_decisions();
        for decision in decisions {
            match decision {
                ResidencyDecision::Load(key) => {
                    let resident_record = self.storage.resident.get(&key);
                    let payload = resident_record.and_then(|record| record.payload.clone());
                    let revision = resident_record
                        .map(|record| record.meta.revision as u32)
                        .unwrap_or(0);
                    if let Some(page) = self
                        .gpu
                        .enqueue_resident_brick(key, revision, payload, false)
                    {
                        self.residency.mark_resident(key, Some(page));
                    }
                }
                ResidencyDecision::Evict(key) => {
                    self.gpu.evict_brick(key);
                    self.residency.mark_unloaded(key);
                }
            }
        }

        self.edit_applier.apply_pending_edits(
            &mut self.commands,
            &mut self.residency,
            &mut self.gpu,
            &mut self.scheduler,
        );
        self.scheduler.advance_frame(&mut self.gpu);

        self.gpu.flush();
    }
}
