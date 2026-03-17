//! Engine2 is the new GPU-first voxel engine path.
//! Phase 1 provides compile-safe module boundaries and stubs.

pub mod app_bridge;
pub mod commands;
pub mod gpu;
pub mod phases;
pub mod render;
pub mod sim;
pub mod types;
pub mod world;

use std::sync::Arc;

use crate::engine2::commands::CommandQueue;
use crate::engine2::gpu::buffers::BufferPoolConfig;
use crate::engine2::gpu::Engine2Gpu;
use crate::engine2::phases::{EditOutput, SimInput, SimOutput, UploadOutput};
use crate::engine2::render::camera::CameraState;
use crate::engine2::render::draw::{DrawInput, DrawPacket, Engine2Drawer};
use crate::engine2::render::extract::{Engine2Extractor, ExtractInput, ExtractOutput};
use crate::engine2::sim::edit_apply::EditApplier;
use crate::engine2::sim::scheduler::SimScheduler;
use crate::engine2::world::procgen::ProcgenInterface;
use crate::engine2::world::residency::{ResidencyDecision, ResidencyStateMap};
use crate::engine2::world::storage::WorldStorage;

/// Root object for the engine2 world path.
///
/// Ownership boundary:
/// - CPU residency/storage remain the cold-state authority for what should be loaded.
/// - Engine2 GPU owns canonical mutable hot-state once uploads are initialized.
#[derive(Debug, Default)]
pub struct Engine2State {
    pub residency: ResidencyStateMap,
    pub commands: CommandQueue,
    pub storage: WorldStorage,
    pub procgen: ProcgenInterface,
    pub gpu: Engine2Gpu,
    pub extractor: Engine2Extractor,
    pub drawer: Engine2Drawer,
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

    /// Frame phase 1: resolve sparse residency transitions.
    pub fn residency_update_step(&mut self) {
        if !self.gpu.is_initialized() {
            return;
        }

        let decisions = self.residency.collect_active_decisions();
        for decision in decisions {
            match decision {
                ResidencyDecision::Load(key) => {
                    let pending_materialization = self.storage.take_pending_materialization(key);
                    let revision = pending_materialization
                        .as_ref()
                        .map(|staged| staged.revision)
                        .unwrap_or_else(|| self.storage.resident_revision(key));
                    if let Some(page) = self.gpu.enqueue_resident_brick(
                        key,
                        revision,
                        pending_materialization.map(|staged| staged.payload),
                        false,
                    ) {
                        self.residency.mark_resident(key, Some(page));
                    }
                }
                ResidencyDecision::Evict(key) => {
                    self.gpu.evict_brick(key);
                    self.residency.mark_unloaded(key);
                }
            }
        }
    }

    /// Frame phase 2: flush staged uploads/page-table metadata to GPU buffers.
    pub fn upload_step(&mut self) -> UploadOutput {
        self.gpu.flush_uploads()
    }

    /// Frame phase 3: apply pending edit commands.
    pub fn command_application_step(&mut self) -> EditOutput {
        self.edit_applier.apply_pending_edits(
            &mut self.commands,
            &mut self.residency,
            &mut self.gpu,
            &mut self.scheduler,
        )
    }

    /// Frame phase 4: advance active/sleep scheduling.
    pub fn active_scheduling_step(&mut self, sim_input: SimInput) -> SimOutput {
        self.scheduler.advance_frame(sim_input)
    }

    /// Frame phase 5: run engine2 extraction ownership.
    pub fn render_extraction_step(&mut self, camera: CameraState, sim: SimOutput) -> ExtractOutput {
        let extract_input = ExtractInput { camera, sim };
        self.extractor.extract(extract_input)
    }

    /// Frame phase 6: prepare draw packet and upload indirect draw arguments.
    pub fn draw_step(&mut self, extract_output: ExtractOutput) -> DrawPacket {
        let draw_packet = self.drawer.prepare(DrawInput::from(extract_output));
        self.gpu.set_draw_indirect_count(draw_packet.indirect_count);
        if let Some(gpu_handles) = self.gpu.context_and_buffers() {
            self.drawer.upload_indirect(gpu_handles, &draw_packet);
        }
        draw_packet
    }

    pub fn queue_upload_step(&mut self, sim_output: &SimOutput) {
        self.gpu.upload_sim_queues(sim_output);
    }
}
