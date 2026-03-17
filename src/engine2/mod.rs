//! Engine2 is the new GPU-first voxel engine path.
//! Phase 1 provides compile-safe module boundaries and stubs.

pub mod app_bridge;
pub mod commands;
pub mod gpu;
pub mod render;
pub mod sim;
pub mod types;
pub mod world;

use crate::engine2::commands::CommandQueue;
use crate::engine2::world::procgen::ProcgenInterface;
use crate::engine2::world::residency::ResidencyStateMap;
use crate::engine2::world::storage::WorldStorage;

/// Root object for the engine2 cold-state world path.
///
/// Ownership boundary: this state owns CPU residency tracking, command staging,
/// and cold storage/procgen context. GPU hot-state allocation remains a future phase.
#[derive(Debug, Default)]
pub struct Engine2State {
    pub residency: ResidencyStateMap,
    pub commands: CommandQueue,
    pub storage: WorldStorage,
    pub procgen: ProcgenInterface,
}
