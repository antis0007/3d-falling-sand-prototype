//! Engine2 is the new GPU-first voxel engine path.
//! Phase 1 provides compile-safe module boundaries and stubs.

pub mod app_bridge;
pub mod commands;
pub mod gpu;
pub mod render;
pub mod sim;
pub mod types;
pub mod world;

/// Root object for the new engine core.
#[derive(Debug, Default)]
pub struct Engine2Core {
    pub residency_epoch: u64,
}
