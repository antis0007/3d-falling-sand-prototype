//! Minimal compile-safe engine2 entry point.

use crate::engine2::render::camera::CameraState;
use crate::engine2::Engine2State;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootAction {
    RunLegacyApp,
}

#[derive(Debug, Default)]
pub struct EngineLoop {
    pub state: Engine2State,
}

pub async fn run() -> anyhow::Result<BootAction> {
    let mut loop_state = EngineLoop::default();
    let _packet = loop_state
        .state
        .prepare_render_packet(CameraState::from_world_position([0.0, 0.0, 0.0], 256.0));
    Ok(BootAction::RunLegacyApp)
}
