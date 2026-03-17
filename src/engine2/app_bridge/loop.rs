//! Minimal compile-safe engine2 entry point.

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
    let _loop_state = EngineLoop::default();
    Ok(BootAction::RunLegacyApp)
}
