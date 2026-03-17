//! Minimal compile-safe engine2 entry point.

use crate::engine2::Engine2Core;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootAction {
    RunLegacyApp,
}

#[derive(Debug, Default)]
pub struct EngineLoop {
    pub core: Engine2Core,
}

pub async fn run() -> anyhow::Result<BootAction> {
    let _loop_state = EngineLoop::default();
    Ok(BootAction::RunLegacyApp)
}
