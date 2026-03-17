//! Engine2-owned app loop bridge.

use std::future::Future;

use crate::engine2::render::camera::CameraState;
use crate::engine2::Engine2State;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyDependency {
    LegacyStreamingAndProcgen,
    LegacyRendererAndPresentation,
    LegacyInputAndUiTooling,
}

#[derive(Debug, Default)]
pub struct EngineLoop {
    pub state: Engine2State,
}

impl EngineLoop {
    /// Orchestrates one engine2 frame with explicit phase ordering.
    pub fn tick_frame(&mut self, camera: CameraState) {
        self.state.residency_update_step();
        self.state.upload_step();
        self.state.command_application_step();
        self.state.active_scheduling_step();
        let extract = self.state.render_extraction_step(camera);
        let _draw = self.state.draw_step(extract);
    }
}

pub async fn run<F, Fut>(legacy_fallback: F) -> anyhow::Result<()>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    let mut loop_state = EngineLoop::default();
    loop_state.tick_frame(CameraState::from_world_position([0.0, 0.0, 0.0], 256.0));

    for dependency in remaining_legacy_dependencies() {
        log::info!("[engine2] phase5 legacy dependency: {:?}", dependency);
    }

    legacy_fallback().await
}

pub fn remaining_legacy_dependencies() -> &'static [LegacyDependency] {
    &[
        LegacyDependency::LegacyStreamingAndProcgen,
        LegacyDependency::LegacyRendererAndPresentation,
        LegacyDependency::LegacyInputAndUiTooling,
    ]
}
