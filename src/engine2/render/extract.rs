//! GPU extraction pass input.

use crate::engine2::render::camera::CameraState;

#[derive(Debug, Default)]
pub struct ExtractInput {
    pub camera: CameraState,
}
