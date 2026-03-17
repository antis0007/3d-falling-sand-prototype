//! GPU extraction pass input and dirty-brick extraction output.

use crate::engine2::gpu::queues::HotQueues;
use crate::engine2::render::camera::CameraState;

/// Minimal extracted surface primitive owned by engine2.
///
/// This phase emits one placeholder cube per extracted brick page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractedPrimitive {
    pub page_slot: u32,
    pub vertex_count: u32,
}

#[derive(Debug, Clone)]
pub struct ExtractInput {
    pub camera: CameraState,
}

impl Default for ExtractInput {
    fn default() -> Self {
        Self {
            camera: CameraState::default(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExtractOutput {
    pub extracted: Vec<ExtractedPrimitive>,
}

#[derive(Debug, Default)]
pub struct Engine2Extractor;

impl Engine2Extractor {
    pub fn extract(&mut self, queues: &mut HotQueues, input: &ExtractInput) -> ExtractOutput {
        let _camera = input.camera;

        // Dirty bricks are promoted into remesh work owned by engine2 extraction.
        queues.consume_dirty_into_remesh();
        let remesh_pages = queues.take_remesh_bricks();

        let mut extracted = Vec::with_capacity(remesh_pages.len());
        for page_slot in remesh_pages {
            // Placeholder geometry for phase 4: one cube (36 vertices) per changed brick.
            extracted.push(ExtractedPrimitive {
                page_slot,
                vertex_count: 36,
            });
            queues.push_extracted(page_slot);
        }

        ExtractOutput { extracted }
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine2Extractor, ExtractInput};
    use crate::engine2::gpu::queues::HotQueues;

    #[test]
    fn consumes_dirty_and_remesh_into_extracted_primitives() {
        let mut queues = HotQueues::default();
        queues.push_dirty(7);
        queues.push_remesh(9);

        let mut extractor = Engine2Extractor;
        let output = extractor.extract(&mut queues, &ExtractInput::default());

        assert_eq!(output.extracted.len(), 2);
        assert_eq!(queues.extracted_bricks(), &[9, 7]);
    }
}
