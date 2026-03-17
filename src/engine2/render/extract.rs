//! GPU extraction pass input and dirty-brick extraction output.

use std::collections::BTreeSet;

use crate::engine2::phases::SimOutput;
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
    pub sim: SimOutput,
}

impl Default for ExtractInput {
    fn default() -> Self {
        Self {
            camera: CameraState::default(),
            sim: SimOutput::default(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExtractOutput {
    pub extracted: Vec<ExtractedPrimitive>,
    pub extracted_pages: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct Engine2Extractor;

impl Engine2Extractor {
    pub fn extract(&mut self, input: ExtractInput) -> ExtractOutput {
        let _camera = input.camera;

        let mut page_set = BTreeSet::new();
        for page in input.sim.active_pages {
            page_set.insert(page);
        }
        for page in input.sim.remesh_pages {
            page_set.insert(page);
        }

        let mut extracted = Vec::with_capacity(page_set.len());
        let mut extracted_pages = Vec::with_capacity(page_set.len());
        for page_slot in page_set {
            // Placeholder geometry for phase 4: one cube (36 vertices) per changed brick.
            extracted.push(ExtractedPrimitive {
                page_slot,
                vertex_count: 36,
            });
            extracted_pages.push(page_slot);
        }

        ExtractOutput {
            extracted,
            extracted_pages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine2Extractor, ExtractInput};
    use crate::engine2::phases::SimOutput;

    #[test]
    fn consumes_sim_remesh_pages_into_extracted_primitives() {
        let mut extractor = Engine2Extractor;
        let output = extractor.extract(ExtractInput {
            sim: SimOutput {
                remesh_pages: vec![9, 7],
                ..SimOutput::default()
            },
            ..ExtractInput::default()
        });

        assert_eq!(output.extracted.len(), 2);
        assert_eq!(output.extracted_pages, vec![7, 9]);
    }
}
