//! Typed frame-phase handoff contracts for engine2 orchestration.

#[derive(Debug, Default, Clone)]
pub struct UploadOutput {
    pub active_pages: Vec<u32>,
    pub dirty_pages: Vec<u32>,
    pub remesh_pages: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
pub struct EditOutput {
    pub dirty_pages: Vec<u32>,
    pub wake_pages: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
pub struct SimInput {
    pub upload: UploadOutput,
    pub edit: EditOutput,
}

#[derive(Debug, Default, Clone)]
pub struct SimOutput {
    pub active_pages: Vec<u32>,
    pub dirty_pages: Vec<u32>,
    pub remesh_pages: Vec<u32>,
}

