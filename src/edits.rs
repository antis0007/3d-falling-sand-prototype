use crate::types::{MaterialId, VoxelCoord};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelEdit {
    pub coord: VoxelCoord,
    pub material: MaterialId,
}

#[derive(Default)]
pub struct EditJournal {
    entries: Vec<VoxelEdit>,
}

impl EditJournal {
    pub fn push(&mut self, edit: VoxelEdit) {
        self.entries.push(edit);
    }

    pub fn entries(&self) -> &[VoxelEdit] {
        &self.entries
    }
}
