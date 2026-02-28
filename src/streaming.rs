use std::collections::HashSet;

use crate::types::ChunkCoord;

pub struct StreamingState {
    resident: HashSet<ChunkCoord>,
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            resident: HashSet::new(),
        }
    }

    pub fn ensure_resident(&mut self, _region: impl IntoIterator<Item = ChunkCoord>) {
        todo!("Chunk residency management will be implemented in a follow-up PR")
    }

    pub fn get_resident_set(&self) -> &HashSet<ChunkCoord> {
        &self.resident
    }
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}
