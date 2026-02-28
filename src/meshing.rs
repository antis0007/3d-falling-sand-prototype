use crate::chunk_store::ChunkStore;
use crate::types::ChunkCoord;

pub struct MeshingSystem;

impl MeshingSystem {
    pub fn mesh_chunk(&self, _store: &ChunkStore, _coord: ChunkCoord) {
        todo!("Chunk meshing from ChunkStore will be implemented in a follow-up PR")
    }
}
