//! Mesh artifact authority domain.
//!
//! - Candidate artifacts are produced by meshing workers.
//! - Render adoption/finalization ownership is tracked separately in
//!   `engine::render` and `engine::gpu_residency`.

use crate::engine::mesh::LodLevel;
use crate::engine::world::ChunkVersion;
use crate::types::ChunkCoord;

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshArtifactKey {
    pub chunk_id: ChunkCoord,
    pub chunk_version: ChunkVersion,
    pub lod: LodLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MeshArtifactHandle(pub u64);

impl MeshArtifactHandle {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArtifactCandidateState {
    pub key: MeshArtifactKey,
    pub handle: MeshArtifactHandle,
    pub pending_finalize: bool,
    pub task_id: u64,
}

#[inline]
pub fn debug_assert_candidate_finalize_state(candidate: &ArtifactCandidateState, finalized: bool) {
    debug_assert!(
        candidate.pending_finalize || !finalized,
        "forbidden state: candidate cannot finalize without pending_finalize=true"
    );
}

impl fmt::Display for MeshArtifactHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
