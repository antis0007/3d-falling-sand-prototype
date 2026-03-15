//! Meshing-domain identity types.
//! Ownership note: mesh artifacts are candidates only; they do not imply
//! visible residency or GPU allocation ownership.

use crate::engine::artifacts::MeshArtifactKey;

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LodLevel(pub u8);

impl LodLevel {
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Debug assertion helper for candidate/finalize pipelines.
#[inline]
pub fn debug_assert_lod_matches_key(lod: LodLevel, key: MeshArtifactKey) {
    debug_assert_eq!(lod, key.lod, "lod must match mesh artifact key lod");
}

impl fmt::Display for LodLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
