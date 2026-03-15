//! Authoritative world-domain identity and snapshot tokens.
//! This module owns compile-time distinctions between mutable chunk truth
//! and immutable snapshot truth consumed by meshing/render pipelines.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ChunkVersion(pub u64);

impl ChunkVersion {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for ChunkVersion {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SnapshotEpoch(pub u64);

impl SnapshotEpoch {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SnapshotId(pub u64);

impl SnapshotId {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ChunkVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for SnapshotEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
