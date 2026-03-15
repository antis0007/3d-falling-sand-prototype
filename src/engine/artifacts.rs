//! Mesh artifact authority domain.
//!
//! This module is the CPU-side source of truth for mesh artifact publication
//! lifecycle and identity. Runtime systems (meshing, renderer, residency glue)
//! must query this registry for artifact publication state instead of keeping a
//! competing lifecycle map.

use crate::engine::mesh::LodLevel;
use crate::engine::world::ChunkVersion;
use crate::types::ChunkCoord;

use std::collections::HashMap;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshArtifactPublicationState {
    Missing,
    Requested,
    Building,
    Published,
    Superseded,
    Retired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshArtifactPayloadKind {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshArtifactMetadata {
    pub task_id: u64,
    pub payload_kind: MeshArtifactPayloadKind,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshArtifactRecord {
    pub key: MeshArtifactKey,
    pub handle: Option<MeshArtifactHandle>,
    pub state: MeshArtifactPublicationState,
    pub metadata: Option<MeshArtifactMetadata>,
}

impl MeshArtifactRecord {
    #[inline]
    pub fn is_published(&self) -> bool {
        matches!(self.state, MeshArtifactPublicationState::Published)
    }
}

#[derive(Default)]
pub struct MeshArtifactRegistry {
    records: HashMap<MeshArtifactKey, MeshArtifactRecord>,
}

impl MeshArtifactRegistry {
    pub fn request_artifact_build(&mut self, key: MeshArtifactKey) -> MeshArtifactRecord {
        let entry = self.records.entry(key).or_insert(MeshArtifactRecord {
            key,
            handle: None,
            state: MeshArtifactPublicationState::Missing,
            metadata: None,
        });
        if matches!(
            entry.state,
            MeshArtifactPublicationState::Missing | MeshArtifactPublicationState::Retired
        ) {
            entry.state = MeshArtifactPublicationState::Requested;
            entry.handle = None;
            entry.metadata = None;
        }
        *entry
    }

    pub fn mark_artifact_building(&mut self, key: MeshArtifactKey) -> MeshArtifactRecord {
        let mut record = self.request_artifact_build(key);
        record.state = MeshArtifactPublicationState::Building;
        self.records.insert(key, record);
        record
    }

    /// Atomically publishes immutable artifact identity + metadata for `key`.
    /// Readers will only ever observe pre-publish state or fully-published state.
    pub fn publish_artifact(
        &mut self,
        key: MeshArtifactKey,
        handle: MeshArtifactHandle,
        metadata: MeshArtifactMetadata,
    ) -> MeshArtifactRecord {
        if let Some(existing) = self.records.get(&key) {
            debug_assert!(
                !(existing.is_published() && existing.handle != Some(handle)),
                "publish must not mutate an already-published artifact payload in place"
            );
        }
        let published = MeshArtifactRecord {
            key,
            handle: Some(handle),
            state: MeshArtifactPublicationState::Published,
            metadata: Some(metadata),
        };
        self.records.insert(key, published);
        debug_assert!(
            published.handle.is_some(),
            "published artifact requires a handle"
        );
        published
    }

    pub fn mark_superseded(&mut self, key: MeshArtifactKey) -> Option<MeshArtifactRecord> {
        let record = self.records.get_mut(&key)?;
        record.state = MeshArtifactPublicationState::Superseded;
        Some(*record)
    }

    pub fn retire_artifact(&mut self, key: MeshArtifactKey) -> Option<MeshArtifactRecord> {
        let record = self.records.get_mut(&key)?;
        record.state = MeshArtifactPublicationState::Retired;
        Some(*record)
    }

    pub fn latest_exact_artifact(&self, key: MeshArtifactKey) -> Option<MeshArtifactRecord> {
        self.records.get(&key).copied()
    }

    pub fn artifact_state(&self, key: MeshArtifactKey) -> MeshArtifactPublicationState {
        self.records
            .get(&key)
            .map(|record| record.state)
            .unwrap_or(MeshArtifactPublicationState::Missing)
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}

pub fn debug_assert_drawable_registry_state_valid(
    key: MeshArtifactKey,
    state: MeshArtifactPublicationState,
) {
    debug_assert!(
        !matches!(
            state,
            MeshArtifactPublicationState::Missing
                | MeshArtifactPublicationState::Superseded
                | MeshArtifactPublicationState::Retired
        ),
        "drawable/candidate cannot be valid while registry state is {:?} for key {:?}",
        state,
        key,
    );
}

impl fmt::Display for MeshArtifactHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
