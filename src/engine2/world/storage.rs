//! CPU cold-state world storage.
//!
//! Ownership boundary: this module tracks known brick metadata and materialization/upload
//! sources, but it is not authoritative for mutable resident voxel state.
//! Once a brick upload is flushed, GPU resident pages are canonical for hot state.

use std::collections::HashMap;

use crate::engine2::types::BrickKey;
use crate::engine2::world::brick::{BrickMeta, BrickPayload};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationSource {
    Procedural,
    Persisted,
}

/// One-shot staging payload produced by a materialization source.
///
/// This is an upload initializer, not a resident mutable copy.
#[derive(Debug, Clone)]
pub struct PendingMaterialization {
    pub revision: u32,
    pub source: MaterializationSource,
    pub payload: BrickPayload,
}

/// Known state for a brick that exists in world space but may not currently be resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownBrickState {
    Unknown,
    Procedural,
    Persisted,
}

#[derive(Debug, Clone)]
pub struct NonResidentBrick {
    pub meta: BrickMeta,
    pub known_state: KnownBrickState,
}

#[derive(Debug, Clone)]
pub struct ResidentBrickRecord {
    pub meta: BrickMeta,
    pub source: MaterializationSource,
    /// Optional CPU-side snapshot for debugging/checkpointing only.
    ///
    /// Non-authoritative: once uploaded, GPU page state is canonical.
    pub debug_snapshot: Option<BrickPayload>,
}

#[derive(Debug, Default)]
pub struct WorldStorage {
    pub nonresident: HashMap<BrickKey, NonResidentBrick>,
    pub resident: HashMap<BrickKey, ResidentBrickRecord>,
    pending_materialization: HashMap<BrickKey, PendingMaterialization>,
}

impl WorldStorage {
    pub fn note_nonresident(&mut self, brick: NonResidentBrick) {
        self.nonresident.insert(brick.meta.key, brick);
    }

    pub fn stage_materialized_brick(
        &mut self,
        key: BrickKey,
        payload: BrickPayload,
        source: MaterializationSource,
    ) {
        let mut meta = self
            .nonresident
            .remove(&key)
            .map(|record| record.meta)
            .unwrap_or_else(|| BrickMeta::new(key));
        meta.revision = meta.revision.saturating_add(1);
        let revision = meta.revision as u32;
        self.resident.insert(
            key,
            ResidentBrickRecord {
                meta,
                source,
                debug_snapshot: None,
            },
        );
        self.pending_materialization.insert(
            key,
            PendingMaterialization {
                revision,
                source,
                payload,
            },
        );
    }

    pub fn take_pending_materialization(
        &mut self,
        key: BrickKey,
    ) -> Option<PendingMaterialization> {
        self.pending_materialization.remove(&key)
    }

    pub fn resident_revision(&self, key: BrickKey) -> u32 {
        self.resident
            .get(&key)
            .map(|record| record.meta.revision as u32)
            .unwrap_or(0)
    }

    pub fn mark_unloaded(&mut self, key: BrickKey, known_state: KnownBrickState) {
        self.pending_materialization.remove(&key);
        if let Some(record) = self.resident.remove(&key) {
            self.nonresident.insert(
                key,
                NonResidentBrick {
                    meta: record.meta,
                    known_state,
                },
            );
        }
    }
}
