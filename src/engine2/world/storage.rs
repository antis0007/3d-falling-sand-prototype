//! CPU cold-state world storage.
//!
//! Ownership boundary: this module tracks known brick metadata and optional loaded payloads,
//! but does not decide residency policy. Residency decisions are owned by
//! `world::residency`.

use std::collections::HashMap;

use crate::engine2::types::BrickKey;
use crate::engine2::world::brick::{BrickMeta, BrickPayload};

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
    pub payload: Option<BrickPayload>,
}

#[derive(Debug, Default)]
pub struct WorldStorage {
    pub nonresident: HashMap<BrickKey, NonResidentBrick>,
    pub resident: HashMap<BrickKey, ResidentBrickRecord>,
}

impl WorldStorage {
    pub fn note_nonresident(&mut self, brick: NonResidentBrick) {
        self.nonresident.insert(brick.meta.key, brick);
    }

    pub fn insert_resident_payload(&mut self, key: BrickKey, payload: BrickPayload) {
        let mut meta = self
            .nonresident
            .remove(&key)
            .map(|record| record.meta)
            .unwrap_or_else(|| BrickMeta::new(key));
        meta.revision = meta.revision.saturating_add(1);
        self.resident.insert(
            key,
            ResidentBrickRecord {
                meta,
                payload: Some(payload),
            },
        );
    }

    pub fn mark_unloaded(&mut self, key: BrickKey, known_state: KnownBrickState) {
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
