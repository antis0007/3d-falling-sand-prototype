//! Shared brick-domain constants and metadata.

use crate::engine2::types::{BrickDim, BrickKey, BRICK_VOLUME_VOXELS};

pub const BRICK_EDGE: u32 = BrickDim::EDGE_U32;
pub const BRICK_VOXEL_CAPACITY: usize = BRICK_VOLUME_VOXELS as usize;

/// Lightweight CPU metadata for a brick known to the world model.
#[derive(Debug, Clone)]
pub struct BrickMeta {
    pub key: BrickKey,
    pub revision: u64,
    pub known_non_empty: bool,
}

impl BrickMeta {
    pub fn new(key: BrickKey) -> Self {
        Self {
            key,
            revision: 0,
            known_non_empty: false,
        }
    }
}

/// Placeholder payload for future materialized brick voxel data.
#[derive(Debug, Clone)]
pub struct BrickPayload {
    pub material_ids: Vec<u16>,
}

impl Default for BrickPayload {
    fn default() -> Self {
        Self {
            material_ids: vec![0; BRICK_VOXEL_CAPACITY],
        }
    }
}
