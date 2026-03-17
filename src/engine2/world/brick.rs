//! Sparse brick storage record.

use crate::engine2::types::BrickKey;

#[derive(Debug, Clone)]
pub struct BrickRecord {
    pub key: BrickKey,
    pub storage_id: u64,
}
