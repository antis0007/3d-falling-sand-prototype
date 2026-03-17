//! Cold storage catalog for streamed bricks.

use crate::engine2::types::BrickKey;

#[derive(Debug, Default)]
pub struct StorageCatalog {
    pub queued_loads: Vec<BrickKey>,
}
