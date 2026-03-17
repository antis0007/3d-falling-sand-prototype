//! Brick-key to GPU-page table.

use std::collections::HashMap;

use crate::engine2::types::{BrickKey, GpuPage};

#[derive(Debug, Default)]
pub struct BrickPageTable {
    pub pages: HashMap<BrickKey, GpuPage>,
}
