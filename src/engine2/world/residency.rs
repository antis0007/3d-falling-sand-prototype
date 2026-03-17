//! CPU-side cold residency metadata keyed by brick.

use std::collections::HashMap;

use crate::engine2::types::{BrickKey, GpuPage};

#[derive(Debug, Default)]
pub struct ResidencyMap {
    pub page_for_brick: HashMap<BrickKey, GpuPage>,
}
