//! Brick-key to GPU-page table.

use std::collections::HashMap;

use crate::engine2::types::{BrickKey, GpuPage, GpuPageHandle};
use crate::engine2::world::residency::{ResidencyDecision, ResidencyStateMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageAllocation {
    pub key: BrickKey,
    pub page: GpuPageHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageEviction {
    pub key: BrickKey,
    pub page: GpuPageHandle,
}

#[derive(Debug, Default)]
pub struct BrickPageTable {
    pages: HashMap<BrickKey, GpuPage>,
    free_slots: Vec<u32>,
    capacity: u32,
}

impl BrickPageTable {
    pub fn with_capacity(capacity: u32) -> Self {
        let mut free_slots = Vec::with_capacity(capacity as usize);
        for idx in (0..capacity).rev() {
            free_slots.push(idx);
        }
        Self {
            pages: HashMap::new(),
            free_slots,
            capacity,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn page_for(&self, key: BrickKey) -> Option<GpuPageHandle> {
        self.pages.get(&key).copied()
    }

    pub fn allocate(&mut self, key: BrickKey) -> Option<GpuPageHandle> {
        if let Some(page) = self.pages.get(&key).copied() {
            return Some(page);
        }
        let slot = self.free_slots.pop()?;
        let page = GpuPage(slot);
        self.pages.insert(key, page);
        Some(page)
    }

    pub fn evict(&mut self, key: BrickKey) -> Option<GpuPageHandle> {
        let page = self.pages.remove(&key)?;
        self.free_slots.push(page.0);
        Some(page)
    }

    pub fn map_residency_decisions(
        &mut self,
        decisions: &[ResidencyDecision],
    ) -> (Vec<PageAllocation>, Vec<PageEviction>) {
        let mut allocs = Vec::new();
        let mut evictions = Vec::new();

        for decision in decisions {
            match decision {
                ResidencyDecision::Load(key) => {
                    if let Some(page) = self.allocate(*key) {
                        allocs.push(PageAllocation { key: *key, page });
                    }
                }
                ResidencyDecision::Evict(key) => {
                    if let Some(page) = self.evict(*key) {
                        evictions.push(PageEviction { key: *key, page });
                    }
                }
            }
        }

        (allocs, evictions)
    }

    pub fn sync_resident_mappings(&self, residency: &mut ResidencyStateMap) {
        for (key, page) in &self.pages {
            residency.mark_resident(*key, Some(*page));
        }
    }
}
