//! Residency source of truth for sparse 16^3 bricks.
//!
//! Ownership boundary: this module owns desired/current residency state transitions and
//! sparse per-brick residency metadata. It does not own payload storage or materialization.
//! GPU page state is canonical after resident pages are initialized.

use std::collections::{HashMap, HashSet};

use crate::engine2::types::{BrickKey, GpuPageHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyState {
    Unloaded,
    PendingLoad,
    Resident,
    Evicting,
}

#[derive(Debug, Clone)]
pub struct ResidencyEntry {
    pub state: ResidencyState,
    pub desired_count: u32,
    pub dirty: bool,
    pub gpu_page: Option<GpuPageHandle>,
    pub last_touched_epoch: u64,
}

impl ResidencyEntry {
    fn new(epoch: u64) -> Self {
        Self {
            state: ResidencyState::Unloaded,
            desired_count: 0,
            dirty: false,
            gpu_page: None,
            last_touched_epoch: epoch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyDecision {
    Load(BrickKey),
    Evict(BrickKey),
}

#[derive(Debug, Default)]
pub struct ResidencyStateMap {
    entries: HashMap<BrickKey, ResidencyEntry>,
    desired: HashSet<BrickKey>,
    epoch: u64,
}

impl ResidencyStateMap {
    pub fn request(&mut self, key: BrickKey) {
        self.epoch = self.epoch.saturating_add(1);
        self.desired.insert(key);
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| ResidencyEntry::new(self.epoch));
        entry.desired_count = entry.desired_count.saturating_add(1);
        if matches!(
            entry.state,
            ResidencyState::Unloaded | ResidencyState::Evicting
        ) {
            entry.state = ResidencyState::PendingLoad;
        }
        entry.last_touched_epoch = self.epoch;
    }

    pub fn release(&mut self, key: BrickKey) {
        self.epoch = self.epoch.saturating_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.desired_count = entry.desired_count.saturating_sub(1);
            if entry.desired_count == 0 {
                self.desired.remove(&key);
                if matches!(
                    entry.state,
                    ResidencyState::Resident | ResidencyState::PendingLoad
                ) {
                    entry.state = ResidencyState::Evicting;
                }
            }
            entry.last_touched_epoch = self.epoch;
        }
    }

    pub fn mark_resident(&mut self, key: BrickKey, gpu_page: Option<GpuPageHandle>) {
        self.epoch = self.epoch.saturating_add(1);
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| ResidencyEntry::new(self.epoch));
        entry.state = ResidencyState::Resident;
        entry.gpu_page = gpu_page;
        entry.last_touched_epoch = self.epoch;
    }

    pub fn mark_unloaded(&mut self, key: BrickKey) {
        self.epoch = self.epoch.saturating_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.state = ResidencyState::Unloaded;
            entry.gpu_page = None;
            entry.dirty = false;
            entry.last_touched_epoch = self.epoch;
        }
    }

    pub fn mark_dirty(&mut self, key: BrickKey) {
        self.epoch = self.epoch.saturating_add(1);
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| ResidencyEntry::new(self.epoch));
        entry.dirty = true;
        entry.last_touched_epoch = self.epoch;
    }

    pub fn entry(&self, key: BrickKey) -> Option<&ResidencyEntry> {
        self.entries.get(&key)
    }

    pub fn collect_active_decisions(&self) -> Vec<ResidencyDecision> {
        self.entries
            .iter()
            .filter_map(|(key, entry)| match entry.state {
                ResidencyState::PendingLoad => Some(ResidencyDecision::Load(*key)),
                ResidencyState::Evicting => Some(ResidencyDecision::Evict(*key)),
                ResidencyState::Unloaded | ResidencyState::Resident => None,
            })
            .collect()
    }

    pub fn desired_count(&self) -> usize {
        self.desired.len()
    }
}
