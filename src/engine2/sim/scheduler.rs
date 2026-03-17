//! High-level simulation scheduler (CPU cold control only).

use std::collections::{BTreeSet, HashMap};

use crate::engine2::gpu::Engine2Gpu;

#[derive(Debug)]
pub struct SimScheduler {
    pub tick_index: u64,
    pub sleep_after_quiet_ticks: u8,
    active_this_frame: Vec<u32>,
    next_frame_active: BTreeSet<u32>,
    quiet_ticks: HashMap<u32, u8>,
}

impl Default for SimScheduler {
    fn default() -> Self {
        Self {
            tick_index: 0,
            sleep_after_quiet_ticks: 3,
            active_this_frame: Vec::new(),
            next_frame_active: BTreeSet::new(),
            quiet_ticks: HashMap::new(),
        }
    }
}

impl SimScheduler {
    pub fn wake_brick(&mut self, page_slot: u32) {
        self.next_frame_active.insert(page_slot);
        self.quiet_ticks.insert(page_slot, 0);
    }

    pub fn advance_frame(&mut self, gpu: &mut Engine2Gpu) {
        self.tick_index = self.tick_index.saturating_add(1);
        self.active_this_frame.clear();
        self.active_this_frame
            .extend(self.next_frame_active.iter().copied());
        self.next_frame_active.clear();

        gpu.queues.clear_active();
        for page in &self.active_this_frame {
            gpu.queues.push_active(*page);

            let quiet_entry = self.quiet_ticks.entry(*page).or_default();
            *quiet_entry = quiet_entry.saturating_add(1);

            if *quiet_entry < self.sleep_after_quiet_ticks {
                self.next_frame_active.insert(*page);
            } else {
                self.quiet_ticks.remove(page);
            }
        }
    }

    pub fn active_this_frame(&self) -> &[u32] {
        &self.active_this_frame
    }
}
