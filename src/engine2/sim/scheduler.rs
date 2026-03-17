//! High-level simulation scheduler (CPU cold control only).

use std::collections::{BTreeSet, HashMap};

use crate::engine2::phases::{SimInput, SimOutput};

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

    pub fn advance_frame(&mut self, input: SimInput) -> SimOutput {
        self.tick_index = self.tick_index.saturating_add(1);

        for page in input.upload.active_pages {
            self.wake_brick(page);
        }
        for page in input.edit.wake_pages {
            self.wake_brick(page);
        }

        self.active_this_frame.clear();
        self.active_this_frame
            .extend(self.next_frame_active.iter().copied());
        self.next_frame_active.clear();

        let mut sim_output = SimOutput {
            active_pages: Vec::with_capacity(self.active_this_frame.len()),
            dirty_pages: input.upload.dirty_pages,
            remesh_pages: input.upload.remesh_pages,
        };
        sim_output
            .remesh_pages
            .extend(sim_output.dirty_pages.iter().copied());

        for page in input.edit.dirty_pages {
            sim_output.dirty_pages.push(page);
            sim_output.remesh_pages.push(page);
        }

        for page in &self.active_this_frame {
            sim_output.active_pages.push(*page);

            let quiet_entry = self.quiet_ticks.entry(*page).or_default();
            *quiet_entry = quiet_entry.saturating_add(1);

            if *quiet_entry < self.sleep_after_quiet_ticks {
                self.next_frame_active.insert(*page);
            } else {
                self.quiet_ticks.remove(page);
            }
        }

        sim_output
    }

    pub fn active_this_frame(&self) -> &[u32] {
        &self.active_this_frame
    }
}
