use std::collections::HashSet;

use crate::types::ChunkCoord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Unloaded,
    Generating,
    Resident,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkItem {
    Generate(ChunkCoord),
    Evict(ChunkCoord),
}

#[derive(Debug)]
pub struct ChunkStreaming {
    pub seed: u64,
    pub resident: HashSet<ChunkCoord>,
    pub generating: HashSet<ChunkCoord>,
    pub max_generate_per_update: usize,
    pub max_evict_per_update: usize,
    work_items: Vec<WorkItem>,
}

impl ChunkStreaming {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            resident: HashSet::new(),
            generating: HashSet::new(),
            max_generate_per_update: 8,
            max_evict_per_update: 8,
            work_items: Vec::new(),
        }
    }

    pub fn residency_of(&self, coord: ChunkCoord) -> Residency {
        if self.resident.contains(&coord) {
            Residency::Resident
        } else if self.generating.contains(&coord) {
            Residency::Generating
        } else {
            Residency::Unloaded
        }
    }

    pub fn desired_set(
        player_chunk: ChunkCoord,
        sim_radius_xz: i32,
        render_radius_xz: i32,
        vertical_radius: i32,
    ) -> HashSet<ChunkCoord> {
        let mut desired = HashSet::new();
        let radius_xz = sim_radius_xz.max(render_radius_xz).max(0);
        let ry = vertical_radius.max(0);

        for dz in -radius_xz..=radius_xz {
            for dy in -ry..=ry {
                for dx in -radius_xz..=radius_xz {
                    desired.insert(ChunkCoord {
                        x: player_chunk.x + dx,
                        y: player_chunk.y + dy,
                        z: player_chunk.z + dz,
                    });
                }
            }
        }
        desired
    }

    pub fn update(&mut self, desired: &HashSet<ChunkCoord>) -> (Vec<ChunkCoord>, Vec<ChunkCoord>) {
        let mut to_generate = Vec::new();
        let mut to_evict = Vec::new();

        for &coord in desired {
            if to_generate.len() >= self.max_generate_per_update {
                break;
            }
            if self.resident.contains(&coord) || self.generating.contains(&coord) {
                continue;
            }
            self.generating.insert(coord);
            self.work_items.push(WorkItem::Generate(coord));
            to_generate.push(coord);
        }

        for &coord in &self.resident {
            if to_evict.len() >= self.max_evict_per_update {
                break;
            }
            if desired.contains(&coord) || self.generating.contains(&coord) {
                continue;
            }
            to_evict.push(coord);
        }

        for coord in &to_evict {
            self.resident.remove(coord);
            self.work_items.push(WorkItem::Evict(*coord));
        }

        (to_generate, to_evict)
    }

    pub fn mark_generated(&mut self, coord: ChunkCoord) {
        self.generating.remove(&coord);
        self.resident.insert(coord);
    }

    pub fn mark_evicted(&mut self, coord: ChunkCoord) {
        self.generating.remove(&coord);
        self.resident.remove(&coord);
    }

    pub fn drain_work_items(&mut self) -> Vec<WorkItem> {
        std::mem::take(&mut self.work_items)
    }
    pub fn clear(&mut self) {
        self.resident.clear();
        self.generating.clear();
        self.work_items.clear();
    }
    /// Optional convenience: clear and set a new seed.
    pub fn reset_with_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.clear();
    }
}

pub struct StreamingState {
    resident: HashSet<ChunkCoord>,
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            resident: HashSet::new(),
        }
    }

    pub fn ensure_resident(&mut self, region: impl IntoIterator<Item = ChunkCoord>) {
        self.resident.extend(region);
    }

    pub fn get_resident_set(&self) -> &HashSet<ChunkCoord> {
        &self.resident
    }
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}
