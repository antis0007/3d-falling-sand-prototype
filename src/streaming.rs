use std::collections::{HashMap, HashSet, VecDeque};

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

#[derive(Debug, Default, Clone, Copy)]
pub struct StreamingUpdateStats {
    pub newly_desired: usize,
    pub queued_generate: usize,
    pub queued_evict: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DesiredChunks {
    pub guaranteed: Vec<ChunkCoord>,
    pub render: Vec<ChunkCoord>,
    pub prefetch: Vec<ChunkCoord>,
    pub generation_order: Vec<ChunkCoord>,
    pub resident_keep: HashSet<ChunkCoord>,
}

#[derive(Debug)]
pub struct ChunkStreaming {
    pub seed: u64,
    pub resident: HashSet<ChunkCoord>,
    pub generating: HashSet<ChunkCoord>,
    pending_generate: VecDeque<ChunkCoord>,
    pending_evict: VecDeque<ChunkCoord>,
    evict_not_desired_since: HashMap<ChunkCoord, u64>,
    queued_evict_set: HashSet<ChunkCoord>,
    pub max_generate_schedule_per_update: usize,
    pub max_evict_schedule_per_update: usize,
    pub eviction_linger_frames: u64,
    work_items: Vec<WorkItem>,
}

impl ChunkStreaming {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            resident: HashSet::new(),
            generating: HashSet::new(),
            pending_generate: VecDeque::new(),
            pending_evict: VecDeque::new(),
            evict_not_desired_since: HashMap::new(),
            queued_evict_set: HashSet::new(),
            max_generate_schedule_per_update: 24,
            max_evict_schedule_per_update: 24,
            eviction_linger_frames: 24,
            work_items: Vec::new(),
        }
    }

    pub fn sort_key(player_chunk: ChunkCoord, coord: ChunkCoord) -> (i64, i32, i32, i32) {
        let dx = i64::from(coord.x - player_chunk.x);
        let dy = i64::from(coord.y - player_chunk.y);
        let dz = i64::from(coord.z - player_chunk.z);
        (dx * dx + dy * dy + dz * dz, coord.x, coord.y, coord.z)
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
        guaranteed_radius_xz: i32,
        render_radius_xz: i32,
        prefetch_radius_xz: Option<i32>,
        vertical_radius: i32,
    ) -> DesiredChunks {
        let guaranteed_radius = guaranteed_radius_xz.max(0);
        let render_radius = render_radius_xz.max(guaranteed_radius);
        let prefetch_radius = prefetch_radius_xz
            .unwrap_or(render_radius)
            .max(render_radius);
        let ry = vertical_radius.max(0);

        let guaranteed = Self::ring_sorted_region(player_chunk, guaranteed_radius, ry, 0);
        let render =
            Self::ring_sorted_region(player_chunk, render_radius, ry, guaranteed_radius + 1);
        let prefetch =
            Self::ring_sorted_region(player_chunk, prefetch_radius, ry, render_radius + 1);

        let mut generation_order =
            Vec::with_capacity(guaranteed.len() + render.len() + prefetch.len());
        generation_order.extend(guaranteed.iter().copied());
        generation_order.extend(render.iter().copied());
        generation_order.extend(prefetch.iter().copied());

        let mut resident_keep = HashSet::with_capacity(guaranteed.len() + render.len());
        resident_keep.extend(guaranteed.iter().copied());
        resident_keep.extend(render.iter().copied());

        DesiredChunks {
            guaranteed,
            render,
            prefetch,
            generation_order,
            resident_keep,
        }
    }

    fn ring_sorted_region(
        player_chunk: ChunkCoord,
        radius_xz: i32,
        vertical_radius: i32,
        start_ring: i32,
    ) -> Vec<ChunkCoord> {
        let radius = radius_xz.max(0);
        let start = start_ring.max(0).min(radius + 1);
        let mut out = Vec::new();

        for ring in start..=radius {
            let mut ring_coords = Vec::new();
            for dz in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs().max(dz.abs()) != ring {
                        continue;
                    }
                    for dy in -vertical_radius..=vertical_radius {
                        ring_coords.push(ChunkCoord {
                            x: player_chunk.x + dx,
                            y: player_chunk.y + dy,
                            z: player_chunk.z + dz,
                        });
                    }
                }
            }
            ring_coords.sort_by_key(|&coord| Self::sort_key(player_chunk, coord));
            out.extend(ring_coords);
        }

        out
    }

    pub fn update(
        &mut self,
        desired_sorted: &[ChunkCoord],
        resident_keep: &HashSet<ChunkCoord>,
        player_chunk: ChunkCoord,
        frame_index: u64,
    ) -> StreamingUpdateStats {
        let mut stats = StreamingUpdateStats::default();

        for &coord in desired_sorted
            .iter()
            .take(self.max_generate_schedule_per_update)
        {
            if self.resident.contains(&coord) || self.generating.contains(&coord) {
                continue;
            }
            self.generating.insert(coord);
            self.pending_generate.push_back(coord);
            self.work_items.push(WorkItem::Generate(coord));
            stats.queued_generate += 1;
        }

        stats.newly_desired = desired_sorted
            .iter()
            .filter(|coord| !self.resident.contains(coord) && !self.generating.contains(coord))
            .count();

        let mut resident_sorted: Vec<ChunkCoord> = self.resident.iter().copied().collect();
        resident_sorted.sort_by_key(|&coord| Self::sort_key(player_chunk, coord));

        for coord in resident_sorted
            .into_iter()
            .rev()
            .take(self.max_evict_schedule_per_update)
        {
            if resident_keep.contains(&coord) || self.generating.contains(&coord) {
                self.evict_not_desired_since.remove(&coord);
                continue;
            }

            let since = self
                .evict_not_desired_since
                .entry(coord)
                .or_insert(frame_index);
            if frame_index.saturating_sub(*since) < self.eviction_linger_frames {
                continue;
            }
            if self.queued_evict_set.insert(coord) {
                self.pending_evict.push_back(coord);
                self.work_items.push(WorkItem::Evict(coord));
                stats.queued_evict += 1;
            }
        }

        stats
    }

    pub fn drain_generate_requests(&mut self, limit: usize) -> Vec<ChunkCoord> {
        let take = limit.min(self.pending_generate.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(coord) = self.pending_generate.pop_front() {
                out.push(coord);
            }
        }
        out
    }

    pub fn drain_evict_requests(&mut self, limit: usize) -> Vec<ChunkCoord> {
        let take = limit.min(self.pending_evict.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(coord) = self.pending_evict.pop_front() {
                self.queued_evict_set.remove(&coord);
                out.push(coord);
            }
        }
        out
    }

    pub fn pending_generate_count(&self) -> usize {
        self.pending_generate.len()
    }

    pub fn pending_evict_count(&self) -> usize {
        self.pending_evict.len()
    }

    pub fn mark_generated(&mut self, coord: ChunkCoord) {
        self.generating.remove(&coord);
        self.resident.insert(coord);
    }

    pub fn mark_generation_dropped(&mut self, coord: ChunkCoord) {
        self.generating.remove(&coord);
    }

    pub fn mark_evicted(&mut self, coord: ChunkCoord) {
        self.generating.remove(&coord);
        self.resident.remove(&coord);
        self.evict_not_desired_since.remove(&coord);
        self.queued_evict_set.remove(&coord);
    }

    pub fn drain_work_items(&mut self) -> Vec<WorkItem> {
        std::mem::take(&mut self.work_items)
    }

    pub fn clear(&mut self) {
        self.resident.clear();
        self.generating.clear();
        self.pending_generate.clear();
        self.pending_evict.clear();
        self.evict_not_desired_since.clear();
        self.queued_evict_set.clear();
        self.work_items.clear();
    }

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
