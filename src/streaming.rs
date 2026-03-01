use std::collections::{HashMap, HashSet, VecDeque};

use glam::{Vec3, Vec4};

use crate::types::ChunkCoord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Unloaded,
    Scheduled,
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
    pub near: Vec<ChunkCoord>,
    pub mid: Vec<ChunkCoord>,
    pub far: Vec<ChunkCoord>,
    pub generation_order: Vec<ChunkCoord>,
    pub generation_scores: HashMap<ChunkCoord, f32>,
    pub resident_keep: HashSet<ChunkCoord>,
}

#[derive(Debug, Clone)]
pub struct VisibilityContext {
    pub camera_pos_chunks: Vec3,
    pub cone_inner_cos: f32,
    pub cone_outer_cos: f32,
    pub frustum_planes: Option<[Vec4; 6]>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChunkLifecycle {
    last_visible_frame: u64,
    last_generated_frame: u64,
    last_evicted_frame: u64,
}

#[derive(Debug)]
pub struct ChunkStreaming {
    pub seed: u64,
    pub resident: HashSet<ChunkCoord>,
    pub scheduled_generate: HashSet<ChunkCoord>,
    pub dispatched_generate: HashSet<ChunkCoord>,
    pending_generate: VecDeque<ChunkCoord>,
    pending_evict: VecDeque<ChunkCoord>,
    evict_not_desired_since: HashMap<ChunkCoord, u64>,
    queued_evict_set: HashSet<ChunkCoord>,
    desired_resident_keep: HashSet<ChunkCoord>,
    chunk_lifecycle: HashMap<ChunkCoord, ChunkLifecycle>,
    pub max_generate_schedule_per_update: usize,
    pub max_evict_schedule_per_update: usize,
    pub eviction_linger_frames: u64,
    pub boundary_eviction_linger_frames: u64,
    pub regen_cooldown_frames: u64,
    work_items: Vec<WorkItem>,
}

impl ChunkStreaming {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            resident: HashSet::new(),
            scheduled_generate: HashSet::new(),
            dispatched_generate: HashSet::new(),
            pending_generate: VecDeque::new(),
            pending_evict: VecDeque::new(),
            evict_not_desired_since: HashMap::new(),
            queued_evict_set: HashSet::new(),
            desired_resident_keep: HashSet::new(),
            chunk_lifecycle: HashMap::new(),
            max_generate_schedule_per_update: 36,
            max_evict_schedule_per_update: 24,
            eviction_linger_frames: 24,
            boundary_eviction_linger_frames: 96,
            regen_cooldown_frames: 24,
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
        } else if self.scheduled_generate.contains(&coord) {
            Residency::Scheduled
        } else if self.dispatched_generate.contains(&coord) {
            Residency::Generating
        } else {
            Residency::Unloaded
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn desired_set(
        player_chunk: ChunkCoord,
        player_velocity: Vec3,
        view_dir: Vec3,
        visibility: Option<&VisibilityContext>,
        near_radius_xz: i32,
        mid_radius_xz: i32,
        far_radius_xz: Option<i32>,
        vertical_radius: i32,
        resident_keep_mid_cap: usize,
        resident_keep_far_cap: usize,
        last_visible_frame: &HashMap<ChunkCoord, u64>,
        frame_index: u64,
    ) -> DesiredChunks {
        let near_radius = near_radius_xz.max(0);
        let mid_radius = mid_radius_xz.max(near_radius);
        let far_radius = far_radius_xz.unwrap_or(mid_radius).max(mid_radius);
        let ry = vertical_radius.max(0);

        let near = Self::ring_sorted_region(player_chunk, near_radius, ry, 0);
        let mid = Self::ring_sorted_region(player_chunk, mid_radius, ry, near_radius + 1);
        let far = Self::ring_sorted_region(player_chunk, far_radius, ry, mid_radius + 1);

        let mut weighted = Vec::with_capacity(near.len() + mid.len() + far.len());
        let view_dir = view_dir.normalize_or_zero();
        let velocity_dir = player_velocity.normalize_or_zero();
        let (cone_inner_cos, cone_outer_cos, frustum_planes, camera_pos_chunks) = visibility
            .map(|ctx| {
                (
                    ctx.cone_inner_cos,
                    ctx.cone_outer_cos,
                    ctx.frustum_planes.as_ref(),
                    ctx.camera_pos_chunks,
                )
            })
            .unwrap_or((
                0.65,
                0.1,
                None,
                Vec3::new(
                    player_chunk.x as f32,
                    player_chunk.y as f32,
                    player_chunk.z as f32,
                ),
            ));

        let near_set: HashSet<_> = near.iter().copied().collect();
        let mid_set: HashSet<_> = mid.iter().copied().collect();
        for coord in near
            .iter()
            .copied()
            .chain(mid.iter().copied())
            .chain(far.iter().copied())
        {
            let dx = i64::from(coord.x - player_chunk.x);
            let dy = i64::from(coord.y - player_chunk.y);
            let dz = i64::from(coord.z - player_chunk.z);
            let distance2 = dx * dx + dy * dy + dz * dz;
            let to_chunk = Vec3::new(dx as f32, dy as f32, dz as f32).normalize_or_zero();
            let view_alignment = view_dir.dot(to_chunk).max(0.0);
            let velocity_alignment = velocity_dir.dot(to_chunk).max(0.0);
            let cone_weight = if cone_inner_cos > cone_outer_cos {
                ((view_alignment - cone_outer_cos) / (cone_inner_cos - cone_outer_cos))
                    .clamp(0.0, 1.0)
            } else {
                1.0
            };

            let frustum_weight = if let Some(planes) = frustum_planes {
                let chunk_center = Vec3::new(
                    coord.x as f32 + 0.5,
                    coord.y as f32 + 0.5,
                    coord.z as f32 + 0.5,
                );
                let relative = chunk_center - camera_pos_chunks;
                let radius = 0.866_025_4;
                let mut min_margin = f32::INFINITY;
                for plane in planes {
                    let margin = plane.truncate().dot(relative) + plane.w;
                    min_margin = min_margin.min(margin);
                }
                if min_margin < -radius {
                    0.0
                } else {
                    ((min_margin + radius) / (radius * 2.0)).clamp(0.0, 1.0)
                }
            } else {
                1.0
            };

            let recent_age = last_visible_frame
                .get(&coord)
                .map(|last| frame_index.saturating_sub(*last))
                .unwrap_or(u64::MAX);
            let recent_visibility = if recent_age == u64::MAX {
                0.0
            } else {
                (1.0 - (recent_age as f32 / 120.0)).clamp(0.0, 1.0)
            };
            let cheb = (coord.x - player_chunk.x)
                .abs()
                .max((coord.y - player_chunk.y).abs())
                .max((coord.z - player_chunk.z).abs());
            let lod_need = if cheb <= near_radius {
                1.0
            } else if cheb <= mid_radius {
                0.65
            } else {
                0.35
            };
            let score = (1.0 / (1.0 + distance2 as f32)) * 0.55
                + (cone_weight * 0.72 + velocity_alignment * 0.28) * 0.25
                + frustum_weight * 0.08
                + recent_visibility * 0.12
                + lod_need * 0.08;
            weighted.push((
                coord,
                score,
                distance2,
                cone_weight,
                recent_visibility,
                lod_need,
            ));
        }
        weighted.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| b.3.total_cmp(&a.3))
                .then_with(|| b.4.total_cmp(&a.4))
                .then_with(|| b.5.total_cmp(&a.5))
                .then_with(|| a.0.x.cmp(&b.0.x))
                .then_with(|| a.0.y.cmp(&b.0.y))
                .then_with(|| a.0.z.cmp(&b.0.z))
        });

        let mut generation_order = Vec::with_capacity(weighted.len());
        let mut generation_scores = HashMap::with_capacity(weighted.len());
        for (coord, score, ..) in weighted {
            generation_order.push(coord);
            generation_scores.insert(coord, score);
        }

        let mut resident_keep = HashSet::with_capacity(
            near.len()
                + resident_keep_mid_cap.min(mid.len())
                + resident_keep_far_cap.min(far.len()),
        );
        resident_keep.extend(near.iter().copied());

        let mut mid_kept = 0usize;
        let mut far_kept = 0usize;
        for &coord in &generation_order {
            if resident_keep.contains(&coord) {
                continue;
            }
            if mid_set.contains(&coord) {
                if mid_kept < resident_keep_mid_cap {
                    resident_keep.insert(coord);
                    mid_kept += 1;
                }
            } else if !near_set.contains(&coord) && far_kept < resident_keep_far_cap {
                resident_keep.insert(coord);
                far_kept += 1;
            }
            if mid_kept >= resident_keep_mid_cap && far_kept >= resident_keep_far_cap {
                break;
            }
        }

        DesiredChunks {
            near,
            mid,
            far,
            generation_order,
            generation_scores,
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
        let mut queued_this_frame = 0usize;

        stats.newly_desired = desired_sorted
            .iter()
            .filter(|coord| {
                !self.resident.contains(coord)
                    && !self.dispatched_generate.contains(coord)
                    && !self.scheduled_generate.contains(coord)
            })
            .count();

        self.desired_resident_keep = resident_keep.clone();
        for &coord in resident_keep {
            self.cancel_queued_evict(coord);
            self.evict_not_desired_since.remove(&coord);
        }

        for &coord in desired_sorted {
            let lifecycle = self.chunk_lifecycle.entry(coord).or_default();
            lifecycle.last_visible_frame = frame_index;
            if self.resident.contains(&coord)
                || self.dispatched_generate.contains(&coord)
                || self.scheduled_generate.contains(&coord)
            {
                continue;
            }
            if lifecycle.last_evicted_frame > 0
                && frame_index.saturating_sub(lifecycle.last_evicted_frame)
                    < self.regen_cooldown_frames
            {
                continue;
            }
            if queued_this_frame >= self.max_generate_schedule_per_update {
                continue;
            }
            self.scheduled_generate.insert(coord);
            self.pending_generate.push_back(coord);
            self.work_items.push(WorkItem::Generate(coord));
            queued_this_frame += 1;
            stats.queued_generate += 1;
        }

        let mut resident_sorted: Vec<ChunkCoord> = self.resident.iter().copied().collect();
        resident_sorted.sort_by_key(|&coord| Self::sort_key(player_chunk, coord));

        for coord in resident_sorted
            .into_iter()
            .rev()
            .take(self.max_evict_schedule_per_update)
        {
            if resident_keep.contains(&coord) || self.dispatched_generate.contains(&coord) {
                self.evict_not_desired_since.remove(&coord);
                continue;
            }

            let since = self
                .evict_not_desired_since
                .entry(coord)
                .or_insert(frame_index);
            let lifecycle = self.chunk_lifecycle.entry(coord).or_default();
            let linger = if frame_index.saturating_sub(lifecycle.last_visible_frame)
                < self.boundary_eviction_linger_frames
            {
                self.boundary_eviction_linger_frames
                    .max(self.eviction_linger_frames)
            } else {
                self.eviction_linger_frames
            };
            if frame_index.saturating_sub(*since) < linger {
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
                self.mark_dispatch_succeeded(coord);
                out.push(coord);
            }
        }
        out
    }

    pub fn next_generation_job(&self) -> Option<ChunkCoord> {
        self.pending_generate.front().copied()
    }

    pub fn mark_dispatch_succeeded(&mut self, coord: ChunkCoord) {
        if let Some(pos) = self
            .pending_generate
            .iter()
            .position(|queued| *queued == coord)
        {
            self.pending_generate.remove(pos);
        }
        self.scheduled_generate.remove(&coord);
        self.dispatched_generate.insert(coord);
    }

    pub fn mark_dispatch_failed_or_deferred(&mut self, coord: ChunkCoord) {
        if !self.scheduled_generate.contains(&coord) {
            self.scheduled_generate.insert(coord);
        }
        if !self.pending_generate.contains(&coord) {
            self.pending_generate.push_back(coord);
        }
    }

    pub fn drain_evict_requests(&mut self, limit: usize) -> Vec<ChunkCoord> {
        let mut out = Vec::with_capacity(limit.min(self.pending_evict.len()));
        while out.len() < limit {
            if let Some(coord) = self.pending_evict.pop_front() {
                self.queued_evict_set.remove(&coord);
                if self.is_valid_evict(coord) {
                    out.push(coord);
                }
            } else {
                break;
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

    pub fn mark_generated(&mut self, coord: ChunkCoord, frame_index: u64) {
        self.dispatched_generate.remove(&coord);
        self.scheduled_generate.remove(&coord);
        self.resident.insert(coord);
        self.chunk_lifecycle
            .entry(coord)
            .or_default()
            .last_generated_frame = frame_index;
    }

    pub fn mark_generation_dropped(&mut self, coord: ChunkCoord) {
        self.dispatched_generate.remove(&coord);
        self.scheduled_generate.remove(&coord);
        self.pending_generate.retain(|queued| *queued != coord);
    }

    pub fn mark_canceled(&mut self, coord: ChunkCoord) {
        self.dispatched_generate.remove(&coord);
        self.scheduled_generate.remove(&coord);
        self.pending_generate.retain(|queued| *queued != coord);
    }

    pub fn mark_evicted(&mut self, coord: ChunkCoord, frame_index: u64) {
        self.dispatched_generate.remove(&coord);
        self.scheduled_generate.remove(&coord);
        self.pending_generate.retain(|queued| *queued != coord);
        self.resident.remove(&coord);
        self.evict_not_desired_since.remove(&coord);
        self.queued_evict_set.remove(&coord);
        self.desired_resident_keep.remove(&coord);
        self.chunk_lifecycle
            .entry(coord)
            .or_default()
            .last_evicted_frame = frame_index;
    }

    pub fn drain_work_items(&mut self) -> Vec<WorkItem> {
        std::mem::take(&mut self.work_items)
    }

    pub fn clear(&mut self) {
        self.resident.clear();
        self.scheduled_generate.clear();
        self.dispatched_generate.clear();
        self.pending_generate.clear();
        self.pending_evict.clear();
        self.evict_not_desired_since.clear();
        self.queued_evict_set.clear();
        self.desired_resident_keep.clear();
        self.chunk_lifecycle.clear();
        self.work_items.clear();
    }

    pub fn reset_with_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.clear();
    }

    pub fn last_visible_frames(&self) -> HashMap<ChunkCoord, u64> {
        self.chunk_lifecycle
            .iter()
            .filter_map(|(coord, life)| {
                if life.last_visible_frame > 0 {
                    Some((*coord, life.last_visible_frame))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl ChunkStreaming {
    fn cancel_queued_evict(&mut self, coord: ChunkCoord) {
        if !self.queued_evict_set.remove(&coord) {
            return;
        }
        self.pending_evict.retain(|queued| *queued != coord);
        self.work_items
            .retain(|item| !matches!(item, WorkItem::Evict(c) if *c == coord));
    }

    fn is_valid_evict(&self, coord: ChunkCoord) -> bool {
        self.resident.contains(&coord)
            && !self.desired_resident_keep.contains(&coord)
            && !self.scheduled_generate.contains(&coord)
            && !self.dispatched_generate.contains(&coord)
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::types::ChunkCoord;

    use super::ChunkStreaming;
    use super::Residency;

    #[test]
    fn scheduling_budget_limits_queued_chunks_not_scan_count() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 1;

        let near_1 = ChunkCoord { x: 0, y: 0, z: 0 };
        let near_2 = ChunkCoord { x: 1, y: 0, z: 0 };
        let far = ChunkCoord { x: 4, y: 0, z: 0 };

        streaming.resident.insert(near_1);
        streaming.resident.insert(near_2);

        let desired_sorted = vec![near_1, near_2, far];
        let resident_keep: HashSet<_> = desired_sorted.iter().copied().collect();

        let stats = streaming.update(&desired_sorted, &resident_keep, near_1, 100);

        assert_eq!(stats.queued_generate, 1);
        assert!(streaming.scheduled_generate.contains(&far));
        assert_eq!(streaming.drain_generate_requests(8), vec![far]);
        assert!(streaming.dispatched_generate.contains(&far));
    }

    #[test]
    fn stale_queued_eviction_is_canceled_when_chunk_becomes_desired_again() {
        let mut streaming = ChunkStreaming::new(1);
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        streaming.eviction_linger_frames = 0;
        streaming.boundary_eviction_linger_frames = 0;
        streaming.max_evict_schedule_per_update = 8;
        streaming.resident.insert(c);

        streaming.update(&[], &HashSet::new(), c, 1);
        assert_eq!(streaming.pending_evict_count(), 1);

        let keep = HashSet::from([c]);
        streaming.update(&[c], &keep, c, 2);
        assert_eq!(streaming.pending_evict_count(), 0);
        assert!(streaming.drain_evict_requests(1).is_empty());
    }

    #[test]
    fn stale_evict_request_is_revalidated_at_drain_time() {
        let mut streaming = ChunkStreaming::new(1);
        let c = ChunkCoord { x: 1, y: 0, z: 0 };
        streaming.eviction_linger_frames = 0;
        streaming.boundary_eviction_linger_frames = 0;
        streaming.max_evict_schedule_per_update = 8;
        streaming.resident.insert(c);

        streaming.update(&[], &HashSet::new(), c, 1);
        assert_eq!(streaming.pending_evict_count(), 1);

        streaming.scheduled_generate.insert(c);
        assert!(streaming.drain_evict_requests(1).is_empty());
    }

    #[test]
    fn newly_desired_stat_is_computed_before_scheduling_mutation() {
        let mut streaming = ChunkStreaming::new(1);
        let a = ChunkCoord { x: 0, y: 0, z: 0 };
        let b = ChunkCoord { x: 2, y: 0, z: 0 };
        streaming.max_generate_schedule_per_update = 1;

        let desired = vec![a, b];
        let keep: HashSet<_> = desired.iter().copied().collect();
        let stats = streaming.update(&desired, &keep, a, 10);

        assert_eq!(stats.newly_desired, 2);
        assert_eq!(stats.queued_generate, 1);
    }

    #[test]
    fn regen_cooldown_does_not_block_never_evicted_chunks() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 4;
        streaming.regen_cooldown_frames = 120;

        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let desired = vec![c];
        let keep = HashSet::from([c]);
        let stats = streaming.update(&desired, &keep, c, 1);

        assert_eq!(stats.queued_generate, 1);
        assert!(streaming.scheduled_generate.contains(&c));
    }

    #[test]
    fn regen_cooldown_blocks_recently_evicted_chunk() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 4;
        streaming.regen_cooldown_frames = 10;

        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        streaming.mark_evicted(c, 5);

        let desired = vec![c];
        let keep = HashSet::from([c]);
        let blocked = streaming.update(&desired, &keep, c, 10);
        assert_eq!(blocked.queued_generate, 0);

        let allowed = streaming.update(&desired, &keep, c, 15);
        assert_eq!(allowed.queued_generate, 1);
    }

    #[test]
    fn residency_reports_scheduled_state() {
        let mut streaming = ChunkStreaming::new(1);
        let c = ChunkCoord { x: 3, y: 0, z: 0 };
        streaming.scheduled_generate.insert(c);
        assert_eq!(streaming.residency_of(c), Residency::Scheduled);
    }
}
