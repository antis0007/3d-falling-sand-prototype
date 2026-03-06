use std::collections::{HashMap, HashSet, VecDeque};

use glam::{Vec3, Vec4};

use crate::types::ChunkCoord;

pub const URGENT_CHUNK_CHEBYSHEV_RADIUS: i32 = 1;
pub const URGENT_CHUNK_VERTICAL_RADIUS: i32 = 1;
const MID_RING_UPWARD_BIAS_BUDGET: i32 = 4;
const FAR_RING_UPWARD_BIAS_BUDGET: i32 = 8;
const DEPTH_PENALTY_START_DELTA_Y: i32 = 3;
const DEPTH_PENALTY_PER_CHUNK: f32 = 0.075;
const ABOVE_PLAYER_RING_BOOST: f32 = 0.12;
const HORIZONTAL_NEIGHBOR_SCHEDULE_FLOOR: usize = 4;

pub fn is_urgent_chunk(player_chunk: ChunkCoord, coord: ChunkCoord) -> bool {
    let chebyshev = (coord.x - player_chunk.x)
        .abs()
        .max((coord.y - player_chunk.y).abs())
        .max((coord.z - player_chunk.z).abs());
    chebyshev <= URGENT_CHUNK_CHEBYSHEV_RADIUS
        && (coord.y - player_chunk.y).abs() <= URGENT_CHUNK_VERTICAL_RADIUS
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerateJobClass {
    Urgent,
    Near,
    Mid,
    Far,
}

impl GenerateJobClass {
    fn priority_rank(self) -> u8 {
        match self {
            Self::Urgent => 3,
            Self::Near => 2,
            Self::Mid => 1,
            Self::Far => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct GenerateQueueMeta {
    class: Option<GenerateJobClass>,
    score: f32,
    enqueue_seq: u64,
    far_version: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StreamingUpdateStats {
    pub newly_desired: usize,
    pub queued_generate: usize,
    pub queued_evict: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QueueAgeStat {
    pub p50: u64,
    pub p95: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QueueAgeTelemetry {
    pub urgent: QueueAgeStat,
    pub near: QueueAgeStat,
    pub mid: QueueAgeStat,
    pub far: QueueAgeStat,
}

#[derive(Debug, Clone, Default)]
pub struct DesiredChunks {
    pub near: Vec<ChunkCoord>,
    pub mid: Vec<ChunkCoord>,
    pub far: Vec<ChunkCoord>,
    pub ultra: Vec<ChunkCoord>,
    pub generation_order: Vec<ChunkCoord>,
    pub generation_scores: HashMap<ChunkCoord, f32>,
    pub resident_keep: HashSet<ChunkCoord>,
}

#[derive(Debug, Clone)]
pub struct VisibilityContext {
    /// Camera position expressed in **chunk-space** coordinates (1.0 == one chunk edge).
    ///
    /// This must use the same basis as [`ChunkStreaming::desired_set`] chunk centers
    /// (`chunk_coord + 0.5`) so frustum weighting stays consistent.
    pub camera_pos_chunks: Vec3,
    pub cone_inner_cos: f32,
    pub cone_outer_cos: f32,
    /// Optional frustum planes in the same **chunk-space** basis as `camera_pos_chunks`.
    ///
    /// Plane equations are evaluated against vectors relative to `camera_pos_chunks` and
    /// chunk centers derived from integer chunk coordinates. Supplying world-space planes
    /// here can silently skew streaming priorities.
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
    generate_meta: HashMap<ChunkCoord, GenerateQueueMeta>,
    enqueue_seq: u64,
    far_generation_version: u64,
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
    pub fn reprioritize_generate_queue(
        &mut self,
        player_chunk: ChunkCoord,
        generation_scores: &HashMap<ChunkCoord, f32>,
    ) {
        if self.pending_generate.len() <= 1 {
            return;
        }

        let mut coords: Vec<_> = self.pending_generate.drain(..).collect();
        coords.sort_by(|a, b| {
            let meta_a = self.generate_meta.get(a).copied().unwrap_or_default();
            let meta_b = self.generate_meta.get(b).copied().unwrap_or_default();
            let urgent_a = is_urgent_chunk(player_chunk, *a);
            let urgent_b = is_urgent_chunk(player_chunk, *b);
            urgent_b
                .cmp(&urgent_a)
                .then_with(|| {
                    meta_b
                        .class
                        .map(GenerateJobClass::priority_rank)
                        .unwrap_or(0)
                        .cmp(
                            &meta_a
                                .class
                                .map(GenerateJobClass::priority_rank)
                                .unwrap_or(0),
                        )
                })
                .then_with(|| {
                    generation_scores
                        .get(b)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&generation_scores.get(a).copied().unwrap_or(0.0))
                })
                .then_with(|| meta_a.enqueue_seq.cmp(&meta_b.enqueue_seq))
                .then_with(|| {
                    Self::sort_key(player_chunk, *a).cmp(&Self::sort_key(player_chunk, *b))
                })
        });
        self.pending_generate.extend(coords);
    }

    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            resident: HashSet::new(),
            scheduled_generate: HashSet::new(),
            dispatched_generate: HashSet::new(),
            pending_generate: VecDeque::new(),
            generate_meta: HashMap::new(),
            enqueue_seq: 0,
            far_generation_version: 0,
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
        ultra_radius_xz: Option<i32>,
        vertical_radius: i32,
        resident_keep_mid_cap: usize,
        resident_keep_far_cap: usize,
        last_visible_frame: &HashMap<ChunkCoord, u64>,
        frame_index: u64,
    ) -> DesiredChunks {
        let near_radius = near_radius_xz.max(0);
        let mid_radius = mid_radius_xz.max(near_radius);
        let far_radius = far_radius_xz.unwrap_or(mid_radius).max(mid_radius);
        let ultra_radius = ultra_radius_xz.unwrap_or(far_radius).max(far_radius);
        let ry = vertical_radius.max(0);

        let near = Self::ring_sorted_region(player_chunk, near_radius, ry, 0, 0);
        let mid = Self::ring_sorted_region(
            player_chunk,
            mid_radius,
            ry,
            near_radius + 1,
            MID_RING_UPWARD_BIAS_BUDGET,
        );
        let far = Self::ring_sorted_region(
            player_chunk,
            far_radius,
            ry,
            mid_radius + 1,
            FAR_RING_UPWARD_BIAS_BUDGET,
        );
        let ultra = Self::ring_sorted_region(
            player_chunk,
            ultra_radius,
            ry,
            far_radius + 1,
            FAR_RING_UPWARD_BIAS_BUDGET,
        );

        let mut weighted = Vec::with_capacity(near.len() + mid.len() + far.len() + ultra.len());
        let view_dir = view_dir.normalize_or_zero();
        let velocity_dir = player_velocity.normalize_or_zero();
        let (cone_inner_cos, cone_outer_cos, frustum_planes_chunk_space, camera_pos_chunk_space) =
            visibility
                .map(|ctx| {
                    Self::debug_validate_visibility_context(player_chunk, ctx, far_radius, ry);
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
            .chain(ultra.iter().copied())
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

            let frustum_weight = if let Some(planes_chunk_space) = frustum_planes_chunk_space {
                // Chunk center in chunk-space coordinates (integer chunk index + half chunk).
                let chunk_center_chunk_space = Vec3::new(
                    coord.x as f32 + 0.5,
                    coord.y as f32 + 0.5,
                    coord.z as f32 + 0.5,
                );
                let camera_to_chunk_center_chunk_space =
                    chunk_center_chunk_space - camera_pos_chunk_space;
                let radius = 0.866_025_4;
                let mut min_margin = f32::INFINITY;
                for plane_chunk_space in planes_chunk_space {
                    let margin = plane_chunk_space
                        .truncate()
                        .dot(camera_to_chunk_center_chunk_space)
                        + plane_chunk_space.w;
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
            let dy_i32 = coord.y - player_chunk.y;
            let lod_need = if cheb <= near_radius {
                1.0
            } else if cheb <= mid_radius {
                0.75
            } else if cheb <= far_radius {
                0.45
            } else {
                0.2
            };
            let depth_penalty = if dy_i32 < -DEPTH_PENALTY_START_DELTA_Y {
                (-dy_i32 - DEPTH_PENALTY_START_DELTA_Y) as f32 * DEPTH_PENALTY_PER_CHUNK
            } else {
                0.0
            };
            let is_near_or_mid = cheb <= mid_radius;
            let above_player_boost = if dy_i32 >= 0 && is_near_or_mid {
                ABOVE_PLAYER_RING_BOOST * (0.7 + 0.3 * cone_weight)
            } else {
                0.0
            };
            let score = (1.0 / (1.0 + distance2 as f32)) * 0.55
                + (cone_weight * 0.72 + velocity_alignment * 0.28) * 0.25
                + frustum_weight * 0.08
                + recent_visibility * 0.12
                + lod_need * 0.08
                + above_player_boost
                - depth_penalty;
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

        let mut generation_scores = HashMap::with_capacity(weighted.len());
        let mut weighted_order = Vec::with_capacity(weighted.len());
        for (coord, score, ..) in weighted {
            generation_scores.insert(coord, score);
            weighted_order.push(coord);
        }

        // Keep urgent/near chunks at the front while preserving weighted order within each class.
        let mut generation_order = Vec::with_capacity(weighted_order.len());
        for &coord in &weighted_order {
            if near_set.contains(&coord) || is_urgent_chunk(player_chunk, coord) {
                generation_order.push(coord);
            }
        }
        for coord in weighted_order {
            if !near_set.contains(&coord) && !is_urgent_chunk(player_chunk, coord) {
                generation_order.push(coord);
            }
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
            ultra,
            generation_order,
            generation_scores,
            resident_keep,
        }
    }

    fn debug_validate_visibility_context(
        player_chunk: ChunkCoord,
        visibility: &VisibilityContext,
        far_radius_xz_chunks: i32,
        vertical_radius_chunks: i32,
    ) {
        debug_assert!(
            visibility.camera_pos_chunks.is_finite(),
            "VisibilityContext.camera_pos_chunks must be finite and in chunk-space units"
        );

        if let Some(planes_chunk_space) = visibility.frustum_planes.as_ref() {
            for plane_chunk_space in planes_chunk_space {
                let normal = plane_chunk_space.truncate();
                debug_assert!(
                    normal.is_finite() && plane_chunk_space.w.is_finite(),
                    "VisibilityContext.frustum_planes must be finite and in chunk-space units"
                );
                debug_assert!(
                    normal.length_squared() > f32::EPSILON,
                    "VisibilityContext.frustum_planes must have non-zero normals"
                );
            }
        }

        // Heuristic contract check: camera chunk-space position should remain near the player
        // chunk when both are expressed in chunk coordinates. This catches common mixed-space
        // usage (e.g., world meters passed as chunk units) during debug runs.
        let expected_camera_chunk_space = Vec3::new(
            player_chunk.x as f32,
            player_chunk.y as f32,
            player_chunk.z as f32,
        );
        let delta = visibility.camera_pos_chunks - expected_camera_chunk_space;
        let allowed_xz = far_radius_xz_chunks as f32 + 2.0;
        let allowed_y = vertical_radius_chunks as f32 + FAR_RING_UPWARD_BIAS_BUDGET as f32 + 2.0;
        debug_assert!(
            delta.x.abs() <= allowed_xz && delta.z.abs() <= allowed_xz && delta.y.abs() <= allowed_y,
            "VisibilityContext appears to mix spaces: expected camera_pos_chunks near player chunk in chunk-space"
        );
    }

    fn ring_sorted_region(
        player_chunk: ChunkCoord,
        radius_xz: i32,
        vertical_radius: i32,
        start_ring: i32,
        upward_bias_budget: i32,
    ) -> Vec<ChunkCoord> {
        let radius = radius_xz.max(0);
        let start = start_ring.max(0).min(radius + 1);
        let min_dy = -vertical_radius;
        let max_dy = vertical_radius + upward_bias_budget.max(0);
        let mut out = Vec::new();

        for ring in start..=radius {
            let mut ring_coords = Vec::new();
            let ring2 = ring * ring;
            let inner2 = (ring.saturating_sub(1)).pow(2);
            for dz in -ring..=ring {
                for dx in -ring..=ring {
                    let dist2 = dx * dx + dz * dz;
                    if dist2 > ring2 || (ring > 0 && dist2 <= inner2) {
                        continue;
                    }
                    for dy in min_dy..=max_dy {
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

        let try_schedule = |streaming: &mut Self, coord: ChunkCoord| -> bool {
            let lifecycle = streaming.chunk_lifecycle.entry(coord).or_default();
            lifecycle.last_visible_frame = frame_index;

            if streaming.resident.contains(&coord)
                || streaming.dispatched_generate.contains(&coord)
                || streaming.scheduled_generate.contains(&coord)
            {
                return false;
            }

            if !is_urgent_chunk(player_chunk, coord)
                && lifecycle.last_evicted_frame > 0
                && frame_index.saturating_sub(lifecycle.last_evicted_frame)
                    < streaming.regen_cooldown_frames
            {
                return false;
            }

            streaming.scheduled_generate.insert(coord);
            streaming.enqueue_generate(coord, GenerateJobClass::Far, 0.0);
            streaming.work_items.push(WorkItem::Generate(coord));
            true
        };

        let horizontal_neighbors: Vec<_> = desired_sorted
            .iter()
            .copied()
            .filter(|coord| Self::is_immediate_horizontal_neighbor(player_chunk, *coord))
            .collect();

        let reserved_neighbor_budget = HORIZONTAL_NEIGHBOR_SCHEDULE_FLOOR
            .min(horizontal_neighbors.len())
            .min(self.max_generate_schedule_per_update);

        for coord in horizontal_neighbors {
            if queued_this_frame >= reserved_neighbor_budget {
                break;
            }
            if try_schedule(self, coord) {
                queued_this_frame += 1;
                stats.queued_generate += 1;
            }
        }

        for &coord in desired_sorted {
            if queued_this_frame >= self.max_generate_schedule_per_update {
                break;
            }
            if try_schedule(self, coord) {
                queued_this_frame += 1;
                stats.queued_generate += 1;
            }
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

    pub fn next_generation_job(&mut self) -> Option<ChunkCoord> {
        let index = self.best_generate_index()?;
        self.pending_generate.remove(index)
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
        self.defer_generation_dispatch(coord, GenerateJobClass::Far);
    }

    pub fn dispatch_generation_for_class<F>(
        &mut self,
        coord: ChunkCoord,
        class: GenerateJobClass,
        mut try_dispatch: F,
    ) -> bool
    where
        F: FnMut(ChunkCoord) -> bool,
    {
        if self.resident.contains(&coord) || self.dispatched_generate.contains(&coord) {
            return false;
        }

        self.scheduled_generate.insert(coord);

        if try_dispatch(coord) {
            self.mark_dispatch_succeeded(coord);
            true
        } else {
            self.defer_generation_dispatch(coord, class);
            false
        }
    }

    fn defer_generation_dispatch(&mut self, coord: ChunkCoord, class: GenerateJobClass) {
        self.scheduled_generate.insert(coord);
        if self.pending_generate.contains(&coord) {
            return;
        }
        if matches!(class, GenerateJobClass::Urgent) {
            self.enqueue_seq = self.enqueue_seq.saturating_add(1);
            self.generate_meta.insert(
                coord,
                GenerateQueueMeta {
                    class: Some(class),
                    score: 0.0,
                    enqueue_seq: self.enqueue_seq,
                    far_version: self.far_generation_version,
                },
            );
            self.pending_generate.push_front(coord);
        } else {
            self.enqueue_generate(coord, class, 0.0);
        }
    }

    pub fn invalidate_far_jobs(&mut self) {
        self.far_generation_version = self.far_generation_version.saturating_add(1);
        let mut to_remove = Vec::new();
        for coord in &self.pending_generate {
            if let Some(meta) = self.generate_meta.get(coord) {
                if matches!(
                    meta.class,
                    Some(GenerateJobClass::Mid | GenerateJobClass::Far)
                ) {
                    to_remove.push(*coord);
                }
            }
        }
        for coord in to_remove {
            self.pending_generate.retain(|queued| *queued != coord);
            self.scheduled_generate.remove(&coord);
            self.generate_meta.remove(&coord);
        }
    }

    pub fn far_generation_version(&self) -> u64 {
        self.far_generation_version
    }

    pub fn queue_age_telemetry(&self) -> QueueAgeTelemetry {
        let mut urgent = Vec::new();
        let mut near = Vec::new();
        let mut mid = Vec::new();
        let mut far = Vec::new();
        for coord in &self.pending_generate {
            let meta = self.generate_meta.get(coord).copied().unwrap_or_default();
            let age = self.enqueue_seq.saturating_sub(meta.enqueue_seq);
            match meta.class.unwrap_or(GenerateJobClass::Far) {
                GenerateJobClass::Urgent => urgent.push(age),
                GenerateJobClass::Near => near.push(age),
                GenerateJobClass::Mid => mid.push(age),
                GenerateJobClass::Far => far.push(age),
            }
        }
        QueueAgeTelemetry {
            urgent: percentile_pair(&mut urgent),
            near: percentile_pair(&mut near),
            mid: percentile_pair(&mut mid),
            far: percentile_pair(&mut far),
        }
    }

    pub fn generation_version_for(&self, coord: ChunkCoord) -> u64 {
        self.generate_meta
            .get(&coord)
            .map(|meta| meta.far_version)
            .unwrap_or(self.far_generation_version)
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
        self.generate_meta.remove(&coord);
    }

    pub fn mark_evicted(&mut self, coord: ChunkCoord, frame_index: u64) {
        self.dispatched_generate.remove(&coord);
        self.scheduled_generate.remove(&coord);
        self.pending_generate.retain(|queued| *queued != coord);
        self.generate_meta.remove(&coord);
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
        self.generate_meta.clear();
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

fn percentile_pair(data: &mut Vec<u64>) -> QueueAgeStat {
    if data.is_empty() {
        return QueueAgeStat::default();
    }
    data.sort_unstable();
    let pick = |q: f32| -> u64 {
        let idx = ((data.len() - 1) as f32 * q).round() as usize;
        data[idx]
    };
    QueueAgeStat {
        p50: pick(0.50),
        p95: pick(0.95),
    }
}

impl ChunkStreaming {
    fn is_immediate_horizontal_neighbor(player_chunk: ChunkCoord, coord: ChunkCoord) -> bool {
        let dx = (coord.x - player_chunk.x).abs();
        let dz = (coord.z - player_chunk.z).abs();
        dx.max(dz) == 1 && coord.y == player_chunk.y
    }

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

    fn enqueue_generate(&mut self, coord: ChunkCoord, class: GenerateJobClass, score: f32) {
        self.enqueue_seq = self.enqueue_seq.saturating_add(1);
        self.generate_meta.insert(
            coord,
            GenerateQueueMeta {
                class: Some(class),
                score,
                enqueue_seq: self.enqueue_seq,
                far_version: self.far_generation_version,
            },
        );
        self.pending_generate.push_back(coord);
    }

    fn best_generate_index(&self) -> Option<usize> {
        self.pending_generate
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let meta_a = self.generate_meta.get(a).copied().unwrap_or_default();
                let meta_b = self.generate_meta.get(b).copied().unwrap_or_default();
                meta_a
                    .class
                    .map(GenerateJobClass::priority_rank)
                    .unwrap_or(0)
                    .cmp(
                        &meta_b
                            .class
                            .map(GenerateJobClass::priority_rank)
                            .unwrap_or(0),
                    )
                    .then_with(|| meta_a.enqueue_seq.cmp(&meta_b.enqueue_seq))
                    .then_with(|| meta_a.score.total_cmp(&meta_b.score))
            })
            .map(|(idx, _)| idx)
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
    use std::collections::{HashMap, HashSet};

    use glam::Vec3;

    use crate::types::ChunkCoord;

    use super::Residency;
    use super::{is_urgent_chunk, ChunkStreaming, GenerateJobClass};

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

        let player_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let c = ChunkCoord { x: 4, y: 0, z: 0 };
        assert!(!is_urgent_chunk(player_chunk, c));
        streaming.mark_evicted(c, 5);

        let desired = vec![c];
        let keep = HashSet::from([c]);
        let blocked = streaming.update(&desired, &keep, player_chunk, 10);
        assert_eq!(blocked.queued_generate, 0);

        let allowed = streaming.update(&desired, &keep, player_chunk, 15);
        assert_eq!(allowed.queued_generate, 1);
    }

    #[test]
    fn regen_cooldown_allows_recently_evicted_urgent_chunk() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 4;
        streaming.regen_cooldown_frames = 10;

        let player_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let urgent = ChunkCoord { x: 1, y: 0, z: 1 };
        assert!(is_urgent_chunk(player_chunk, urgent));
        streaming.mark_evicted(urgent, 5);

        let desired = vec![urgent];
        let keep = HashSet::from([urgent]);
        let stats = streaming.update(&desired, &keep, player_chunk, 10);

        assert_eq!(stats.queued_generate, 1);
        assert!(streaming.scheduled_generate.contains(&urgent));
    }

    #[test]
    fn deferred_urgent_dispatch_stays_queued_at_front() {
        let mut streaming = ChunkStreaming::new(1);
        let urgent = ChunkCoord { x: 1, y: 0, z: 1 };
        let other = ChunkCoord { x: 4, y: 0, z: 0 };

        streaming.pending_generate.push_back(other);

        assert!(!streaming.dispatch_generation_for_class(
            urgent,
            GenerateJobClass::Urgent,
            |_coord| false,
        ));

        assert!(streaming.scheduled_generate.contains(&urgent));
        assert_eq!(streaming.pending_generate.pop_front(), Some(urgent));
        assert_eq!(streaming.pending_generate.pop_front(), Some(other));
    }

    #[test]
    fn deferred_background_dispatch_remains_pending() {
        let mut streaming = ChunkStreaming::new(1);
        let coord = ChunkCoord { x: 5, y: 0, z: 0 };

        assert!(
            !streaming.dispatch_generation_for_class(coord, GenerateJobClass::Far, |_coord| false,)
        );

        assert!(streaming.scheduled_generate.contains(&coord));
        assert_eq!(streaming.next_generation_job(), Some(coord));
    }

    #[test]
    fn residency_reports_scheduled_state() {
        let mut streaming = ChunkStreaming::new(1);
        let c = ChunkCoord { x: 3, y: 0, z: 0 };
        streaming.scheduled_generate.insert(c);
        assert_eq!(streaming.residency_of(c), Residency::Scheduled);
    }

    #[test]
    fn reprioritize_queue_promotes_urgent_and_high_score_chunks() {
        let mut streaming = ChunkStreaming::new(1);
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let far = ChunkCoord { x: 8, y: 0, z: 0 };
        let near = ChunkCoord { x: 2, y: 0, z: 0 };
        let urgent = ChunkCoord { x: 1, y: 0, z: 1 };
        streaming.pending_generate.push_back(far);
        streaming.pending_generate.push_back(near);
        streaming.pending_generate.push_back(urgent);

        let mut scores = HashMap::new();
        scores.insert(far, 0.8);
        scores.insert(near, 0.2);
        scores.insert(urgent, 0.1);

        streaming.reprioritize_generate_queue(player, &scores);

        assert_eq!(streaming.pending_generate.pop_front(), Some(urgent));
        assert_eq!(streaming.pending_generate.pop_front(), Some(far));
        assert_eq!(streaming.pending_generate.pop_front(), Some(near));
    }

    #[test]
    fn desired_set_keeps_surface_adjacent_chunks_when_player_is_low() {
        let player = ChunkCoord { x: 0, y: -10, z: 0 };
        let desired = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            None,
            1,
            3,
            Some(5),
            Some(7),
            3,
            32,
            32,
            &HashMap::new(),
            0,
        );

        let near_surface = ChunkCoord { x: 0, y: 0, z: 4 };
        let deep_below = ChunkCoord { x: 0, y: -13, z: 4 };

        assert!(desired.generation_scores.contains_key(&near_surface));
        assert!(desired.generation_scores.contains_key(&deep_below));
        assert!(desired.generation_scores[&near_surface] > desired.generation_scores[&deep_below]);

        let near_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == near_surface)
            .unwrap();
        let deep_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == deep_below)
            .unwrap();
        assert!(near_idx < deep_idx);
    }

    #[test]
    fn deep_below_chunks_do_not_displace_near_above_chunks_with_finite_budget() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 1;

        let player = ChunkCoord { x: 0, y: -6, z: 0 };
        let desired = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            None,
            1,
            3,
            Some(5),
            Some(7),
            3,
            32,
            32,
            &HashMap::new(),
            0,
        );

        let near_above = ChunkCoord { x: 1, y: 0, z: 4 };
        let deep_below = ChunkCoord { x: 1, y: -9, z: 4 };
        assert!(desired.generation_scores[&near_above] > desired.generation_scores[&deep_below]);

        let desired_pair =
            if desired.generation_scores[&near_above] >= desired.generation_scores[&deep_below] {
                vec![near_above, deep_below]
            } else {
                vec![deep_below, near_above]
            };
        let keep_pair: HashSet<_> = desired_pair.iter().copied().collect();
        let stats = streaming.update(&desired_pair, &keep_pair, player, 1);

        assert_eq!(stats.queued_generate, 1);
        assert!(streaming.scheduled_generate.contains(&near_above));
        assert!(!streaming.scheduled_generate.contains(&deep_below));
    }

    #[test]
    fn immediate_horizontal_neighbors_are_scheduled_with_budget_floor() {
        let mut streaming = ChunkStreaming::new(1);
        streaming.max_generate_schedule_per_update = 2;

        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let neighbor_a = ChunkCoord { x: 1, y: 0, z: 0 };
        let neighbor_b = ChunkCoord { x: 0, y: 0, z: 1 };
        let high_y = ChunkCoord { x: 0, y: 4, z: 0 };

        let desired = vec![high_y, neighbor_a, neighbor_b];
        let keep: HashSet<_> = desired.iter().copied().collect();
        let stats = streaming.update(&desired, &keep, player, 1);

        assert_eq!(stats.queued_generate, 2);
        assert!(streaming.scheduled_generate.contains(&neighbor_a));
        assert!(streaming.scheduled_generate.contains(&neighbor_b));
        assert!(!streaming.scheduled_generate.contains(&high_y));
    }

    #[test]
    fn desired_set_weighted_order_prioritizes_front_over_rear_at_equal_distance() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let front = ChunkCoord { x: 0, y: 0, z: 2 };
        let rear = ChunkCoord { x: 0, y: 0, z: -2 };

        let visibility = super::VisibilityContext {
            camera_pos_chunks: Vec3::ZERO,
            cone_inner_cos: 0.9,
            cone_outer_cos: 0.0,
            frustum_planes: Some([glam::vec4(0.0, 0.0, 1.0, -0.5); 6]),
        };

        let desired = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Some(&visibility),
            0,
            1,
            Some(2),
            Some(2),
            0,
            8,
            8,
            &HashMap::new(),
            0,
        );

        let old_ring_only_order: Vec<_> = desired
            .near
            .iter()
            .chain(desired.mid.iter())
            .chain(desired.far.iter())
            .chain(desired.ultra.iter())
            .copied()
            .collect();

        let old_front_idx = old_ring_only_order
            .iter()
            .position(|coord| *coord == front)
            .unwrap();
        let old_rear_idx = old_ring_only_order
            .iter()
            .position(|coord| *coord == rear)
            .unwrap();
        let weighted_front_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == front)
            .unwrap();
        let weighted_rear_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == rear)
            .unwrap();

        assert!(old_rear_idx < old_front_idx);
        assert!(weighted_front_idx < weighted_rear_idx);
    }

    #[test]
    fn desired_set_resident_keep_mid_far_follow_weighted_visibility_order() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let mid_front = ChunkCoord { x: 0, y: 0, z: 1 };
        let mid_rear = ChunkCoord { x: 0, y: 0, z: -1 };
        let far_front = ChunkCoord { x: 0, y: 0, z: 2 };
        let far_rear = ChunkCoord { x: 0, y: 0, z: -2 };

        let visibility = super::VisibilityContext {
            camera_pos_chunks: Vec3::ZERO,
            cone_inner_cos: 0.95,
            cone_outer_cos: 0.0,
            frustum_planes: Some([glam::vec4(0.0, 0.0, 1.0, -0.5); 6]),
        };

        let desired = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Some(&visibility),
            0,
            1,
            Some(2),
            Some(2),
            0,
            1,
            1,
            &HashMap::new(),
            0,
        );

        let weighted_mid_pick = desired
            .generation_order
            .iter()
            .copied()
            .find(|coord| desired.mid.contains(coord))
            .unwrap();
        let weighted_far_pick = desired
            .generation_order
            .iter()
            .copied()
            .find(|coord| !desired.near.contains(coord) && !desired.mid.contains(coord))
            .unwrap();

        assert!(desired.resident_keep.contains(&weighted_mid_pick));
        assert!(desired.resident_keep.contains(&weighted_far_pick));

        let mid_front_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == mid_front)
            .unwrap();
        let mid_rear_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == mid_rear)
            .unwrap();
        let far_front_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == far_front)
            .unwrap();
        let far_rear_idx = desired
            .generation_order
            .iter()
            .position(|coord| *coord == far_rear)
            .unwrap();

        assert!(mid_front_idx < mid_rear_idx);
        assert!(far_front_idx < far_rear_idx);
    }

    #[test]
    fn desired_set_uses_chunk_space_camera_and_planes_for_frustum_weight() {
        let player = ChunkCoord { x: 0, y: 0, z: 0 };
        let target = ChunkCoord { x: 2, y: 0, z: 0 };

        let near_radius = 0;
        let mid_radius = 1;
        let far_radius = Some(2);

        // Plane normal points +X in camera-relative chunk-space.
        let include_target = super::VisibilityContext {
            camera_pos_chunks: Vec3::ZERO,
            cone_inner_cos: 1.0,
            cone_outer_cos: 1.0,
            frustum_planes: Some([glam::vec4(1.0, 0.0, 0.0, -2.0); 6]),
        };
        let exclude_target = super::VisibilityContext {
            camera_pos_chunks: Vec3::ZERO,
            cone_inner_cos: 1.0,
            cone_outer_cos: 1.0,
            frustum_planes: Some([glam::vec4(1.0, 0.0, 0.0, -5.0); 6]),
        };

        let base = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::ZERO,
            None,
            near_radius,
            mid_radius,
            far_radius,
            far_radius,
            0,
            64,
            64,
            &HashMap::new(),
            0,
        );
        let boosted = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::ZERO,
            Some(&include_target),
            near_radius,
            mid_radius,
            far_radius,
            far_radius,
            0,
            64,
            64,
            &HashMap::new(),
            0,
        );
        let culled = ChunkStreaming::desired_set(
            player,
            Vec3::ZERO,
            Vec3::ZERO,
            Some(&exclude_target),
            near_radius,
            mid_radius,
            far_radius,
            far_radius,
            0,
            64,
            64,
            &HashMap::new(),
            0,
        );

        let base_score = base.generation_scores[&target];
        let boosted_score = boosted.generation_scores[&target];
        let culled_score = culled.generation_scores[&target];

        assert!(boosted_score > base_score);
        assert!(base_score > culled_score);
    }
}
