use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::chunk_store::ChunkStore;
use crate::sim::{material, Phase, XorShift32};
use crate::simulation::{SimulationPhaseClass, SimulationStepMetadata, SimulationStepStats};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::EMPTY;

const STONE: u16 = 1;
const WOOD: u16 = 2;
const WATER: u16 = 5;
const LAVA: u16 = 6;
const ACID: u16 = 7;
const SMOKE: u16 = 8;
const STEAM: u16 = 9;
const FIRE_GAS: u16 = 11;
const TORCH: u16 = 12;
const EMBER_HOT: u16 = 13;
const EMBER_WARM: u16 = 14;
const EMBER_ASH: u16 = 15;
const DIRT: u16 = 16;
const TURF: u16 = 17;
const BUSH: u16 = 18;
const GRASS: u16 = 19;
const PLANT: u16 = 20;
const WEED: u16 = 21;
const TREE_SEED: u16 = 22;
const LEAVES: u16 = 23;
const DEAD_LEAF: u16 = 24;

pub type Rng = XorShift32;

const BASE_CHUNK_COOLDOWN_TICKS: u8 = 8;
const BASE_MAX_CHUNKS_PER_STEP: usize = 96;
const MIN_CHUNKS_PER_STEP: usize = 24;
const MAX_DYNAMIC_CHUNKS_PER_STEP: usize = 192;
const CHUNK_STARVATION_WAIT_TICKS: u16 = 10;
const CHUNK_VOLUME: usize = (CHUNK_SIZE_VOXELS as usize).pow(3);
const CHUNK_WORDS: usize = CHUNK_VOLUME.div_ceil(64);
const MAX_FRONTIER_PROCESSED_PER_SUBSTEP: usize = 768;
const MIN_FRONTIER_PROCESSED_PER_SUBSTEP: usize = 64;
const SIM_WORK_BUDGET_MS: f32 = 2.8;

fn phase_likely_to_move(phase: Phase) -> bool {
    matches!(phase, Phase::Gas | Phase::Liquid | Phase::Powder)
}

fn mat_likely_to_move_or_react(mat_id: u16) -> bool {
    if mat_id == EMPTY {
        return false;
    }
    let mat = material(mat_id);
    phase_likely_to_move(mat.phase)
        || mat.transforms_on_contact.is_some()
        || mat.flammable
        || matches!(
            mat_id,
            FIRE_GAS | LAVA | ACID | STEAM | SMOKE | EMBER_HOT | EMBER_WARM
        )
}

#[derive(Default)]
struct ChunkFrontierState {
    active_bits: Vec<u64>,
    frontier: Vec<u16>,
    carry_frontier: Vec<u16>,
    pending_writes: Vec<(VoxelCoord, u16)>,
    activation_centers: Vec<VoxelCoord>,
    moved_sources_bits: Vec<u64>,
    claimed_destinations_bits: Vec<u64>,
    rotate_offset: usize,
    seed_scan_cursor: usize,
    seed_scan_complete: bool,
    cooldown_ticks: u8,
    recent_activity: u16,
    moving_avg_processed: f32,
    dormant_water_bits: Vec<u64>,
    wait_ticks: u16,
}

impl ChunkFrontierState {
    fn new() -> Self {
        Self {
            active_bits: vec![0; CHUNK_WORDS],
            frontier: Vec::with_capacity(512),
            carry_frontier: Vec::with_capacity(512),
            pending_writes: Vec::with_capacity(1024),
            activation_centers: Vec::with_capacity(512),
            moved_sources_bits: vec![0; CHUNK_WORDS],
            claimed_destinations_bits: vec![0; CHUNK_WORDS],
            rotate_offset: 0,
            seed_scan_cursor: 0,
            seed_scan_complete: false,
            cooldown_ticks: 0,
            recent_activity: 0,
            moving_avg_processed: 0.0,
            dormant_water_bits: vec![0; CHUNK_WORDS],
            wait_ticks: 0,
        }
    }

    fn enqueue_local(&mut self, idx: u16) {
        let idx_usize = idx as usize;
        let word = idx_usize / 64;
        let bit = 1u64 << (idx_usize % 64);
        if (self.active_bits[word] & bit) != 0 {
            return;
        }
        self.active_bits[word] |= bit;
        self.frontier.push(idx);
    }

    fn clear_active(&mut self, idx: u16) {
        let idx_usize = idx as usize;
        let word = idx_usize / 64;
        let bit = 1u64 << (idx_usize % 64);
        self.active_bits[word] &= !bit;
    }

    fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    fn clear_step_scratch(&mut self) {
        self.pending_writes.clear();
        self.activation_centers.clear();
        self.carry_frontier.clear();
        self.moved_sources_bits.fill(0);
        self.claimed_destinations_bits.fill(0);
    }

    fn mark_source_moved(&mut self, idx: u16) {
        let i = idx as usize;
        self.moved_sources_bits[i / 64] |= 1u64 << (i % 64);
    }

    fn source_moved(&self, idx: u16) -> bool {
        let i = idx as usize;
        (self.moved_sources_bits[i / 64] & (1u64 << (i % 64))) != 0
    }

    fn mark_water_dormant(&mut self, idx: u16) {
        let i = idx as usize;
        self.dormant_water_bits[i / 64] |= 1u64 << (i % 64);
    }

    fn clear_water_dormant(&mut self, idx: u16) {
        let i = idx as usize;
        self.dormant_water_bits[i / 64] &= !(1u64 << (i % 64));
    }

    fn water_dormant(&self, idx: u16) -> bool {
        let i = idx as usize;
        (self.dormant_water_bits[i / 64] & (1u64 << (i % 64))) != 0
    }
    fn try_claim_local_destination(&mut self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = 1u64 << (idx % 64);
        if (self.claimed_destinations_bits[word] & bit) != 0 {
            return false;
        }
        self.claimed_destinations_bits[word] |= bit;
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimSoAChunkBuffer {
    pub material_ids: Vec<u16>,
    pub phases: Vec<u8>,
    pub settled: Vec<u8>,
}

#[derive(Default)]
pub struct SimWorld {
    chunks: HashMap<ChunkCoord, ChunkFrontierState>,
}

impl SimWorld {
    pub fn notify_voxel_edit(&mut self, coord: VoxelCoord) {
        self.enqueue_with_neighbors(coord);
    }

    pub fn step_region(
        &mut self,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
        metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        self.seed_region_frontier_if_needed(store, region);
        self.enqueue_dirty_voxels(store, region);
        let step_start = Instant::now();

        let mut candidates = self.prioritized_chunks(center, region);
        let mut stats = SimulationStepStats::default();
        let max_chunks_this_step = dynamic_chunk_budget(
            candidates.len(),
            step_start.elapsed().as_secs_f32() * 1000.0,
        );
        if candidates.len() > max_chunks_this_step {
            stats.skipped_active_chunks += candidates.len() - max_chunks_this_step;
            candidates.truncate(max_chunks_this_step);
        }

        let mut total_wait_ticks = 0usize;
        for chunk_coord in candidates {
            if !store.is_chunk_loaded(chunk_coord) {
                self.chunks.remove(&chunk_coord);
                continue;
            }

            let mut activation_centers = Vec::new();
            {
                let Some(state) = self.chunks.get_mut(&chunk_coord) else {
                    continue;
                };
                if state.is_empty() {
                    continue;
                }

                stats.stepped_chunks += 1;
                state.clear_step_scratch();
                let mut moved_any = false;
                let mut processed_matching_phase = false;
                let mut encountered_other_phase = false;

                let frontier_len = state.frontier.len();
                let elapsed_ms = step_start.elapsed().as_secs_f32() * 1000.0;
                let mut process_cap = MAX_FRONTIER_PROCESSED_PER_SUBSTEP;
                if elapsed_ms > SIM_WORK_BUDGET_MS {
                    process_cap = (MAX_FRONTIER_PROCESSED_PER_SUBSTEP / 2)
                        .max(MIN_FRONTIER_PROCESSED_PER_SUBSTEP);
                }
                let ema_cap = state.moving_avg_processed as usize;
                if ema_cap > 0 {
                    process_cap = process_cap.min(ema_cap.saturating_add(128));
                }
                process_cap = process_cap.min(frontier_len).max(frontier_len.min(1));
                let start = if frontier_len > 0 {
                    state.rotate_offset % frontier_len
                } else {
                    0
                };

                for processed in 0..frontier_len {
                    let pos = (start + processed) % frontier_len;
                    let idx = state.frontier[pos];
                    if processed >= process_cap {
                        state.carry_frontier.push(idx);
                        continue;
                    }
                    state.clear_active(idx);
                    if state.source_moved(idx) {
                        continue;
                    }
                    let source = local_index_to_world(chunk_coord, idx);
                    if state.water_dormant(idx) {
                        let source_id = store.get_voxel(source);
                        if source_id == WATER && water_cell_locally_settled(store, source) {
                            continue;
                        }
                        state.clear_water_dormant(idx);
                    }
                    let (source_chunk, _) = voxel_to_chunk(source);
                    if source_chunk != chunk_coord || !store.is_chunk_loaded(source_chunk) {
                        continue;
                    }

                    let mat_id = store.get_voxel(source);
                    if mat_id == EMPTY {
                        continue;
                    }
                    if react_voxel(store, source, mat_id, rng) {
                        moved_any = true;
                        activation_centers.push(source);
                        continue;
                    }

                    let mat = material(mat_id);
                    if mat_id == WATER && water_cell_locally_settled(store, source) {
                        state.mark_water_dormant(idx);
                        continue;
                    }
                    if !phase_matches_class(mat.phase, metadata.phase_class) {
                        encountered_other_phase = true;
                        continue;
                    }
                    processed_matching_phase = true;

                    if matches!(mat.phase, Phase::Gas)
                        && metadata.boundary_dissipation_strength > 0.0
                        && chunk_chebyshev_distance(chunk_coord, center)
                            > metadata.core_radius_chunks
                        && rng.chance(metadata.boundary_dissipation_strength.clamp(0.0, 1.0))
                    {
                        state.pending_writes.push((source, EMPTY));
                        stats.boundary_dissipated_particles += 1;
                        moved_any = true;
                        state.activation_centers.push(source);
                        break;
                    }

                    let candidates = movement_candidates(source, mat_id, mat.phase, rng);

                    for destination in candidates {
                        let (destination_chunk, _) = voxel_to_chunk(destination);
                        if !store.is_chunk_loaded(destination_chunk) {
                            if mat.phase == Phase::Gas {
                                let escape_chance = gas_boundary_escape_chance(
                                    source,
                                    source_chunk,
                                    center,
                                    destination.y > source.y,
                                );
                                if rng.chance(escape_chance) {
                                    state.pending_writes.push((source, EMPTY));
                                    state.mark_source_moved(idx);
                                    moved_any = true;
                                    state.activation_centers.push(source);
                                    break;
                                }
                            }
                            continue;
                        }
                        if destination_chunk == chunk_coord {
                            let local = voxel_to_chunk(destination).1;
                            let local_idx = local_to_index(
                                local[0] as usize,
                                local[1] as usize,
                                local[2] as usize,
                            );
                            if !state.try_claim_local_destination(local_idx) {
                                continue;
                            }
                        }

                        let target_id = store.get_voxel(destination);
                        if target_id == mat_id {
                            continue;
                        }
                        let target_mat = material(target_id);
                        if !can_displace(
                            mat.phase,
                            mat.density,
                            target_mat.phase,
                            target_mat.density,
                            target_id,
                        ) {
                            continue;
                        }

                        state.pending_writes.push((source, EMPTY));
                        state.pending_writes.push((destination, mat_id));
                        state.mark_source_moved(idx);
                        moved_any = true;
                        state.activation_centers.push(source);
                        state.activation_centers.push(destination);
                        break;
                    }
                }

                for (coord, mat_id) in state.pending_writes.iter().copied() {
                    store.set_voxel(coord, mat_id);
                }

                if !state.carry_frontier.is_empty() {
                    state.frontier.clear();
                    state.frontier.append(&mut state.carry_frontier);
                } else {
                    state.frontier.clear();
                }
                state.rotate_offset = state.rotate_offset.wrapping_add(17);
                let processed_count = process_cap as f32;
                state.moving_avg_processed =
                    (state.moving_avg_processed * 0.85) + (processed_count * 0.15);

                if !processed_matching_phase && encountered_other_phase {
                    stats.skipped_chunks += 1;
                }

                if moved_any {
                    state.cooldown_ticks = 0;
                    state.wait_ticks = 0;
                    state.recent_activity = state.recent_activity.saturating_add(4);
                } else {
                    state.cooldown_ticks =
                        adaptive_chunk_cooldown(state.recent_activity, frontier_len);
                    state.wait_ticks = state.wait_ticks.saturating_add(1);
                    state.recent_activity = state.recent_activity.saturating_sub(1);
                }
                stats.processed_frontier_voxels += process_cap;
                total_wait_ticks += state.wait_ticks as usize;
                activation_centers.append(&mut state.activation_centers);
            }

            for center in activation_centers {
                enqueue_world_neighbors_to_chunks(&mut self.chunks, center, region);
            }
        }

        if stats.stepped_chunks > 0 {
            stats.avg_chunk_wait_ticks = total_wait_ticks as f32 / stats.stepped_chunks as f32;
        }

        stats
    }

    pub fn build_soa_for_chunk(
        &self,
        store: &ChunkStore,
        chunk_coord: ChunkCoord,
    ) -> Option<SimSoAChunkBuffer> {
        let chunk = store.get_chunk(chunk_coord)?;
        let mut material_ids = Vec::with_capacity(chunk.iter_raw().len());
        let mut phases = Vec::with_capacity(chunk.iter_raw().len());
        let mut settled = Vec::with_capacity(chunk.iter_raw().len());

        let active_state = self.chunks.get(&chunk_coord);
        for (idx, &mat_id) in chunk.iter_raw().iter().enumerate() {
            material_ids.push(mat_id);
            phases.push(material(mat_id).phase as u8);
            let x = (idx % CHUNK_SIZE_VOXELS as usize) as i32;
            let y = ((idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize) as i32;
            let z = (idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)) as i32;
            let is_active = active_state
                .map(|state| {
                    let idx = local_to_index(x as usize, y as usize, z as usize);
                    let word = idx / 64;
                    let bit = 1u64 << (idx % 64);
                    (state.active_bits[word] & bit) != 0
                })
                .unwrap_or(false);
            settled.push((!is_active) as u8);
        }

        Some(SimSoAChunkBuffer {
            material_ids,
            phases,
            settled,
        })
    }

    fn enqueue_dirty_voxels(&mut self, store: &mut ChunkStore, region: &HashSet<ChunkCoord>) {
        for coord in store.take_sim_dirty_voxels() {
            self.enqueue_with_neighbors_if_in_region(coord, region);
        }
    }

    fn seed_region_frontier_if_needed(&mut self, store: &ChunkStore, region: &HashSet<ChunkCoord>) {
        const SEED_SCAN_BATCH: usize = 1024;
        for &chunk_coord in region {
            if !store.is_chunk_loaded(chunk_coord) {
                continue;
            }
            let mut seed_worlds = Vec::new();
            let state = self
                .chunks
                .entry(chunk_coord)
                .or_insert_with(ChunkFrontierState::new);
            if state.seed_scan_complete {
                continue;
            }
            let Some(chunk) = store.get_chunk(chunk_coord) else {
                continue;
            };
            let base = chunk_to_world_min(chunk_coord);
            let raw = chunk.iter_raw();
            let end = (state.seed_scan_cursor + SEED_SCAN_BATCH).min(raw.len());
            for idx in state.seed_scan_cursor..end {
                let mat_id = raw[idx];
                if !mat_likely_to_move_or_react(mat_id) {
                    continue;
                }
                if mat_id == WATER {
                    let local_idx = idx as u16;
                    if state.water_dormant(local_idx) {
                        continue;
                    }
                }
                let x = idx % CHUNK_SIZE_VOXELS as usize;
                let y = (idx / CHUNK_SIZE_VOXELS as usize) % CHUNK_SIZE_VOXELS as usize;
                let z = idx / (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize);
                let world = VoxelCoord {
                    x: base.x + x as i32,
                    y: base.y + y as i32,
                    z: base.z + z as i32,
                };
                seed_worlds.push(world);
            }
            state.seed_scan_cursor = end;
            if state.seed_scan_cursor >= raw.len() {
                state.seed_scan_complete = true;
            }
            for world in seed_worlds {
                self.enqueue_with_neighbors_if_in_region(world, region);
            }
        }
    }

    fn enqueue_with_neighbors_if_in_region(
        &mut self,
        center: VoxelCoord,
        region: &HashSet<ChunkCoord>,
    ) {
        enqueue_world_neighbors_to_chunks(&mut self.chunks, center, region);
    }

    fn enqueue_with_neighbors(&mut self, center: VoxelCoord) {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let world = VoxelCoord {
                        x: center.x + dx,
                        y: center.y + dy,
                        z: center.z + dz,
                    };
                    let (chunk_coord, local) = voxel_to_chunk(world);
                    let idx =
                        local_to_index(local[0] as usize, local[1] as usize, local[2] as usize);
                    let state = self
                        .chunks
                        .entry(chunk_coord)
                        .or_insert_with(ChunkFrontierState::new);
                    let local_idx = idx as u16;
                    state.clear_water_dormant(local_idx);
                    state.enqueue_local(local_idx);
                    state.cooldown_ticks = 0;
                }
            }
        }
    }

    fn prioritized_chunks(
        &mut self,
        center: ChunkCoord,
        region: &HashSet<ChunkCoord>,
    ) -> Vec<ChunkCoord> {
        let mut out = Vec::new();
        for (&coord, state) in &mut self.chunks {
            if !region.contains(&coord) {
                continue;
            }
            if state.is_empty() {
                if state.cooldown_ticks > 0 {
                    state.cooldown_ticks -= 1;
                }
                state.wait_ticks = state.wait_ticks.saturating_add(1);
                continue;
            }
            if state.cooldown_ticks > 0 {
                state.cooldown_ticks -= 1;
                state.wait_ticks = state.wait_ticks.saturating_add(1);
                continue;
            }
            out.push((coord, state.recent_activity, state.wait_ticks));
        }

        out.sort_by_key(|(coord, activity, wait_ticks)| {
            let dx = i64::from(coord.x - center.x);
            let dy = i64::from(coord.y - center.y);
            let dz = i64::from(coord.z - center.z);
            let dist2 = dx * dx + dy * dy + dz * dz;
            let starved = *wait_ticks >= CHUNK_STARVATION_WAIT_TICKS;
            (
                !starved,
                dist2,
                std::cmp::Reverse(*activity),
                std::cmp::Reverse(*wait_ticks),
                coord.x,
                coord.y,
                coord.z,
            )
        });
        out.into_iter().map(|(coord, _, _)| coord).collect()
    }
}

fn adaptive_chunk_cooldown(recent_activity: u16, frontier_len: usize) -> u8 {
    let pressure =
        (frontier_len as f32 / MAX_FRONTIER_PROCESSED_PER_SUBSTEP as f32).clamp(0.0, 2.0);
    let activity = (recent_activity as f32 / 12.0).clamp(0.0, 1.0);
    let cooldown = BASE_CHUNK_COOLDOWN_TICKS as f32 + pressure * 2.0 - activity * 4.0;
    cooldown.round().clamp(1.0, 14.0) as u8
}

fn dynamic_chunk_budget(active_chunks: usize, elapsed_ms: f32) -> usize {
    let load = (active_chunks as f32 / BASE_MAX_CHUNKS_PER_STEP as f32).clamp(0.2, 2.5);
    let time_pressure = (elapsed_ms / SIM_WORK_BUDGET_MS).clamp(0.0, 2.5);
    let budget = BASE_MAX_CHUNKS_PER_STEP as f32 / load + 48.0 - 20.0 * time_pressure;
    budget.round().clamp(
        MIN_CHUNKS_PER_STEP as f32,
        MAX_DYNAMIC_CHUNKS_PER_STEP as f32,
    ) as usize
}

fn water_cell_locally_settled(store: &ChunkStore, source: VoxelCoord) -> bool {
    let below = offset_voxel(source, 0, -1, 0);
    let below_id = store.get_voxel(below);
    if below_id == EMPTY || material(below_id).phase == Phase::Gas {
        return false;
    }

    let mut lateral_empty = false;
    for [dx, _, dz] in neighbor_dirs4() {
        let lateral = offset_voxel(source, dx, 0, dz);
        let lateral_id = store.get_voxel(lateral);
        if lateral_id == EMPTY {
            lateral_empty = true;
            break;
        }
        let lateral_phase = material(lateral_id).phase;
        if lateral_phase == Phase::Gas {
            lateral_empty = true;
            break;
        }
        let below_lateral = offset_voxel(source, dx, -1, dz);
        let below_lateral_id = store.get_voxel(below_lateral);
        if below_lateral_id == EMPTY || material(below_lateral_id).phase == Phase::Gas {
            return false;
        }
        if lateral_id != WATER {
            return false;
        }
    }

    !lateral_empty
}

fn chunk_chebyshev_distance(a: ChunkCoord, b: ChunkCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

fn phase_matches_class(phase: Phase, class: Option<SimulationPhaseClass>) -> bool {
    match class {
        None => true,
        Some(SimulationPhaseClass::Gas) => matches!(phase, Phase::Gas),
        Some(SimulationPhaseClass::SolidsLiquidsPowders) => !matches!(phase, Phase::Gas),
    }
}

fn react_voxel(store: &mut ChunkStore, p: VoxelCoord, id: u16, rng: &mut Rng) -> bool {
    let mut reacted = false;
    let mat = material(id);

    if id == FIRE_GAS {
        let mut neighbors = neighbor_dirs6();
        rng.shuffle(&mut neighbors);
        for [dx, dy, dz] in neighbors {
            let np = offset_voxel(p, dx, dy, dz);
            let nid = store.get_voxel(np);
            if nid != EMPTY && material(nid).flammable && rng.chance(0.2) {
                let replacement = if nid == WOOD {
                    EMBER_HOT
                } else if rng.chance(0.5) {
                    FIRE_GAS
                } else {
                    SMOKE
                };
                store.set_voxel(np, replacement);
                reacted = true;
            }
        }

        if rng.chance(0.06) {
            store.set_voxel(p, SMOKE);
            reacted = true;
        } else if rng.chance(0.08) {
            store.set_voxel(p, EMPTY);
            reacted = true;
        }
    }

    if id == STEAM
        && rng.chance(0.02)
        && !has_neighbor(store, p, LAVA)
        && !has_neighbor(store, p, FIRE_GAS)
    {
        store.set_voxel(p, WATER);
        reacted = true;
    }

    if id == SMOKE && rng.chance(0.015) {
        store.set_voxel(p, EMPTY);
        reacted = true;
    }

    if id == ACID {
        let mut neighbors = neighbor_dirs6();
        rng.shuffle(&mut neighbors);
        for [dx, dy, dz] in neighbors {
            let np = offset_voxel(p, dx, dy, dz);
            let nid = store.get_voxel(np);
            if !is_acid_dissolvable(nid) || !rng.chance(0.24) {
                continue;
            }
            store.set_voxel(np, EMPTY);
            reacted = true;
            if rng.chance(0.40) {
                let byproduct = if rng.chance(0.55) { STEAM } else { SMOKE };
                let _ = spawn_reaction_product(store, np, byproduct, rng);
            }
            if rng.chance(0.10) {
                store.set_voxel(p, EMPTY);
            }
            break;
        }
    }

    if id == LAVA || id == WATER {
        let mut neighbors = neighbor_dirs6();
        rng.shuffle(&mut neighbors);
        for [dx, dy, dz] in neighbors {
            let np = offset_voxel(p, dx, dy, dz);
            let nid = store.get_voxel(np);
            let nmat = material(nid);

            if ((id == LAVA && nid == WATER) || (id == WATER && nid == LAVA)) && rng.chance(0.35) {
                store.set_voxel(p, STONE);
                let replacement = if rng.chance(0.60) { STEAM } else { EMPTY };
                store.set_voxel(np, replacement);
                reacted = true;
                break;
            }

            if id == LAVA
                && nid != LAVA
                && nmat.transforms_on_contact
                    == Some(crate::sim::ContactReaction::LavaCoolsToWaterOrSteam)
                && rng.chance(0.45)
            {
                let replacement = if rng.chance(0.65) { WATER } else { STEAM };
                store.set_voxel(np, replacement);
                reacted = true;
                continue;
            }

            if id == LAVA && nmat.flammable && rng.chance(0.18) {
                let replacement = if rng.chance(0.55) { FIRE_GAS } else { SMOKE };
                store.set_voxel(np, replacement);
                let _ = spawn_reaction_product(store, np, SMOKE, rng);
                reacted = true;
            }
        }
    }

    if mat.flammable && rng.chance(0.06) {
        for [dx, dy, dz] in neighbor_dirs6() {
            let np = offset_voxel(p, dx, dy, dz);
            let nid = store.get_voxel(np);
            if nid == LAVA || nid == FIRE_GAS {
                let replacement = if rng.chance(0.6) { FIRE_GAS } else { SMOKE };
                store.set_voxel(p, replacement);
                let _ = spawn_reaction_product(store, p, SMOKE, rng);
                reacted = true;
                break;
            }
        }
    }

    if id == TORCH {
        if rng.chance(0.55) {
            reacted |= spawn_reaction_product(store, p, FIRE_GAS, rng);
        }
        if rng.chance(0.18) {
            reacted |= spawn_reaction_product(store, p, SMOKE, rng);
        }
    }

    if id == WOOD && has_ignition_neighbor(store, p) && rng.chance(0.52) {
        store.set_voxel(p, EMBER_HOT);
        let _ = spawn_reaction_product(store, p, FIRE_GAS, rng);
        reacted = true;
    }

    if id == EMBER_HOT {
        if rng.chance(0.52) {
            let _ = spawn_reaction_product(store, p, FIRE_GAS, rng);
            reacted = true;
        }
        if rng.chance(0.03) {
            store.set_voxel(p, EMBER_WARM);
            reacted = true;
        }
    } else if id == EMBER_WARM {
        if rng.chance(0.26) {
            let _ = spawn_reaction_product(store, p, SMOKE, rng);
            reacted = true;
        }
        if rng.chance(0.02) {
            store.set_voxel(p, EMBER_ASH);
            reacted = true;
        }
    } else if id == EMBER_ASH && rng.chance(0.003) {
        store.set_voxel(p, EMPTY);
        reacted = true;
    }

    if id == LEAVES && !has_tree_support(store, p) {
        store.set_voxel(p, DEAD_LEAF);
        reacted = true;
    }

    if matches!(id, BUSH | GRASS) && !has_solid_support_below(store, p) {
        store.set_voxel(p, EMPTY);
        reacted = true;
    }

    if id == DIRT && is_exposed_to_sky(store, p) && rng.chance(0.008) {
        store.set_voxel(p, TURF);
        reacted = true;
    }

    if id == TURF {
        let above = store.get_voxel(offset_voxel(p, 0, 1, 0));
        if above != EMPTY && material(above).phase != Phase::Gas {
            if rng.chance(0.20) {
                store.set_voxel(p, DIRT);
                reacted = true;
            }
        } else if rng.chance(0.0004) {
            let grow_id = if rng.chance(0.55) { GRASS } else { BUSH };
            let above_pos = offset_voxel(p, 0, 1, 0);
            if store.get_voxel(above_pos) == EMPTY && has_solid_support_below(store, above_pos) {
                store.set_voxel(above_pos, grow_id);
                reacted = true;
            }
        }
    }

    if id == PLANT {
        if !has_neighbor(store, p, WATER) && rng.chance(0.008) {
            store.set_voxel(p, WEED);
            reacted = true;
        } else if has_neighbor(store, p, WATER) && rng.chance(0.012) {
            let np = offset_voxel(p, 0, 1, 0);
            if store.get_voxel(np) == EMPTY && has_solid_support_below(store, np) {
                store.set_voxel(np, PLANT);
                reacted = true;
            }
        }
    }

    if id == WEED {
        if has_neighbor(store, p, WATER) && rng.chance(0.08) {
            store.set_voxel(p, PLANT);
            reacted = true;
        } else if rng.chance(0.006) {
            let np = offset_voxel(p, 0, 1, 0);
            if store.get_voxel(np) == EMPTY && has_solid_support_below(store, np) {
                store.set_voxel(np, WEED);
                reacted = true;
            }
        }
    }

    if id == TREE_SEED && has_solid_support_below(store, p) && rng.chance(0.015) {
        store.set_voxel(p, WOOD);
        reacted = true;
    }

    reacted
}

fn is_acid_dissolvable(id: u16) -> bool {
    if id == EMPTY {
        return false;
    }
    let mat = material(id);
    matches!(mat.phase, Phase::Solid | Phase::Powder) && !mat.acid_resistant
}

fn has_neighbor(store: &ChunkStore, p: VoxelCoord, target: u16) -> bool {
    neighbor_dirs6()
        .into_iter()
        .any(|[dx, dy, dz]| store.get_voxel(offset_voxel(p, dx, dy, dz)) == target)
}

fn has_solid_support_below(store: &ChunkStore, p: VoxelCoord) -> bool {
    let below = store.get_voxel(offset_voxel(p, 0, -1, 0));
    below != EMPTY && material(below).phase != Phase::Gas
}

fn has_tree_support(store: &ChunkStore, p: VoxelCoord) -> bool {
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if dx * dx + dy * dy + dz * dz > 5 {
                    continue;
                }
                let nid = store.get_voxel(offset_voxel(p, dx, dy, dz));
                if matches!(nid, WOOD | LEAVES) {
                    return true;
                }
            }
        }
    }
    false
}

fn is_exposed_to_sky(store: &ChunkStore, p: VoxelCoord) -> bool {
    for y in (p.y + 1)..=(p.y + CHUNK_SIZE_VOXELS * 2) {
        if store.get_voxel(VoxelCoord { x: p.x, y, z: p.z }) != EMPTY {
            return false;
        }
    }
    true
}

fn spawn_reaction_product(
    store: &mut ChunkStore,
    origin: VoxelCoord,
    product: u16,
    rng: &mut Rng,
) -> bool {
    let mut dirs = neighbor_dirs6();
    rng.shuffle(&mut dirs);
    let product_phase = material(product).phase;
    for [dx, dy, dz] in dirs {
        let np = offset_voxel(origin, dx, dy, dz);
        if !store.is_voxel_chunk_loaded(np) {
            if product_phase == Phase::Gas {
                return true;
            }
            continue;
        }
        if store.get_voxel(np) == EMPTY {
            store.set_voxel(np, product);
            return true;
        }
    }
    false
}

fn has_ignition_neighbor(store: &ChunkStore, p: VoxelCoord) -> bool {
    for dz in -2..=2 {
        for dy in -1..=2 {
            for dx in -2..=2 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                if dx * dx + dy * dy + dz * dz > 5 {
                    continue;
                }
                let nid = store.get_voxel(offset_voxel(p, dx, dy, dz));
                if matches!(nid, LAVA | FIRE_GAS | TORCH | EMBER_HOT | EMBER_WARM) {
                    return true;
                }
            }
        }
    }
    false
}

fn neighbor_dirs6() -> [[i32; 3]; 6] {
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
}

fn neighbor_dirs4() -> [[i32; 3]; 4] {
    [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]
}

fn offset_voxel(p: VoxelCoord, dx: i32, dy: i32, dz: i32) -> VoxelCoord {
    VoxelCoord {
        x: p.x + dx,
        y: p.y + dy,
        z: p.z + dz,
    }
}

pub fn step_region_profiled(
    store: &mut ChunkStore,
    region: &HashSet<ChunkCoord>,
    center: ChunkCoord,
    rng: &mut Rng,
) -> usize {
    let mut sim = SimWorld::default();
    sim.step_region(
        store,
        region,
        center,
        rng,
        SimulationStepMetadata::default(),
    )
    .stepped_chunks
}

pub fn step_region(store: &mut ChunkStore, region: &HashSet<ChunkCoord>, rng: &mut Rng) {
    let mut sim = SimWorld::default();
    let _ = sim.step_region(
        store,
        region,
        ChunkCoord { x: 0, y: 0, z: 0 },
        rng,
        SimulationStepMetadata::default(),
    );
}

fn enqueue_world_neighbors_to_chunks(
    chunks: &mut HashMap<ChunkCoord, ChunkFrontierState>,
    center: VoxelCoord,
    region: &HashSet<ChunkCoord>,
) {
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let world = VoxelCoord {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                };
                let (chunk_coord, local) = voxel_to_chunk(world);
                if !region.contains(&chunk_coord) {
                    continue;
                }
                let idx = local_to_index(local[0] as usize, local[1] as usize, local[2] as usize);
                let state = chunks
                    .entry(chunk_coord)
                    .or_insert_with(ChunkFrontierState::new);
                let local_idx = idx as u16;
                state.clear_water_dormant(local_idx);
                state.enqueue_local(local_idx);
                state.cooldown_ticks = 0;
            }
        }
    }
}

fn local_to_index(x: usize, y: usize, z: usize) -> usize {
    x + y * CHUNK_SIZE_VOXELS as usize
        + z * (CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize)
}

fn local_index_to_world(chunk_coord: ChunkCoord, idx: u16) -> VoxelCoord {
    let base = chunk_to_world_min(chunk_coord);
    let idx = idx as usize;
    let sz = CHUNK_SIZE_VOXELS as usize;
    let x = idx % sz;
    let y = (idx / sz) % sz;
    let z = idx / (sz * sz);
    VoxelCoord {
        x: base.x + x as i32,
        y: base.y + y as i32,
        z: base.z + z as i32,
    }
}

fn movement_candidates(
    source: VoxelCoord,
    mat_id: u16,
    phase: Phase,
    rng: &mut Rng,
) -> Vec<VoxelCoord> {
    let mut cardinal_lateral = vec![
        VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z,
        },
        VoxelCoord {
            x: source.x + 1,
            y: source.y,
            z: source.z,
        },
        VoxelCoord {
            x: source.x,
            y: source.y,
            z: source.z - 1,
        },
        VoxelCoord {
            x: source.x,
            y: source.y,
            z: source.z + 1,
        },
    ];
    rng.shuffle(&mut cardinal_lateral);

    let mut diagonal_lateral = vec![
        VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z - 1,
        },
        VoxelCoord {
            x: source.x + 1,
            y: source.y,
            z: source.z - 1,
        },
        VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z + 1,
        },
        VoxelCoord {
            x: source.x + 1,
            y: source.y,
            z: source.z + 1,
        },
    ];
    rng.shuffle(&mut diagonal_lateral);

    match phase {
        Phase::Powder => {
            let mut downward_diagonal = vec![
                VoxelCoord {
                    x: source.x - 1,
                    y: source.y - 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x + 1,
                    y: source.y - 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y - 1,
                    z: source.z - 1,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y - 1,
                    z: source.z + 1,
                },
            ];
            rng.shuffle(&mut downward_diagonal);
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y - 1,
                z: source.z,
            }];
            candidates.extend(downward_diagonal);
            candidates
        }
        Phase::Liquid => {
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y - 1,
                z: source.z,
            }];
            match mat_id {
                LAVA => {
                    candidates.extend(cardinal_lateral.into_iter().take(2));
                    if rng.chance(0.2) {
                        candidates.extend(diagonal_lateral.into_iter().take(1));
                    }
                }
                WATER => {
                    candidates.extend(cardinal_lateral);
                    candidates.extend(diagonal_lateral.into_iter().take(2));
                }
                ACID => {
                    candidates.extend(cardinal_lateral);
                    candidates.extend(diagonal_lateral.into_iter().take(1));
                }
                _ => {
                    candidates.extend(cardinal_lateral);
                }
            }
            candidates
        }
        Phase::Gas => {
            let mut upward_diagonal = vec![
                VoxelCoord {
                    x: source.x - 1,
                    y: source.y + 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x + 1,
                    y: source.y + 1,
                    z: source.z,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y + 1,
                    z: source.z - 1,
                },
                VoxelCoord {
                    x: source.x,
                    y: source.y + 1,
                    z: source.z + 1,
                },
            ];
            rng.shuffle(&mut upward_diagonal);
            let mut candidates = vec![VoxelCoord {
                x: source.x,
                y: source.y + 1,
                z: source.z,
            }];
            let mat = material(mat_id);
            let lateral = (mat.flow_speed.max(1) as usize).min(4);
            candidates.extend(upward_diagonal.into_iter().take(lateral));
            candidates.extend(cardinal_lateral.into_iter().take(lateral));
            candidates
        }
        Phase::Solid => Vec::new(),
    }
}

fn gas_boundary_escape_chance(
    source: VoxelCoord,
    source_chunk: ChunkCoord,
    center_chunk: ChunkCoord,
    is_upward_attempt: bool,
) -> f32 {
    let local = voxel_to_chunk(source).1;
    let max_local_y = (CHUNK_SIZE_VOXELS - 1) as f32;
    let height = (local[1] as f32 / max_local_y).clamp(0.0, 1.0);

    let dx = (source_chunk.x - center_chunk.x).abs() as f32;
    let dy = (source_chunk.y - center_chunk.y).abs() as f32;
    let dz = (source_chunk.z - center_chunk.z).abs() as f32;
    let distance = ((dx * dx) + (dy * dy) + (dz * dz)).sqrt();
    let distance_factor = (distance / 4.0).clamp(0.0, 1.0);

    let mut chance = 0.06 + (height * 0.24) + (distance_factor * 0.20);
    if is_upward_attempt {
        chance += 0.18 + (height * distance_factor * 0.22);
    }
    chance.clamp(0.0, 0.95)
}

fn can_displace(
    mover_phase: Phase,
    mover_density: i16,
    target_phase: Phase,
    target_density: i16,
    target_id: u16,
) -> bool {
    if target_id == EMPTY {
        return true;
    }

    match mover_phase {
        Phase::Powder | Phase::Liquid => {
            target_phase == Phase::Gas || mover_density > target_density
        }
        Phase::Gas => target_phase == Phase::Gas && mover_density < target_density,
        Phase::Solid => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::XorShift32;
    use crate::world::{Chunk as LegacyChunk, EMPTY};

    #[test]
    fn skips_cross_chunk_moves_into_unloaded_destination_chunks() {
        let mut store = ChunkStore::new();
        let source_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let source_world = chunk_to_world_min(source_chunk);
        let source = VoxelCoord {
            x: source_world.x,
            y: source_world.y,
            z: source_world.z,
        };

        let mut chunk = LegacyChunk::new();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    chunk.set(x, y, z, 1);
                }
            }
        }
        chunk.set(0, 0, 0, 5);
        store.insert_chunk(source_chunk, chunk);

        let region = HashSet::from([source_chunk]);
        let mut rng = XorShift32::new(1);
        let mut sim = SimWorld::default();

        sim.step_region(
            &mut store,
            &region,
            source_chunk,
            &mut rng,
            SimulationStepMetadata::default(),
        );

        assert_eq!(store.get_voxel(source), 5);
        let destination = VoxelCoord {
            x: source.x - 1,
            y: source.y,
            z: source.z,
        };
        assert_eq!(store.get_voxel(destination), EMPTY);
        assert!(!store.is_voxel_chunk_loaded(destination));
    }

    #[test]
    fn deterministic_frontier_matches_reference_for_simple_fall() {
        let mut store = ChunkStore::new();
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(c);
        let top = VoxelCoord {
            x: base.x + 3,
            y: base.y + 5,
            z: base.z + 3,
        };
        store.set_voxel(top, 3);

        let region = HashSet::from([c]);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(top);

        let mut rng_a = XorShift32::new(42);
        let mut rng_b = XorShift32::new(42);
        let mut store_ref = ChunkStore::new();
        store_ref.set_voxel(top, 3);
        let mut sim_ref = SimWorld::default();
        sim_ref.notify_voxel_edit(top);

        for _ in 0..4 {
            sim.step_region(
                &mut store,
                &region,
                c,
                &mut rng_a,
                SimulationStepMetadata::default(),
            );
            sim_ref.step_region(
                &mut store_ref,
                &region,
                c,
                &mut rng_b,
                SimulationStepMetadata::default(),
            );
        }

        assert_eq!(
            store.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            }),
            3
        );
        assert_eq!(
            store.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            }),
            store_ref.get_voxel(VoxelCoord {
                x: top.x,
                y: top.y - 4,
                z: top.z
            })
        );
    }

    #[test]
    fn soa_layout_captures_settled_bit() {
        let mut store = ChunkStore::new();
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(c);
        let v = VoxelCoord {
            x: base.x + 1,
            y: base.y + 1,
            z: base.z + 1,
        };
        store.set_voxel(v, 5);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(v);
        let region = HashSet::from([c]);
        let mut rng = XorShift32::new(7);
        sim.step_region(
            &mut store,
            &region,
            c,
            &mut rng,
            SimulationStepMetadata::default(),
        );

        let soa = sim
            .build_soa_for_chunk(&store, c)
            .expect("chunk should exist");
        assert_eq!(soa.material_ids.len(), (CHUNK_SIZE_VOXELS as usize).pow(3));
        assert_eq!(soa.phases.len(), soa.material_ids.len());
        assert_eq!(soa.settled.len(), soa.material_ids.len());
    }

    #[test]
    fn gas_escape_chance_increases_with_height_distance_and_upward_bias() {
        let center = ChunkCoord { x: 0, y: 0, z: 0 };
        let near_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let far_chunk = ChunkCoord { x: 3, y: 0, z: 3 };

        let low = VoxelCoord { x: 0, y: 0, z: 0 };
        let high = VoxelCoord {
            x: 0,
            y: CHUNK_SIZE_VOXELS - 1,
            z: 0,
        };

        let low_chance = gas_boundary_escape_chance(low, near_chunk, center, false);
        let high_chance = gas_boundary_escape_chance(high, near_chunk, center, false);
        let far_upward_chance = gas_boundary_escape_chance(high, far_chunk, center, true);

        assert!(high_chance > low_chance);
        assert!(far_upward_chance > high_chance);
    }

    #[test]
    fn gas_dissipates_against_unloaded_boundary_without_chunk_spawn() {
        let mut store = ChunkStore::new();
        let source_chunk = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(source_chunk);
        let top_center = VoxelCoord {
            x: base.x + (CHUNK_SIZE_VOXELS / 2),
            y: base.y + CHUNK_SIZE_VOXELS - 1,
            z: base.z + (CHUNK_SIZE_VOXELS / 2),
        };

        let mut chunk = LegacyChunk::new();
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    chunk.set(x, y, z, STONE);
                }
            }
        }
        chunk.set(
            (CHUNK_SIZE_VOXELS / 2) as usize,
            (CHUNK_SIZE_VOXELS - 1) as usize,
            (CHUNK_SIZE_VOXELS / 2) as usize,
            SMOKE,
        );
        store.insert_chunk(source_chunk, chunk);

        let region = HashSet::from([source_chunk]);
        let mut rng = XorShift32::new(3);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(top_center);

        for _ in 0..64 {
            sim.step_region(
                &mut store,
                &region,
                source_chunk,
                &mut rng,
                SimulationStepMetadata::default(),
            );
            if store.get_voxel(top_center) == EMPTY {
                break;
            }
        }

        assert_eq!(store.get_voxel(top_center), EMPTY);
        assert!(!store.is_chunk_loaded(ChunkCoord {
            x: source_chunk.x,
            y: source_chunk.y + 1,
            z: source_chunk.z,
        }));
    }
    #[test]
    fn flat_water_pool_goes_quiet_until_disturbed() {
        let mut store = ChunkStore::new();
        let c = ChunkCoord { x: 0, y: 0, z: 0 };
        let base = chunk_to_world_min(c);
        for z in 2..=8 {
            for x in 2..=8 {
                store.set_voxel(
                    VoxelCoord {
                        x: base.x + x,
                        y: base.y,
                        z: base.z + z,
                    },
                    STONE,
                );
                let boundary = x == 2 || x == 8 || z == 2 || z == 8;
                store.set_voxel(
                    VoxelCoord {
                        x: base.x + x,
                        y: base.y + 1,
                        z: base.z + z,
                    },
                    if boundary { STONE } else { WATER },
                );
            }
        }

        let region = HashSet::from([c]);
        let mut sim = SimWorld::default();
        sim.notify_voxel_edit(VoxelCoord {
            x: base.x + 5,
            y: base.y + 1,
            z: base.z + 5,
        });
        let mut rng = XorShift32::new(99);

        let _ = sim.step_region(
            &mut store,
            &region,
            c,
            &mut rng,
            SimulationStepMetadata::default(),
        );
        for _ in 0..4 {
            let _ = sim.step_region(
                &mut store,
                &region,
                c,
                &mut rng,
                SimulationStepMetadata::default(),
            );
        }

        let mut quiet_before = Vec::new();
        for z in 3..8 {
            for x in 3..8 {
                quiet_before.push(store.get_voxel(VoxelCoord {
                    x: base.x + x,
                    y: base.y + 1,
                    z: base.z + z,
                }));
            }
        }
        let quiet_stats = sim.step_region(
            &mut store,
            &region,
            c,
            &mut rng,
            SimulationStepMetadata::default(),
        );
        let mut quiet_after = Vec::new();
        for z in 3..8 {
            for x in 3..8 {
                quiet_after.push(store.get_voxel(VoxelCoord {
                    x: base.x + x,
                    y: base.y + 1,
                    z: base.z + z,
                }));
            }
        }
        assert_eq!(quiet_before, quiet_after);
        let disturb = VoxelCoord {
            x: base.x + 2,
            y: base.y + 1,
            z: base.z + 5,
        };
        store.set_voxel(disturb, WATER);
        sim.notify_voxel_edit(disturb);
        let wake_stats = sim.step_region(
            &mut store,
            &region,
            c,
            &mut rng,
            SimulationStepMetadata::default(),
        );
        assert!(wake_stats.processed_frontier_voxels > 0);
        assert!(quiet_stats.processed_frontier_voxels <= wake_stats.processed_frontier_voxels);
    }
}
