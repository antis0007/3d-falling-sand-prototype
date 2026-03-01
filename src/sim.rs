use crate::world::{MaterialId, World, CHUNK_SIZE, EMPTY};

const STONE: MaterialId = 1;
const WOOD: MaterialId = 2;
const WATER: MaterialId = 5;
const LAVA: MaterialId = 6;
const ACID: MaterialId = 7;
const SMOKE: MaterialId = 8;
const STEAM: MaterialId = 9;
const FIRE_GAS: MaterialId = 11;
const TORCH: MaterialId = 12;
const EMBER_HOT: MaterialId = 13;
const EMBER_WARM: MaterialId = 14;
const EMBER_ASH: MaterialId = 15;
const DIRT: MaterialId = 16;
const TURF: MaterialId = 17;
const BUSH: MaterialId = 18;
const GRASS: MaterialId = 19;
const PLANT: MaterialId = 20;
const WEED: MaterialId = 21;
const TREE_SEED: MaterialId = 22;
const LEAVES: MaterialId = 23;
const DEAD_LEAF: MaterialId = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Solid,
    Powder,
    Liquid,
    Gas,
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    // Future hooks: pressure/temperature/reaction coefficients can live here for advanced fluids/gases.
    pub id: MaterialId,
    pub name: &'static str,
    pub color: [u8; 4],
    pub phase: Phase,
    pub density: i16,
    pub flammable: bool,
    pub acid_resistant: bool,
    pub melts_from_lava: bool,
    pub transforms_on_contact: Option<ContactReaction>,
    pub flow_speed: u8,
    pub viscosity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactReaction {
    LavaCoolsToWaterOrSteam,
    WaterVsLava,
    BurnsIntoSmoke,
}

pub const MATERIALS: [Material; 25] = [
    Material {
        id: 0,
        name: "Empty",
        color: [0, 0, 0, 0],
        phase: Phase::Gas,
        density: -10,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 1,
        name: "Stone",
        color: [120, 120, 120, 255],
        phase: Phase::Solid,
        density: 100,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 2,
        name: "Wood",
        color: [122, 81, 46, 255],
        phase: Phase::Solid,
        density: 80,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 3,
        name: "Sand",
        color: [194, 178, 128, 255],
        phase: Phase::Powder,
        density: 60,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 4,
        name: "Snow",
        color: [230, 235, 240, 255],
        phase: Phase::Powder,
        density: 30,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: true,
        transforms_on_contact: Some(ContactReaction::LavaCoolsToWaterOrSteam),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 5,
        name: "Water",
        color: [64, 120, 220, 190],
        phase: Phase::Liquid,
        density: 20,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::WaterVsLava),
        flow_speed: 5,
        viscosity: 0.05,
    },
    Material {
        id: 6,
        name: "Lava",
        color: [230, 100, 30, 220],
        phase: Phase::Liquid,
        density: 40,
        flammable: false,
        acid_resistant: true,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::LavaCoolsToWaterOrSteam),
        flow_speed: 1,
        viscosity: 0.92,
    },
    Material {
        id: 7,
        name: "Acid",
        color: [60, 220, 90, 210],
        phase: Phase::Liquid,
        density: 25,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 3,
        viscosity: 0.32,
    },
    Material {
        id: 8,
        name: "Smoke",
        color: [120, 120, 120, 140],
        phase: Phase::Gas,
        density: 5,
        flammable: false,
        acid_resistant: true,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 2,
        viscosity: 0.05,
    },
    Material {
        id: 9,
        name: "Steam",
        color: [190, 190, 210, 120],
        phase: Phase::Gas,
        density: 2,
        flammable: false,
        acid_resistant: true,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 3,
        viscosity: 0.05,
    },
    Material {
        id: 10,
        name: "Steel",
        color: [112, 128, 140, 255],
        phase: Phase::Solid,
        density: 160,
        flammable: false,
        acid_resistant: true,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 11,
        name: "Fire Gas",
        color: [255, 155, 72, 120],
        phase: Phase::Gas,
        density: 1,
        flammable: false,
        acid_resistant: true,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 4,
        viscosity: 0.0,
    },
    Material {
        id: 12,
        name: "Torch",
        color: [255, 184, 96, 255],
        phase: Phase::Solid,
        density: 100,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 13,
        name: "Ember (Hot)",
        color: [255, 105, 38, 255],
        phase: Phase::Solid,
        density: 70,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 14,
        name: "Ember (Warm)",
        color: [168, 76, 52, 255],
        phase: Phase::Solid,
        density: 70,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 15,
        name: "Ember Ash",
        color: [90, 84, 84, 255],
        phase: Phase::Powder,
        density: 45,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 16,
        name: "Dirt",
        color: [121, 88, 56, 255],
        phase: Phase::Solid,
        density: 65,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 17,
        name: "Turf",
        color: [114, 108, 67, 255],
        phase: Phase::Solid,
        density: 65,
        flammable: false,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: None,
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 18,
        name: "Bush",
        color: [72, 156, 64, 220],
        phase: Phase::Solid,
        density: 10,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 19,
        name: "Grass",
        color: [94, 186, 72, 220],
        phase: Phase::Solid,
        density: 8,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 20,
        name: "Plant",
        color: [94, 186, 72, 255],
        phase: Phase::Solid,
        density: 50,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 21,
        name: "Weed",
        color: [78, 146, 62, 255],
        phase: Phase::Solid,
        density: 50,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 22,
        name: "Tree Seed",
        color: [142, 92, 52, 255],
        phase: Phase::Solid,
        density: 55,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 23,
        name: "Leaves",
        color: [70, 156, 66, 255],
        phase: Phase::Solid,
        density: 35,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
    Material {
        id: 24,
        name: "Dead Leaves",
        color: [170, 114, 60, 255],
        phase: Phase::Powder,
        density: 14,
        flammable: true,
        acid_resistant: false,
        melts_from_lava: false,
        transforms_on_contact: Some(ContactReaction::BurnsIntoSmoke),
        flow_speed: 0,
        viscosity: 1.0,
    },
];

pub fn material(id: MaterialId) -> &'static Material {
    MATERIALS.get(id as usize).unwrap_or(&MATERIALS[0])
}

#[derive(Clone)]
pub struct SimState {
    pub running: bool,
    pub accumulator: f32,
    pub fixed_dt: f32,
    pub rng: XorShift32,
}

impl Default for SimState {
    fn default() -> Self {
        Self {
            running: false,
            accumulator: 0.0,
            fixed_dt: 1.0 / 60.0,
            rng: XorShift32::new(0xC0FFEE11),
        }
    }
}

#[derive(Clone)]
pub struct XorShift32 {
    state: u32,
}
impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }
    pub fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    pub fn shuffle<T>(&mut self, arr: &mut [T]) {
        for i in (1..arr.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            arr.swap(i, j);
        }
    }

    pub fn chance(&mut self, p: f32) -> bool {
        let threshold = (p.clamp(0.0, 1.0) * (u32::MAX as f32)) as u32;
        self.next() <= threshold
    }
}

const WATER_BASE_EVAPORATION_CHANCE: f32 = 0.006;
const WATER_HOT_EVAPORATION_CHANCE: f32 = 0.035;
const WATER_HOTSPOT_RADIUS_SQ: i32 = 9;
const WATER_ISOLATION_RADIUS_SQ: i32 = 9;

pub fn step(world: &mut World, rng: &mut XorShift32) {
    let mut all_chunks = Vec::with_capacity(world.chunks.len());
    for cz in 0..world.chunks_dims[2] {
        for cy in 0..world.chunks_dims[1] {
            for cx in 0..world.chunks_dims[0] {
                all_chunks.push(world.chunk_index(cx, cy, cz));
            }
        }
    }
    step_selected_chunks(world, rng, &all_chunks);
}

pub fn step_selected_chunks(world: &mut World, rng: &mut XorShift32, chunk_indices: &[usize]) {
    let dims = world.dims;
    for &cidx in chunk_indices {
        let [cx, cy, cz] = chunk_index_to_coord(world, cidx);
        let active_list: Vec<u16> = world.chunks[cidx].active.iter().copied().collect();
        for local_u16 in active_list {
            let local = local_u16 as usize;
            if !world.chunks[cidx].active.contains(&local_u16) {
                continue;
            }
            let lx = local % CHUNK_SIZE;
            let ly = (local / CHUNK_SIZE) % CHUNK_SIZE;
            let lz = local / (CHUNK_SIZE * CHUNK_SIZE);
            let wx = (cx * CHUNK_SIZE + lx) as i32;
            let wy = (cy * CHUNK_SIZE + ly) as i32;
            let wz = (cz * CHUNK_SIZE + lz) as i32;
            if wx < 0
                || wy < 0
                || wz < 0
                || wx >= dims[0] as i32
                || wy >= dims[1] as i32
                || wz >= dims[2] as i32
            {
                world.chunks[cidx].active.remove(&local_u16);
                continue;
            }
            let id = world.get(wx, wy, wz);
            if id == EMPTY || (material(id).phase == Phase::Solid && !is_reactive_solid(id)) {
                world.chunks[cidx].active.remove(&local_u16);
                continue;
            }
            let moved = step_voxel(world, [wx, wy, wz], id, rng);
            if moved {
                world.chunks[cidx].active.remove(&local_u16);
            } else {
                if is_reactive_solid(id) {
                    world.chunks[cidx].settled[local] = 0;
                    continue;
                }
                let settle = &mut world.chunks[cidx].settled[local];
                *settle = settle.saturating_add(1);
                if *settle > 5 {
                    world.chunks[cidx].active.remove(&local_u16);
                }
            }
        }
    }
}

fn chunk_index_to_coord(world: &World, cidx: usize) -> [usize; 3] {
    let w = world.chunks_dims[0];
    let h = world.chunks_dims[1];
    let cz = cidx / (w * h);
    let rem = cidx % (w * h);
    let cy = rem / w;
    let cx = rem % w;
    [cx, cy, cz]
}

fn try_move(world: &mut World, from: [i32; 3], to: [i32; 3], id: MaterialId) -> bool {
    let dst = world.get(to[0], to[1], to[2]);
    if dst == EMPTY || material(dst).density < material(id).density {
        if !world.set(to[0], to[1], to[2], id) {
            return false;
        }
        let _ = world.set(from[0], from[1], from[2], dst);
        world.activate_neighbors(to[0], to[1], to[2]);
        return true;
    }
    false
}

// Rule dispatch point: swap this function for more realistic velocity/pressure-based solvers later.
pub fn step_voxel(world: &mut World, p: [i32; 3], id: MaterialId, rng: &mut XorShift32) -> bool {
    if react_voxel(world, p, id, rng) {
        return true;
    }

    let id = world.get(p[0], p[1], p[2]);
    if id == EMPTY {
        return true;
    }

    let mat = material(id);
    match mat.phase {
        Phase::Solid => {
            if id == LEAVES {
                return update_leaf_lifecycle(world, p, rng);
            }
            if id == BUSH || id == GRASS {
                if !has_solid_support_below(world, p) {
                    return world.set(p[0], p[1], p[2], EMPTY);
                }
            }
            false
        }
        Phase::Powder => {
            if id == DEAD_LEAF {
                if p[1] > 0 && try_move(world, p, [p[0], p[1] - 1, p[2]], id) {
                    return true;
                }
                if rng.chance(0.0012) {
                    return world.set(p[0], p[1], p[2], DIRT);
                }
                if rng.chance(0.0006) {
                    return world.set(p[0], p[1], p[2], EMPTY);
                }
            }
            if p[1] == 0 {
                return world.set(p[0], p[1], p[2], EMPTY);
            }
            if try_move(world, p, [p[0], p[1] - 1, p[2]], id) {
                return true;
            }
            let mut dirs = down_diagonals();
            rng.shuffle(&mut dirs);
            for d in dirs {
                let to = [p[0] + d[0], p[1] - 1, p[2] + d[1]];
                if is_diagonal_blocked(world, p, to) {
                    continue;
                }
                if try_move(world, p, to, id) {
                    return true;
                }
            }
            false
        }
        Phase::Liquid => {
            if p[1] == 0 {
                return world.set(p[0], p[1], p[2], EMPTY);
            }
            if id == WATER {
                let is_isolated =
                    !has_same_neighbor_in_radius(world, p, WATER, WATER_ISOLATION_RADIUS_SQ);
                if is_isolated {
                    let near_heat = has_heat_source_nearby(world, p, WATER_HOTSPOT_RADIUS_SQ);
                    let chance = if near_heat {
                        WATER_HOT_EVAPORATION_CHANCE
                    } else {
                        WATER_BASE_EVAPORATION_CHANCE
                    };
                    if rng.chance(chance) {
                        return world.set(p[0], p[1], p[2], EMPTY);
                    }
                }
            }
            if try_move(world, p, [p[0], p[1] - 1, p[2]], id) {
                return true;
            }

            let current_cohesion = liquid_cohesion_score(world, p, id);
            let surface_tension = liquid_surface_tension(material(id));
            let pressure = liquid_pressure_head(world, p, id);

            let mut diag = down_diagonals();
            rng.shuffle(&mut diag);
            let lateral_attempts = material(id).flow_speed.max(1) as usize;
            let mut best_diagonal_move: Option<([i32; 3], i32)> = None;
            for d in diag.into_iter().take(lateral_attempts.min(8)) {
                let to = [p[0] + d[0], p[1] - 1, p[2] + d[1]];
                if is_diagonal_blocked(world, p, to) {
                    continue;
                }
                if !can_displace(world, to, id) {
                    continue;
                }
                let cohesion = liquid_cohesion_score(world, to, id);
                best_diagonal_move = match best_diagonal_move {
                    Some((_, best)) if best >= cohesion => best_diagonal_move,
                    _ => Some((to, cohesion)),
                };
            }

            if let Some((to, cohesion)) = best_diagonal_move {
                let cohesion_delta = cohesion - current_cohesion;
                if cohesion_delta >= -1
                    || rng.chance((1.0 - surface_tension) * 0.5 + pressure * 0.2)
                {
                    return try_move(world, p, to, id);
                }
            }

            let cohesion_neighbors = count_same_neighbors(world, p, id);
            if cohesion_neighbors > 1
                && rng.chance(
                    (material(id).viscosity * 0.75 + surface_tension * 0.2).clamp(0.0, 0.97),
                )
            {
                return false;
            }

            let mut side = side_dirs();
            rng.shuffle(&mut side);
            let mut best_side_move: Option<([i32; 3], i32)> = None;
            for d in side.into_iter().take(lateral_attempts.min(8)) {
                let to = [p[0] + d[0], p[1], p[2] + d[1]];
                if !can_displace(world, to, id) {
                    continue;
                }
                let cohesion = liquid_cohesion_score(world, to, id);
                best_side_move = match best_side_move {
                    Some((_, best)) if best >= cohesion => best_side_move,
                    _ => Some((to, cohesion)),
                };
            }

            if let Some((to, cohesion)) = best_side_move {
                let cohesion_delta = cohesion - current_cohesion;
                if cohesion_delta >= 0
                    || (pressure > 0.25 && cohesion_delta >= -1)
                    || rng.chance((1.0 - surface_tension) * 0.22 + pressure * 0.32)
                {
                    return try_move(world, p, to, id);
                }
            }
            false
        }
        Phase::Gas => {
            if p[1] >= world.dims[1] as i32 - 1 {
                return world.set(p[0], p[1], p[2], EMPTY);
            }
            if try_move(world, p, [p[0], p[1] + 1, p[2]], id) {
                return true;
            }
            let mut diag = down_diagonals();
            rng.shuffle(&mut diag);
            for d in diag
                .into_iter()
                .take(material(id).flow_speed.max(1) as usize)
            {
                if try_move(world, p, [p[0] + d[0], p[1] + 1, p[2] + d[1]], id) {
                    return true;
                }
            }
            let mut side = side_dirs();
            rng.shuffle(&mut side);
            for d in side {
                if try_move(world, p, [p[0] + d[0], p[1], p[2] + d[1]], id) {
                    return true;
                }
            }
            false
        }
    }
}

fn react_voxel(world: &mut World, p: [i32; 3], id: MaterialId, rng: &mut XorShift32) -> bool {
    let mut reacted = false;
    let mat = material(id);

    if id == FIRE_GAS {
        let mut neighbors = neighbor_dirs6();
        rng.shuffle(&mut neighbors);
        for [dx, dy, dz] in neighbors {
            let np = [p[0] + dx, p[1] + dy, p[2] + dz];
            let nid = world.get(np[0], np[1], np[2]);
            if nid == EMPTY {
                continue;
            }
            if material(nid).flammable && rng.chance(0.2) {
                let replacement = if nid == WOOD {
                    EMBER_HOT
                } else if rng.chance(0.5) {
                    FIRE_GAS
                } else {
                    SMOKE
                };
                let _ = world.set(np[0], np[1], np[2], replacement);
                reacted = true;
            }
        }

        if rng.chance(0.06) {
            let _ = world.set(p[0], p[1], p[2], SMOKE);
            reacted = true;
        } else if rng.chance(0.08) {
            let _ = world.set(p[0], p[1], p[2], EMPTY);
            reacted = true;
        }
    }

    let mut neighbors = neighbor_dirs6();
    rng.shuffle(&mut neighbors);

    if id == ACID {
        for [dx, dy, dz] in neighbors {
            let np = [p[0] + dx, p[1] + dy, p[2] + dz];
            let nid = world.get(np[0], np[1], np[2]);
            if !is_acid_dissolvable(nid) || !rng.chance(0.24) {
                continue;
            }

            if world.set(np[0], np[1], np[2], EMPTY) {
                reacted = true;
                if rng.chance(0.40) {
                    let byproduct = if rng.chance(0.55) { STEAM } else { SMOKE };
                    let _ = spawn_reaction_product(world, np, byproduct, rng);
                }
                if rng.chance(0.10) {
                    let _ = world.set(p[0], p[1], p[2], EMPTY);
                }
                break;
            }
        }
    }

    if id == LAVA || id == WATER {
        let mut neighbors = neighbor_dirs6();
        rng.shuffle(&mut neighbors);
        for [dx, dy, dz] in neighbors {
            let np = [p[0] + dx, p[1] + dy, p[2] + dz];
            let nid = world.get(np[0], np[1], np[2]);
            let nmat = material(nid);

            if ((id == LAVA && nid == WATER) || (id == WATER && nid == LAVA)) && rng.chance(0.35) {
                let _ = world.set(p[0], p[1], p[2], STONE);
                let replacement = if rng.chance(0.60) { STEAM } else { EMPTY };
                let _ = world.set(np[0], np[1], np[2], replacement);
                reacted = true;
                break;
            }

            if id == LAVA
                && nid != LAVA
                && nmat.transforms_on_contact == Some(ContactReaction::LavaCoolsToWaterOrSteam)
                && rng.chance(0.45)
            {
                let replacement = if rng.chance(0.65) { WATER } else { STEAM };
                let _ = world.set(np[0], np[1], np[2], replacement);
                reacted = true;
                continue;
            }

            if id == LAVA && nmat.melts_from_lava && rng.chance(0.45) {
                let _ = world.set(np[0], np[1], np[2], STONE);
                let _ = spawn_reaction_product(world, np, STEAM, rng);
                reacted = true;
                continue;
            }

            if id == LAVA && nmat.flammable && rng.chance(0.18) {
                let replacement = if rng.chance(0.55) { FIRE_GAS } else { SMOKE };
                let _ = world.set(np[0], np[1], np[2], replacement);
                let _ = spawn_reaction_product(world, np, SMOKE, rng);
                reacted = true;
            } else if id == LAVA && nmat.flammable && rng.chance(0.12) {
                let _ = spawn_reaction_product(world, np, FIRE_GAS, rng);
                reacted = true;
            }
        }
    }

    if mat.flammable && rng.chance(0.06) {
        for [dx, dy, dz] in neighbor_dirs6() {
            let np = [p[0] + dx, p[1] + dy, p[2] + dz];
            let nid = world.get(np[0], np[1], np[2]);
            if nid == LAVA || nid == FIRE_GAS {
                let replacement = if rng.chance(0.6) { FIRE_GAS } else { SMOKE };
                let _ = world.set(p[0], p[1], p[2], replacement);
                let _ = spawn_reaction_product(world, p, SMOKE, rng);
                reacted = true;
                break;
            }
        }
    }

    if id == TORCH {
        if rng.chance(0.55) {
            reacted |= spawn_reaction_product(world, p, FIRE_GAS, rng);
        }
        if rng.chance(0.18) {
            reacted |= spawn_reaction_product(world, p, SMOKE, rng);
        }
    }

    if id == WOOD && has_ignition_neighbor(world, p) && rng.chance(0.52) {
        reacted |= world.set(p[0], p[1], p[2], EMBER_HOT);
        let _ = spawn_reaction_product(world, p, FIRE_GAS, rng);
    }

    if id == EMBER_HOT {
        if rng.chance(0.52) {
            let _ = spawn_reaction_product(world, p, FIRE_GAS, rng);
            reacted = true;
        }
        if rng.chance(0.03) {
            reacted |= world.set(p[0], p[1], p[2], EMBER_WARM);
        }
    } else if id == EMBER_WARM {
        if rng.chance(0.26) {
            let _ = spawn_reaction_product(world, p, SMOKE, rng);
            reacted = true;
        }
        if rng.chance(0.02) {
            reacted |= world.set(p[0], p[1], p[2], EMBER_ASH);
        }
    } else if id == EMBER_ASH && rng.chance(0.003) {
        reacted |= world.set(p[0], p[1], p[2], EMPTY);
    }

    if id == PLANT {
        let has_water = has_neighbor(world, p, WATER);
        if !has_water && rng.chance(0.008) {
            reacted |= world.set(p[0], p[1], p[2], WEED);
        } else if has_water {
            reacted |= try_grow_plant(world, p, PLANT, rng, true, 0.012);
        }

        if rng.chance(0.0025) {
            reacted |= world.set(p[0], p[1], p[2], WOOD);
        }
    }

    if id == WEED {
        let has_water = has_neighbor(world, p, WATER);
        if has_water && rng.chance(0.08) {
            reacted |= world.set(p[0], p[1], p[2], PLANT);
        } else {
            reacted |= try_grow_plant(world, p, WEED, rng, false, 0.006);
        }

        if rng.chance(0.0015) {
            reacted |= world.set(p[0], p[1], p[2], WOOD);
        }
    }

    if id == DIRT {
        if is_exposed_to_sky(world, p) && rng.chance(0.008) {
            reacted |= world.set(p[0], p[1], p[2], TURF);
        }
    }

    if id == TURF {
        let above = world.get(p[0], p[1] + 1, p[2]);
        if above != EMPTY && material(above).phase != Phase::Gas {
            if rng.chance(0.20) {
                reacted |= world.set(p[0], p[1], p[2], DIRT);
            }
        } else if rng.chance(0.0004) {
            let grow_id = if rng.chance(0.55) { GRASS } else { BUSH };
            reacted |= try_spawn_surface_plant(world, p, grow_id, rng);
        }
    }

    if id == TREE_SEED {
        reacted |= germinate_tree(world, p, rng);
    }

    if id == WOOD && try_absorb_water_for_tree_growth(world, p, rng) {
        reacted = true;
    }

    if reacted {
        world.activate_neighbors(p[0], p[1], p[2]);
    }
    reacted
}

fn is_reactive_solid(id: MaterialId) -> bool {
    matches!(
        id,
        WOOD | TORCH
            | EMBER_HOT
            | EMBER_WARM
            | PLANT
            | WEED
            | TREE_SEED
            | DIRT
            | TURF
            | LEAVES
            | BUSH
            | GRASS
            | DEAD_LEAF
    )
}

fn has_neighbor(world: &World, p: [i32; 3], target: MaterialId) -> bool {
    neighbor_dirs6()
        .into_iter()
        .any(|[dx, dy, dz]| world.get(p[0] + dx, p[1] + dy, p[2] + dz) == target)
}

fn has_ignition_neighbor(world: &World, p: [i32; 3]) -> bool {
    for dz in -2..=2 {
        for dy in -1..=2 {
            for dx in -2..=2 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                if dx * dx + dy * dy + dz * dz > 5 {
                    continue;
                }
                let nid = world.get(p[0] + dx, p[1] + dy, p[2] + dz);
                if matches!(nid, LAVA | FIRE_GAS | TORCH | EMBER_HOT | EMBER_WARM) {
                    return true;
                }
            }
        }
    }
    false
}

fn try_grow_plant(
    world: &mut World,
    p: [i32; 3],
    plant_id: MaterialId,
    rng: &mut XorShift32,
    needs_water: bool,
    grow_chance: f32,
) -> bool {
    if needs_water && !has_neighbor(world, p, WATER) {
        return false;
    }

    let same_neighbors = neighbor_dirs6()
        .into_iter()
        .filter(|[dx, dy, dz]| world.get(p[0] + dx, p[1] + dy, p[2] + dz) == plant_id)
        .count();
    if same_neighbors >= 3 {
        return false;
    }

    let mut dirs = [
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 1, 1],
        [0, 1, -1],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];
    rng.shuffle(&mut dirs);
    for [dx, dy, dz] in dirs {
        if !rng.chance(grow_chance) {
            continue;
        }
        let np = [p[0] + dx, p[1] + dy, p[2] + dz];
        if world.get(np[0], np[1], np[2]) != EMPTY {
            continue;
        }
        let below = world.get(np[0], np[1] - 1, np[2]);
        if below == EMPTY || material(below).phase == Phase::Gas {
            continue;
        }
        return world.set(np[0], np[1], np[2], plant_id);
    }
    false
}

fn try_spawn_surface_plant(
    world: &mut World,
    p: [i32; 3],
    plant_id: MaterialId,
    rng: &mut XorShift32,
) -> bool {
    let np = [p[0], p[1] + 1, p[2]];
    if world.get(np[0], np[1], np[2]) != EMPTY {
        return false;
    }
    if !has_solid_support_below(world, np) {
        return false;
    }
    if rng.chance(0.2) {
        return world.set(np[0], np[1], np[2], plant_id);
    }
    false
}

fn update_leaf_lifecycle(world: &mut World, p: [i32; 3], rng: &mut XorShift32) -> bool {
    if has_tree_support(world, p) {
        return false;
    }

    if p[1] > 0 {
        let below = [p[0], p[1] - 1, p[2]];
        if world.get(below[0], below[1], below[2]) == EMPTY && try_move(world, p, below, LEAVES) {
            return true;
        }
    }

    if world.set(p[0], p[1], p[2], DEAD_LEAF) {
        return true;
    }

    if rng.chance(0.002) {
        return world.set(p[0], p[1], p[2], DIRT);
    }
    if rng.chance(0.0015) {
        return world.set(p[0], p[1], p[2], EMPTY);
    }
    false
}

fn has_tree_support(world: &World, p: [i32; 3]) -> bool {
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                if dx * dx + dy * dy + dz * dz > 5 {
                    continue;
                }
                let nid = world.get(p[0] + dx, p[1] + dy, p[2] + dz);
                if matches!(nid, WOOD | LEAVES) {
                    return true;
                }
            }
        }
    }
    false
}

fn has_solid_support_below(world: &World, p: [i32; 3]) -> bool {
    let below = world.get(p[0], p[1] - 1, p[2]);
    below != EMPTY && material(below).phase != Phase::Gas
}

fn has_same_neighbor_in_radius(world: &World, p: [i32; 3], id: MaterialId, radius_sq: i32) -> bool {
    for dz in -3..=3 {
        for dy in -3..=3 {
            for dx in -3..=3 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 == 0 || dist2 > radius_sq {
                    continue;
                }
                if world.get(p[0] + dx, p[1] + dy, p[2] + dz) == id {
                    return true;
                }
            }
        }
    }
    false
}

fn has_heat_source_nearby(world: &World, p: [i32; 3], radius_sq: i32) -> bool {
    for dz in -3..=3 {
        for dy in -3..=3 {
            for dx in -3..=3 {
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 > radius_sq {
                    continue;
                }
                let nid = world.get(p[0] + dx, p[1] + dy, p[2] + dz);
                if matches!(nid, LAVA | FIRE_GAS | TORCH | EMBER_HOT | EMBER_WARM) {
                    return true;
                }
            }
        }
    }
    false
}

fn is_exposed_to_sky(world: &World, p: [i32; 3]) -> bool {
    for y in p[1] + 1..world.dims[1] as i32 {
        if world.get(p[0], y, p[2]) != EMPTY {
            return false;
        }
    }
    true
}

fn germinate_tree(world: &mut World, p: [i32; 3], rng: &mut XorShift32) -> bool {
    if !world.set(p[0], p[1], p[2], WOOD) {
        return false;
    }

    let trunk_height = 4 + (rng.next() % 6) as i32;
    let mut top = p;
    for i in 1..=trunk_height {
        let np = [p[0], p[1] + i, p[2]];
        if world.get(np[0], np[1], np[2]) != EMPTY {
            break;
        }
        let _ = world.set(np[0], np[1], np[2], WOOD);
        top = np;
    }

    let branch_count = 3 + (rng.next() % 4) as i32;
    for _ in 0..branch_count {
        grow_branch(world, top, rng);
    }

    grow_leaf_crown(world, top, 3 + (rng.next() % 2) as i32, rng);
    true
}

fn grow_branch(world: &mut World, from: [i32; 3], rng: &mut XorShift32) {
    let dirs = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
    ];
    let mut pos = from;
    let dir = dirs[(rng.next() as usize) % dirs.len()];
    let len = 2 + (rng.next() % 4) as i32;
    for i in 0..len {
        pos = [
            pos[0] + dir[0],
            pos[1] + if i % 2 == 0 { 1 } else { 0 },
            pos[2] + dir[2],
        ];
        if world.get(pos[0], pos[1], pos[2]) != EMPTY {
            break;
        }
        let _ = world.set(pos[0], pos[1], pos[2], WOOD);
    }
    grow_leaf_crown(world, pos, 2, rng);
}

fn grow_leaf_crown(world: &mut World, center: [i32; 3], radius: i32, rng: &mut XorShift32) {
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 > radius * radius {
                    continue;
                }
                if dist2 > (radius - 1).max(0) * (radius - 1).max(0) && rng.chance(0.25) {
                    continue;
                }
                let np = [center[0] + dx, center[1] + dy, center[2] + dz];
                if world.get(np[0], np[1], np[2]) == EMPTY {
                    let _ = world.set(np[0], np[1], np[2], LEAVES);
                }
            }
        }
    }
}

fn try_absorb_water_for_tree_growth(world: &mut World, p: [i32; 3], rng: &mut XorShift32) -> bool {
    let mut water_neighbor: Option<[i32; 3]> = None;
    for [dx, dy, dz] in neighbor_dirs6() {
        let np = [p[0] + dx, p[1] + dy, p[2] + dz];
        if world.get(np[0], np[1], np[2]) == WATER {
            water_neighbor = Some(np);
            break;
        }
    }

    let Some(water_pos) = water_neighbor else {
        return false;
    };

    if !world.set(water_pos[0], water_pos[1], water_pos[2], EMPTY) {
        return false;
    }

    let mut growth_dirs = [
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 1, 1],
        [0, 1, -1],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];
    rng.shuffle(&mut growth_dirs);
    for [dx, dy, dz] in growth_dirs {
        let np = [p[0] + dx, p[1] + dy, p[2] + dz];
        if world.get(np[0], np[1], np[2]) != EMPTY {
            continue;
        }
        let grow_wood = dy > 0 && rng.chance(0.55);
        let id = if grow_wood { WOOD } else { LEAVES };
        return world.set(np[0], np[1], np[2], id);
    }

    true
}

fn is_acid_dissolvable(id: MaterialId) -> bool {
    if id == EMPTY {
        return false;
    }
    let mat = material(id);
    matches!(mat.phase, Phase::Solid | Phase::Powder) && !mat.acid_resistant
}

fn can_displace(world: &World, to: [i32; 3], id: MaterialId) -> bool {
    let dst = world.get(to[0], to[1], to[2]);
    dst == EMPTY || material(dst).density < material(id).density
}

fn liquid_surface_tension(mat: &Material) -> f32 {
    (0.2 + mat.viscosity * 0.75).clamp(0.08, 0.97)
}

fn liquid_pressure_head(world: &World, p: [i32; 3], id: MaterialId) -> f32 {
    let mut depth = 0;
    while depth < 4 {
        if world.get(p[0], p[1] + depth, p[2]) != id {
            break;
        }
        depth += 1;
    }
    ((depth as f32) - 1.0).max(0.0) / 4.0
}

fn liquid_cohesion_score(world: &World, p: [i32; 3], id: MaterialId) -> i32 {
    let mut score = 0;

    for [dx, dz] in side_dirs() {
        if world.get(p[0] + dx, p[1], p[2] + dz) == id {
            score += 2;
        }
    }

    let below = world.get(p[0], p[1] - 1, p[2]);
    if below == id {
        score += 3;
    } else if below != EMPTY {
        score += 1;
    }

    if world.get(p[0], p[1] + 1, p[2]) == id {
        score += 1;
    }

    score
}

fn count_same_neighbors(world: &World, p: [i32; 3], id: MaterialId) -> usize {
    neighbor_dirs6()
        .into_iter()
        .filter(|[dx, dy, dz]| world.get(p[0] + dx, p[1] + dy, p[2] + dz) == id)
        .count()
}

fn is_diagonal_blocked(world: &World, from: [i32; 3], to: [i32; 3]) -> bool {
    let dx = to[0] - from[0];
    let dz = to[2] - from[2];
    if dx == 0 && dz == 0 {
        return false;
    }

    let side_x = world.get(from[0] + dx, from[1], from[2]) != EMPTY;
    let side_z = world.get(from[0], from[1], from[2] + dz) != EMPTY;
    let under = world.get(from[0], to[1], from[2]) != EMPTY;
    (side_x && side_z) || (under && (side_x || side_z))
}

fn spawn_reaction_product(
    world: &mut World,
    origin: [i32; 3],
    product: MaterialId,
    rng: &mut XorShift32,
) -> bool {
    let mut dirs = neighbor_dirs6();
    rng.shuffle(&mut dirs);
    for [dx, dy, dz] in dirs {
        let np = [origin[0] + dx, origin[1] + dy, origin[2] + dz];
        if world.get(np[0], np[1], np[2]) == EMPTY {
            return world.set(np[0], np[1], np[2], product);
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

fn down_diagonals() -> [[i32; 2]; 8] {
    [
        [-1, -1],
        [0, -1],
        [1, -1],
        [-1, 0],
        [1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
    ]
}
fn side_dirs() -> [[i32; 2]; 8] {
    [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
    ]
}
