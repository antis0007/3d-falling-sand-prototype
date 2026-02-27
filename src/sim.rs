use crate::world::{MaterialId, World, CHUNK_SIZE, EMPTY};

const STONE: MaterialId = 1;
const WATER: MaterialId = 5;
const LAVA: MaterialId = 6;
const ACID: MaterialId = 7;
const SMOKE: MaterialId = 8;
const STEAM: MaterialId = 9;
const FIRE_GAS: MaterialId = 11;

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

pub const MATERIALS: [Material; 12] = [
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
        melts_from_lava: true,
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
        melts_from_lava: true,
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
        flow_speed: 4,
        viscosity: 0.15,
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
        flow_speed: 2,
        viscosity: 0.82,
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

pub fn step(world: &mut World, rng: &mut XorShift32) {
    let dims = world.dims;
    for cz in 0..world.chunks_dims[2] {
        for cy in 0..world.chunks_dims[1] {
            for cx in 0..world.chunks_dims[0] {
                let cidx = world.chunk_index(cx, cy, cz);
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
                    if id == EMPTY || material(id).phase == Phase::Solid {
                        world.chunks[cidx].active.remove(&local_u16);
                        continue;
                    }
                    let moved = step_voxel(world, [wx, wy, wz], id, rng);
                    if moved {
                        world.chunks[cidx].active.remove(&local_u16);
                    } else {
                        let settle = &mut world.chunks[cidx].settled[local];
                        *settle = settle.saturating_add(1);
                        if *settle > 5 {
                            world.chunks[cidx].active.remove(&local_u16);
                        }
                    }
                }
            }
        }
    }
}

fn try_move(world: &mut World, from: [i32; 3], to: [i32; 3], id: MaterialId) -> bool {
    let dst = world.get(to[0], to[1], to[2]);
    if dst == EMPTY || material(dst).density < material(id).density {
        world.set(to[0], to[1], to[2], id);
        world.set(from[0], from[1], from[2], dst);
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
        Phase::Solid => false,
        Phase::Powder => {
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
            if try_move(world, p, [p[0], p[1] - 1, p[2]], id) {
                return true;
            }
            let mut diag = down_diagonals();
            rng.shuffle(&mut diag);
            let lateral_attempts = material(id).flow_speed.max(1) as usize;
            for d in diag.into_iter().take(lateral_attempts.min(8)) {
                let to = [p[0] + d[0], p[1] - 1, p[2] + d[1]];
                if is_diagonal_blocked(world, p, to) {
                    continue;
                }
                if try_move(world, p, to, id) {
                    return true;
                }
            }

            let cohesion_neighbors = count_same_neighbors(world, p, id);
            if cohesion_neighbors > 1 && rng.chance(material(id).viscosity.clamp(0.0, 0.95)) {
                return false;
            }

            let mut side = side_dirs();
            rng.shuffle(&mut side);
            for d in side.into_iter().take(lateral_attempts.min(8)) {
                if try_move(world, p, [p[0] + d[0], p[1], p[2] + d[1]], id) {
                    return true;
                }
            }
            false
        }
        Phase::Gas => {
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
                let replacement = if rng.chance(0.5) { FIRE_GAS } else { SMOKE };
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
                && (nmat.transforms_on_contact == Some(ContactReaction::LavaCoolsToWaterOrSteam)
                    || nmat.melts_from_lava)
                && rng.chance(0.45)
            {
                let replacement = if rng.chance(0.65) { WATER } else { STEAM };
                let _ = world.set(np[0], np[1], np[2], replacement);
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

    if reacted {
        world.activate_neighbors(p[0], p[1], p[2]);
    }
    reacted
}

fn is_acid_dissolvable(id: MaterialId) -> bool {
    if id == EMPTY {
        return false;
    }
    let mat = material(id);
    matches!(mat.phase, Phase::Solid | Phase::Powder) && !mat.acid_resistant
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
