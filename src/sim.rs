use crate::world::{MaterialId, World, CHUNK_SIZE, EMPTY};

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
}

pub const MATERIALS: [Material; 10] = [
    Material {
        id: 0,
        name: "Empty",
        color: [0, 0, 0, 0],
        phase: Phase::Gas,
        density: -10,
    },
    Material {
        id: 1,
        name: "Stone",
        color: [120, 120, 120, 255],
        phase: Phase::Solid,
        density: 100,
    },
    Material {
        id: 2,
        name: "Wood",
        color: [122, 81, 46, 255],
        phase: Phase::Solid,
        density: 80,
    },
    Material {
        id: 3,
        name: "Sand",
        color: [194, 178, 128, 255],
        phase: Phase::Powder,
        density: 60,
    },
    Material {
        id: 4,
        name: "Snow",
        color: [230, 235, 240, 255],
        phase: Phase::Powder,
        density: 30,
    },
    Material {
        id: 5,
        name: "Water",
        color: [64, 120, 220, 190],
        phase: Phase::Liquid,
        density: 20,
    },
    Material {
        id: 6,
        name: "Lava",
        color: [230, 100, 30, 220],
        phase: Phase::Liquid,
        density: 40,
    },
    Material {
        id: 7,
        name: "Acid",
        color: [60, 220, 90, 210],
        phase: Phase::Liquid,
        density: 25,
    },
    Material {
        id: 8,
        name: "Smoke",
        color: [120, 120, 120, 140],
        phase: Phase::Gas,
        density: 5,
    },
    Material {
        id: 9,
        name: "Steam",
        color: [190, 190, 210, 120],
        phase: Phase::Gas,
        density: 2,
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
                if try_move(world, p, [p[0] + d[0], p[1] - 1, p[2] + d[1]], id) {
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
            for d in diag {
                if try_move(world, p, [p[0] + d[0], p[1] - 1, p[2] + d[1]], id) {
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
        Phase::Gas => {
            if try_move(world, p, [p[0], p[1] + 1, p[2]], id) {
                return true;
            }
            let mut diag = down_diagonals();
            rng.shuffle(&mut diag);
            for d in diag {
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
