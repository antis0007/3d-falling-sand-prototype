use crate::world::{MaterialId, World, EMPTY};

const STONE: MaterialId = 1;
const WOOD: MaterialId = 2;
const SAND: MaterialId = 3;
const WATER: MaterialId = 5;
const DIRT: MaterialId = 16;
const TURF: MaterialId = 17;
const BUSH: MaterialId = 18;
const GRASS: MaterialId = 19;
const LEAVES: MaterialId = 23;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeType {
    Forest,
    Desert,
    River,
    Lake,
}

#[derive(Clone, Copy, Debug)]
pub struct ProcGenConfig {
    pub dims: [usize; 3],
    pub seed: u64,
    pub base_height: f32,
    pub terrain_variation: f32,
    pub cave_density: f32,
    pub cave_depth_start: usize,
    pub tree_density: f32,
    pub water_level: usize,
}

impl ProcGenConfig {
    pub fn for_size(size: usize, seed: u64) -> Self {
        let water_level = (size / 3).clamp(8, size.saturating_sub(4));
        Self {
            dims: [size, size, size],
            seed,
            base_height: size as f32 * 0.32,
            terrain_variation: size as f32 * 0.14,
            cave_density: 0.18,
            cave_depth_start: size / 10,
            tree_density: 0.035,
            water_level,
        }
    }
}

pub fn generate_world(config: ProcGenConfig) -> World {
    let mut world = World::new(config.dims);
    world.clear();

    let biome_grid = build_chunk_biome_field(&config);

    base_terrain_pass(&mut world, &config, &biome_grid);
    cave_carve_pass(&mut world, &config, &biome_grid);
    surface_layering_pass(&mut world, &config, &biome_grid);
    biome_decoration_pass(&mut world, &config, &biome_grid);
    vegetation_pass(&mut world, &config, &biome_grid);

    world
}

fn build_chunk_biome_field(config: &ProcGenConfig) -> Vec<BiomeType> {
    let chunk_dims = [
        config.dims[0] / 16,
        config.dims[1] / 16,
        config.dims[2] / 16,
    ];
    let mut out = vec![BiomeType::Forest; chunk_dims[0] * chunk_dims[2]];
    for cz in 0..chunk_dims[2] {
        for cx in 0..chunk_dims[0] {
            let noise = hash01(config.seed, cx as i32, 0, cz as i32);
            let biome = if noise < 0.2 {
                BiomeType::River
            } else if noise < 0.4 {
                BiomeType::Desert
            } else if noise < 0.52 {
                BiomeType::Lake
            } else {
                BiomeType::Forest
            };
            out[cx + cz * chunk_dims[0]] = biome;
        }
    }
    out
}

fn sample_biome_weights(
    config: &ProcGenConfig,
    biome_grid: &[BiomeType],
    x: usize,
    z: usize,
) -> [f32; 4] {
    let chunk_dims_x = config.dims[0] / 16;
    let fx = x as f32 / 16.0;
    let fz = z as f32 / 16.0;
    let x0 = fx.floor() as i32;
    let z0 = fz.floor() as i32;
    let tx = fx.fract();
    let tz = fz.fract();

    let mut weights = [0.0; 4];
    for oz in 0..=1 {
        for ox in 0..=1 {
            let sx = (x0 + ox).clamp(0, chunk_dims_x as i32 - 1) as usize;
            let sz = (z0 + oz).clamp(0, config.dims[2] as i32 / 16 - 1) as usize;
            let anchor = biome_grid[sx + sz * chunk_dims_x];
            let wx = if ox == 0 { 1.0 - tx } else { tx };
            let wz = if oz == 0 { 1.0 - tz } else { tz };
            let w = wx * wz;
            weights[biome_index(anchor)] += w;
        }
    }

    let warp = hash_signed(config.seed ^ 0xA02B_DB10, x as i32, 0, z as i32) * 0.15;
    weights[biome_index(BiomeType::River)] =
        (weights[biome_index(BiomeType::River)] + warp).max(0.0);
    normalize_weights(weights)
}

fn base_terrain_pass(world: &mut World, config: &ProcGenConfig, biome_grid: &[BiomeType]) {
    for z in 0..config.dims[2] {
        for x in 0..config.dims[0] {
            let weights = sample_biome_weights(config, biome_grid, x, z);
            let terrain_n = hash_signed(config.seed ^ 0x55AA_9102, x as i32, 0, z as i32);
            let mut height = config.base_height + terrain_n * config.terrain_variation;

            height += weights[biome_index(BiomeType::River)] * -3.0;
            height += weights[biome_index(BiomeType::Lake)] * -2.0;
            height += weights[biome_index(BiomeType::Desert)] * 1.5;

            let h = height
                .round()
                .clamp(4.0, (config.dims[1].saturating_sub(3)) as f32) as i32;
            for y in 0..=h {
                let _ = world.set(x as i32, y, z as i32, STONE);
            }
        }
    }
}

fn cave_carve_pass(world: &mut World, config: &ProcGenConfig, biome_grid: &[BiomeType]) {
    for z in 0..config.dims[2] as i32 {
        for y in config.cave_depth_start as i32..config.dims[1] as i32 {
            for x in 0..config.dims[0] as i32 {
                if world.get(x, y, z) == EMPTY {
                    continue;
                }
                let weights = sample_biome_weights(config, biome_grid, x as usize, z as usize);
                let cave_bias = 0.06 * weights[biome_index(BiomeType::Desert)];
                let cave_noise = hash01(config.seed ^ 0xCAFE_BA5E, x, y, z);
                if cave_noise < (config.cave_density + cave_bias) {
                    let _ = world.set(x, y, z, EMPTY);
                }
            }
        }
    }
}

fn surface_layering_pass(world: &mut World, config: &ProcGenConfig, biome_grid: &[BiomeType]) {
    for z in 0..config.dims[2] as i32 {
        for x in 0..config.dims[0] as i32 {
            if let Some(surface_y) = surface_y(world, x, z) {
                let weights = sample_biome_weights(config, biome_grid, x as usize, z as usize);
                let sand_depth =
                    (1.0 + 2.0 * weights[biome_index(BiomeType::Desert)]).round() as i32;
                for d in 0..3 {
                    let y = surface_y - d;
                    if y <= 0 {
                        continue;
                    }
                    if d == 0 {
                        let top = if weights[biome_index(BiomeType::Desert)] > 0.6 {
                            SAND
                        } else {
                            TURF
                        };
                        let _ = world.set(x, y, z, top);
                    } else if d <= sand_depth {
                        let fill = if weights[biome_index(BiomeType::Desert)] > 0.5 {
                            SAND
                        } else {
                            DIRT
                        };
                        let _ = world.set(x, y, z, fill);
                    }
                }
            }
        }
    }
}

fn biome_decoration_pass(world: &mut World, config: &ProcGenConfig, biome_grid: &[BiomeType]) {
    for z in 0..config.dims[2] as i32 {
        for x in 0..config.dims[0] as i32 {
            let weights = sample_biome_weights(config, biome_grid, x as usize, z as usize);
            if let Some(surface_y) = surface_y(world, x, z) {
                if weights[biome_index(BiomeType::River)] > 0.45 {
                    let channel = (config.water_level as i32 - 2).max(2);
                    for y in channel..=config.water_level as i32 {
                        if world.get(x, y, z) == EMPTY {
                            let _ = world.set(x, y, z, WATER);
                        }
                    }
                    for y in channel.saturating_sub(1)..channel {
                        let _ = world.set(x, y, z, SAND);
                    }
                }

                if weights[biome_index(BiomeType::Lake)] > 0.5 {
                    let basin_floor = (config.water_level as i32 - 3).max(2);
                    for y in basin_floor..=config.water_level as i32 {
                        if world.get(x, y, z) == EMPTY || y <= surface_y {
                            let _ = world.set(x, y, z, WATER);
                        }
                    }
                }
            }
        }
    }
}

fn vegetation_pass(world: &mut World, config: &ProcGenConfig, biome_grid: &[BiomeType]) {
    for z in 1..config.dims[2] as i32 - 1 {
        for x in 1..config.dims[0] as i32 - 1 {
            let Some(y) = surface_y(world, x, z) else {
                continue;
            };
            let ground = world.get(x, y, z);
            if !(ground == TURF || ground == DIRT || ground == SAND) {
                continue;
            }
            if world.get(x, y + 1, z) != EMPTY {
                continue;
            }
            let near_water = has_water_neighbor(world, x, y, z);
            let weights = sample_biome_weights(config, biome_grid, x as usize, z as usize);
            let forest = weights[biome_index(BiomeType::Forest)];
            let desert = weights[biome_index(BiomeType::Desert)];
            let river =
                weights[biome_index(BiomeType::River)] + weights[biome_index(BiomeType::Lake)];
            let mut tree_p = config.tree_density * (0.3 + forest * 1.4);
            tree_p *= 1.0 - desert * 0.95;
            if near_water {
                tree_p *= (1.0 - river * 0.8).max(0.05);
            }

            let tree_roll = hash01(config.seed ^ 0x0F0E_0D0C, x, y + 1, z);
            if tree_roll < tree_p && can_place_tree(world, x, y + 1, z) {
                place_tree(world, config.seed, x, y + 1, z);
                continue;
            }

            let flora_roll = hash01(config.seed ^ 0x0BAD_B002, x, y + 1, z);
            if forest > 0.4 && flora_roll < 0.08 {
                let _ = world.set(x, y + 1, z, if flora_roll < 0.03 { BUSH } else { GRASS });
            }
        }
    }
}

fn can_place_tree(world: &World, x: i32, y: i32, z: i32) -> bool {
    for ty in 0..8 {
        if world.get(x, y + ty, z) != EMPTY {
            return false;
        }
    }
    !has_water_neighbor(world, x, y - 1, z)
}

fn place_tree(world: &mut World, seed: u64, x: i32, y: i32, z: i32) {
    let trunk_h = 4 + (hash01(seed ^ 0x1346_AAA1, x, y, z) * 3.0) as i32;
    for ty in 0..trunk_h {
        let _ = world.set(x, y + ty, z, WOOD);
    }
    let top = y + trunk_h;
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 > 6 {
                    continue;
                }
                let px = x + dx;
                let py = top + dy;
                let pz = z + dz;
                if px <= 0
                    || py <= 0
                    || pz <= 0
                    || px >= world.dims[0] as i32 - 1
                    || py >= world.dims[1] as i32 - 1
                    || pz >= world.dims[2] as i32 - 1
                {
                    continue;
                }
                if world.get(px, py, pz) == EMPTY {
                    let _ = world.set(px, py, pz, LEAVES);
                }
            }
        }
    }
}

fn has_water_neighbor(world: &World, x: i32, y: i32, z: i32) -> bool {
    for dz in -1..=1 {
        for dx in -1..=1 {
            if world.get(x + dx, y, z + dz) == WATER || world.get(x + dx, y + 1, z + dz) == WATER {
                return true;
            }
        }
    }
    false
}

fn surface_y(world: &World, x: i32, z: i32) -> Option<i32> {
    for y in (0..world.dims[1] as i32).rev() {
        if world.get(x, y, z) != EMPTY && world.get(x, y, z) != WATER {
            return Some(y);
        }
    }
    None
}

fn biome_index(biome: BiomeType) -> usize {
    match biome {
        BiomeType::Forest => 0,
        BiomeType::Desert => 1,
        BiomeType::River => 2,
        BiomeType::Lake => 3,
    }
}

fn normalize_weights(mut w: [f32; 4]) -> [f32; 4] {
    let sum = w.iter().sum::<f32>().max(1e-6);
    for v in &mut w {
        *v /= sum;
    }
    w
}

fn hash01(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    let h = hash_u64(seed, x, y, z);
    (h as f64 / u64::MAX as f64) as f32
}

fn hash_signed(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    hash01(seed, x, y, z) * 2.0 - 1.0
}

fn hash_u64(seed: u64, x: i32, y: i32, z: i32) -> u64 {
    let mut v = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (z as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 30;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}
