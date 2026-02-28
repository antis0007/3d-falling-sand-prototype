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
const BIOME_CLUSTER_MACROS: i32 = 3;
const MACROCHUNK_SIZE: i32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeType {
    Forest,
    Desert,
    River,
    Lake,
    Ocean,
}

impl BiomeType {
    pub fn label(self) -> &'static str {
        match self {
            BiomeType::Forest => "Forest",
            BiomeType::Desert => "Desert",
            BiomeType::River => "River",
            BiomeType::Lake => "Lake",
            BiomeType::Ocean => "Ocean",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProcGenConfig {
    pub dims: [usize; 3],
    pub world_origin: [i32; 3],
    pub seed: u64,
    pub sea_level: i32,
    pub terrain_scale: f32,
    pub cave_density: f32,
    pub tree_density: f32,
}

impl ProcGenConfig {
    pub fn for_size(size: usize, seed: u64) -> Self {
        let sea_level = 22.min(size as i32 - 8).max(12);
        Self {
            dims: [size, size, size],
            world_origin: [0, 0, 0],
            seed,
            sea_level,
            terrain_scale: 1.0,
            cave_density: 0.13,
            tree_density: 0.012,
        }
    }

    pub fn with_origin(mut self, world_origin: [i32; 3]) -> Self {
        self.world_origin = world_origin;
        self
    }
}

pub fn generate_world(config: ProcGenConfig) -> World {
    let mut world = World::new(config.dims);
    world.clear();
    let heights = build_heightmap(&config);

    base_terrain_pass(&mut world, &config, &heights);
    cave_carve_pass(&mut world, &config);
    surface_layering_pass(&mut world, &config, &heights);
    biome_water_pass(&mut world, &config, &heights);
    vegetation_pass(&mut world, &config);

    world
}

pub fn biome_hint_at_world(config: &ProcGenConfig, x: i32, z: i32) -> BiomeType {
    let weights = biome_weights(config.seed, x, z);
    dominant_biome(weights)
}

fn build_heightmap(config: &ProcGenConfig) -> Vec<i32> {
    let mut heights = vec![0; config.dims[0] * config.dims[2]];
    for z in 0..config.dims[2] as i32 {
        for x in 0..config.dims[0] as i32 {
            let wx = config.world_origin[0] + x;
            let wz = config.world_origin[2] + z;
            let mut sum = 0.0;
            let mut n = 0.0;
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let sx = wx + dx;
                    let sz = wz + dz;
                    let w = biome_weights(config.seed, sx, sz);
                    sum += terrain_height(config, sx, sz, w) as f32;
                    n += 1.0;
                }
            }
            let w0 = biome_weights(config.seed, wx, wz);
            let raw = terrain_height(config, wx, wz, w0) as f32;
            let h = (raw * 0.6 + (sum / n) * 0.4)
                .round()
                .clamp(6.0, (config.dims[1] as i32 - 4) as f32) as i32;
            heights[x as usize + z as usize * config.dims[0]] = h;
        }
    }
    heights
}

fn base_terrain_pass(world: &mut World, _config: &ProcGenConfig, heights: &[i32]) {
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
            let h = heights[lx as usize + lz as usize * world.dims[0]];
            for y in 0..=h {
                let _ = world.set(lx, y, lz, STONE);
            }
        }
    }
}

fn cave_carve_pass(world: &mut World, config: &ProcGenConfig) {
    let max_y = world.dims[1] as i32 - 2;
    for lz in 0..world.dims[2] as i32 {
        for ly in 4..max_y {
            for lx in 0..world.dims[0] as i32 {
                if world.get(lx, ly, lz) == EMPTY {
                    continue;
                }
                let wx = config.world_origin[0] + lx;
                let wy = ly;
                let wz = config.world_origin[2] + lz;
                let weights = biome_weights(config.seed, wx, wz);
                let top =
                    terrain_height(config, wx, wz, weights).clamp(6, world.dims[1] as i32 - 4);
                if ly >= top - 4 {
                    continue;
                }
                let depth_fade = 1.0 - (ly as f32 / max_y as f32).powf(1.3);
                let cave = fbm3(
                    config.seed ^ 0xCC77AA11,
                    wx as f32 * 0.035,
                    wy as f32 * 0.042,
                    wz as f32 * 0.035,
                    4,
                );
                let cave_warp = fbm3(
                    config.seed ^ 0x1077BEEF,
                    wx as f32 * 0.02,
                    wy as f32 * 0.02,
                    wz as f32 * 0.02,
                    3,
                );
                let value = 0.7 * cave + 0.3 * cave_warp;
                let threshold = 0.63 + (1.0 - config.cave_density.clamp(0.05, 0.25)) * 0.18;
                if value > threshold && depth_fade > 0.08 {
                    let _ = world.set(lx, ly, lz, EMPTY);
                }
            }
        }
    }
}

fn surface_layering_pass(world: &mut World, config: &ProcGenConfig, heights: &[i32]) {
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
            let top_y = heights[lx as usize + lz as usize * world.dims[0]];
            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;
            let weights = biome_weights(config.seed, wx, wz);

            let desert = weights[biome_index(BiomeType::Desert)];
            let ocean = weights[biome_index(BiomeType::Ocean)];
            let shore_w = smoothstep((ocean - 0.22) / 0.35);
            let coastal = shore_w > 0.18 && top_y <= config.sea_level + 3;
            let dirt_depth = (4.0 + 3.0 * (1.0 - desert)).round() as i32;
            let sand_depth = if coastal {
                (3.0 + shore_w * 4.0).round() as i32
            } else {
                (2.0 + 3.0 * desert).round() as i32
            };

            for d in 0..(dirt_depth + sand_depth + 2) {
                let y = top_y - d;
                if y <= 1 {
                    continue;
                }
                let block = if d == 0 {
                    if coastal || desert > 0.58 {
                        SAND
                    } else {
                        TURF
                    }
                } else if (desert > 0.58 || coastal) && d <= sand_depth {
                    SAND
                } else if d <= dirt_depth {
                    DIRT
                } else {
                    STONE
                };
                let _ = world.set(lx, y, lz, block);
            }
        }
    }
}

fn biome_water_pass(world: &mut World, config: &ProcGenConfig, heights: &[i32]) {
    let sea_level = config.sea_level;
    for lz in 1..world.dims[2] as i32 - 1 {
        for lx in 1..world.dims[0] as i32 - 1 {
            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;
            let weights = biome_weights(config.seed, wx, wz);
            let ocean_w = weights[biome_index(BiomeType::Ocean)];
            let river_w = weights[biome_index(BiomeType::River)].max(river_meander_signal(
                config.seed,
                wx,
                wz,
            ));
            let lake_w = weights[biome_index(BiomeType::Lake)];
            let surface = heights[lx as usize + lz as usize * world.dims[0]];

            let lowland_neighbors =
                count_neighbors_below_heightmap(heights, world.dims[0], lx, lz, sea_level - 1);
            if ocean_w > 0.42 && surface <= sea_level - 1 && lowland_neighbors >= 4 {
                for y in (sea_level - 12).max(2)..=sea_level {
                    if world.get(lx, y, lz) == EMPTY {
                        let _ = world.set(lx, y, lz, WATER);
                    }
                }
                for y in (sea_level - 13).max(1)..=(sea_level - 12).max(1) {
                    let _ = world.set(lx, y, lz, SAND);
                }
                continue;
            }

            let wetness = river_w.max(lake_w);
            if wetness < 0.58 {
                continue;
            }

            let depth = (1.0 + 1.2 * wetness).round() as i32;
            let floor = (surface - depth).max(2);
            let neigh_min =
                neighbor_min_surface_heightmap(heights, world.dims[0], lx, lz).unwrap_or(surface);
            let top = (surface - 1).min(sea_level - 1).min(neigh_min - 1);
            if top < floor {
                continue;
            }

            for y in floor..=surface {
                let _ = world.set(lx, y, lz, EMPTY);
            }
            for y in floor..=top {
                let _ = world.set(lx, y, lz, WATER);
            }
            for y in (floor - 1).max(1)..=floor {
                if world.get(lx, y, lz) != WATER {
                    let _ = world.set(lx, y, lz, SAND);
                }
            }
        }
    }
}

fn count_neighbors_below_heightmap(
    heights: &[i32],
    width: usize,
    x: i32,
    z: i32,
    max_height: i32,
) -> i32 {
    let mut count = 0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dz == 0 {
                continue;
            }
            let nx = x + dx;
            let nz = z + dz;
            if nx < 0 || nz < 0 || nx >= width as i32 || nz >= (heights.len() / width) as i32 {
                continue;
            }
            let idx = nx as usize + nz as usize * width;
            if heights[idx] <= max_height {
                count += 1;
            }
        }
    }
    count
}

fn neighbor_min_surface_heightmap(heights: &[i32], width: usize, x: i32, z: i32) -> Option<i32> {
    let mut min_h: Option<i32> = None;
    for dz in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dz == 0 {
                continue;
            }
            let nx = x + dx;
            let nz = z + dz;
            if nx < 0 || nz < 0 || nx >= width as i32 || nz >= (heights.len() / width) as i32 {
                continue;
            }
            let idx = nx as usize + nz as usize * width;
            let h = heights[idx];
            min_h = Some(match min_h {
                Some(curr) => curr.min(h),
                None => h,
            });
        }
    }
    min_h
}

fn vegetation_pass(world: &mut World, config: &ProcGenConfig) {
    for lz in 2..world.dims[2] as i32 - 2 {
        for lx in 2..world.dims[0] as i32 - 2 {
            let Some(top_y) = surface_y(world, lx, lz) else {
                continue;
            };
            let ground = world.get(lx, top_y, lz);
            if ground != TURF && ground != DIRT && ground != SAND {
                continue;
            }
            if world.get(lx, top_y + 1, lz) != EMPTY {
                continue;
            }

            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;
            let weights = biome_weights(config.seed, wx, wz);
            let forest = weights[biome_index(BiomeType::Forest)];
            let desert = weights[biome_index(BiomeType::Desert)];
            let wet =
                weights[biome_index(BiomeType::River)] + weights[biome_index(BiomeType::Lake)];

            let mut tree_p = config.tree_density * (0.25 + 1.8 * forest) * (1.0 - desert).powf(2.5);
            if near_water(world, lx, top_y, lz) {
                tree_p *= (1.0 - wet * 0.8).max(0.05);
            }

            let roll = hash01(config.seed ^ 0x11117777, wx, top_y, wz);
            if roll < tree_p && can_place_tree(world, lx, top_y + 1, lz) {
                place_tree(world, config.seed, wx, lx, top_y + 1, lz);
                continue;
            }

            let flora_roll = hash01(config.seed ^ 0x22224444, wx, top_y, wz);
            if forest > 0.4 && ground != SAND && flora_roll < 0.07 {
                let _ = world.set(
                    lx,
                    top_y + 1,
                    lz,
                    if flora_roll < 0.026 { BUSH } else { GRASS },
                );
            }
        }
    }
}

pub fn find_safe_spawn(world: &World, seed: u64) -> [f32; 3] {
    let cx = (world.dims[0] / 2) as i32;
    let cz = (world.dims[2] / 2) as i32;

    for r in 0..(world.dims[0].min(world.dims[2]) as i32 / 2) {
        for dz in -r..=r {
            for dx in -r..=r {
                if r > 0 && dx.abs() < r && dz.abs() < r {
                    continue;
                }
                let x = (cx + dx).clamp(2, world.dims[0] as i32 - 3);
                let z = (cz + dz).clamp(2, world.dims[2] as i32 - 3);
                let bias = hash01(seed ^ 0xABCD0001, x, 0, z);
                if bias < 0.08 {
                    continue;
                }
                if let Some(y) = valid_surface_spawn_y(world, x, z) {
                    return [x as f32 + 0.5, y as f32 + 3.4, z as f32 + 0.5];
                }
            }
        }
    }

    [
        cx as f32 + 0.5,
        (world.dims[1] as f32 * 0.7).max(8.0),
        cz as f32 + 0.5,
    ]
}

fn valid_surface_spawn_y(world: &World, x: i32, z: i32) -> Option<i32> {
    let top = surface_y(world, x, z)?;
    let base = world.get(x, top, z);
    if base == WATER || base == EMPTY {
        return None;
    }
    for head in 1..=3 {
        if world.get(x, top + head, z) != EMPTY {
            return None;
        }
    }
    if near_water(world, x, top, z) {
        return None;
    }
    Some(top)
}

fn terrain_height(config: &ProcGenConfig, x: i32, z: i32, weights: [f32; 5]) -> i32 {
    let scale = config.terrain_scale.max(0.25);
    let continental = fbm2(
        config.seed ^ 0xA1000001,
        x as f32 * 0.0035 * scale,
        z as f32 * 0.0035 * scale,
        5,
    );
    let hills = fbm2(
        config.seed ^ 0xA1000002,
        x as f32 * 0.008 * scale,
        z as f32 * 0.008 * scale,
        4,
    );
    let detail = fbm2(
        config.seed ^ 0xA1000003,
        x as f32 * 0.022 * scale,
        z as f32 * 0.022 * scale,
        2,
    );

    let broad = (continental - 0.45) * 34.0;
    let rolling = (hills - 0.5) * 12.0;
    let micro = (detail - 0.5) * 2.5;

    let mut h = config.sea_level as f32 + broad + rolling + micro;

    let flatness = fbm2(
        config.seed ^ 0xA1000004,
        x as f32 * 0.0022,
        z as f32 * 0.0022,
        3,
    );
    if flatness > 0.55 {
        let plateau = (h / 3.0).round() * 3.0;
        h = h * 0.45 + plateau * 0.55;
    }

    let river_signal = river_meander_signal(config.seed, x, z);
    let river_w = weights[biome_index(BiomeType::River)].max(river_signal);
    let bank_lower = (river_w * 4.0).clamp(0.0, 4.0);
    let ocean_w = weights[biome_index(BiomeType::Ocean)];

    h += weights[biome_index(BiomeType::Desert)] * 1.3;
    h -= bank_lower;
    h -= weights[biome_index(BiomeType::Lake)] * 3.2;
    h = h * (1.0 - ocean_w * 0.55) + (config.sea_level as f32 - 6.0) * ocean_w * 0.55;

    h.round() as i32
}

fn river_meander_signal(seed: u64, x: i32, z: i32) -> f32 {
    let warp_x = (fbm2(seed ^ 0x77110011, x as f32 * 0.004, z as f32 * 0.004, 3) - 0.5) * 180.0;
    let warp_z = (fbm2(seed ^ 0x77110012, x as f32 * 0.004, z as f32 * 0.004, 3) - 0.5) * 180.0;
    let nx = (x as f32 + warp_x) * 0.0065;
    let nz = (z as f32 + warp_z) * 0.0065;
    let path = ((nx.sin() + 0.6 * (nz * 1.7).cos()) * 0.5 + 0.5).clamp(0.0, 1.0);
    (1.0 - ((path - 0.5).abs() * 3.4)).clamp(0.0, 1.0)
}

fn biome_weights(seed: u64, x: i32, z: i32) -> [f32; 5] {
    let macro_x = floor_div(x, MACROCHUNK_SIZE);
    let macro_z = floor_div(z, MACROCHUNK_SIZE);
    let cluster_scale = BIOME_CLUSTER_MACROS as f32;

    let warp_x = (fbm2(seed ^ 0x5522AA11, x as f32 * 0.0018, z as f32 * 0.0018, 3) - 0.5) * 1.2;
    let warp_z = (fbm2(seed ^ 0x5522AA12, x as f32 * 0.0018, z as f32 * 0.0018, 3) - 0.5) * 1.2;

    let fx = macro_x as f32 / cluster_scale + warp_x;
    let fz = macro_z as f32 / cluster_scale + warp_z;
    let x0 = fx.floor() as i32;
    let z0 = fz.floor() as i32;
    let tx = fx.fract();
    let tz = fz.fract();

    let mut weights = [0.0; 5];
    for oz in 0..=1 {
        for ox in 0..=1 {
            let ax = x0 + ox;
            let az = z0 + oz;
            let biome = biome_anchor(seed, ax, az);
            let wx = if ox == 0 { 1.0 - tx } else { tx };
            let wz = if oz == 0 { 1.0 - tz } else { tz };
            weights[biome_index(biome)] += wx * wz;
        }
    }

    let warp = fbm2(seed ^ 0xF009_1201, x as f32 * 0.0024, z as f32 * 0.0024, 3) - 0.5;
    let river = river_meander_signal(seed, x, z);
    let continental = fbm2(seed ^ 0xAD991100, x as f32 * 0.0016, z as f32 * 0.0016, 4);
    weights[biome_index(BiomeType::River)] =
        (weights[biome_index(BiomeType::River)] * 0.45 + river * 0.75 + warp * 0.1).max(0.0);
    weights[biome_index(BiomeType::Lake)] =
        (weights[biome_index(BiomeType::Lake)] + warp * 0.08).max(0.0);
    weights[biome_index(BiomeType::Ocean)] =
        (weights[biome_index(BiomeType::Ocean)] + (continental - 0.58).max(0.0) * 1.4).max(0.0);

    normalize(weights)
}

fn biome_anchor(seed: u64, chunk_x: i32, chunk_z: i32) -> BiomeType {
    let v = hash01(seed ^ 0xFA112233, chunk_x, 0, chunk_z);
    if v < 0.26 {
        BiomeType::Desert
    } else if v < 0.36 {
        BiomeType::Lake
    } else if v < 0.78 {
        BiomeType::Forest
    } else {
        BiomeType::Ocean
    }
}

fn floor_div(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn dominant_biome(weights: [f32; 5]) -> BiomeType {
    let mut best = 0;
    for i in 1..5 {
        if weights[i] > weights[best] {
            best = i;
        }
    }
    match best {
        0 => BiomeType::Forest,
        1 => BiomeType::Desert,
        2 => BiomeType::River,
        3 => BiomeType::Lake,
        _ => BiomeType::Ocean,
    }
}

fn can_place_tree(world: &World, x: i32, y: i32, z: i32) -> bool {
    for ty in 0..9 {
        if world.get(x, y + ty, z) != EMPTY {
            return false;
        }
    }
    !near_water(world, x, y - 1, z)
}

fn place_tree(world: &mut World, seed: u64, world_x: i32, x: i32, y: i32, z: i32) {
    let trunk_h = 4 + (hash01(seed ^ 0x71335599, world_x, y, z) * 4.0) as i32;
    for ty in 0..trunk_h {
        let _ = world.set(x, y + ty, z, WOOD);
    }

    let top = y + trunk_h;
    for dz in -2..=2 {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let dist = dx * dx + dz * dz + (dy * dy * 2 / 3);
                if dist > 6 {
                    continue;
                }
                let px = x + dx;
                let py = top + dy;
                let pz = z + dz;
                if px <= 1
                    || pz <= 1
                    || py <= 1
                    || px >= world.dims[0] as i32 - 2
                    || pz >= world.dims[2] as i32 - 2
                    || py >= world.dims[1] as i32 - 2
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

fn surface_y(world: &World, x: i32, z: i32) -> Option<i32> {
    for y in (1..world.dims[1] as i32).rev() {
        let m = world.get(x, y, z);
        if m != EMPTY && m != WATER {
            return Some(y);
        }
    }
    None
}

fn near_water(world: &World, x: i32, y: i32, z: i32) -> bool {
    for dz in -2..=2 {
        for dx in -2..=2 {
            if world.get(x + dx, y, z + dz) == WATER || world.get(x + dx, y + 1, z + dz) == WATER {
                return true;
            }
        }
    }
    false
}

fn biome_index(b: BiomeType) -> usize {
    match b {
        BiomeType::Forest => 0,
        BiomeType::Desert => 1,
        BiomeType::River => 2,
        BiomeType::Lake => 3,
        BiomeType::Ocean => 4,
    }
}

fn normalize(mut w: [f32; 5]) -> [f32; 5] {
    let s = w.iter().sum::<f32>().max(1e-6);
    for v in &mut w {
        *v /= s;
    }
    w
}

fn fbm2(seed: u64, mut x: f32, mut z: f32, octaves: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise_2d(seed, x * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
        x += 17.0;
        z += 29.0;
    }
    (sum / norm).clamp(0.0, 1.0)
}

fn fbm3(seed: u64, x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for i in 0..octaves {
        sum += amp * value_noise_3d(seed ^ (i as u64 * 0x9E37), x * freq, y * freq, z * freq);
        norm += amp;
        amp *= 0.55;
        freq *= 1.95;
    }
    (sum / norm).clamp(0.0, 1.0)
}

fn value_noise_2d(seed: u64, x: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let tx = smoothstep(x - xi as f32);
    let tz = smoothstep(z - zi as f32);

    let v00 = hash01(seed, xi, 0, zi);
    let v10 = hash01(seed, xi + 1, 0, zi);
    let v01 = hash01(seed, xi, 0, zi + 1);
    let v11 = hash01(seed, xi + 1, 0, zi + 1);

    let a = lerp(v00, v10, tx);
    let b = lerp(v01, v11, tx);
    lerp(a, b, tz)
}

fn value_noise_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let tx = smoothstep(x - xi as f32);
    let ty = smoothstep(y - yi as f32);
    let tz = smoothstep(z - zi as f32);

    let mut c = [0.0f32; 8];
    let mut idx = 0;
    for oz in 0..=1 {
        for oy in 0..=1 {
            for ox in 0..=1 {
                c[idx] = hash01(seed, xi + ox, yi + oy, zi + oz);
                idx += 1;
            }
        }
    }

    let x00 = lerp(c[0], c[1], tx);
    let x10 = lerp(c[2], c[3], tx);
    let x01 = lerp(c[4], c[5], tx);
    let x11 = lerp(c[6], c[7], tx);
    let y0 = lerp(x00, x10, ty);
    let y1 = lerp(x01, x11, ty);
    lerp(y0, y1, tz)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hash01(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    let h = hash_u64(seed, x, y, z);
    (h as f64 / u64::MAX as f64) as f32
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
