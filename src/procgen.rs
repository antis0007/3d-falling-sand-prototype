use std::collections::VecDeque;

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
const BIOME_COUNT: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeType {
    Forest,
    Plains,
    Highlands,
    Desert,
    River,
    Lake,
    Ocean,
}

impl BiomeType {
    pub fn label(self) -> &'static str {
        match self {
            BiomeType::Forest => "Forest",
            BiomeType::Plains => "Plains",
            BiomeType::Highlands => "Highlands",
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
        let sea_level = 18.min(size as i32 - 10).max(10);
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
    shoreline_transition_pass(&mut world, &config, &heights);
    biome_water_pass(&mut world, &config);
    enforce_subsea_materials_pass(&mut world, &config);
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
            heights[x as usize + z as usize * config.dims[0]] =
                sampled_surface_height(config, wx, wz);
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
            let plains = weights[biome_index(BiomeType::Plains)];
            let highlands = weights[biome_index(BiomeType::Highlands)];
            let ocean = weights[biome_index(BiomeType::Ocean)];
            let river = weights[biome_index(BiomeType::River)].max(river_meander_signal(
                config.seed,
                wx,
                wz,
            ));
            let slope = local_slope_heightmap(heights, world.dims[0], lx, lz);
            let shore_w = smoothstep((ocean - 0.24) / 0.34);
            let near_sea_band = top_y <= config.sea_level + 4;
            let coastal = shore_w > 0.18 && near_sea_band;
            let cliffy_coast = coastal && slope >= 3;
            let subsea_ocean = top_y <= config.sea_level && ocean > 0.55;
            let river_bank = river > 0.48 && top_y <= config.sea_level + 2;
            let dirt_depth = (4.0 + 3.0 * (1.0 - desert) + plains * 2.0 - highlands * 1.2)
                .round()
                .clamp(2.0, 8.0) as i32;
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
                    if subsea_ocean {
                        if coastal || river_bank || desert > 0.4 {
                            SAND
                        } else {
                            STONE
                        }
                    } else if cliffy_coast || (highlands > 0.58 && top_y > config.sea_level + 4) {
                        if top_y > config.sea_level + 5 {
                            STONE
                        } else {
                            TURF
                        }
                    } else if coastal || river_bank || desert > 0.58 {
                        SAND
                    } else {
                        TURF
                    }
                } else if subsea_ocean && y <= config.sea_level {
                    if d <= sand_depth + 1 {
                        SAND
                    } else {
                        STONE
                    }
                } else if (desert > 0.58 || coastal || river_bank) && d <= sand_depth {
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

fn local_slope_heightmap(heights: &[i32], width: usize, x: i32, z: i32) -> i32 {
    let depth = (heights.len() / width) as i32;
    let center = heights[x as usize + z as usize * width];
    let mut max_delta = 0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dz == 0 {
                continue;
            }
            let nx = (x + dx).clamp(0, width as i32 - 1);
            let nz = (z + dz).clamp(0, depth - 1);
            let h = heights[nx as usize + nz as usize * width];
            max_delta = max_delta.max((h - center).abs());
        }
    }
    max_delta
}

fn shoreline_transition_pass(world: &mut World, config: &ProcGenConfig, heights: &[i32]) {
    let sea = config.sea_level;
    for lz in 1..world.dims[2] as i32 - 1 {
        for lx in 1..world.dims[0] as i32 - 1 {
            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;
            let w = biome_weights(config.seed, wx, wz);
            let ocean = w[biome_index(BiomeType::Ocean)];
            if ocean > 0.55 {
                continue;
            }

            let mut near_ocean = false;
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let nw = biome_weights(config.seed, wx + dx, wz + dz);
                    if nw[biome_index(BiomeType::Ocean)] > 0.62 {
                        near_ocean = true;
                        break;
                    }
                }
                if near_ocean {
                    break;
                }
            }
            if !near_ocean {
                continue;
            }

            let top = heights[lx as usize + lz as usize * world.dims[0]];
            if top >= sea - 1 {
                continue;
            }

            let berm_top = (sea - 1).min(world.dims[1] as i32 - 2);
            let stone_top = (berm_top - 2).max(2);
            for y in (top + 1).max(2)..=stone_top {
                let _ = world.set(lx, y, lz, STONE);
            }
            for y in (stone_top + 1).max(2)..=berm_top {
                let _ = world.set(lx, y, lz, SAND);
            }
        }
    }
}

fn build_surface_heightmap_from_world(world: &World) -> Vec<i32> {
    let mut heights = vec![0; world.dims[0] * world.dims[2]];
    for z in 0..world.dims[2] as i32 {
        for x in 0..world.dims[0] as i32 {
            let mut top = 1;
            for y in (1..world.dims[1] as i32).rev() {
                if world.get(x, y, z) != EMPTY {
                    top = y;
                    break;
                }
            }
            heights[x as usize + z as usize * world.dims[0]] = top;
        }
    }
    heights
}

fn biome_water_pass(world: &mut World, config: &ProcGenConfig) {
    let sea_level = config.sea_level;
    let heights = build_surface_heightmap_from_world(world);
    let width = world.dims[0];
    let depth = world.dims[2];
    let mut wet_mask = vec![false; width * depth];
    let mut channel_floor = vec![0i32; width * depth];
    let mut water_target = vec![0i32; width * depth];
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
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
            let surface = heights[lx as usize + lz as usize * width];

            let lowland_neighbors =
                count_neighbors_below_heightmap(&heights, width, config, lx, lz, sea_level - 1);
            let lowland_area =
                local_lowland_fraction(&heights, width, config, lx, lz, sea_level - 1, 3);
            let ocean_dominant = ocean_w > 0.66;
            if ocean_dominant {
                let depth_variation = ((fbm2(
                    config.seed ^ 0x0CEA_0010,
                    wx as f32 * 0.0032,
                    wz as f32 * 0.0032,
                    3,
                ) - 0.5)
                    * 3.0)
                    .round() as i32;
                let target_depth = (10 + depth_variation).clamp(8, 14);
                let floor = (sea_level - target_depth).max(2);
                let qualifies =
                    surface <= sea_level - 1 && lowland_neighbors >= 6 && lowland_area >= 0.60;

                if qualifies {
                    for y in 1..floor {
                        let _ = world.set(lx, y, lz, STONE);
                    }
                    for y in floor..=sea_level {
                        let _ = world.set(lx, y, lz, WATER);
                    }
                    for y in (floor - 2).max(1)..floor {
                        let _ = world.set(lx, y, lz, SAND);
                    }
                    continue;
                }
            }

            let wetness = river_w.max(lake_w);
            if wetness < 0.58 {
                continue;
            }

            let global_plane = quantized_water_plane(config, wx, wz);
            if surface > global_plane + 2 {
                continue;
            }

            let depth = (1.0 + 1.2 * wetness).round() as i32;
            let floor = (surface - depth).max(2);
            if floor >= surface {
                continue;
            }

            for y in floor..=surface {
                let _ = world.set(lx, y, lz, EMPTY);
            }
            for y in (floor - 1).max(1)..=floor {
                if world.get(lx, y, lz) != WATER {
                    let _ = world.set(lx, y, lz, SAND);
                }
            }

            let idx = lx as usize + lz as usize * width;
            wet_mask[idx] = true;
            channel_floor[idx] = floor;
            water_target[idx] = global_plane;
        }
    }

    let mut visited = vec![false; width * depth];
    for lz in 0..depth as i32 {
        for lx in 0..width as i32 {
            let start = lx as usize + lz as usize * width;
            if visited[start] || !wet_mask[start] {
                continue;
            }

            let mut queue = VecDeque::new();
            let mut region = Vec::new();
            queue.push_back((lx, lz));
            visited[start] = true;

            while let Some((cx, cz)) = queue.pop_front() {
                region.push((cx, cz));
                for (nx, nz) in [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)] {
                    if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                        continue;
                    }
                    let nidx = nx as usize + nz as usize * width;
                    if visited[nidx] || !wet_mask[nidx] {
                        continue;
                    }
                    visited[nidx] = true;
                    queue.push_back((nx, nz));
                }
            }

            let mut avg_target = 0.0;
            for (rx, rz) in &region {
                let ridx = *rx as usize + *rz as usize * width;
                avg_target += water_target[ridx] as f32;
            }
            avg_target /= region.len() as f32;

            let mut spill = i32::MAX;
            for (rx, rz) in &region {
                let r_idx = *rx as usize + *rz as usize * width;
                let r_floor = channel_floor[r_idx] + 1;
                for (nx, nz) in [
                    (*rx - 1, *rz),
                    (*rx + 1, *rz),
                    (*rx, *rz - 1),
                    (*rx, *rz + 1),
                ] {
                    let n_surface = height_sample_with_fallback(&heights, width, config, nx, nz);
                    if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                        spill = spill.min(n_surface.max(r_floor));
                        continue;
                    }
                    let nidx = nx as usize + nz as usize * width;
                    if !wet_mask[nidx] {
                        spill = spill.min(n_surface.max(r_floor));
                    }
                }
            }

            let mut basin_target = avg_target.round() as i32;
            if spill != i32::MAX {
                basin_target = basin_target.min(spill);
            }

            for (rx, rz) in region {
                let ridx = rx as usize + rz as usize * width;
                let wx = config.world_origin[0] + rx;
                let wz = config.world_origin[2] + rz;
                let slope = drainage_step(config.seed, wx, wz);
                let surface = heights[ridx];
                let floor = channel_floor[ridx];
                let mut top = basin_target + slope;
                top = top.min(surface - 1).max(floor);
                for y in floor..=top {
                    let _ = world.set(rx, y, rz, WATER);
                }
            }
        }
    }
}

fn sampled_global_water_potential(config: &ProcGenConfig, x: i32, z: i32) -> f32 {
    let low_noise = fbm2(
        config.seed ^ 0xD00D_1001,
        x as f32 * 0.0014,
        z as f32 * 0.0014,
        4,
    );
    let dir_x = fbm2(
        config.seed ^ 0xD00D_1002,
        x as f32 * 0.001,
        z as f32 * 0.001,
        2,
    ) * 2.0
        - 1.0;
    let dir_z = fbm2(
        config.seed ^ 0xD00D_1003,
        x as f32 * 0.001,
        z as f32 * 0.001,
        2,
    ) * 2.0
        - 1.0;
    let len = (dir_x * dir_x + dir_z * dir_z).sqrt().max(1e-4);
    let flow_x = dir_x / len;
    let flow_z = dir_z / len;
    let outlet_proj = (x as f32 * flow_x + z as f32 * flow_z) * 0.0018;
    low_noise * 0.7 - outlet_proj * 0.3
}

fn quantized_water_plane(config: &ProcGenConfig, x: i32, z: i32) -> i32 {
    let potential = sampled_global_water_potential(config, x, z);
    let plane_size = 2.0;
    let base = config.sea_level as f32 - 3.0;
    let raw = base + (potential - 0.5) * 8.0;
    (raw / plane_size).round() as i32 * plane_size as i32
}

fn drainage_step(seed: u64, x: i32, z: i32) -> i32 {
    let g = fbm2(seed ^ 0xD00D_1004, x as f32 * 0.003, z as f32 * 0.003, 2);
    if g > 0.75 {
        1
    } else if g < 0.25 {
        -1
    } else {
        0
    }
}

fn count_neighbors_below_heightmap(
    heights: &[i32],
    width: usize,
    config: &ProcGenConfig,
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
            let h = height_sample_with_fallback(heights, width, config, x + dx, z + dz);
            if h <= max_height {
                count += 1;
            }
        }
    }
    count
}

fn local_lowland_fraction(
    heights: &[i32],
    width: usize,
    config: &ProcGenConfig,
    x: i32,
    z: i32,
    max_height: i32,
    radius: i32,
) -> f32 {
    let mut low = 0;
    let mut total = 0;
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            total += 1;
            let h = height_sample_with_fallback(heights, width, config, x + dx, z + dz);
            if h <= max_height {
                low += 1;
            }
        }
    }
    low as f32 / total as f32
}

fn height_sample_with_fallback(
    heights: &[i32],
    width: usize,
    config: &ProcGenConfig,
    x: i32,
    z: i32,
) -> i32 {
    let depth = heights.len() / width;
    if x >= 0 && z >= 0 && x < width as i32 && z < depth as i32 {
        return heights[x as usize + z as usize * width];
    }
    let wx = config.world_origin[0] + x;
    let wz = config.world_origin[2] + z;
    sampled_surface_height(config, wx, wz)
}

fn explicit_rapid_or_fall(seed: u64, x0: i32, z0: i32, x1: i32, z1: i32) -> bool {
    let a = drainage_step(seed, x0, z0);
    let b = drainage_step(seed, x1, z1);
    (a - b).abs() >= 2
}

fn enforce_subsea_materials_pass(world: &mut World, config: &ProcGenConfig) {
    let sea = config.sea_level;
    for z in 0..world.dims[2] as i32 {
        for x in 0..world.dims[0] as i32 {
            for y in 1..=sea.min(world.dims[1] as i32 - 2) {
                let m = world.get(x, y, z);
                if m == TURF || m == DIRT || m == GRASS || m == BUSH {
                    let replacement = if y >= sea - 2 { SAND } else { STONE };
                    let _ = world.set(x, y, z, replacement);
                }
                if m == WATER && world.get(x, y - 1, z) == EMPTY {
                    let _ = world.set(x, y - 1, z, STONE);
                }
            }
        }
    }
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
            let plains = weights[biome_index(BiomeType::Plains)];
            let highlands = weights[biome_index(BiomeType::Highlands)];
            let desert = weights[biome_index(BiomeType::Desert)];
            let wet =
                weights[biome_index(BiomeType::River)] + weights[biome_index(BiomeType::Lake)];

            let mut tree_p = config.tree_density
                * (0.22 + 1.7 * forest + 0.45 * plains)
                * (1.0 - desert).powf(2.5)
                * (1.0 - highlands * 0.45).max(0.35);
            if near_water(world, lx, top_y, lz) {
                tree_p *= (1.0 - wet * 0.8).max(0.05);
            }

            let roll = hash01(config.seed ^ 0x11117777, wx, top_y, wz);
            if roll < tree_p && can_place_tree(world, lx, top_y + 1, lz) {
                place_tree(world, config.seed, wx, lx, top_y + 1, lz);
                continue;
            }

            let flora_roll = hash01(config.seed ^ 0x22224444, wx, top_y, wz);
            if (forest > 0.4 || plains > 0.5) && ground != SAND && flora_roll < 0.09 {
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

fn sampled_surface_height(config: &ProcGenConfig, x: i32, z: i32) -> i32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            let sx = x + dx;
            let sz = z + dz;
            let w = biome_weights(config.seed, sx, sz);
            sum += terrain_height(config, sx, sz, w) as f32;
            n += 1.0;
        }
    }
    let w0 = biome_weights(config.seed, x, z);
    let raw = terrain_height(config, x, z, w0) as f32;
    (raw * 0.6 + (sum / n) * 0.4)
        .round()
        .clamp(6.0, (config.dims[1] as i32 - 4) as f32) as i32
}

fn terrain_height(config: &ProcGenConfig, x: i32, z: i32, weights: [f32; BIOME_COUNT]) -> i32 {
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

    h += weights[biome_index(BiomeType::Desert)] * 1.2;
    h += weights[biome_index(BiomeType::Highlands)] * 4.2;
    h += weights[biome_index(BiomeType::Plains)] * 0.4;
    h -= bank_lower;
    h -= weights[biome_index(BiomeType::Lake)] * 3.0;

    let ocean_depth_shape = (fbm2(
        config.seed ^ 0x0CEA_0001,
        x as f32 * 0.003,
        z as f32 * 0.003,
        3,
    ) - 0.5)
        * 3.0;
    let ocean_floor_target = config.sea_level as f32 - 12.0 + ocean_depth_shape;
    h = h * (1.0 - ocean_w * 0.45) + (config.sea_level as f32 - 5.0) * ocean_w * 0.45;
    h = h * (1.0 - ocean_w * 0.82) + ocean_floor_target * ocean_w * 0.82;

    let coast_w = smoothstep((ocean_w - 0.50) / 0.30);
    let coast_target = config.sea_level as f32;
    h = h * (1.0 - coast_w * 0.30) + coast_target * coast_w * 0.30;

    if ocean_w > 0.70 {
        h = h.min(config.sea_level as f32 - 1.0);
    }

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

fn biome_weights(seed: u64, x: i32, z: i32) -> [f32; BIOME_COUNT] {
    let macro_x = floor_div(x, MACROCHUNK_SIZE);
    let macro_z = floor_div(z, MACROCHUNK_SIZE);
    let cluster_scale = BIOME_CLUSTER_MACROS as f32;

    let warp_x = (fbm2(seed ^ 0x5522AA11, x as f32 * 0.0018, z as f32 * 0.0018, 3) - 0.5) * 1.2;
    let warp_z = (fbm2(seed ^ 0x5522AA12, x as f32 * 0.0018, z as f32 * 0.0018, 3) - 0.5) * 1.2;

    let fx = macro_x as f32 / cluster_scale + warp_x;
    let fz = macro_z as f32 / cluster_scale + warp_z;
    let x0 = fx.floor() as i32;
    let z0 = fz.floor() as i32;
    let tx = (fx - x0 as f32).clamp(0.0, 1.0);
    let tz = (fz - z0 as f32).clamp(0.0, 1.0);

    let mut weights = [0.0; BIOME_COUNT];
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
        (weights[biome_index(BiomeType::Ocean)] + (continental - 0.60).max(0.0) * 1.5).max(0.0);
    let relief = fbm2(seed ^ 0xCC99_1010, x as f32 * 0.0025, z as f32 * 0.0025, 3);
    weights[biome_index(BiomeType::Highlands)] =
        (weights[biome_index(BiomeType::Highlands)] + (relief - 0.56).max(0.0) * 1.25).max(0.0);
    weights[biome_index(BiomeType::Plains)] =
        (weights[biome_index(BiomeType::Plains)] + (0.60 - relief).max(0.0) * 0.9).max(0.0);

    normalize(weights)
}

fn biome_anchor(seed: u64, chunk_x: i32, chunk_z: i32) -> BiomeType {
    let v = hash01(seed ^ 0xFA112233, chunk_x, 0, chunk_z);
    if v < 0.20 {
        BiomeType::Desert
    } else if v < 0.34 {
        BiomeType::Plains
    } else if v < 0.50 {
        BiomeType::Highlands
    } else if v < 0.60 {
        BiomeType::Lake
    } else if v < 0.86 {
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

fn dominant_biome(weights: [f32; BIOME_COUNT]) -> BiomeType {
    let mut best = 0;
    for i in 1..BIOME_COUNT {
        if weights[i] > weights[best] {
            best = i;
        }
    }
    match best {
        0 => BiomeType::Forest,
        1 => BiomeType::Plains,
        2 => BiomeType::Highlands,
        3 => BiomeType::Desert,
        4 => BiomeType::River,
        5 => BiomeType::Lake,
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
        BiomeType::Plains => 1,
        BiomeType::Highlands => 2,
        BiomeType::Desert => 3,
        BiomeType::River => 4,
        BiomeType::Lake => 5,
        BiomeType::Ocean => 6,
    }
}

fn normalize(mut w: [f32; BIOME_COUNT]) -> [f32; BIOME_COUNT] {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn water_surface(world: &World, x: i32, z: i32) -> Option<i32> {
        for y in (1..world.dims[1] as i32).rev() {
            if world.get(x, y, z) == WATER {
                return Some(y);
            }
        }
        None
    }

    #[test]
    fn adjacent_river_cells_have_continuous_levels() {
        let config = ProcGenConfig::for_size(64, 0xBADC0FFE).with_origin([0, 0, 0]);
        let world = generate_world(config);
        let heights = build_surface_heightmap_from_world(&world);

        for z in 0..world.dims[2] as i32 {
            for x in 0..world.dims[0] as i32 {
                let Some(level) = water_surface(&world, x, z) else {
                    continue;
                };

                for (nx, nz) in [(x + 1, z), (x, z + 1)] {
                    if nx >= world.dims[0] as i32 || nz >= world.dims[2] as i32 {
                        continue;
                    }
                    let Some(other_level) = water_surface(&world, nx, nz) else {
                        continue;
                    };
                    let wx0 = config.world_origin[0] + x;
                    let wz0 = config.world_origin[2] + z;
                    let wx1 = config.world_origin[0] + nx;
                    let wz1 = config.world_origin[2] + nz;
                    let diff = (level - other_level).abs();
                    let rapid = explicit_rapid_or_fall(config.seed, wx0, wz0, wx1, wz1);
                    let terrain_drop = (heights[x as usize + z as usize * world.dims[0]]
                        - heights[nx as usize + nz as usize * world.dims[0]])
                        .abs();
                    if !rapid && terrain_drop < 3 {
                        assert!(
                            diff <= 1,
                            "water step too large at ({x},{z}) -> ({nx},{nz}): {diff}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_border_waterline_uses_global_context() {
        let size = 64;
        let seed = 0x1234_5678;
        let left_cfg = ProcGenConfig::for_size(size, seed).with_origin([0, 0, 0]);
        let right_cfg = ProcGenConfig::for_size(size, seed).with_origin([size as i32, 0, 0]);
        let left = generate_world(left_cfg);
        let right = generate_world(right_cfg);

        for z in 0..size as i32 {
            let a = water_surface(&left, size as i32 - 1, z);
            let b = water_surface(&right, 0, z);
            if let (Some(al), Some(bl)) = (a, b) {
                assert!(
                    (al - bl).abs() <= 1,
                    "border waterline mismatch at z={z}: {al} vs {bl}"
                );
            }
        }
    }
}
