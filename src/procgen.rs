use std::collections::{HashMap, VecDeque};

#[cfg(feature = "procgen-profile")]
use std::time::{Duration, Instant};

use crate::chunk_store::{Chunk, ChunkStore, NeighborDirtyPolicy};
use crate::types::ChunkCoord;
use crate::world::{MaterialId, World, CHUNK_SIZE, EMPTY};

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
const DEFAULT_WORLD_MIN_Y: i32 = -256;
const DEFAULT_WORLD_MAX_Y: i32 = 384;

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(default)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn hash3(seed: u64, x: i32, y: i32, z: i32) -> u64 {
    let mut h = seed;
    h ^= (x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h ^= (y as i64 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h ^= (z as i64 as u64).wrapping_mul(0x165667B19E3779F9);
    splitmix64(h)
}

pub fn generate_chunk(seed: u64, c: ChunkCoord) -> Chunk {
    generate_chunk_direct(seed, c)
}

pub fn generate_chunk_direct(seed: u64, c: ChunkCoord) -> Chunk {
    let chunk_size = CHUNK_SIZE as i32;
    let chunk_origin = [c.x * chunk_size, c.y * chunk_size, c.z * chunk_size];
    let config = ProcGenConfig::for_size(CHUNK_SIZE, seed).with_origin(chunk_origin);
    generate_chunk_from_config(config)
}

fn generate_chunk_from_config(config: ProcGenConfig) -> Chunk {
    let mut world = World::new(config.dims);
    world.clear_empty();
    let timings = ProcGenPassTimings::default();
    let stages = ProcGenStages::default();
    let (heights, columns) = build_column_cache(&config, &timings);
    let hydrology = build_hydrology_cache_for_chunk(&config, &heights, &columns, &timings);

    if stages.base_relief {
        base_terrain_pass(&mut world, &config, &heights, &timings);
    }
    if stages.erosion_valley {
        cave_carve_pass_chunk(&mut world, &config, &columns, &timings);
    }
    if stages.material_painting {
        surface_layering_pass_chunk(&mut world, &config, &columns, &hydrology, &timings);
        shoreline_transition_pass(&mut world, &config, &heights, &columns, &timings);
    }
    if stages.channel_extraction || stages.basin_filling {
        hydrology_fill_pass(
            &mut world, &config, &heights, &columns, &hydrology, &timings,
        );
        remove_unsupported_hanging_water_pass(&mut world, &config, &timings);
    }
    enforce_subsea_materials_pass(&mut world, &config, &timings);
    if stages.vegetation {
        vegetation_pass_chunk(&mut world, &config, &columns, &hydrology, &timings);
    }
    world.finalize_generation_side_effects();
    timings.log_total(config.world_origin);

    let legacy = world
        .chunks
        .into_iter()
        .next()
        .unwrap_or_else(crate::world::Chunk::new);
    legacy.into()
}

pub fn apply_generated_chunk(store: &mut ChunkStore, c: ChunkCoord, chunk: Chunk) {
    store.insert_chunk_with_policy(c, chunk, true, NeighborDirtyPolicy::GeneratedConditional);
}

#[derive(Default)]
struct ProcGenPassTimings {
    #[cfg(feature = "procgen-profile")]
    rows: std::sync::Mutex<Vec<(&'static str, Duration)>>,
}

#[cfg(feature = "procgen-profile")]
struct ScopedPassTimer<'a> {
    owner: &'a ProcGenPassTimings,
    name: &'static str,
    start: Instant,
}

#[cfg(not(feature = "procgen-profile"))]
struct ScopedPassTimer<'a>(std::marker::PhantomData<&'a ProcGenPassTimings>);

impl ProcGenPassTimings {
    fn scoped(&self, name: &'static str) -> ScopedPassTimer<'_> {
        #[cfg(feature = "procgen-profile")]
        {
            ScopedPassTimer {
                owner: self,
                name,
                start: Instant::now(),
            }
        }

        #[cfg(not(feature = "procgen-profile"))]
        {
            let _ = self;
            let _ = name;
            ScopedPassTimer(std::marker::PhantomData)
        }
    }

    fn log_total(&self, origin: [i32; 3]) {
        #[cfg(feature = "procgen-profile")]
        {
            let rows = self.rows.lock().expect("procgen timing lock");
            let total: Duration = rows.iter().map(|(_, d)| *d).sum();
            log::info!(
                "procgen {:?} total {:.2}ms",
                origin,
                total.as_secs_f64() * 1000.0
            );
            for (name, d) in rows.iter() {
                log::info!("procgen pass {}: {:.2}ms", name, d.as_secs_f64() * 1000.0);
            }
        }

        #[cfg(not(feature = "procgen-profile"))]
        {
            let _ = self;
            let _ = origin;
        }
    }
}

#[cfg(feature = "procgen-profile")]
impl Drop for ScopedPassTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.owner
            .rows
            .lock()
            .expect("procgen timing lock")
            .push((self.name, elapsed));
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HydroFeature {
    None,
    RiverChannel,
    LakeBasin,
    OceanShelf,
    Coast,
}

impl HydroFeature {
    pub fn label(self) -> &'static str {
        match self {
            HydroFeature::None => "None",
            HydroFeature::RiverChannel => "River channel",
            HydroFeature::LakeBasin => "Lake basin",
            HydroFeature::OceanShelf => "Ocean shelf",
            HydroFeature::Coast => "Coast",
        }
    }
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
    pub global_min_y: i32,
    pub global_max_y: i32,
    pub surface_band_center: i32,
    pub deep_cave_start: i32,
    pub sky_ceiling_start: i32,
    pub terrain_scale: f32,
    pub cave_density: f32,
    pub tree_density: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerticalStratum {
    Deep,
    Surface,
    Sky,
}

#[derive(Clone, Copy)]
struct ProcGenControl<'a> {
    pub epoch: u64,
    pub should_cancel: &'a dyn Fn(u64) -> bool,
}

#[derive(Clone, Copy)]
struct ColumnGenData {
    wx: i32,
    wz: i32,
    weights: [f32; BIOME_COUNT],
    climate: ClimateSample,
    surface_height: i32,
    slope: i32,
    coastal: bool,
    river: bool,
    ocean: bool,
    stratum: VerticalBiomeStratum,
    landmark: Option<LandmarkKind>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ClimateSample {
    temperature: f32,
    moisture: f32,
    continentality: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerticalBiomeStratum {
    WetlandValley,
    Lowland,
    DryPlateau,
    Alpine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LandmarkKind {
    BoulderField,
    Ravine,
    Oasis,
    DeadwoodGrove,
}

#[derive(Clone, Copy, Debug)]
struct VegetationIntent {
    wx: i32,
    wy: i32,
    wz: i32,
    material: MaterialId,
}

#[derive(Clone, Copy)]
struct ProcGenStages {
    base_relief: bool,
    erosion_valley: bool,
    channel_extraction: bool,
    basin_filling: bool,
    material_painting: bool,
    vegetation: bool,
}

impl Default for ProcGenStages {
    fn default() -> Self {
        Self {
            base_relief: true,
            erosion_valley: true,
            channel_extraction: true,
            basin_filling: true,
            material_painting: true,
            vegetation: true,
        }
    }
}

#[derive(Clone)]
struct HydrologyData {
    river_weight: Vec<f32>,
    ocean_weight: Vec<f32>,
    lake_level: Vec<Option<i32>>,
    river_level: Vec<Option<i32>>,
}

#[derive(Clone, Copy, Debug)]
struct HydrologyColumnContext {
    surface_height: i32,
    flow_accum: f32,
    sink_owner: u64,
    spill_level: Option<i32>,
}

#[derive(Clone, Copy)]
struct HydrologyContextCell {
    local_idx: Option<usize>,
    surface_height: i32,
    ocean_weight: f32,
    river_weight: f32,
}

struct HydrologyContext {
    width: usize,
    depth: usize,
    cells: Vec<HydrologyContextCell>,
}

fn sample_climate(seed: u64, wx: i32, wz: i32) -> ClimateSample {
    let temperature = (fbm2(
        seed ^ 0x6C11_A001,
        wx as f32 * 0.0018,
        wz as f32 * 0.0018,
        4,
    ) * 0.75
        + fbm2(
            seed ^ 0x6C11_A002,
            wx as f32 * 0.0060,
            wz as f32 * 0.0060,
            2,
        ) * 0.25)
        .clamp(0.0, 1.0);
    let moisture = (fbm2(
        seed ^ 0x6C11_B001,
        wx as f32 * 0.0020,
        wz as f32 * 0.0020,
        4,
    ) * 0.7
        + fbm2(
            seed ^ 0x6C11_B002,
            wx as f32 * 0.0080,
            wz as f32 * 0.0080,
            2,
        ) * 0.3)
        .clamp(0.0, 1.0);
    let continentality = fbm2(
        seed ^ 0x6C11_C001,
        wx as f32 * 0.0016,
        wz as f32 * 0.0016,
        4,
    )
    .clamp(0.0, 1.0);
    ClimateSample {
        temperature,
        moisture,
        continentality,
    }
}

fn classify_vertical_biome_stratum(
    config: &ProcGenConfig,
    col: &ColumnGenData,
    surface_height: i32,
) -> VerticalBiomeStratum {
    let sea = config.sea_level_local();
    if surface_height <= sea + 1 && (col.river || col.climate.moisture > 0.60 || col.coastal) {
        return VerticalBiomeStratum::WetlandValley;
    }
    if surface_height >= sea + 22 || col.slope >= 6 || col.climate.temperature < 0.32 {
        return VerticalBiomeStratum::Alpine;
    }
    if surface_height >= sea + 10
        && col.climate.moisture < 0.38
        && col.climate.continentality > 0.55
        && col.slope <= 4
    {
        return VerticalBiomeStratum::DryPlateau;
    }
    VerticalBiomeStratum::Lowland
}

fn sample_landmark(
    seed: u64,
    wx: i32,
    wz: i32,
    climate: ClimateSample,
    slope: i32,
) -> Option<LandmarkKind> {
    let coarse = fbm2(
        seed ^ 0xDEAD_B001,
        wx as f32 * 0.0008,
        wz as f32 * 0.0008,
        3,
    );
    let mask = hash01(seed ^ 0xDEAD_B002, wx.div_euclid(12), 0, wz.div_euclid(12));
    if coarse > 0.86 && slope >= 5 && mask < 0.42 {
        return Some(LandmarkKind::Ravine);
    }
    if coarse > 0.82 && slope <= 3 && mask < 0.26 {
        return Some(LandmarkKind::BoulderField);
    }
    if climate.moisture < 0.30 && climate.temperature > 0.66 && coarse < 0.15 && mask < 0.30 {
        return Some(LandmarkKind::Oasis);
    }
    if climate.moisture < 0.44 && climate.temperature < 0.48 && coarse > 0.72 && mask < 0.22 {
        return Some(LandmarkKind::DeadwoodGrove);
    }
    None
}
impl ProcGenConfig {
    pub fn for_size(size: usize, seed: u64) -> Self {
        let sea_level = 18.min(size as i32 - 10).max(10);
        let configured_min_y = env_i32("SAND_WORLD_MIN_Y", DEFAULT_WORLD_MIN_Y);
        let configured_max_y = env_i32("SAND_WORLD_MAX_Y", DEFAULT_WORLD_MAX_Y);
        let global_min_y = configured_min_y.min(configured_max_y - 1);
        let global_max_y = configured_max_y.max(global_min_y + 1);

        Self {
            dims: [size, size, size],
            world_origin: [0, 0, 0],
            seed,
            sea_level,
            global_min_y,
            global_max_y,
            surface_band_center: 0,
            deep_cave_start: -64,
            sky_ceiling_start: 160,
            terrain_scale: 1.0,
            cave_density: 0.13,
            tree_density: 0.012,
        }
    }

    pub fn with_origin(mut self, world_origin: [i32; 3]) -> Self {
        self.world_origin = world_origin;
        self
    }

    fn sea_level_world(self) -> i32 {
        self.surface_band_center + self.sea_level
    }

    fn sea_level_local(self) -> i32 {
        self.sea_level_world() - self.world_origin[1]
    }

    fn stratum_for_world_y(self, world_y: i32) -> VerticalStratum {
        if world_y < self.deep_cave_start {
            VerticalStratum::Deep
        } else if world_y >= self.sky_ceiling_start {
            VerticalStratum::Sky
        } else {
            VerticalStratum::Surface
        }
    }
}

fn generate_world(config: ProcGenConfig) -> World {
    let never_cancel = |_epoch: u64| false;
    generate_world_with_control(
        config,
        ProcGenControl {
            epoch: 0,
            should_cancel: &never_cancel,
        },
    )
    .unwrap_or_else(|| {
        let mut world = World::new(config.dims);
        world.clear_empty();
        world
    })
}

fn generate_world_with_control(
    config: ProcGenConfig,
    control: ProcGenControl<'_>,
) -> Option<World> {
    let mut world = World::new(config.dims);
    world.clear_empty();
    let timings = ProcGenPassTimings::default();
    let stages = ProcGenStages::default();
    let (heights, columns) = build_column_cache(&config, &timings);
    let hydrology = build_hydrology_cache(&config, &heights, &columns, &timings);

    if (control.should_cancel)(control.epoch) {
        return None;
    }

    if stages.base_relief {
        base_terrain_pass(&mut world, &config, &heights, &timings);
    }
    if (control.should_cancel)(control.epoch) {
        return None;
    }
    if stages.erosion_valley {
        cave_carve_pass(&mut world, &config, &columns, &timings);
    }
    if (control.should_cancel)(control.epoch) {
        return None;
    }
    if stages.material_painting {
        surface_layering_pass(&mut world, &config, &columns, &hydrology, &timings);
        shoreline_transition_pass(&mut world, &config, &heights, &columns, &timings);
    }
    if stages.channel_extraction || stages.basin_filling {
        hydrology_fill_pass(
            &mut world, &config, &heights, &columns, &hydrology, &timings,
        );
        remove_unsupported_hanging_water_pass(&mut world, &config, &timings);
    }
    enforce_subsea_materials_pass(&mut world, &config, &timings);
    if stages.vegetation {
        vegetation_pass(&mut world, &config, &columns, &hydrology, &timings);
    }
    world.finalize_generation_side_effects();
    timings.log_total(config.world_origin);

    Some(world)
}

pub fn biome_hint_at_world(config: &ProcGenConfig, x: i32, z: i32) -> BiomeType {
    base_biome_at_world(config, x, z)
}

pub fn base_biome_at_world(config: &ProcGenConfig, x: i32, z: i32) -> BiomeType {
    let weights = biome_weights(config.seed, x, z, sample_climate(config.seed, x, z));
    dominant_base_biome(weights)
}

pub fn hydro_feature_at_world(config: &ProcGenConfig, x: i32, z: i32) -> HydroFeature {
    let weights = biome_weights(config.seed, x, z, sample_climate(config.seed, x, z));
    let ocean = weights[biome_index(BiomeType::Ocean)];
    if ocean > 0.62 {
        return HydroFeature::OceanShelf;
    }
    if ocean > 0.50 {
        return HydroFeature::Coast;
    }
    let lake = weights[biome_index(BiomeType::Lake)];
    if lake > 0.54 {
        return HydroFeature::LakeBasin;
    }
    let river = weights[biome_index(BiomeType::River)].max(river_meander_signal(config.seed, x, z));
    if river > 0.52 {
        return HydroFeature::RiverChannel;
    }
    HydroFeature::None
}

fn build_heightmap(config: &ProcGenConfig) -> Vec<i32> {
    build_column_cache(config, &ProcGenPassTimings::default()).0
}

fn build_column_cache(
    config: &ProcGenConfig,
    timings: &ProcGenPassTimings,
) -> (Vec<i32>, Vec<ColumnGenData>) {
    let _timer = timings.scoped("build_column_cache");
    let mut heights = vec![0; config.dims[0] * config.dims[2]];
    let mut columns = vec![
        ColumnGenData {
            wx: 0,
            wz: 0,
            weights: [0.0; BIOME_COUNT],
            climate: ClimateSample::default(),
            surface_height: 0,
            slope: 0,
            coastal: false,
            river: false,
            ocean: false,
            stratum: VerticalBiomeStratum::Lowland,
            landmark: None,
        };
        config.dims[0] * config.dims[2]
    ];
    for z in 0..config.dims[2] as i32 {
        for x in 0..config.dims[0] as i32 {
            let wx = config.world_origin[0] + x;
            let wz = config.world_origin[2] + z;
            let idx = x as usize + z as usize * config.dims[0];
            let climate = sample_climate(config.seed, wx, wz);
            let weights = biome_weights(config.seed, wx, wz, climate);
            let surface_height = sampled_surface_height(config, wx, wz, Some(weights));
            heights[idx] = surface_height;
            columns[idx] = ColumnGenData {
                wx,
                wz,
                weights,
                climate,
                surface_height,
                slope: 0,
                coastal: false,
                river: false,
                ocean: false,
                stratum: VerticalBiomeStratum::Lowland,
                landmark: None,
            };
        }
    }
    for z in 0..config.dims[2] as i32 {
        for x in 0..config.dims[0] as i32 {
            let idx = x as usize + z as usize * config.dims[0];
            let col = &mut columns[idx];
            let ocean = col.weights[biome_index(BiomeType::Ocean)];
            let river = col.weights[biome_index(BiomeType::River)].max(river_meander_signal(
                config.seed,
                col.wx,
                col.wz,
            ));
            let shore_w = smoothstep((ocean - 0.24) / 0.34);
            let near_sea_band = col.surface_height <= config.sea_level_local() + 4;
            col.slope = slope_at_world(config, col.wx, col.wz);
            col.coastal = shore_w > 0.18 && near_sea_band;
            col.river = river > 0.48;
            col.ocean = ocean > 0.55;
            col.stratum = classify_vertical_biome_stratum(config, col, col.surface_height);
            col.landmark = sample_landmark(config.seed, col.wx, col.wz, col.climate, col.slope);
        }
    }
    (heights, columns)
}

fn build_hydrology_cache(
    config: &ProcGenConfig,
    _heights: &[i32],
    columns: &[ColumnGenData],
    timings: &ProcGenPassTimings,
) -> HydrologyData {
    let _timer = timings.scoped("channel_extraction");
    let width = config.dims[0];
    let depth = config.dims[2];
    let len = width * depth;
    let pad_x = width as i32;
    let pad_z = depth as i32;
    build_hydrology_cache_impl(config, columns, len, width, depth, pad_x, pad_z)
}

fn build_hydrology_cache_for_chunk(
    config: &ProcGenConfig,
    _heights: &[i32],
    columns: &[ColumnGenData],
    timings: &ProcGenPassTimings,
) -> HydrologyData {
    let _timer = timings.scoped("channel_extraction_chunk");
    let width = config.dims[0];
    let depth = config.dims[2];
    let len = width * depth;
    let pad_x = (width as i32).max(16);
    let pad_z = (depth as i32).max(16);
    build_hydrology_cache_impl(config, columns, len, width, depth, pad_x, pad_z)
}

fn build_hydrology_cache_impl(
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    len: usize,
    width: usize,
    depth: usize,
    pad_x: i32,
    pad_z: i32,
) -> HydrologyData {
    let ctx_width = width + pad_x as usize * 2;
    let ctx_depth = depth + pad_z as usize * 2;
    let mut context = HydrologyContext {
        width: ctx_width,
        depth: ctx_depth,
        cells: Vec::with_capacity(ctx_width * ctx_depth),
    };
    for cz in 0..ctx_depth as i32 {
        for cx in 0..ctx_width as i32 {
            let wx = config.world_origin[0] + cx - pad_x;
            let wz = config.world_origin[2] + cz - pad_z;
            let local_x = wx - config.world_origin[0];
            let local_z = wz - config.world_origin[2];
            let local_idx =
                if local_x >= 0 && local_z >= 0 && local_x < width as i32 && local_z < depth as i32
                {
                    Some(local_x as usize + local_z as usize * width)
                } else {
                    None
                };
            let local_col = local_idx.map(|idx| columns[idx]);
            let local_weights = local_col.map(|col| col.weights).unwrap_or_else(|| {
                biome_weights(config.seed, wx, wz, sample_climate(config.seed, wx, wz))
            });
            let surface_height = local_col
                .map(|col| col.surface_height)
                .unwrap_or_else(|| sampled_surface_height(config, wx, wz, Some(local_weights)));
            let ocean = local_weights[biome_index(BiomeType::Ocean)];
            let river = local_weights[biome_index(BiomeType::River)].max(river_meander_signal(
                config.seed,
                wx,
                wz,
            ));
            let accum_seed = smoothstep((river - 0.24) / 0.65);
            let ocean_weight = (ocean * 0.75
                + smoothstep(
                    (config.sea_level_local() as f32 + 3.0 - surface_height as f32) / 7.0,
                ) * 0.25)
                .clamp(0.0, 1.0);
            let river_weight = (accum_seed * 0.75 + river * 0.4).clamp(0.0, 1.0);
            context.cells.push(HydrologyContextCell {
                local_idx,
                surface_height,
                ocean_weight,
                river_weight,
            });
        }
    }

    let ctx_len = context.width * context.depth;
    let mut flow_to = vec![None; ctx_len];
    for z in 0..context.depth as i32 {
        for x in 0..context.width as i32 {
            let idx = x as usize + z as usize * context.width;
            let center = context.cells[idx].surface_height;
            let mut best = center;
            let mut best_idx = None;
            for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                if nx < 0 || nz < 0 || nx >= context.width as i32 || nz >= context.depth as i32 {
                    continue;
                }
                let nidx = nx as usize + nz as usize * context.width;
                let nh = context.cells[nidx].surface_height;
                if nh < best {
                    best = nh;
                    best_idx = Some(nidx);
                }
            }
            flow_to[idx] = best_idx;
        }
    }

    let mut order: Vec<usize> = (0..ctx_len).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(context.cells[i].surface_height));
    let mut flow_accum = vec![1.0f32; ctx_len];
    for idx in order {
        if let Some(down) = flow_to[idx] {
            flow_accum[down] += flow_accum[idx];
        }
    }

    let mut ocean_weight = vec![0.0; len];
    let mut river_weight = vec![0.0; len];
    for idx in 0..ctx_len {
        if let Some(local_idx) = context.cells[idx].local_idx {
            let accum_norm = (flow_accum[idx].ln() / 6.0).clamp(0.0, 1.0);
            ocean_weight[local_idx] = context.cells[idx].ocean_weight;
            river_weight[local_idx] = smoothstep(
                (accum_norm * 0.9 + context.cells[idx].river_weight * 0.75 - 0.38) / 0.45,
            );
        }
    }

    let mut distance = vec![-1i32; ctx_len];
    for idx in 0..ctx_len {
        if distance[idx] >= 0 {
            continue;
        }
        let mut chain = Vec::new();
        let mut cur = idx;
        let mut guard = 0;
        while distance[cur] < 0 && guard < 256 {
            chain.push(cur);
            guard += 1;
            match flow_to[cur] {
                Some(next) => cur = next,
                None => {
                    distance[cur] = 0;
                    break;
                }
            }
        }
        let mut dist = distance[cur].max(0);
        while let Some(cell) = chain.pop() {
            if cell == cur {
                continue;
            }
            dist += 1;
            distance[cell] = dist;
        }
    }

    let mut sink_owner = vec![None::<usize>; ctx_len];
    for i in 0..ctx_len {
        let mut cur = i;
        let mut sink = None;
        for _ in 0..128 {
            if let Some(n) = flow_to[cur] {
                cur = n;
            } else {
                sink = Some(cur);
                break;
            }
        }
        sink_owner[i] = sink;
    }

    let mut sink_cells: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, sink) in sink_owner.iter().enumerate() {
        if let Some(s) = sink {
            sink_cells.entry(*s).or_default().push(i);
        }
    }
    let mut sink_spill = vec![None::<i32>; ctx_len];
    for (sink, cells) in &sink_cells {
        let mut in_sink = vec![false; ctx_len];
        for &i in cells {
            in_sink[i] = true;
        }
        let mut spill = i32::MAX;
        for &i in cells {
            let x = (i % context.width) as i32;
            let z = (i / context.width) as i32;
            for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                if nx < 0 || nz < 0 || nx >= context.width as i32 || nz >= context.depth as i32 {
                    continue;
                }
                let ni = nx as usize + nz as usize * context.width;
                if !in_sink[ni] {
                    spill = spill.min(context.cells[ni].surface_height);
                }
            }
        }
        if spill != i32::MAX {
            sink_spill[*sink] = Some(spill - 1);
        }
    }

    let mut river_level = vec![None; len];
    let mut river_mask = vec![false; ctx_len];
    for i in 0..ctx_len {
        river_mask[i] = context.cells[i].river_weight > 0.45;
    }
    let mut channel_level_ctx = vec![None::<i32>; ctx_len];
    for i in 0..ctx_len {
        if !river_mask[i] {
            continue;
        }
        let surface = context.cells[i].surface_height;
        let downstream_surface = flow_to[i]
            .map(|d| context.cells[d].surface_height)
            .unwrap_or(surface);
        let local_slope = (surface - downstream_surface).max(0) as f32;
        let dist_term = (distance[i].max(0) as f32) * 0.16;
        let slope_term = (1.0 / (local_slope + 1.0)).clamp(0.0, 1.0);
        let flow_term = flow_accum[i].ln().max(0.0) * 0.34;
        let incision = (1.0 + dist_term + slope_term + flow_term).round() as i32;
        let mut level = surface - incision.max(1);
        if let Some(sink) = sink_owner[i] {
            if let Some(spill_level) = sink_spill[sink] {
                level = level.min(spill_level);
            }
        }
        level = level.min(surface - 1);
        channel_level_ctx[i] = Some(level.max(1));
    }

    let mut topo = (0..ctx_len).collect::<Vec<_>>();
    topo.sort_by_key(|&i| std::cmp::Reverse(distance[i]));
    for idx in topo {
        let Some(level) = channel_level_ctx[idx] else {
            continue;
        };
        let Some(down) = flow_to[idx] else {
            continue;
        };
        if !river_mask[down] {
            continue;
        }
        if let Some(down_level) = channel_level_ctx[down] {
            if down_level > level {
                channel_level_ctx[down] = Some(level);
            }
        } else {
            channel_level_ctx[down] = Some(level);
        }
    }

    for i in 0..ctx_len {
        if let (Some(local_idx), Some(level)) = (context.cells[i].local_idx, channel_level_ctx[i]) {
            river_level[local_idx] = Some(level.min(context.cells[i].surface_height - 1));
        }
    }

    let mut basin_cells: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..ctx_len {
        if context.cells[i].ocean_weight > 0.55 || context.cells[i].river_weight > 0.55 {
            continue;
        }
        if let Some(s) = sink_owner[i] {
            basin_cells.entry(s).or_default().push(i);
        }
    }
    let mut lake_level = vec![None; len];
    for (_sink, cells) in basin_cells {
        if cells.len() < 4 {
            continue;
        }
        let mut in_basin = vec![false; ctx_len];
        for &i in &cells {
            in_basin[i] = true;
        }
        let mut spill = i32::MAX;
        for &i in &cells {
            let x = (i % context.width) as i32;
            let z = (i / context.width) as i32;
            for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                if nx < 0 || nz < 0 || nx >= context.width as i32 || nz >= context.depth as i32 {
                    continue;
                }
                let ni = nx as usize + nz as usize * context.width;
                if !in_basin[ni] {
                    spill = spill.min(context.cells[ni].surface_height);
                }
            }
        }
        if spill == i32::MAX {
            continue;
        }
        let level = (spill - 1).min(config.sea_level_local() + 10);
        for &i in &cells {
            if let Some(local_idx) = context.cells[i].local_idx {
                if context.cells[i].surface_height <= level {
                    lake_level[local_idx] = Some(level);
                }
            }
        }
    }
    HydrologyData {
        river_weight,
        ocean_weight,
        lake_level,
        river_level,
    }
}

fn pack_column_key(wx: i32, wz: i32) -> u64 {
    let x = (wx as u32) as u64;
    let z = (wz as u32) as u64;
    (x << 32) | z
}

fn base_terrain_pass(
    world: &mut World,
    config: &ProcGenConfig,
    heights: &[i32],
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("base_terrain_pass");
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
            let surface_local = heights[lx as usize + lz as usize * world.dims[0]];
            let surface_world = config.world_origin[1] + surface_local;
            for ly in 0..world.dims[1] as i32 {
                let world_y = config.world_origin[1] + ly;
                if world_y < config.global_min_y || world_y > config.global_max_y {
                    continue;
                }
                if world_y > surface_world {
                    continue;
                }
                if config.stratum_for_world_y(world_y) == VerticalStratum::Sky {
                    continue;
                }
                let _ = world.set_raw_no_side_effects(lx, ly, lz, STONE);
            }
        }
    }
}

fn cave_carve_pass(
    world: &mut World,
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("cave_carve_pass");
    let deep_cave_start = config.sea_level - 10;
    let surface_falloff_start = config.sea_level + 18;
    let max_y = world.dims[1] as i32 - 2;
    for lz in 0..world.dims[2] as i32 {
        for ly in 2..max_y {
            for lx in 0..world.dims[0] as i32 {
                if world.get(lx, ly, lz) == EMPTY {
                    continue;
                }
                let wx = config.world_origin[0] + lx;
                let world_y = config.world_origin[1] + ly;
                let wz = config.world_origin[2] + lz;
                if world_y < config.global_min_y || world_y > config.global_max_y {
                    continue;
                }
                let stratum = config.stratum_for_world_y(world_y);
                if stratum == VerticalStratum::Sky {
                    continue;
                }
                let col = &columns[lx as usize + lz as usize * world.dims[0]];
                let top_world = config.world_origin[1] + col.surface_height;
                if world_y >= top_world - 4 {
                    continue;
                }
                let depth_profile = smoothstep(((deep_cave_start - world_y) as f32) / 26.0);
                let near_surface_falloff =
                    1.0 - smoothstep(((world_y - surface_falloff_start) as f32) / 20.0);
                let global_depth = (depth_profile * near_surface_falloff).clamp(0.0, 1.0);

                let large_caverns = fbm3(
                    config.seed ^ 0xCC77AA11,
                    wx as f32 * 0.013,
                    world_y as f32 * 0.015,
                    wz as f32 * 0.013,
                    5,
                );

                let warp = fbm3(
                    config.seed ^ 0x1077BEEF,
                    wx as f32 * 0.02,
                    world_y as f32 * 0.02,
                    wz as f32 * 0.02,
                    3,
                ) - 0.5;
                let tunnel_network = fbm3(
                    config.seed ^ 0x7AA1_0444,
                    (wx as f32 + warp * 26.0) * 0.055,
                    (world_y as f32 + warp * 18.0) * 0.07,
                    (wz as f32 - warp * 26.0) * 0.055,
                    4,
                );

                let w = col.weights;
                let moisture = (w[biome_index(BiomeType::Forest)] * 0.8
                    + w[biome_index(BiomeType::River)]
                    + w[biome_index(BiomeType::Lake)] * 0.9
                    + w[biome_index(BiomeType::Ocean)] * 0.7
                    + w[biome_index(BiomeType::Plains)] * 0.35
                    - w[biome_index(BiomeType::Desert)] * 0.9)
                    .clamp(0.0, 1.0);
                let density_bias = (1.0 - config.cave_density.clamp(0.05, 0.25)) * 0.16;

                let cavern_threshold =
                    (0.74 - global_depth * 0.18 + moisture * 0.06 + density_bias).clamp(0.50, 0.90);
                let tunnel_threshold =
                    (0.70 - global_depth * 0.26 + moisture * 0.10 + density_bias * 0.7)
                        .clamp(0.42, 0.88);
                let carve = large_caverns > cavern_threshold || tunnel_network > tunnel_threshold;
                if carve {
                    let _ = world.set_raw_no_side_effects(lx, ly, lz, EMPTY);
                }
            }
        }
    }
}

fn cave_carve_pass_chunk(
    world: &mut World,
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    timings: &ProcGenPassTimings,
) {
    cave_carve_pass(world, config, columns, timings);
}

fn surface_layering_pass(
    world: &mut World,
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    hydrology: &HydrologyData,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("material_painting");
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
            let idx = lx as usize + lz as usize * world.dims[0];
            let col = &columns[idx];
            let top_y = col.surface_height;
            let weights = col.weights;
            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;

            let desert = weights[biome_index(BiomeType::Desert)];
            let marine_influence = hydrology.ocean_weight[idx]
                .max(weights[biome_index(BiomeType::Lake)])
                .max(if hydrology.lake_level[idx].is_some() {
                    0.75
                } else {
                    0.0
                });
            let marine_substrate = top_y <= config.sea_level_local() && marine_influence > 0.38;

            let desert_surface = if top_y <= config.sea_level_local() || marine_substrate {
                0.0
            } else {
                desert
            };
            let plains = weights[biome_index(BiomeType::Plains)];
            let highlands = weights[biome_index(BiomeType::Highlands)];
            let coastal = smoothstep((hydrology.ocean_weight[idx] - 0.35) / 0.45);
            let river_bank = smoothstep((hydrology.river_weight[idx] - 0.30) / 0.55);
            let slope = col.slope as f32;
            let climate = col.climate;
            let aridity = (climate.temperature * (1.0 - climate.moisture)).clamp(0.0, 1.0);
            let blend_mask = fbm2(
                config.seed ^ 0x1133_77AA,
                wx as f32 * 0.010,
                wz as f32 * 0.010,
                2,
            );
            let depth_mod = fbm2(
                config.seed ^ 0x5A17_9C20,
                wx as f32 * 0.0065,
                wz as f32 * 0.0065,
                3,
            );
            let anti_stripe = (blend_mask - 0.5) * 0.26 + (depth_mod - 0.5) * 0.12;

            let dirt_depth = (3.0 + 2.2 * plains + 1.4 * (1.0 - desert_surface) - 0.9 * highlands
                + climate.moisture * 1.6
                - aridity * 0.9)
                + anti_stripe * 5.0;
            let dirt_depth = dirt_depth.round().clamp(2.0, 8.0) as i32;
            let sand_depth = (1.0
                + 3.5 * desert_surface
                + 3.0 * coastal
                + 2.0 * river_bank
                + aridity * 2.0
                + anti_stripe * 4.0)
                .round()
                .clamp(1.0, 10.0) as i32;
            let sediment_context =
                coastal > 0.38 || river_bank > 0.42 || (desert_surface > 0.55 && slope < 4.0);
            let deep_stone_context =
                highlands > 0.58 || slope > 4.2 || top_y > config.sea_level_local() + 10;

            for d in 0..(dirt_depth + sand_depth + 2) {
                let y = top_y - d;
                if y <= 1 {
                    continue;
                }
                let sand_bias = (desert_surface * 0.8 + coastal * 0.7 + river_bank * 0.6
                    - slope * 0.05
                    + anti_stripe)
                    .clamp(0.0, 1.0);
                let rock_bias =
                    (highlands * 0.75 + slope * 0.12 - anti_stripe * 0.5).clamp(0.0, 1.0);
                let mut top_cover = if rock_bias > 0.65 && top_y > config.sea_level_local() + 4 {
                    STONE
                } else if sand_bias > 0.52 && sediment_context {
                    SAND
                } else {
                    TURF
                };
                match col.stratum {
                    VerticalBiomeStratum::WetlandValley => {
                        if top_cover == STONE {
                            top_cover = DIRT;
                        }
                    }
                    VerticalBiomeStratum::Lowland => {}
                    VerticalBiomeStratum::DryPlateau => {
                        if top_cover == TURF {
                            top_cover = if aridity > 0.58 { SAND } else { DIRT };
                        }
                    }
                    VerticalBiomeStratum::Alpine => {
                        top_cover = if slope > 3.0 { STONE } else { DIRT };
                    }
                }
                if matches!(
                    col.landmark,
                    Some(LandmarkKind::BoulderField | LandmarkKind::Ravine)
                ) {
                    top_cover = STONE;
                }
                if matches!(col.landmark, Some(LandmarkKind::Oasis)) {
                    top_cover = TURF;
                }
                let cover_depth = if top_cover == SAND {
                    sand_depth.min(3)
                } else {
                    1
                };

                let mut block = if d < cover_depth {
                    top_cover
                } else if d < cover_depth + dirt_depth {
                    if top_cover == STONE {
                        STONE
                    } else if top_cover == SAND && d <= cover_depth && sediment_context {
                        SAND
                    } else {
                        DIRT
                    }
                } else {
                    STONE
                };

                let sand_cap = cover_depth + (sand_depth / 2).max(1);
                if block == SAND && d > sand_cap {
                    block = if deep_stone_context { STONE } else { DIRT };
                }
                if block == SAND && deep_stone_context && !sediment_context {
                    block = STONE;
                }
                if block == SAND && !sediment_context {
                    block = DIRT;
                }
                if d >= cover_depth + dirt_depth {
                    block = STONE;
                }

                if d > 0 && top_cover == TURF && block == TURF {
                    block = DIRT;
                }

                if d > cover_depth + dirt_depth + 1 && block != STONE {
                    block = STONE;
                }

                block = if d <= sand_depth
                    && sand_bias > 0.35
                    && sediment_context
                    && !deep_stone_context
                {
                    SAND
                } else {
                    block
                };
                let _ = world.set_raw_no_side_effects(lx, y, lz, block);
            }
        }
    }
}

fn surface_layering_pass_chunk(
    world: &mut World,
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    hydrology: &HydrologyData,
    timings: &ProcGenPassTimings,
) {
    surface_layering_pass(world, config, columns, hydrology, timings);
}

fn slope_at_world(config: &ProcGenConfig, wx: i32, wz: i32) -> i32 {
    let center = sampled_surface_height(config, wx, wz, None);
    let mut max_delta = 0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dz == 0 {
                continue;
            }
            let h = sampled_surface_height(config, wx + dx, wz + dz, None);
            max_delta = max_delta.max((h - center).abs());
        }
    }
    max_delta
}

fn shoreline_transition_pass(
    world: &mut World,
    config: &ProcGenConfig,
    heights: &[i32],
    columns: &[ColumnGenData],
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("shoreline_transition_pass");
    let sea = config.sea_level_local();
    for lz in 1..world.dims[2] as i32 - 1 {
        for lx in 1..world.dims[0] as i32 - 1 {
            let col = &columns[lx as usize + lz as usize * world.dims[0]];
            let ocean = col.weights[biome_index(BiomeType::Ocean)];
            if ocean > 0.55 {
                continue;
            }

            let mut near_ocean = false;
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let nx = (lx + dx).clamp(0, world.dims[0] as i32 - 1);
                    let nz = (lz + dz).clamp(0, world.dims[2] as i32 - 1);
                    let ncol = &columns[nx as usize + nz as usize * world.dims[0]];
                    if ncol.weights[biome_index(BiomeType::Ocean)] > 0.62 {
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
                let _ = world.set_raw_no_side_effects(lx, y, lz, STONE);
            }
            for y in (stone_top + 1).max(2)..=berm_top {
                let _ = world.set_raw_no_side_effects(lx, y, lz, SAND);
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

fn flood_fill_columns(
    width: usize,
    depth: usize,
    start: usize,
    allowed: &[bool],
    visited: &mut [bool],
) -> Vec<usize> {
    let mut q = VecDeque::new();
    let mut cells = Vec::new();
    q.push_back(start);
    visited[start] = true;
    while let Some(idx) = q.pop_front() {
        cells.push(idx);
        let x = (idx % width) as i32;
        let z = (idx / width) as i32;
        for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
            if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                continue;
            }
            let ni = nx as usize + nz as usize * width;
            if visited[ni] || !allowed[ni] {
                continue;
            }
            visited[ni] = true;
            q.push_back(ni);
        }
    }
    cells
}

fn hydrology_fill_pass(
    world: &mut World,
    config: &ProcGenConfig,
    heights: &[i32],
    columns: &[ColumnGenData],
    hydrology: &HydrologyData,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("hydrology_fill_pass");
    let sea_level = config.sea_level_local();
    let width = world.dims[0];
    let depth = world.dims[2];
    let len = width * depth;

    let mut ocean_reachable = vec![false; len];
    let mut open = VecDeque::new();
    for z in 0..depth as i32 {
        for x in 0..width as i32 {
            let border = x == 0 || z == 0 || x == width as i32 - 1 || z == depth as i32 - 1;
            if !border {
                continue;
            }
            let idx = x as usize + z as usize * width;
            if heights[idx] <= sea_level {
                ocean_reachable[idx] = true;
                open.push_back(idx);
            }
        }
    }

    while let Some(idx) = open.pop_front() {
        let x = (idx % width) as i32;
        let z = (idx / width) as i32;
        for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
            if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                continue;
            }
            let ni = nx as usize + nz as usize * width;
            if ocean_reachable[ni] {
                continue;
            }
            let saddle = heights[idx].max(heights[ni]);
            if saddle <= sea_level {
                ocean_reachable[ni] = true;
                open.push_back(ni);
            }
        }
    }

    for idx in 0..len {
        if !ocean_reachable[idx] {
            continue;
        }
        let x = (idx % width) as i32;
        let z = (idx / width) as i32;
        let floor = (heights[idx] + 1).max(1);
        apply_channel_edit(world, x, z, floor, sea_level, 1, SAND, true);
    }

    for z in 0..depth as i32 {
        for x in 0..width as i32 {
            let idx = x as usize + z as usize * width;
            let surface = heights[idx];
            if surface < 1 || world.get(x, surface, z) == EMPTY {
                continue;
            }
            let wx = config.world_origin[0] + x;
            let wz = config.world_origin[2] + z;
            let river_w = hydrology.river_weight[idx];
            let lake_w = columns[idx].weights[biome_index(BiomeType::Lake)];
            let open_sky = is_column_open_to_sky(world, x, z, surface + 1);
            let estuary = ocean_reachable[idx] || hydrology.ocean_weight[idx] > 0.62;
            let aquifer = explicit_aquifer_flag(config.seed, wx, wz);

            if let Some(level) = hydrology.lake_level[idx] {
                if estuary {
                    continue;
                }
                let lake_fill = smoothstep((lake_w + 0.35 - 0.35) / 0.65);
                if lake_fill > 0.08 {
                    if !open_sky && !aquifer {
                        continue;
                    }
                    let top = level.max(surface).min(sea_level + 4);
                    let depth_hint = (2.0 + lake_fill * 4.0) as i32;
                    let floor = (surface - depth_hint).min(top);
                    if floor < 1 || top < floor {
                        continue;
                    }
                    if !estuary && world.get(x, floor - 1, z) == EMPTY {
                        continue;
                    }
                    apply_channel_edit(world, x, z, floor, top, 2, SAND, true);
                }
            }

            if let Some(level) = hydrology.river_level[idx] {
                let river_fill = smoothstep((river_w - 0.24) / 0.50);
                if river_fill <= 0.02 {
                    continue;
                }
                if estuary {
                    continue;
                }
                let top = level.min(surface).min(sea_level + 1);
                let channel_depth = (2.0 + river_fill * 4.5).round() as i32;
                let floor = top - channel_depth;
                if floor < 1 || top < floor {
                    continue;
                }
                let can_fill = open_sky || top >= surface - 1;
                if !can_fill {
                    continue;
                }
                if !estuary && world.get(x, floor - 1, z) == EMPTY {
                    continue;
                }
                let bank_blend = smoothstep(
                    (river_fill + columns[idx].weights[biome_index(BiomeType::Desert)] * 0.2 - 0.2)
                        / 0.8,
                );
                let bank_material = if bank_blend > 0.05 { SAND } else { DIRT };
                apply_channel_edit(world, x, z, floor, top, 3, bank_material, true);
            }
        }
    }
}

fn apply_channel_edit(
    world: &mut World,
    x: i32,
    z: i32,
    floor: i32,
    top: i32,
    bank_depth: i32,
    bank_material: MaterialId,
    expose_opening: bool,
) {
    if top < floor {
        return;
    }
    // 1) Carve channel profile.
    for y in floor..=top {
        let _ = world.set_raw_no_side_effects(x, y, z, EMPTY);
    }
    // 2) Expose top opening/banks.
    if expose_opening {
        for y in (top + 1)..world.dims[1] as i32 {
            if world.get(x, y, z) == EMPTY {
                break;
            }
            let _ = world.set_raw_no_side_effects(x, y, z, EMPTY);
        }
    }
    // 3) Fill water to target level.
    for y in floor..=top {
        let _ = world.set_raw_no_side_effects(x, y, z, WATER);
    }
    // 4) Paint/substitute bank materials after carve/fill.
    for y in (floor - bank_depth).max(1)..floor {
        let _ = world.set_raw_no_side_effects(x, y, z, bank_material);
    }
}

fn is_column_open_to_sky(world: &World, x: i32, z: i32, start_y: i32) -> bool {
    for y in start_y.max(0)..world.dims[1] as i32 {
        let mat = world.get(x, y, z);
        if mat != EMPTY && mat != WATER {
            return false;
        }
    }
    true
}

fn explicit_aquifer_flag(seed: u64, wx: i32, wz: i32) -> bool {
    fbm2(seed ^ 0xA811_22CC, wx as f32 * 0.021, wz as f32 * 0.021, 3) > 0.79
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
    sampled_surface_height(config, wx, wz, None)
}

#[cfg(test)]
fn explicit_rapid_or_fall(seed: u64, x0: i32, z0: i32, x1: i32, z1: i32) -> bool {
    let a = river_meander_signal(seed, x0, z0);
    let b = river_meander_signal(seed, x1, z1);
    (a - b).abs() > 0.55
}

fn remove_unsupported_hanging_water_pass(
    world: &mut World,
    config: &ProcGenConfig,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("remove_unsupported_hanging_water_pass");
    let sea = config.sea_level_local();
    let width = world.dims[0] as i32;
    let height = world.dims[1] as i32;
    let depth = world.dims[2] as i32;
    let mut supported = vec![false; world.dims[0] * world.dims[1] * world.dims[2]];
    let mut open = VecDeque::new();

    let idx3 = |x: i32, y: i32, z: i32, dims: [usize; 3]| -> usize {
        x as usize + y as usize * dims[0] + z as usize * dims[0] * dims[1]
    };

    for z in 0..depth {
        for x in 0..width {
            for y in 1..height {
                if world.get(x, y, z) != WATER {
                    continue;
                }
                let boundary_source =
                    (x == 0 || z == 0 || x == width - 1 || z == depth - 1) && y <= sea;
                let grounded = world.get(x, y - 1, z) != EMPTY;
                if boundary_source || grounded {
                    let idx = idx3(x, y, z, world.dims);
                    if !supported[idx] {
                        supported[idx] = true;
                        open.push_back((x, y, z));
                    }
                }
            }
        }
    }

    while let Some((x, y, z)) = open.pop_front() {
        for (nx, ny, nz) in [
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y - 1, z),
            (x, y + 1, z),
            (x, y, z - 1),
            (x, y, z + 1),
        ] {
            if nx < 0 || ny < 1 || nz < 0 || nx >= width || ny >= height || nz >= depth {
                continue;
            }
            if world.get(nx, ny, nz) != WATER {
                continue;
            }
            let ni = idx3(nx, ny, nz, world.dims);
            if supported[ni] {
                continue;
            }
            supported[ni] = true;
            open.push_back((nx, ny, nz));
        }
    }

    for z in 0..depth {
        for x in 0..width {
            for y in 1..height {
                if world.get(x, y, z) != WATER {
                    continue;
                }
                let idx = idx3(x, y, z, world.dims);
                let mut exposed = 0;
                for (nx, ny, nz) in [
                    (x - 1, y, z),
                    (x + 1, y, z),
                    (x, y - 1, z),
                    (x, y + 1, z),
                    (x, y, z - 1),
                    (x, y, z + 1),
                ] {
                    if world.get(nx, ny, nz) == EMPTY {
                        exposed += 1;
                    }
                }
                let hanging = world.get(x, y - 1, z) == EMPTY;
                if !supported[idx] && hanging && exposed >= 4 {
                    let _ = world.set_raw_no_side_effects(x, y, z, EMPTY);
                }
            }
        }
    }
}

fn enforce_subsea_materials_pass(
    world: &mut World,
    config: &ProcGenConfig,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("enforce_subsea_materials_pass");
    let sea = config.sea_level_local();
    for z in 0..world.dims[2] as i32 {
        for x in 0..world.dims[0] as i32 {
            let Some(top) = surface_y(world, x, z) else {
                continue;
            };
            if top > sea {
                continue;
            }
            for y in 1..=sea.min(world.dims[1] as i32 - 2) {
                let m = world.get(x, y, z);
                if m == TURF || m == DIRT || m == GRASS || m == BUSH || m == LEAVES || m == WOOD {
                    let replacement = if y >= sea - 2 { SAND } else { STONE };
                    let _ = world.set_raw_no_side_effects(x, y, z, replacement);
                }
                if m == WATER && world.get(x, y - 1, z) == EMPTY {
                    let _ = world.set_raw_no_side_effects(x, y - 1, z, STONE);
                }
            }
        }
    }
}

fn vegetation_pass(
    world: &mut World,
    config: &ProcGenConfig,
    columns: &[ColumnGenData],
    hydrology: &HydrologyData,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("vegetation_pass");
    let sea_level = config.sea_level_local();
    let mut intents = Vec::new();
    for lz in 0..world.dims[2] as i32 {
        for lx in 0..world.dims[0] as i32 {
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

            let col = &columns[lx as usize + lz as usize * world.dims[0]];
            let idx = lx as usize + lz as usize * world.dims[0];
            if hydrology_wet_candidate(hydrology, col.surface_height, sea_level, idx)
                && col.stratum != VerticalBiomeStratum::WetlandValley
            {
                continue;
            }

            let climate = col.climate;
            let wx = col.wx;
            let wz = col.wz;
            let weights = col.weights;
            let forest = weights[biome_index(BiomeType::Forest)];
            let plains = weights[biome_index(BiomeType::Plains)];
            let highlands = weights[biome_index(BiomeType::Highlands)];
            let desert = weights[biome_index(BiomeType::Desert)];
            let wet =
                weights[biome_index(BiomeType::River)] + weights[biome_index(BiomeType::Lake)];
            let ocean = weights[biome_index(BiomeType::Ocean)];

            let mut tree_p = config.tree_density
                * (0.16 + 1.45 * forest + 0.35 * plains + climate.moisture * 0.55)
                * (1.0 - desert * (1.2 + climate.temperature)).powf(2.2)
                * (1.0 - highlands * 0.30).max(0.22)
                * (1.0 - ocean * 0.94).max(0.02);

            match col.stratum {
                VerticalBiomeStratum::WetlandValley => tree_p *= 1.28,
                VerticalBiomeStratum::Lowland => {}
                VerticalBiomeStratum::DryPlateau => tree_p *= 0.45,
                VerticalBiomeStratum::Alpine => tree_p *= 0.16,
            }

            if col.coastal
                || col.ocean
                || (ground == SAND && col.landmark != Some(LandmarkKind::Oasis))
            {
                tree_p *= 0.03;
            }
            if near_water(world, lx, top_y, lz) {
                tree_p *= (1.0 - wet * 0.7).max(0.05);
            }
            if matches!(col.landmark, Some(LandmarkKind::DeadwoodGrove)) {
                tree_p *= 0.5;
            }

            let roll = hash01(config.seed ^ 0x1111_7777, wx, top_y, wz);
            if roll < tree_p {
                let base_world_y = config.world_origin[1] + top_y + 1;
                if can_place_tree(world, lx, top_y + 1, lz) {
                    stage_tree_intents(
                        &mut intents,
                        config.seed,
                        wx,
                        wz,
                        base_world_y,
                        col.stratum,
                        col.landmark,
                    );
                    continue;
                }
            }

            let flora_roll = hash01(config.seed ^ 0x2222_4444, wx, top_y, wz);
            let mut flora_p = 0.04 + 0.08 * forest + 0.05 * plains + climate.moisture * 0.08;
            flora_p *= match col.stratum {
                VerticalBiomeStratum::WetlandValley => 1.4,
                VerticalBiomeStratum::Lowland => 1.0,
                VerticalBiomeStratum::DryPlateau => 0.6,
                VerticalBiomeStratum::Alpine => 0.4,
            };
            if matches!(
                col.landmark,
                Some(LandmarkKind::BoulderField | LandmarkKind::Ravine)
            ) {
                flora_p *= 0.35;
            }
            if flora_roll < flora_p {
                let flora_material = vegetation_ground_cover(col, flora_roll, ground);
                intents.push(VegetationIntent {
                    wx,
                    wy: config.world_origin[1] + top_y + 1,
                    wz,
                    material: flora_material,
                });
            }
        }
    }
    apply_vegetation_intents(world, config, &intents);
}

fn vegetation_pass_chunk(
    world: &mut World,
    config: &ProcGenConfig,
    _columns: &[ColumnGenData],
    _hydrology: &HydrologyData,
    timings: &ProcGenPassTimings,
) {
    let _timer = timings.scoped("vegetation_pass_chunk_worldspace");
    let mut intents = Vec::new();
    let chunk_radius = 2;

    for lz in -chunk_radius..(world.dims[2] as i32 + chunk_radius) {
        for lx in -chunk_radius..(world.dims[0] as i32 + chunk_radius) {
            let wx = config.world_origin[0] + lx;
            let wz = config.world_origin[2] + lz;
            let climate = sample_climate(config.seed, wx, wz);
            let weights = biome_weights(config.seed, wx, wz, climate);
            let surface_height = sampled_surface_height(config, wx, wz, Some(weights));
            let slope = slope_at_world(config, wx, wz);
            let ocean = weights[biome_index(BiomeType::Ocean)];
            let shore_w = smoothstep((ocean - 0.24) / 0.34);
            let coastal = shore_w > 0.18 && surface_height <= config.sea_level_local() + 4;
            let stratum = classify_vertical_biome_stratum(
                config,
                &ColumnGenData {
                    wx,
                    wz,
                    weights,
                    climate,
                    surface_height,
                    slope,
                    coastal,
                    river: false,
                    ocean: ocean > 0.55,
                    stratum: VerticalBiomeStratum::Lowland,
                    landmark: sample_landmark(config.seed, wx, wz, climate, slope),
                },
                surface_height,
            );
            let landmark = sample_landmark(config.seed, wx, wz, climate, slope);

            let forest = weights[biome_index(BiomeType::Forest)];
            let plains = weights[biome_index(BiomeType::Plains)];
            let highlands = weights[biome_index(BiomeType::Highlands)];
            let desert = weights[biome_index(BiomeType::Desert)];
            let wet =
                weights[biome_index(BiomeType::River)] + weights[biome_index(BiomeType::Lake)];

            let mut tree_p = config.tree_density
                * (0.16 + 1.45 * forest + 0.35 * plains + climate.moisture * 0.55)
                * (1.0 - desert * (1.2 + climate.temperature)).powf(2.2)
                * (1.0 - highlands * 0.30).max(0.22)
                * (1.0 - ocean * 0.94).max(0.02);

            match stratum {
                VerticalBiomeStratum::WetlandValley => tree_p *= 1.28,
                VerticalBiomeStratum::Lowland => {}
                VerticalBiomeStratum::DryPlateau => tree_p *= 0.45,
                VerticalBiomeStratum::Alpine => tree_p *= 0.16,
            }

            if coastal || ocean > 0.55 {
                tree_p *= 0.03;
            }
            tree_p *= (1.0 - wet * 0.7).max(0.05);
            if matches!(landmark, Some(LandmarkKind::DeadwoodGrove)) {
                tree_p *= 0.5;
            }

            let roll = hash01(config.seed ^ 0x1111_7777, wx, surface_height, wz);
            if roll >= tree_p {
                continue;
            }

            let base_world_y = surface_height + 1;
            stage_tree_intents(
                &mut intents,
                config.seed,
                wx,
                wz,
                base_world_y,
                stratum,
                landmark,
            );
        }
    }

    apply_vegetation_intents(world, config, &intents);
}

fn apply_vegetation_intents(
    world: &mut World,
    config: &ProcGenConfig,
    intents: &[VegetationIntent],
) {
    for intent in intents {
        let lx = intent.wx - config.world_origin[0];
        let ly = intent.wy - config.world_origin[1];
        let lz = intent.wz - config.world_origin[2];
        if lx < 0
            || ly < 0
            || lz < 0
            || lx >= world.dims[0] as i32
            || ly >= world.dims[1] as i32
            || lz >= world.dims[2] as i32
        {
            continue;
        }
        if world.get(lx, ly, lz) == EMPTY {
            let _ = world.set_raw_no_side_effects(lx, ly, lz, intent.material);
        }
    }
}

fn vegetation_ground_cover(col: &ColumnGenData, flora_roll: f32, ground: MaterialId) -> MaterialId {
    if matches!(col.landmark, Some(LandmarkKind::DeadwoodGrove)) {
        return BUSH;
    }
    if matches!(col.stratum, VerticalBiomeStratum::Alpine) {
        return if flora_roll < 0.03 { BUSH } else { GRASS };
    }
    if matches!(col.stratum, VerticalBiomeStratum::WetlandValley) && ground != SAND {
        return if flora_roll < 0.04 { BUSH } else { GRASS };
    }
    if ground == SAND {
        return if flora_roll < 0.018 { BUSH } else { GRASS };
    }
    if flora_roll < 0.026 {
        BUSH
    } else {
        GRASS
    }
}

fn stage_tree_intents(
    intents: &mut Vec<VegetationIntent>,
    seed: u64,
    wx: i32,
    wz: i32,
    base_world_y: i32,
    stratum: VerticalBiomeStratum,
    landmark: Option<LandmarkKind>,
) {
    let mut trunk_h = 4 + (hash01(seed ^ 0x7133_5599, wx, base_world_y, wz) * 4.0) as i32;
    if matches!(stratum, VerticalBiomeStratum::Alpine) {
        trunk_h = trunk_h.saturating_sub(2).max(2);
    }
    if matches!(landmark, Some(LandmarkKind::DeadwoodGrove)) {
        trunk_h = trunk_h.saturating_sub(1).max(2);
    }

    for ty in 0..trunk_h {
        intents.push(VegetationIntent {
            wx,
            wy: base_world_y + ty,
            wz,
            material: WOOD,
        });
    }

    if matches!(landmark, Some(LandmarkKind::DeadwoodGrove)) {
        return;
    }

    let radius = if matches!(stratum, VerticalBiomeStratum::Alpine) {
        1
    } else {
        2
    };
    let top = base_world_y + trunk_h;
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist = dx * dx + dz * dz + (dy * dy * 2 / 3);
                let cap = if radius == 1 { 2 } else { 6 };
                if dist > cap {
                    continue;
                }
                intents.push(VegetationIntent {
                    wx: wx + dx,
                    wy: top + dy,
                    wz: wz + dz,
                    material: LEAVES,
                });
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

fn sampled_surface_height(
    config: &ProcGenConfig,
    x: i32,
    z: i32,
    cached_weights: Option<[f32; BIOME_COUNT]>,
) -> i32 {
    let w0 = cached_weights
        .unwrap_or_else(|| biome_weights(config.seed, x, z, sample_climate(config.seed, x, z)));
    let raw = terrain_height(config, x, z, w0) as f32;

    // Use a light, slope-aware cross filter (5 taps) instead of an aggressive
    // 3x3 box filter so we keep local relief while still suppressing spikes.
    let mut cross_sum = raw;
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let sx = x + dx;
        let sz = z + dz;
        let w = biome_weights(config.seed, sx, sz, sample_climate(config.seed, sx, sz));
        cross_sum += terrain_height(config, sx, sz, w) as f32;
    }
    let cross_avg = cross_sum / 5.0;

    let local_slope = (raw - cross_avg).abs();
    let ocean_w = w0[biome_index(BiomeType::Ocean)];
    let plains_w = w0[biome_index(BiomeType::Plains)];
    let smooth_strength =
        (0.10 + plains_w * 0.08 + ocean_w * 0.20 - local_slope * 0.035).clamp(0.04, 0.28);

    let world_surface = (raw * (1.0 - smooth_strength) + cross_avg * smooth_strength)
        .round()
        .clamp(config.global_min_y as f32, config.global_max_y as f32)
        as i32;
    world_surface - config.world_origin[1]
}

fn terrain_height(config: &ProcGenConfig, x: i32, z: i32, weights: [f32; BIOME_COUNT]) -> i32 {
    let scale = config.terrain_scale.max(0.25);
    let continental = fbm2(
        config.seed ^ 0xA1000001,
        x as f32 * 0.0032 * scale,
        z as f32 * 0.0032 * scale,
        5,
    );
    let ridge = fbm2(
        config.seed ^ 0xA1000002,
        x as f32 * 0.0078 * scale,
        z as f32 * 0.0078 * scale,
        4,
    );
    let detail = fbm2(
        config.seed ^ 0xA1000003,
        x as f32 * 0.024 * scale,
        z as f32 * 0.024 * scale,
        3,
    );

    let highlands = weights[biome_index(BiomeType::Highlands)];
    let plains = weights[biome_index(BiomeType::Plains)];
    let forest = weights[biome_index(BiomeType::Forest)];
    let desert = weights[biome_index(BiomeType::Desert)];
    let lake = weights[biome_index(BiomeType::Lake)];
    let river = weights[biome_index(BiomeType::River)].max(river_meander_signal(config.seed, x, z));
    let ocean_w = weights[biome_index(BiomeType::Ocean)];

    let sea_level_world = config.sea_level_world() as f32;
    let inland_core = smoothstep((continental - 0.54) / 0.24);
    let broad = (continental - 0.44) * (30.0 + highlands * 24.0 + inland_core * 6.0);
    let continental_target = sea_level_world + broad;
    let desert_continental_scale = smoothstep((continental_target - sea_level_world + 2.0) / 8.0);
    let desert_land = desert * desert_continental_scale;

    let biome_amp = plains * 8.8 + forest * 10.8 + desert_land * 7.0 + highlands * 25.6;
    let biome_rough = plains * 0.28 + forest * 0.52 + desert_land * 0.38 + highlands * 1.14;

    let rough = (ridge - 0.5) * (8.5 + biome_amp * 0.32);
    let micro = (detail - 0.5) * (2.8 + biome_rough * 8.0);

    let mut inland = sea_level_world + broad + rough + micro;
    let interior_elevation_boost = inland_core * (1.0 - ocean_w).clamp(0.0, 1.0);
    inland += highlands * (8.0 + 6.2 * interior_elevation_boost);
    inland += forest * 1.6;
    inland += desert_land * 0.8;
    inland += plains * (0.6 + 2.0 * interior_elevation_boost);

    let valley = smoothstep((river - 0.28) / 0.62);
    let valley_cut = valley * (3.5 + highlands * 2.0 + plains * 1.0);
    inland -= valley_cut;
    inland -= lake * 4.2;

    let ocean_depth_shape = (fbm2(
        config.seed ^ 0x0CEA_0001,
        x as f32 * 0.003,
        z as f32 * 0.003,
        3,
    ) - 0.5)
        * 3.0;
    let ocean_floor_target = sea_level_world - 11.0 + ocean_depth_shape;
    let coast_w = smoothstep((ocean_w - 0.60) / 0.24);
    let coast_target = sea_level_world - 1.0;

    let mut h = inland;
    h = h * (1.0 - ocean_w * 0.74) + ocean_floor_target * ocean_w * 0.74;
    h = h * (1.0 - coast_w * 0.24) + coast_target * coast_w * 0.24;

    if ocean_w > 0.76 {
        h = h.min(sea_level_world - 1.0);
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

fn biome_weights(seed: u64, x: i32, z: i32, climate: ClimateSample) -> [f32; BIOME_COUNT] {
    let mut weights = [0.0; BIOME_COUNT];
    let macro_x = x as f32 / MACROCHUNK_SIZE as f32;
    let macro_z = z as f32 / MACROCHUNK_SIZE as f32;
    let cluster_scale = BIOME_CLUSTER_MACROS as f32;
    let coarse_warp_x = (fbm2(seed ^ 0x5522AA11, macro_x * 0.16, macro_z * 0.16, 4) - 0.5) * 2.2;
    let coarse_warp_z = (fbm2(seed ^ 0x5522AA12, macro_x * 0.16, macro_z * 0.16, 4) - 0.5) * 2.2;
    let fine_warp_x =
        (fbm2(seed ^ 0x2211_8899, x as f32 * 0.0021, z as f32 * 0.0021, 2) - 0.5) * 0.7;
    let fine_warp_z =
        (fbm2(seed ^ 0x2211_889A, x as f32 * 0.0021, z as f32 * 0.0021, 2) - 0.5) * 0.7;
    let warped_x = macro_x / cluster_scale + coarse_warp_x + fine_warp_x;
    let warped_z = macro_z / cluster_scale + coarse_warp_z + fine_warp_z;

    for i in 0..BIOME_COUNT {
        let jitter_x = hash01(
            seed ^ 0xABCD_0000 ^ i as u64,
            warped_x.floor() as i32,
            0,
            warped_z.floor() as i32,
        ) - 0.5;
        let jitter_z = hash01(
            seed ^ 0xABCD_1000 ^ i as u64,
            warped_x.floor() as i32,
            0,
            warped_z.floor() as i32,
        ) - 0.5;
        let field = fbm2(
            seed ^ 0x4100_0000 ^ (i as u64).wrapping_mul(0x9E37_79B9),
            (warped_x + jitter_x * 0.42) * 0.95 + i as f32 * 0.217,
            (warped_z + jitter_z * 0.42) * 0.95 - i as f32 * 0.173,
            4,
        );
        let ridge = fbm2(
            seed ^ 0x4600_0000 ^ (i as u64).wrapping_mul(0xBF58_476D),
            (warped_x - jitter_z * 0.28) * 1.4,
            (warped_z + jitter_x * 0.28) * 1.4,
            2,
        );
        weights[i] = (field * 0.72 + ridge * 0.28).max(0.01);
    }

    let warp = fbm2(seed ^ 0xF009_1201, x as f32 * 0.0024, z as f32 * 0.0024, 3) - 0.5;
    let river = river_meander_signal(seed, x, z);
    let continental = fbm2(seed ^ 0xAD991100, x as f32 * 0.0016, z as f32 * 0.0016, 4);
    let coast_noise = fbm2(seed ^ 0xD1CE_4001, x as f32 * 0.0032, z as f32 * 0.0032, 2);
    let coastal_band = smoothstep((0.52 - continental) / 0.28);
    let deep_ocean_bias = smoothstep((0.46 - continental) / 0.32)
        * (0.75 + 0.25 * smoothstep((0.64 - climate.continentality) / 0.28));
    let coast_ocean_bias = coastal_band * (coast_noise - 0.46).max(0.0) * 0.70;
    let desert_continental_scale = smoothstep((continental - 0.64) / 0.20)
        * smoothstep((climate.temperature - 0.45) / 0.35)
        * smoothstep((0.56 - climate.moisture) / 0.42);
    weights[biome_index(BiomeType::Desert)] *= desert_continental_scale;

    let hydro_boost = smoothstep((climate.moisture - 0.46) / 0.35);
    weights[biome_index(BiomeType::River)] = (weights[biome_index(BiomeType::River)] * 0.40
        + river * 0.82
        + hydro_boost * 0.22
        + warp * 0.1)
        .max(0.0);
    weights[biome_index(BiomeType::Lake)] =
        (weights[biome_index(BiomeType::Lake)] + warp * 0.08 + hydro_boost * 0.16).max(0.0);
    weights[biome_index(BiomeType::Ocean)] = (weights[biome_index(BiomeType::Ocean)]
        + deep_ocean_bias * 1.35
        + coast_ocean_bias
        + (0.50 - climate.continentality).max(0.0) * 0.55
        - (continental - 0.63).max(0.0) * 0.50)
        .max(0.0);
    let relief = fbm2(seed ^ 0xCC99_1010, x as f32 * 0.0025, z as f32 * 0.0025, 3);
    weights[biome_index(BiomeType::Highlands)] = (weights[biome_index(BiomeType::Highlands)]
        + (relief - 0.56).max(0.0) * 1.25
        + (0.42 - climate.temperature).max(0.0) * 0.35)
        .max(0.0);
    weights[biome_index(BiomeType::Forest)] = (weights[biome_index(BiomeType::Forest)]
        + hydro_boost * 0.55
        + (0.60 - climate.continentality).max(0.0) * 0.25)
        .max(0.0);
    weights[biome_index(BiomeType::Plains)] = (weights[biome_index(BiomeType::Plains)]
        + (0.60 - relief).max(0.0) * 0.9
        + (0.64 - climate.moisture).max(0.0) * 0.2)
        .max(0.0);

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

fn dominant_biome(weights: [f32; BIOME_COUNT]) -> BiomeType {
    dominant_base_biome(weights)
}

fn dominant_base_biome(weights: [f32; BIOME_COUNT]) -> BiomeType {
    let mut best = 0;
    for i in 1..=3 {
        if weights[i] > weights[best] {
            best = i;
        }
    }
    match best {
        0 => BiomeType::Forest,
        1 => BiomeType::Plains,
        2 => BiomeType::Highlands,
        _ => BiomeType::Desert,
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

fn hydrology_wet_candidate(
    hydrology: &HydrologyData,
    surface_height: i32,
    sea_level: i32,
    idx: usize,
) -> bool {
    (hydrology.ocean_weight[idx] > 0.05 && surface_height <= sea_level)
        || hydrology.lake_level[idx].is_some_and(|level| surface_height <= level)
        || hydrology.river_level[idx].is_some_and(|level| surface_height <= level + 1)
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
    use std::collections::VecDeque;

    fn is_surface_material(mat: MaterialId) -> bool {
        matches!(mat, TURF | SAND | STONE)
    }

    fn is_subsoil(mat: MaterialId) -> bool {
        matches!(mat, DIRT | SAND)
    }

    fn is_ground(mat: MaterialId) -> bool {
        matches!(mat, STONE | DIRT | SAND | TURF)
    }

    fn water_surface(world: &World, x: i32, z: i32) -> Option<i32> {
        for y in (1..world.dims[1] as i32).rev() {
            if world.get(x, y, z) == WATER {
                return Some(y);
            }
        }
        None
    }

    fn cave_agreement(a: &World, b: &World, axis: char) -> (usize, usize) {
        let heights_a = build_surface_heightmap_from_world(a);
        let heights_b = build_surface_heightmap_from_world(b);
        let mut matches = 0usize;
        let mut total = 0usize;
        let max_y = (a.dims[1] as i32 - 2).min(b.dims[1] as i32 - 2);
        for t in 0..a.dims[2] as i32 {
            for y in 4..max_y {
                let (ax, az, bx, bz, top_a, top_b) = if axis == 'x' {
                    (
                        a.dims[0] as i32 - 1,
                        t,
                        0,
                        t,
                        heights_a[a.dims[0] - 1 + t as usize * a.dims[0]],
                        heights_b[t as usize * b.dims[0]],
                    )
                } else {
                    (
                        t,
                        a.dims[2] as i32 - 1,
                        t,
                        0,
                        heights_a[t as usize + (a.dims[2] - 1) * a.dims[0]],
                        heights_b[t as usize],
                    )
                };

                if y >= top_a.min(top_b) - 4 {
                    continue;
                }
                total += 1;
                let cave_a = a.get(ax, y, az) == EMPTY;
                let cave_b = b.get(bx, y, bz) == EMPTY;
                if cave_a == cave_b {
                    matches += 1;
                }
            }
        }
        (matches, total)
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
        let center_cfg = ProcGenConfig::for_size(size, seed).with_origin([0, 0, 0]);
        let east_cfg = ProcGenConfig::for_size(size, seed).with_origin([size as i32, 0, 0]);
        let south_cfg = ProcGenConfig::for_size(size, seed).with_origin([0, 0, size as i32]);

        let center = generate_world(center_cfg);
        let east = generate_world(east_cfg);
        let south = generate_world(south_cfg);

        for z in 0..size as i32 {
            let a = water_surface(&center, size as i32 - 1, z);
            let b = water_surface(&east, 0, z);
            if let (Some(al), Some(bl)) = (a, b) {
                assert!(
                    (al - bl).abs() <= 1,
                    "east/west border waterline mismatch at z={z}: {al} vs {bl}"
                );
            }
        }

        for x in 0..size as i32 {
            let a = water_surface(&center, x, size as i32 - 1);
            let b = water_surface(&south, x, 0);
            if let (Some(al), Some(bl)) = (a, b) {
                assert!(
                    (al - bl).abs() <= 1,
                    "north/south border waterline mismatch at x={x}: {al} vs {bl}"
                );
            }
        }
    }

    #[test]
    fn below_sea_connected_basins_are_water_filled() {
        let config = ProcGenConfig::for_size(64, 0x0CEA_0123).with_origin([0, 0, 0]);
        let world = generate_world(config);
        let sea = config.sea_level_local();
        let heights = build_surface_heightmap_from_world(&world);
        let width = world.dims[0];
        let depth = world.dims[2];
        let mut reachable = vec![false; width * depth];
        let mut q = VecDeque::new();

        for z in 0..depth as i32 {
            for x in 0..width as i32 {
                let border = x == 0 || z == 0 || x == width as i32 - 1 || z == depth as i32 - 1;
                if !border {
                    continue;
                }
                let idx = x as usize + z as usize * width;
                if heights[idx] <= sea {
                    reachable[idx] = true;
                    q.push_back(idx);
                }
            }
        }

        while let Some(idx) = q.pop_front() {
            let x = (idx % width) as i32;
            let z = (idx / width) as i32;
            for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                    continue;
                }
                let ni = nx as usize + nz as usize * width;
                if reachable[ni] {
                    continue;
                }
                if heights[idx].max(heights[ni]) <= sea {
                    reachable[ni] = true;
                    q.push_back(ni);
                }
            }
        }

        for z in 0..depth as i32 {
            for x in 0..width as i32 {
                let idx = x as usize + z as usize * width;
                if !reachable[idx] || heights[idx] >= sea {
                    continue;
                }
                let mut wet = false;
                for y in (heights[idx] + 1).max(1)..=sea {
                    if world.get(x, y, z) == WATER {
                        wet = true;
                        break;
                    }
                }
                assert!(
                    wet,
                    "reachable basin column ({x},{z}) below sea level is dry"
                );
            }
        }
    }

    #[test]
    fn no_shoreline_retreat_with_adjacent_floating_water_columns() {
        let config = ProcGenConfig::for_size(64, 0x51DE_C0DE).with_origin([0, 0, 0]);
        let world = generate_world(config);
        let sea = config.sea_level_local();

        for z in 1..world.dims[2] as i32 - 1 {
            for x in 1..world.dims[0] as i32 - 1 {
                let Some(level) = water_surface(&world, x, z) else {
                    continue;
                };
                if (level - sea).abs() > 1 {
                    continue;
                }
                let left_shore =
                    world.get(x - 1, sea, z) == EMPTY && world.get(x - 1, sea - 1, z) != WATER;
                let right_shore =
                    world.get(x + 1, sea, z) == EMPTY && world.get(x + 1, sea - 1, z) != WATER;
                if !(left_shore || right_shore) {
                    continue;
                }
                let floating = world.get(x, sea, z) == WATER && world.get(x, sea - 1, z) == EMPTY;
                assert!(
                    !floating,
                    "floating shoreline water column at ({x},{z}) creates retreat artifact"
                );
            }
        }
    }

    #[test]
    fn inland_river_channels_can_stay_above_local_sea_level() {
        let config = ProcGenConfig::for_size(64, 0xD00D_A11E).with_origin([0, 96, 0]);
        let timings = ProcGenPassTimings::default();
        let (heights, columns) = build_column_cache(&config, &timings);
        let hydrology = build_hydrology_cache(&config, &heights, &columns, &timings);
        let sea = config.sea_level_local();

        let mut saw_river_channel = false;
        let mut found_above_sea = false;
        for (idx, level) in hydrology.river_level.iter().enumerate() {
            let Some(level) = level else {
                continue;
            };
            saw_river_channel = true;
            if columns[idx].surface_height > sea + 1 && *level > sea {
                found_above_sea = true;
                break;
            }
        }

        if saw_river_channel {
            assert!(
                found_above_sea,
                "expected inland river levels to remain above local sea level"
            );
        }
    }

    #[test]
    fn high_altitude_chunks_do_not_spawn_unsupported_water() {
        let config = ProcGenConfig::for_size(64, 0x51DE_C0DE).with_origin([0, 320, 0]);
        let world = generate_world(config);

        for z in 0..world.dims[2] as i32 {
            for y in 1..world.dims[1] as i32 {
                for x in 0..world.dims[0] as i32 {
                    if world.get(x, y, z) != WATER {
                        continue;
                    }
                    assert_ne!(
                        world.get(x, y - 1, z),
                        EMPTY,
                        "unsupported floating water at ({x},{y},{z}) in high-altitude chunk"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_border_caves_use_global_context() {
        let size = 64;
        let seed = 0xAA55_7711;
        let center_cfg = ProcGenConfig::for_size(size, seed).with_origin([0, 0, 0]);
        let east_cfg = ProcGenConfig::for_size(size, seed).with_origin([size as i32, 0, 0]);
        let south_cfg = ProcGenConfig::for_size(size, seed).with_origin([0, 0, size as i32]);

        let center = generate_world(center_cfg);
        let east = generate_world(east_cfg);
        let south = generate_world(south_cfg);

        let (ew_matches, ew_total) = cave_agreement(&center, &east, 'x');
        let (ns_matches, ns_total) = cave_agreement(&center, &south, 'z');
        let ew_ratio = ew_matches as f32 / ew_total.max(1) as f32;
        let ns_ratio = ns_matches as f32 / ns_total.max(1) as f32;

        assert!(
            ew_ratio > 0.60,
            "east/west cave border mismatch too high: {:.2}% agreement ({}/{})",
            ew_ratio * 100.0,
            ew_matches,
            ew_total
        );
        assert!(
            ns_ratio > 0.60,
            "north/south cave border mismatch too high: {:.2}% agreement ({}/{})",
            ns_ratio * 100.0,
            ns_matches,
            ns_total
        );
    }

    #[test]
    fn independently_generated_chunks_keep_boundary_columns_continuous() {
        let seed = 0xABCD_1234;
        let center_coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let east_coord = ChunkCoord { x: 1, y: 0, z: 0 };
        let south_coord = ChunkCoord { x: 0, y: 0, z: 1 };

        let center = generate_chunk_direct(seed, center_coord);
        let east = generate_chunk_direct(seed, east_coord);
        let south = generate_chunk_direct(seed, south_coord);

        let mut ew_mismatches = 0usize;
        let mut ns_mismatches = 0usize;
        let total = CHUNK_SIZE * CHUNK_SIZE;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                let a = center.get(CHUNK_SIZE - 1, y, z);
                let b = east.get(0, y, z);
                if a != b {
                    ew_mismatches += 1;
                }
            }
        }

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                let a = center.get(x, y, CHUNK_SIZE - 1);
                let b = south.get(x, y, 0);
                if a != b {
                    ns_mismatches += 1;
                }
            }
        }

        let ew_match_ratio = 1.0 - ew_mismatches as f32 / total as f32;
        let ns_match_ratio = 1.0 - ns_mismatches as f32 / total as f32;
        assert!(
            ew_match_ratio > 0.985,
            "east/west seam continuity degraded: {:.2}% match ({}/{})",
            ew_match_ratio * 100.0,
            total - ew_mismatches,
            total
        );
        assert!(
            ns_match_ratio > 0.985,
            "north/south seam continuity degraded: {:.2}% match ({}/{})",
            ns_match_ratio * 100.0,
            total - ns_mismatches,
            total
        );
    }

    fn largest_land_component_ratio(seed: u64, min_x: i32, min_z: i32, size: i32) -> f32 {
        let mut land = vec![false; (size * size) as usize];
        let mut land_count = 0usize;
        for dz in 0..size {
            for dx in 0..size {
                let x = min_x + dx;
                let z = min_z + dz;
                let w = biome_weights(seed, x, z, sample_climate(seed, x, z));
                let is_land = w[biome_index(BiomeType::Ocean)] < 0.52;
                let idx = (dx + dz * size) as usize;
                land[idx] = is_land;
                land_count += usize::from(is_land);
            }
        }
        if land_count == 0 {
            return 0.0;
        }

        let mut visited = vec![false; land.len()];
        let mut largest = 0usize;
        for start in 0..land.len() {
            if visited[start] || !land[start] {
                continue;
            }
            let mut q = VecDeque::new();
            q.push_back(start);
            visited[start] = true;
            let mut count = 0usize;
            while let Some(idx) = q.pop_front() {
                count += 1;
                let x = (idx as i32) % size;
                let z = (idx as i32) / size;
                for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                    if nx < 0 || nz < 0 || nx >= size || nz >= size {
                        continue;
                    }
                    let ni = (nx + nz * size) as usize;
                    if !visited[ni] && land[ni] {
                        visited[ni] = true;
                        q.push_back(ni);
                    }
                }
            }
            largest = largest.max(count);
        }
        largest as f32 / land_count as f32
    }

    #[test]
    fn inland_macro_regions_form_contiguous_non_ocean_clusters() {
        let seed = 0x4242_9911;
        let windows = [(-96, -96), (0, 0), (96, -64), (160, 64)];
        let mut ratios = Vec::new();
        for (x, z) in windows {
            ratios.push(largest_land_component_ratio(seed, x, z, 48));
        }
        let avg_ratio = ratios.iter().copied().sum::<f32>() / ratios.len() as f32;
        assert!(
            avg_ratio > 0.68,
            "expected large contiguous inland regions, got average largest component ratio {avg_ratio:.3} from {ratios:?}"
        );
    }

    #[test]
    fn sampled_surface_height_preserves_broad_height_range() {
        let config = ProcGenConfig::for_size(64, 0xA17E_600D).with_origin([0, 0, 0]);
        let mut sampled_min = i32::MAX;
        let mut sampled_max = i32::MIN;
        let mut raw_min = i32::MAX;
        let mut raw_max = i32::MIN;

        for z in (0..256).step_by(2) {
            for x in (0..256).step_by(2) {
                let wx = config.world_origin[0] + x - 96;
                let wz = config.world_origin[2] + z - 96;
                let h = sampled_surface_height(&config, wx, wz, None);
                sampled_min = sampled_min.min(h);
                sampled_max = sampled_max.max(h);

                let w = biome_weights(config.seed, wx, wz, sample_climate(config.seed, wx, wz));
                let raw = terrain_height(&config, wx, wz, w) - config.world_origin[1];
                raw_min = raw_min.min(raw);
                raw_max = raw_max.max(raw);
            }
        }

        let sampled_span = sampled_max - sampled_min;
        let raw_span = raw_max - raw_min;
        assert!(
            sampled_span >= 9,
            "expected minimum terrain relief after smoothing, got sampled span {sampled_span} ({sampled_min}..{sampled_max})"
        );
        assert!(
            sampled_span as f32 >= raw_span as f32 * 0.45,
            "smoothing removed too much relief: sampled span {sampled_span}, raw span {raw_span}"
        );
    }

    #[test]
    fn adjacent_chunks_keep_surface_height_continuous() {
        let size = 64i32;
        let seed = 0x6E77_A55E;
        let center_cfg = ProcGenConfig::for_size(size as usize, seed).with_origin([0, 0, 0]);
        let east_cfg = ProcGenConfig::for_size(size as usize, seed).with_origin([size, 0, 0]);
        let south_cfg = ProcGenConfig::for_size(size as usize, seed).with_origin([0, 0, size]);
        let timings = ProcGenPassTimings::default();
        let (center_h, _) = build_column_cache(&center_cfg, &timings);
        let (east_h, _) = build_column_cache(&east_cfg, &timings);
        let (south_h, _) = build_column_cache(&south_cfg, &timings);

        for z in 0..size {
            let center_idx = (size - 1 + z * size) as usize;
            let east_idx = (z * size) as usize;
            let a = center_h[center_idx];
            let b = east_h[east_idx];
            assert!(
                (a - b).abs() <= 1,
                "east/west sampled surface discontinuity at z={z}: {a} vs {b}"
            );
        }

        for x in 0..size {
            let center_idx = (x + (size - 1) * size) as usize;
            let south_idx = x as usize;
            let a = center_h[center_idx];
            let b = south_h[south_idx];
            assert!(
                (a - b).abs() <= 1,
                "north/south sampled surface discontinuity at x={x}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn surface_layers_stay_monotonic_in_vertical_columns() {
        let config = ProcGenConfig::for_size(64, 0x3300_4411).with_origin([0, 0, 0]);
        let world = generate_world(config);

        for z in 0..world.dims[2] as i32 {
            for x in 0..world.dims[0] as i32 {
                let mut top = None;
                for y in (2..world.dims[1] as i32).rev() {
                    if is_ground(world.get(x, y, z)) {
                        top = Some(y);
                        break;
                    }
                }
                let Some(top_y) = top else {
                    continue;
                };

                let top_mat = world.get(x, top_y, z);
                assert!(
                    is_surface_material(top_mat),
                    "unexpected top material at ({x},{top_y},{z}): {top_mat}"
                );

                let mut seen_subsoil = false;
                let mut seen_parent = false;
                for d in 1..=12 {
                    let y = top_y - d;
                    if y <= 1 {
                        break;
                    }
                    let mat = world.get(x, y, z);
                    if mat == EMPTY || mat == WATER {
                        continue;
                    }
                    if mat == STONE {
                        seen_parent = true;
                        continue;
                    }
                    if is_subsoil(mat) {
                        assert!(
                            !seen_parent,
                            "subsoil appeared below parent rock at ({x},{y},{z}); mat={mat}"
                        );
                        seen_subsoil = true;
                    }
                    if mat == SAND && seen_parent {
                        panic!("sand appeared below stone at ({x},{y},{z})");
                    }
                }

                assert!(
                    seen_subsoil || top_mat == STONE,
                    "expected subsoil under top cover at ({x},{top_y},{z}); top={top_mat}"
                );
            }
        }
    }

    #[test]
    fn avoids_short_run_alternating_surface_stripes() {
        let config = ProcGenConfig::for_size(64, 0x9090_4422).with_origin([0, 0, 0]);
        let world = generate_world(config);
        let heights = build_surface_heightmap_from_world(&world);

        let mut alternating_pairs = 0usize;
        let mut sampled_pairs = 0usize;

        for z in 0..world.dims[2] as i32 {
            for x in 0..world.dims[0] as i32 - 3 {
                let mut seq = [EMPTY; 4];
                for i in 0..4 {
                    let sx = x + i as i32;
                    let h = heights[sx as usize + z as usize * world.dims[0]];
                    seq[i] = world.get(sx, h, z);
                }
                if seq.iter().all(|m| is_surface_material(*m)) {
                    sampled_pairs += 1;
                    if seq[0] == seq[2] && seq[1] == seq[3] && seq[0] != seq[1] {
                        alternating_pairs += 1;
                    }
                }
            }
        }

        for x in 0..world.dims[0] as i32 {
            for z in 0..world.dims[2] as i32 - 3 {
                let mut seq = [EMPTY; 4];
                for i in 0..4 {
                    let sz = z + i as i32;
                    let h = heights[x as usize + sz as usize * world.dims[0]];
                    seq[i] = world.get(x, h, sz);
                }
                if seq.iter().all(|m| is_surface_material(*m)) {
                    sampled_pairs += 1;
                    if seq[0] == seq[2] && seq[1] == seq[3] && seq[0] != seq[1] {
                        alternating_pairs += 1;
                    }
                }
            }
        }

        let ratio = alternating_pairs as f32 / sampled_pairs.max(1) as f32;
        assert!(
            ratio < 0.08,
            "alternating stripe ratio too high: {:.2}% ({}/{})",
            ratio * 100.0,
            alternating_pairs,
            sampled_pairs
        );
    }
    #[test]
    fn trees_continue_across_lateral_chunk_boundaries() {
        let seed = 0xA51CEu64;
        let west = generate_chunk_direct(seed, ChunkCoord { x: 0, y: 0, z: 0 });
        let east = generate_chunk_direct(seed, ChunkCoord { x: 1, y: 0, z: 0 });
        let side = CHUNK_SIZE;

        let mut found = false;
        for z in 0..side {
            for y in 0..side {
                let west_edge = west.get(side - 1, y, z);
                let east_edge = east.get(0, y, z);
                if matches!(west_edge, WOOD | LEAVES) && matches!(east_edge, WOOD | LEAVES) {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        assert!(
            found,
            "expected at least one cross-border tree voxel continuity sample"
        );
    }

    #[test]
    fn trees_continue_into_chunk_above() {
        let seed = 0xF00DBA5Eu64;
        let base = generate_chunk_direct(seed, ChunkCoord { x: 0, y: 0, z: 0 });
        let above = generate_chunk_direct(seed, ChunkCoord { x: 0, y: 1, z: 0 });
        let side = CHUNK_SIZE;

        let mut found = false;
        for z in 0..side {
            for x in 0..side {
                let lower_top = base.get(x, side - 1, z);
                let upper_bottom = above.get(x, 0, z);
                if matches!(lower_top, WOOD | LEAVES) && matches!(upper_bottom, WOOD | LEAVES) {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        assert!(
            found,
            "expected at least one tree voxel continuity sample across vertical chunk border"
        );
    }
}
