use crate::procgen::{generate_world, ProcGenConfig};
use crate::world::World;
use std::collections::BTreeMap;

pub const MACROCHUNK_SIZE: i32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MacroChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl MacroChunkCoord {
    pub fn from_world_voxel(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: floor_div(x, MACROCHUNK_SIZE),
            y: floor_div(y, MACROCHUNK_SIZE),
            z: floor_div(z, MACROCHUNK_SIZE),
        }
    }

    pub fn world_origin(self) -> [i32; 3] {
        [
            self.x * MACROCHUNK_SIZE,
            self.y * MACROCHUNK_SIZE,
            self.z * MACROCHUNK_SIZE,
        ]
    }
}

#[derive(Clone)]
pub struct MacroChunk {
    pub coord: MacroChunkCoord,
    pub world: World,
}

impl MacroChunk {
    pub fn generate(coord: MacroChunkCoord, seed: u64) -> Self {
        let config = ProcGenConfig::for_size(MACROCHUNK_SIZE as usize, seed)
            .with_origin(coord.world_origin());
        Self {
            coord,
            world: generate_world(config),
        }
    }
}

pub struct WorldManager {
    loaded: BTreeMap<MacroChunkCoord, MacroChunk>,
    pub seed: u64,
}

impl WorldManager {
    pub fn new(seed: u64) -> Self {
        Self {
            loaded: BTreeMap::new(),
            seed,
        }
    }

    pub fn ensure_player_neighborhood(&mut self, player_world_pos: [i32; 3]) {
        let center = MacroChunkCoord::from_world_voxel(
            player_world_pos[0],
            player_world_pos[1],
            player_world_pos[2],
        );

        for z in (center.z - 1)..=(center.z + 1) {
            for y in (center.y - 1)..=(center.y + 1) {
                for x in (center.x - 1)..=(center.x + 1) {
                    let coord = MacroChunkCoord { x, y, z };
                    self.loaded
                        .entry(coord)
                        .or_insert_with(|| MacroChunk::generate(coord, self.seed));
                }
            }
        }

        self.loaded.retain(|coord, _| {
            (coord.x - center.x).abs() <= 1
                && (coord.y - center.y).abs() <= 1
                && (coord.z - center.z).abs() <= 1
        });
    }

    pub fn iter_loaded(&self) -> impl Iterator<Item = (&MacroChunkCoord, &MacroChunk)> {
        self.loaded.iter()
    }

    pub fn loaded_len(&self) -> usize {
        self.loaded.len()
    }

    pub fn global_to_local(
        coord: MacroChunkCoord,
        world_x: i32,
        world_y: i32,
        world_z: i32,
    ) -> [i32; 3] {
        [
            world_x - coord.x * MACROCHUNK_SIZE,
            world_y - coord.y * MACROCHUNK_SIZE,
            world_z - coord.z * MACROCHUNK_SIZE,
        ]
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
