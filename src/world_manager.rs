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
}

#[derive(Clone)]
pub struct MacroChunk {
    pub coord: MacroChunkCoord,
    pub world: World,
}

impl MacroChunk {
    pub fn new(coord: MacroChunkCoord) -> Self {
        Self {
            coord,
            world: World::new([
                MACROCHUNK_SIZE as usize,
                MACROCHUNK_SIZE as usize,
                MACROCHUNK_SIZE as usize,
            ]),
        }
    }
}

pub struct WorldManager {
    loaded: BTreeMap<MacroChunkCoord, MacroChunk>,
}

impl WorldManager {
    pub fn with_origin_radius(radius: i32) -> Self {
        let mut loaded = BTreeMap::new();
        for z in -radius..=radius {
            for y in -radius..=radius {
                for x in -radius..=radius {
                    let coord = MacroChunkCoord { x, y, z };
                    loaded.insert(coord, MacroChunk::new(coord));
                }
            }
        }
        Self { loaded }
    }

    pub fn get_macrochunk_mut(&mut self, coord: MacroChunkCoord) -> Option<&mut MacroChunk> {
        self.loaded.get_mut(&coord)
    }

    pub fn iter_loaded(&self) -> impl Iterator<Item = (&MacroChunkCoord, &MacroChunk)> {
        self.loaded.iter()
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
