use crate::procgen::ProcGenConfig;
use crate::world::{MaterialId, World};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkResidency {
    Unloaded,
    Generating,
    Resident,
}

#[derive(Clone)]
struct StreamChunk {
    residency: ChunkResidency,
    world: Option<World>,
    has_persistent_edits: bool,
}

impl StreamChunk {
    fn unloaded() -> Self {
        Self {
            residency: ChunkResidency::Unloaded,
            world: None,
            has_persistent_edits: false,
        }
    }
}

pub struct WorldStream {
    base_config: ProcGenConfig,
    chunks: BTreeMap<[i32; 3], StreamChunk>,
}

impl WorldStream {
    pub fn new(base_config: ProcGenConfig) -> Self {
        Self {
            base_config,
            chunks: BTreeMap::new(),
        }
    }

    pub fn chunk_origin(&self, coord: [i32; 3]) -> [i32; 3] {
        [
            coord[0] * self.base_config.dims[0] as i32,
            coord[1] * self.base_config.dims[1] as i32,
            coord[2] * self.base_config.dims[2] as i32,
        ]
    }

    pub fn make_config(&self, coord: [i32; 3]) -> ProcGenConfig {
        self.base_config.with_origin(self.chunk_origin(coord))
    }

    pub fn state(&self, coord: [i32; 3]) -> ChunkResidency {
        self.chunks
            .get(&coord)
            .map(|chunk| chunk.residency)
            .unwrap_or(ChunkResidency::Unloaded)
    }

    pub fn mark_generating(&mut self, coord: [i32; 3]) {
        let chunk = self
            .chunks
            .entry(coord)
            .or_insert_with(StreamChunk::unloaded);
        if chunk.residency == ChunkResidency::Unloaded {
            chunk.residency = ChunkResidency::Generating;
        }
    }

    pub fn apply_generated(&mut self, coord: [i32; 3], world: World) {
        let chunk = self
            .chunks
            .entry(coord)
            .or_insert_with(StreamChunk::unloaded);
        if chunk.residency == ChunkResidency::Resident && chunk.has_persistent_edits {
            return;
        }
        chunk.residency = ChunkResidency::Resident;
        chunk.world = Some(world.clone());
        self.validate_seams(coord, &world);
    }

    pub fn cancel_generation(&mut self, coord: [i32; 3]) {
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            if chunk.residency == ChunkResidency::Generating {
                chunk.residency = ChunkResidency::Unloaded;
            }
            if chunk.residency == ChunkResidency::Unloaded
                && chunk.world.is_none()
                && !chunk.has_persistent_edits
            {
                self.chunks.remove(&coord);
            }
        }
    }

    pub fn persist_coord_state(&mut self, coord: [i32; 3], world: &World) {
        let chunk = self
            .chunks
            .entry(coord)
            .or_insert_with(StreamChunk::unloaded);
        chunk.residency = ChunkResidency::Resident;
        chunk.world = Some(world.clone());
        chunk.has_persistent_edits = true;
    }

    pub fn load_coord_into_world(&self, coord: [i32; 3], world: &mut World) -> bool {
        let Some(chunk) = self.chunks.get(&coord) else {
            return false;
        };
        if chunk.residency != ChunkResidency::Resident {
            return false;
        }
        let Some(data) = &chunk.world else {
            return false;
        };
        world.chunks.clone_from(&data.chunks);
        for chunk in &mut world.chunks {
            chunk.dirty_mesh = true;
            if chunk.meshed_version == chunk.voxel_version {
                chunk.meshed_version = chunk.meshed_version.saturating_sub(1);
            }
        }
        true
    }

    pub fn resident_world(&self, coord: [i32; 3]) -> Option<&World> {
        let chunk = self.chunks.get(&coord)?;
        if chunk.residency != ChunkResidency::Resident {
            return None;
        }
        chunk.world.as_ref()
    }

    pub fn deterministic_face_signature(world: &World, axis: usize, side: i32) -> Vec<MaterialId> {
        let mut data = Vec::new();
        let max_x = world.dims[0] as i32 - 1;
        let max_y = world.dims[1] as i32 - 1;
        let max_z = world.dims[2] as i32 - 1;
        match axis {
            0 => {
                let x = if side < 0 { 0 } else { max_x };
                for z in 0..=max_z {
                    for y in 0..=max_y {
                        data.push(world.get(x, y, z));
                    }
                }
            }
            1 => {
                let y = if side < 0 { 0 } else { max_y };
                for z in 0..=max_z {
                    for x in 0..=max_x {
                        data.push(world.get(x, y, z));
                    }
                }
            }
            _ => {
                let z = if side < 0 { 0 } else { max_z };
                for y in 0..=max_y {
                    for x in 0..=max_x {
                        data.push(world.get(x, y, z));
                    }
                }
            }
        }
        data
    }

    fn validate_seams(&self, coord: [i32; 3], world: &World) {
        const NEIGHBORS: [([i32; 3], usize, i32); 6] = [
            ([1, 0, 0], 0, 1),
            ([-1, 0, 0], 0, -1),
            ([0, 1, 0], 1, 1),
            ([0, -1, 0], 1, -1),
            ([0, 0, 1], 2, 1),
            ([0, 0, -1], 2, -1),
        ];
        for (delta, axis, side) in NEIGHBORS {
            let neighbor_coord = [
                coord[0] + delta[0],
                coord[1] + delta[1],
                coord[2] + delta[2],
            ];
            let Some(neighbor) = self.chunks.get(&neighbor_coord) else {
                continue;
            };
            let Some(neighbor_world) = &neighbor.world else {
                continue;
            };
            if neighbor.has_persistent_edits {
                continue;
            }
            if Self::deterministic_face_signature(neighbor_world, axis, -side)
                != Self::deterministic_face_signature(world, axis, side)
            {
                log::warn!(
                    "Seam validation detected neighbor edge mismatch at {:?} vs {:?}",
                    coord,
                    neighbor_coord
                );
            }
        }
    }
}

pub fn floor_div(a: i32, b: i32) -> i32 {
    let mut q = a / b;
    let r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}
