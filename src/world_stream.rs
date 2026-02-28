use crate::procgen::ProcGenConfig;
use crate::world::{MaterialId, World, EMPTY};
use crate::world_bounds::ProceduralWorldBounds;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyChangeEvent {
    pub coord: [i32; 3],
    pub old: ChunkResidency,
    pub new: ChunkResidency,
}

pub struct WorldStream {
    base_config: ProcGenConfig,
    bounds: ProceduralWorldBounds,
    chunks: BTreeMap<[i32; 3], StreamChunk>,
    residency_changes: Vec<ResidencyChangeEvent>,
}

impl WorldStream {
    pub fn new(base_config: ProcGenConfig, bounds: ProceduralWorldBounds) -> Self {
        Self {
            base_config,
            bounds,
            chunks: BTreeMap::new(),
            residency_changes: Vec::new(),
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

    pub fn take_residency_changes(&mut self) -> Vec<ResidencyChangeEvent> {
        std::mem::take(&mut self.residency_changes)
    }

    pub fn neighbor_coords(coord: [i32; 3]) -> [[i32; 3]; 6] {
        [
            [coord[0] + 1, coord[1], coord[2]],
            [coord[0] - 1, coord[1], coord[2]],
            [coord[0], coord[1] + 1, coord[2]],
            [coord[0], coord[1] - 1, coord[2]],
            [coord[0], coord[1], coord[2] + 1],
            [coord[0], coord[1], coord[2] - 1],
        ]
    }

    fn update_residency(&mut self, coord: [i32; 3], new_residency: ChunkResidency) {
        let chunk = self
            .chunks
            .entry(coord)
            .or_insert_with(StreamChunk::unloaded);
        let old = chunk.residency;
        if old != new_residency {
            chunk.residency = new_residency;
            self.residency_changes.push(ResidencyChangeEvent {
                coord,
                old,
                new: new_residency,
            });
        }
    }

    pub fn mark_generating(&mut self, coord: [i32; 3]) {
        if self.state(coord) == ChunkResidency::Unloaded {
            self.update_residency(coord, ChunkResidency::Generating);
        }
    }

    pub fn apply_generated(&mut self, coord: [i32; 3], world: World) {
        if self.state(coord) == ChunkResidency::Resident
            && self
                .chunks
                .get(&coord)
                .map(|chunk| chunk.has_persistent_edits)
                .unwrap_or(false)
        {
            return;
        }
        self.update_residency(coord, ChunkResidency::Resident);
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            chunk.world = Some(world.clone());
        }
        self.validate_seams(coord, &world);
    }

    pub fn cancel_generation(&mut self, coord: [i32; 3]) {
        if self.state(coord) == ChunkResidency::Generating {
            self.update_residency(coord, ChunkResidency::Unloaded);
        }
        if let Some(chunk) = self.chunks.get(&coord) {
            if chunk.residency == ChunkResidency::Unloaded
                && chunk.world.is_none()
                && !chunk.has_persistent_edits
            {
                self.chunks.remove(&coord);
            }
        }
    }

    pub fn persist_coord_state(&mut self, coord: [i32; 3], world: &World) {
        self.update_residency(coord, ChunkResidency::Resident);
        let chunk = self
            .chunks
            .entry(coord)
            .or_insert_with(StreamChunk::unloaded);
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

    pub fn resident_world_mut(&mut self, coord: [i32; 3]) -> Option<&mut World> {
        let chunk = self.chunks.get_mut(&coord)?;
        if chunk.residency != ChunkResidency::Resident {
            return None;
        }
        chunk.world.as_mut()
    }

    pub fn resident_coords(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.chunks.iter().filter_map(|(coord, chunk)| {
            (chunk.residency == ChunkResidency::Resident && chunk.world.is_some()).then_some(*coord)
        })
    }

    pub fn resident_coords_sorted_by_distance(&self, center: [i32; 3]) -> Vec<[i32; 3]> {
        let mut coords: Vec<[i32; 3]> = self.resident_coords().collect();
        coords.sort_by_key(|coord| {
            let dx = coord[0] - center[0];
            let dy = coord[1] - center[1];
            let dz = coord[2] - center[2];
            dx * dx + dy * dy + dz * dz
        });
        coords
    }

    pub fn sample_global_voxel(&self, global: [i32; 3]) -> MaterialId {
        self.sample_global_voxel_known(global).unwrap_or(EMPTY)
    }

    pub fn sample_global_voxel_known(&self, global: [i32; 3]) -> Option<MaterialId> {
        let dims = self.base_config.dims;
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Some(EMPTY);
        }
        if !self.bounds.contains_global_y(global[1]) {
            return Some(EMPTY);
        }
        let dx = dims[0] as i32;
        let dy = dims[1] as i32;
        let dz = dims[2] as i32;

        let coord = [
            floor_div(global[0], dx),
            floor_div(global[1], dy),
            floor_div(global[2], dz),
        ];
        let local = [
            global[0] - coord[0] * dx,
            global[1] - coord[1] * dy,
            global[2] - coord[2] * dz,
        ];
        let world = self.resident_world(coord)?;
        Some(world.get(local[0], local[1], local[2]))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::ProcGenConfig;

    fn test_stream() -> WorldStream {
        let cfg = ProcGenConfig::for_size(64, 1234);
        let bounds = ProceduralWorldBounds::new(-1, 1, cfg.dims[1] as i32);
        WorldStream::new(cfg, bounds)
    }

    #[test]
    fn residency_change_events_include_neighbor_load_transitions() {
        let mut stream = test_stream();
        let coord = [1, 0, 0];
        stream.mark_generating(coord);
        let world = World::new([64, 64, 64]);
        stream.apply_generated(coord, world);

        let events = stream.take_residency_changes();
        assert!(events
            .iter()
            .any(|e| e.coord == coord && e.new == ChunkResidency::Generating));
        assert!(events
            .iter()
            .any(|e| e.coord == coord && e.new == ChunkResidency::Resident));
    }
}
