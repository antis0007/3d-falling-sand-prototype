use crate::world::World;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshDetail {
    Full,
    Low,
}

#[derive(Clone)]
pub struct StreamChunk {
    pub world: World,
    pub voxel_version: u64,
    pub meshed_version: u64,
    pub last_render_frame: u64,
    pub last_sim_frame: u64,
}

pub struct StreamingManager {
    chunks: BTreeMap<[i32; 3], StreamChunk>,
    sim_active: BTreeSet<[i32; 3]>,
    render_active: BTreeSet<[i32; 3]>,
    frame: u64,
    pub macro_size: i32,
    pub sim_radius: i32,
    pub render_radius: i32,
    pub render_budget: usize,
    pub extreme_lod_distance: i32,
}

impl StreamingManager {
    pub fn new(macro_size: i32, render_radius: i32, render_budget: usize) -> Self {
        Self {
            chunks: BTreeMap::new(),
            sim_active: BTreeSet::new(),
            render_active: BTreeSet::new(),
            frame: 0,
            macro_size,
            sim_radius: 1,
            render_radius,
            render_budget,
            extreme_lod_distance: render_radius.saturating_sub(1).max(2),
        }
    }

    pub fn insert_snapshot(&mut self, origin: [i32; 3], world: World) {
        let current = self
            .chunks
            .get(&origin)
            .map(|c| c.voxel_version)
            .unwrap_or(0);
        self.chunks.insert(
            origin,
            StreamChunk {
                world,
                voxel_version: current.saturating_add(1),
                meshed_version: 0,
                last_render_frame: self.frame,
                last_sim_frame: self.frame,
            },
        );
    }

    pub fn get_snapshot(&self, origin: [i32; 3]) -> Option<&World> {
        self.chunks.get(&origin).map(|c| &c.world)
    }

    pub fn mark_voxel_change(&mut self, origin: [i32; 3]) {
        if let Some(chunk) = self.chunks.get_mut(&origin) {
            chunk.voxel_version = chunk.voxel_version.saturating_add(1);
            chunk.last_sim_frame = self.frame;
        }
    }

    pub fn mark_mesh_built(&mut self, origin: [i32; 3]) {
        if let Some(chunk) = self.chunks.get_mut(&origin) {
            chunk.meshed_version = chunk.voxel_version;
            chunk.last_render_frame = self.frame;
        }
    }

    pub fn needs_mesh_rebuild(&self, origin: [i32; 3]) -> bool {
        self.chunks
            .get(&origin)
            .map(|c| c.meshed_version != c.voxel_version)
            .unwrap_or(false)
    }

    pub fn update_residency(&mut self, player_macro: [i32; 3]) {
        self.frame = self.frame.wrapping_add(1);

        let mut next_sim = BTreeSet::new();
        for dy in -self.sim_radius..=self.sim_radius {
            for dz in -self.sim_radius..=self.sim_radius {
                for dx in -self.sim_radius..=self.sim_radius {
                    next_sim.insert([
                        player_macro[0] + dx,
                        player_macro[1] + dy,
                        player_macro[2] + dz,
                    ]);
                }
            }
        }

        let mut next_render = BTreeSet::new();
        for dy in -self.render_radius..=self.render_radius {
            for dz in -self.render_radius..=self.render_radius {
                for dx in -self.render_radius..=self.render_radius {
                    next_render.insert([
                        player_macro[0] + dx,
                        player_macro[1] + dy,
                        player_macro[2] + dz,
                    ]);
                }
            }
        }

        self.sim_active = next_sim;
        self.render_active = next_render;
        self.evict_far_render_only(player_macro);
    }

    pub fn next_missing_render_origin(&self) -> Option<[i32; 3]> {
        self.render_active
            .iter()
            .find(|coord| !self.chunks.contains_key(&self.to_origin(**coord)))
            .copied()
            .map(|coord| self.to_origin(coord))
    }

    pub fn set_render_radius_from_budget(&mut self, budget_chunks: usize) {
        self.render_budget = budget_chunks.max(27);
        let mut radius: i32 = 1;
        while ((radius * 2 + 1) * (radius * 2 + 1) * (radius * 2 + 1)) as usize
            <= self.render_budget
        {
            radius += 1;
        }
        self.render_radius = radius.saturating_sub(1).max(self.sim_radius);
    }

    pub fn mesh_detail_for_origin(&self, origin: [i32; 3], player_macro: [i32; 3]) -> MeshDetail {
        let coord = [
            floor_div(origin[0], self.macro_size),
            floor_div(origin[1], self.macro_size),
            floor_div(origin[2], self.macro_size),
        ];
        let distance = (coord[0] - player_macro[0])
            .abs()
            .max((coord[1] - player_macro[1]).abs())
            .max((coord[2] - player_macro[2]).abs());
        if distance >= self.extreme_lod_distance {
            MeshDetail::Low
        } else {
            MeshDetail::Full
        }
    }

    fn evict_far_render_only(&mut self, player_macro: [i32; 3]) {
        let mut candidates: Vec<([i32; 3], i32, u64)> = self
            .chunks
            .iter()
            .filter_map(|(origin, chunk)| {
                let coord = [
                    floor_div(origin[0], self.macro_size),
                    floor_div(origin[1], self.macro_size),
                    floor_div(origin[2], self.macro_size),
                ];
                if self.sim_active.contains(&coord) {
                    return None;
                }
                let dist = (coord[0] - player_macro[0])
                    .abs()
                    .max((coord[1] - player_macro[1]).abs())
                    .max((coord[2] - player_macro[2]).abs());
                Some((*origin, dist, chunk.last_render_frame))
            })
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

        let mut idx = 0;
        while self.chunks.len() > self.render_budget && idx < candidates.len() {
            let (origin, _, _) = candidates[idx];
            self.chunks.remove(&origin);
            idx += 1;
        }
    }

    fn to_origin(&self, macro_coord: [i32; 3]) -> [i32; 3] {
        [
            macro_coord[0] * self.macro_size,
            macro_coord[1] * self.macro_size,
            macro_coord[2] * self.macro_size,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_residency_radius_one_tracks_27_sim_and_render_coords() {
        let mut manager = StreamingManager::new(32, 1, 64);
        manager.sim_radius = 1;

        manager.update_residency([0, 0, 0]);

        assert_eq!(manager.sim_active.len(), 27);
        assert_eq!(manager.render_active.len(), 27);
        assert!(manager.sim_active.contains(&[0, 1, 0]));
        assert!(manager.render_active.contains(&[0, -1, 0]));
    }

    #[test]
    fn render_budget_radius_uses_cubic_volume() {
        let mut manager = StreamingManager::new(32, 1, 8);
        manager.set_render_radius_from_budget(26);
        assert_eq!(manager.render_budget, 27);
        assert_eq!(manager.render_radius, 1);

        manager.set_render_radius_from_budget(125);
        assert_eq!(manager.render_radius, 2);
    }

    #[test]
    fn mesh_detail_is_y_aware() {
        let manager = StreamingManager::new(32, 4, 512);
        let player = [0, 0, 0];

        let low_detail_origin = [0, manager.extreme_lod_distance * manager.macro_size, 0];
        assert_eq!(
            manager.mesh_detail_for_origin(low_detail_origin, player),
            MeshDetail::Low
        );
    }
}
