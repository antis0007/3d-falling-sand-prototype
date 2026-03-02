use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim_world::{step_region_profiled, Rng};
use crate::types::{ChunkCoord, VoxelCoord};
use crate::world::EMPTY;

const BRICK_EDGE: i32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationMode {
    CandidateSwapCaFallback,
    GpuFluidPipeline,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialVoxel {
    pub material_id: u16,
    pub density: f32,
}

#[derive(Default)]
pub struct SparseVoxelPages {
    pages: HashMap<[i32; 3], Vec<MaterialVoxel>>,
}

impl SparseVoxelPages {
    pub fn set(&mut self, coord: VoxelCoord, material_id: u16, density: f32) {
        let page_coord = [
            coord.x.div_euclid(BRICK_EDGE),
            coord.y.div_euclid(BRICK_EDGE),
            coord.z.div_euclid(BRICK_EDGE),
        ];
        let local = [
            coord.x.rem_euclid(BRICK_EDGE) as usize,
            coord.y.rem_euclid(BRICK_EDGE) as usize,
            coord.z.rem_euclid(BRICK_EDGE) as usize,
        ];
        let idx = local[0]
            + local[1] * BRICK_EDGE as usize
            + local[2] * BRICK_EDGE as usize * BRICK_EDGE as usize;
        let page = self.pages.entry(page_coord).or_insert_with(|| {
            vec![
                MaterialVoxel {
                    material_id: EMPTY,
                    density: 0.0,
                };
                (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize
            ]
        });
        page[idx] = MaterialVoxel {
            material_id,
            density,
        };
    }

    pub fn get(&self, coord: VoxelCoord) -> Option<MaterialVoxel> {
        let page_coord = [
            coord.x.div_euclid(BRICK_EDGE),
            coord.y.div_euclid(BRICK_EDGE),
            coord.z.div_euclid(BRICK_EDGE),
        ];
        let page = self.pages.get(&page_coord)?;
        let local = [
            coord.x.rem_euclid(BRICK_EDGE) as usize,
            coord.y.rem_euclid(BRICK_EDGE) as usize,
            coord.z.rem_euclid(BRICK_EDGE) as usize,
        ];
        let idx = local[0]
            + local[1] * BRICK_EDGE as usize
            + local[2] * BRICK_EDGE as usize * BRICK_EDGE as usize;
        Some(page[idx])
    }
}

#[derive(Default)]
pub struct MacVelocityGrid {
    u: HashMap<VoxelCoord, f32>,
    v: HashMap<VoxelCoord, f32>,
    w: HashMap<VoxelCoord, f32>,
}

impl MacVelocityGrid {
    fn add_impulse(&mut self, coord: VoxelCoord, velocity: [f32; 3], strength: f32) {
        *self.u.entry(coord).or_insert(0.0) += velocity[0] * strength;
        *self.v.entry(coord).or_insert(0.0) += velocity[1] * strength;
        *self.w.entry(coord).or_insert(0.0) += velocity[2] * strength;
    }

    fn velocity_at(&self, coord: VoxelCoord) -> [f32; 3] {
        [
            self.u.get(&coord).copied().unwrap_or(0.0),
            self.v.get(&coord).copied().unwrap_or(0.0),
            self.w.get(&coord).copied().unwrap_or(0.0),
        ]
    }

    fn damp_all(&mut self, factor: f32) {
        self.u.values_mut().for_each(|v| *v *= factor);
        self.v.values_mut().for_each(|v| *v *= factor);
        self.w.values_mut().for_each(|v| *v *= factor);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SimCommand {
    PlaceOrErase {
        coord: VoxelCoord,
        material_id: u16,
        density: f32,
    },
    LaunchVoxel {
        coord: VoxelCoord,
        material_id: u16,
        velocity: [f32; 3],
        density: f32,
    },
    Explosion {
        center: VoxelCoord,
        impulse: [f32; 3],
        radius: i32,
    },
}

#[derive(Default)]
pub struct PhysicsGpuSimulator {
    occupancy: SparseVoxelPages,
    mac_grid: MacVelocityGrid,
    per_voxel_velocity: HashMap<VoxelCoord, [f32; 3]>,
    command_buffer: Vec<SimCommand>,
    touched_voxels: HashSet<VoxelCoord>,
}

impl PhysicsGpuSimulator {
    pub fn queue_command(&mut self, command: SimCommand) {
        self.command_buffer.push(command);
    }

    pub fn queue_place_edit(&mut self, coord: VoxelCoord, material_id: u16) {
        self.queue_command(SimCommand::PlaceOrErase {
            coord,
            material_id,
            density: if material_id == EMPTY { 0.0 } else { 1.0 },
        });
    }

    pub fn step(
        &mut self,
        mode: SimulationMode,
        store: &mut ChunkStore,
        region: &HashSet<ChunkCoord>,
        center: ChunkCoord,
        rng: &mut Rng,
    ) -> usize {
        if matches!(mode, SimulationMode::CandidateSwapCaFallback) {
            self.command_buffer.clear();
            self.touched_voxels.clear();
            return step_region_profiled(store, region, center, rng);
        }

        self.transfer_edit_events_to_command_buffer();
        self.inject_momentum();
        self.advect_material();
        self.project_pressure();
        self.resolve_collisions();
        self.write_back_occupancy(store);
        region.len()
    }

    fn transfer_edit_events_to_command_buffer(&mut self) {
        for cmd in self.command_buffer.drain(..) {
            match cmd {
                SimCommand::PlaceOrErase {
                    coord,
                    material_id,
                    density,
                } => {
                    self.occupancy.set(coord, material_id, density);
                    self.touched_voxels.insert(coord);
                    if material_id == EMPTY {
                        self.per_voxel_velocity.remove(&coord);
                    }
                }
                SimCommand::LaunchVoxel {
                    coord,
                    material_id,
                    velocity,
                    density,
                } => {
                    self.occupancy.set(coord, material_id, density);
                    self.per_voxel_velocity.insert(coord, velocity);
                    self.touched_voxels.insert(coord);
                }
                SimCommand::Explosion {
                    center,
                    impulse,
                    radius,
                } => {
                    let r = radius.max(0);
                    for dz in -r..=r {
                        for dy in -r..=r {
                            for dx in -r..=r {
                                if dx * dx + dy * dy + dz * dz > r * r {
                                    continue;
                                }
                                self.mac_grid.add_impulse(
                                    VoxelCoord {
                                        x: center.x + dx,
                                        y: center.y + dy,
                                        z: center.z + dz,
                                    },
                                    impulse,
                                    0.2,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn inject_momentum(&mut self) {
        for (&coord, &velocity) in &self.per_voxel_velocity {
            self.mac_grid.add_impulse(coord, velocity, 0.1);
        }
    }

    fn advect_material(&mut self) {
        let touched: Vec<_> = self.touched_voxels.iter().copied().collect();
        for coord in touched {
            let vel = self.mac_grid.velocity_at(coord);
            if vel == [0.0, 0.0, 0.0] {
                continue;
            }
            let destination = VoxelCoord {
                x: coord.x + vel[0].round() as i32,
                y: coord.y + vel[1].round() as i32,
                z: coord.z + vel[2].round() as i32,
            };
            let Some(cell) = self.occupancy.get(coord) else {
                continue;
            };
            if cell.material_id == EMPTY {
                continue;
            }
            self.occupancy
                .set(destination, cell.material_id, cell.density);
            self.occupancy.set(coord, EMPTY, 0.0);
            self.touched_voxels.insert(destination);
        }
    }

    fn project_pressure(&mut self) {
        // Placeholder: approximate pressure projection by damping divergence-heavy velocity.
        self.mac_grid.damp_all(0.92);
    }

    fn resolve_collisions(&mut self) {
        let mut to_remove = Vec::new();
        for (&coord, velocity) in &mut self.per_voxel_velocity {
            if coord.y <= 0 {
                velocity[1] = velocity[1].abs() * 0.25;
            }
            *velocity = [velocity[0] * 0.95, velocity[1] * 0.95, velocity[2] * 0.95];
            if velocity[0].abs() + velocity[1].abs() + velocity[2].abs() < 0.05 {
                to_remove.push(coord);
            }
        }
        for coord in to_remove {
            self.per_voxel_velocity.remove(&coord);
        }
    }

    fn write_back_occupancy(&mut self, store: &mut ChunkStore) {
        for coord in self.touched_voxels.drain() {
            if let Some(cell) = self.occupancy.get(coord) {
                store.set_voxel(coord, cell.material_id);
            }
        }
    }
}
