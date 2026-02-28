use crate::chunk_store::ChunkStore;
use crate::renderer::{Vertex, VOXEL_SIZE};
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use glam::Vec3;

const TURF_ID: MaterialId = 17;
const BUSH_ID: MaterialId = 18;
const GRASS_ID: MaterialId = 19;

pub struct MeshingSystem;

impl MeshingSystem {
    pub fn mesh_chunk(
        &self,
        store: &ChunkStore,
        coord: ChunkCoord,
        origin_voxel: VoxelCoord,
    ) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3) {
        mesh_chunk(store, coord, origin_voxel)
    }
}

pub fn mesh_chunk(
    store: &ChunkStore,
    coord: ChunkCoord,
    origin_voxel: VoxelCoord,
) -> (Vec<Vertex>, Vec<u32>, Vec3, Vec3) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();

    let chunk_world_min = chunk_to_world_min(coord);
    let min = relative_voxel_to_world(chunk_world_min, origin_voxel);
    let max = min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);

    let Some(chunk) = store.get_chunk(coord) else {
        return (verts, inds, min, max);
    };

    if chunk.iter_raw().iter().all(|&v| v == EMPTY) {
        return (verts, inds, min, max);
    }

    for lz in 0..CHUNK_SIZE_VOXELS {
        for ly in 0..CHUNK_SIZE_VOXELS {
            for lx in 0..CHUNK_SIZE_VOXELS {
                let world_voxel = VoxelCoord {
                    x: chunk_world_min.x + lx,
                    y: chunk_world_min.y + ly,
                    z: chunk_world_min.z + lz,
                };
                let id = chunk.get(lx as usize, ly as usize, lz as usize);
                if id == EMPTY {
                    continue;
                }

                if matches!(id, BUSH_ID | GRASS_ID) {
                    add_crossed_billboard(
                        store,
                        world_voxel,
                        origin_voxel,
                        id,
                        &mut verts,
                        &mut inds,
                    );
                    continue;
                }

                let color = material(id).color;
                add_voxel_faces(
                    store,
                    world_voxel,
                    origin_voxel,
                    id,
                    color,
                    &mut verts,
                    &mut inds,
                );
            }
        }
    }

    (verts, inds, min, max)
}

fn add_voxel_faces(
    store: &ChunkStore,
    world_voxel: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let dirs = [
        (
            [1, 0, 0],
            [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
            0.88,
        ),
        (
            [-1, 0, 0],
            [[0., 0., 1.], [0., 1., 1.], [0., 1., 0.], [0., 0., 0.]],
            0.73,
        ),
        (
            [0, 1, 0],
            [[0., 1., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.]],
            1.0,
        ),
        (
            [0, -1, 0],
            [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
            0.58,
        ),
        (
            [0, 0, 1],
            [[1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.]],
            0.8,
        ),
        (
            [0, 0, -1],
            [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
            0.66,
        ),
    ];

    for (d, quad, shade) in dirs {
        let neighbor = VoxelCoord {
            x: world_voxel.x + d[0],
            y: world_voxel.y + d[1],
            z: world_voxel.z + d[2],
        };
        if is_face_occluded(id, store.get_voxel(neighbor).unwrap_or(EMPTY)) {
            continue;
        }

        let b = verts.len() as u32;
        let face_color = turf_face_color(id, d, color);
        let shaded = shade_color(face_color, shade);
        for v in quad {
            verts.push(Vertex {
                pos: [
                    (world_voxel.x - origin_voxel.x) as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    (world_voxel.y - origin_voxel.y) as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    (world_voxel.z - origin_voxel.z) as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color: shaded,
            });
        }
        inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
}

fn add_crossed_billboard(
    store: &ChunkStore,
    world_voxel: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let above = store.get_voxel(VoxelCoord {
        x: world_voxel.x,
        y: world_voxel.y + 1,
        z: world_voxel.z,
    });
    if above.unwrap_or(EMPTY) != EMPTY {
        return;
    }

    let color = material(id).color;
    let quads = [
        [
            [0.15, 0.0, 0.15],
            [0.15, 1.0, 0.15],
            [0.85, 1.0, 0.85],
            [0.85, 0.0, 0.85],
        ],
        [
            [0.15, 0.0, 0.85],
            [0.15, 1.0, 0.85],
            [0.85, 1.0, 0.15],
            [0.85, 0.0, 0.15],
        ],
    ];

    for quad in quads {
        let b = verts.len() as u32;
        for v in quad {
            verts.push(Vertex {
                pos: [
                    (world_voxel.x - origin_voxel.x) as f32 * VOXEL_SIZE + v[0] * VOXEL_SIZE,
                    (world_voxel.y - origin_voxel.y) as f32 * VOXEL_SIZE + v[1] * VOXEL_SIZE,
                    (world_voxel.z - origin_voxel.z) as f32 * VOXEL_SIZE + v[2] * VOXEL_SIZE,
                ],
                color,
            });
        }
        inds.extend_from_slice(&[
            b,
            b + 1,
            b + 2,
            b,
            b + 2,
            b + 3,
            b,
            b + 2,
            b + 1,
            b,
            b + 3,
            b + 2,
        ]);
    }
}

fn is_face_occluded(self_id: MaterialId, neighbor_id: MaterialId) -> bool {
    if neighbor_id == EMPTY {
        return false;
    }

    if neighbor_id == self_id {
        return true;
    }

    if matches!(neighbor_id, GRASS_ID | BUSH_ID) {
        return false;
    }

    let neighbor = material(neighbor_id);
    if neighbor.color[3] < 255 {
        return false;
    }

    matches!(neighbor.phase, Phase::Solid | Phase::Powder)
}

fn turf_face_color(id: MaterialId, dir: [i32; 3], fallback: [u8; 4]) -> [u8; 4] {
    if id != TURF_ID {
        return fallback;
    }
    if dir == [0, 1, 0] {
        return [92, 171, 78, 255];
    }
    if dir == [0, -1, 0] {
        return [121, 88, 56, 255];
    }
    [116, 103, 61, 255]
}

fn shade_color(color: [u8; 4], shade: f32) -> [u8; 4] {
    [
        ((color[0] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[1] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        ((color[2] as f32 * shade).round().clamp(0.0, 255.0)) as u8,
        color[3],
    ]
}

fn relative_voxel_to_world(voxel: VoxelCoord, origin_voxel: VoxelCoord) -> Vec3 {
    Vec3::new(
        (voxel.x - origin_voxel.x) as f32,
        (voxel.y - origin_voxel.y) as f32,
        (voxel.z - origin_voxel.z) as f32,
    ) * VOXEL_SIZE
}
