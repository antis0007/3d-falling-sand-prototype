use crate::chunk_store::{ChunkMeshingInput, ChunkStore};
use crate::renderer::{Vertex, VOXEL_SIZE};
use crate::sim::{material, Phase};
use crate::types::{chunk_to_world_min, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};
use crate::world::{MaterialId, EMPTY};
use glam::Vec3;

const TURF_ID: MaterialId = 17;
const BUSH_ID: MaterialId = 18;
const GRASS_ID: MaterialId = 19;
const CHUNK_SIDE: usize = CHUNK_SIZE_VOXELS as usize;
const CHUNK_VOLUME: usize = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

#[derive(Default)]
pub struct MeshedChunk {
    pub opaque_verts: Vec<Vertex>,
    pub opaque_inds: Vec<u32>,
    pub transparent_verts: Vec<Vertex>,
    pub transparent_inds: Vec<u32>,
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NeighborSample {
    Known(MaterialId),
    Unknown,
}

pub struct MeshingSystem;

impl MeshingSystem {
    pub fn mesh_chunk(
        &self,
        store: &ChunkStore,
        coord: ChunkCoord,
        origin_voxel: VoxelCoord,
    ) -> MeshedChunk {
        mesh_chunk(store, coord, origin_voxel)
    }
}

pub fn mesh_chunk(store: &ChunkStore, coord: ChunkCoord, origin_voxel: VoxelCoord) -> MeshedChunk {
    let mut opaque_verts = Vec::with_capacity(CHUNK_VOLUME * 12);
    let mut opaque_inds = Vec::with_capacity(CHUNK_VOLUME * 18);
    let mut transparent_verts = Vec::new();
    let mut transparent_inds = Vec::new();

    let chunk_world_min = chunk_to_world_min(coord);
    let min = relative_voxel_to_world(chunk_world_min, origin_voxel);
    let max = min + Vec3::splat(CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE);

    let Some(mesh_input) = store.build_meshing_input(coord) else {
        return MeshedChunk {
            min,
            max,
            ..MeshedChunk::default()
        };
    };
    let chunk = mesh_input.voxels;

    if chunk.iter().all(|&v| v == EMPTY) {
        return MeshedChunk {
            min,
            max,
            ..MeshedChunk::default()
        };
    }

    if let Some(fill_id) = fully_solid_fill(chunk) {
        add_uniform_chunk_shell_optimized(
            &mesh_input,
            chunk_world_min,
            origin_voxel,
            fill_id,
            &mut opaque_verts,
            &mut opaque_inds,
        );
        return MeshedChunk {
            opaque_verts,
            opaque_inds,
            transparent_verts,
            transparent_inds,
            min,
            max,
        };
    }

    for lz in 0..CHUNK_SIZE_VOXELS {
        for ly in 0..CHUNK_SIZE_VOXELS {
            for lx in 0..CHUNK_SIZE_VOXELS {
                let lx_u = lx as usize;
                let ly_u = ly as usize;
                let lz_u = lz as usize;
                let world_voxel = VoxelCoord {
                    x: chunk_world_min.x + lx,
                    y: chunk_world_min.y + ly,
                    z: chunk_world_min.z + lz,
                };
                let id = chunk[voxel_index(lx_u, ly_u, lz_u)];
                if id == EMPTY {
                    continue;
                }

                if matches!(id, BUSH_ID | GRASS_ID) {
                    add_crossed_billboard(
                        &mesh_input,
                        lx_u,
                        ly_u,
                        lz_u,
                        world_voxel,
                        origin_voxel,
                        id,
                        &mut opaque_verts,
                        &mut opaque_inds,
                    );
                    continue;
                }

                let color = material(id).color;
                let (verts, inds) = if is_transparent_material(id) {
                    (&mut transparent_verts, &mut transparent_inds)
                } else {
                    (&mut opaque_verts, &mut opaque_inds)
                };
                add_voxel_faces(
                    &mesh_input,
                    lx_u,
                    ly_u,
                    lz_u,
                    world_voxel,
                    origin_voxel,
                    id,
                    color,
                    verts,
                    inds,
                );
            }
        }
    }

    MeshedChunk {
        opaque_verts,
        opaque_inds,
        transparent_verts,
        transparent_inds,
        min,
        max,
    }
}

fn add_voxel_faces(
    mesh_input: &ChunkMeshingInput<'_>,
    lx: usize,
    ly: usize,
    lz: usize,
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
        let neighbor = neighbor_voxel(mesh_input, lx, ly, lz, d);
        if is_face_occluded(id, neighbor) {
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
    mesh_input: &ChunkMeshingInput<'_>,
    lx: usize,
    ly: usize,
    lz: usize,
    world_voxel: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let above = neighbor_voxel(mesh_input, lx, ly, lz, [0, 1, 0]);
    if is_occluding_sample(above) {
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

fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    (z * CHUNK_SIDE + y) * CHUNK_SIDE + x
}

fn neighbor_voxel(
    mesh_input: &ChunkMeshingInput<'_>,
    x: usize,
    y: usize,
    z: usize,
    dir: [i32; 3],
) -> NeighborSample {
    match dir {
        [1, 0, 0] => {
            if x + 1 < CHUNK_SIDE {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x + 1, y, z)])
            } else {
                face_border_sample(
                    mesh_input,
                    1,
                    mesh_input.pos_x[ChunkMeshingInput::border_index(y, z)],
                )
            }
        }
        [-1, 0, 0] => {
            if x > 0 {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x - 1, y, z)])
            } else {
                face_border_sample(
                    mesh_input,
                    0,
                    mesh_input.neg_x[ChunkMeshingInput::border_index(y, z)],
                )
            }
        }
        [0, 1, 0] => {
            if y + 1 < CHUNK_SIDE {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x, y + 1, z)])
            } else {
                face_border_sample(
                    mesh_input,
                    3,
                    mesh_input.pos_y[ChunkMeshingInput::border_index(x, z)],
                )
            }
        }
        [0, -1, 0] => {
            if y > 0 {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x, y - 1, z)])
            } else {
                face_border_sample(
                    mesh_input,
                    2,
                    mesh_input.neg_y[ChunkMeshingInput::border_index(x, z)],
                )
            }
        }
        [0, 0, 1] => {
            if z + 1 < CHUNK_SIDE {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x, y, z + 1)])
            } else {
                face_border_sample(
                    mesh_input,
                    5,
                    mesh_input.pos_z[ChunkMeshingInput::border_index(x, y)],
                )
            }
        }
        [0, 0, -1] => {
            if z > 0 {
                NeighborSample::Known(mesh_input.voxels[voxel_index(x, y, z - 1)])
            } else {
                face_border_sample(
                    mesh_input,
                    4,
                    mesh_input.neg_z[ChunkMeshingInput::border_index(x, y)],
                )
            }
        }
        _ => NeighborSample::Known(EMPTY),
    }
}

fn face_border_sample(
    mesh_input: &ChunkMeshingInput<'_>,
    face_idx: u8,
    material: MaterialId,
) -> NeighborSample {
    if (mesh_input.known_neighbor_mask & (1 << face_idx)) != 0 {
        NeighborSample::Known(material)
    } else {
        NeighborSample::Unknown
    }
}

fn fully_solid_fill(voxels: &[MaterialId]) -> Option<MaterialId> {
    let fill = *voxels.first()?;
    if fill == EMPTY || matches!(fill, BUSH_ID | GRASS_ID) {
        return None;
    }
    if !is_occluding_material(fill) {
        return None;
    }
    voxels.iter().all(|&v| v == fill).then_some(fill)
}

fn add_uniform_chunk_shell_optimized(
    mesh_input: &ChunkMeshingInput<'_>,
    chunk_world_min: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let color = material(id).color;
    if try_add_uniform_chunk_face_quads(
        mesh_input,
        chunk_world_min,
        origin_voxel,
        id,
        color,
        verts,
        inds,
    ) {
        return;
    }
    add_uniform_chunk_shell(mesh_input, chunk_world_min, origin_voxel, id, verts, inds);
}

fn try_add_uniform_chunk_face_quads(
    mesh_input: &ChunkMeshingInput<'_>,
    chunk_world_min: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    color: [u8; 4],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) -> bool {
    let face_checks = [
        ([1, 0, 0], 1),
        ([-1, 0, 0], 0),
        ([0, 1, 0], 3),
        ([0, -1, 0], 2),
        ([0, 0, 1], 5),
        ([0, 0, -1], 4),
    ];
    for (dir, border_face_idx) in face_checks {
        let visible = uniform_face_fully_visible(mesh_input, id, dir, border_face_idx);
        if !visible {
            return false;
        }
    }

    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [1, 0, 0],
        verts,
        inds,
    );
    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [-1, 0, 0],
        verts,
        inds,
    );
    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [0, 1, 0],
        verts,
        inds,
    );
    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [0, -1, 0],
        verts,
        inds,
    );
    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [0, 0, 1],
        verts,
        inds,
    );
    add_chunk_face_quad(
        chunk_world_min,
        origin_voxel,
        id,
        color,
        [0, 0, -1],
        verts,
        inds,
    );
    true
}

fn uniform_face_fully_visible(
    mesh_input: &ChunkMeshingInput<'_>,
    id: MaterialId,
    dir: [i32; 3],
    border_face_idx: u8,
) -> bool {
    if (mesh_input.known_neighbor_mask & (1 << border_face_idx)) == 0 {
        return false;
    }
    for v in 0..CHUNK_SIDE {
        for u in 0..CHUNK_SIDE {
            let idx = ChunkMeshingInput::border_index(u, v);
            let neighbor = match dir {
                [1, 0, 0] => NeighborSample::Known(mesh_input.pos_x[idx]),
                [-1, 0, 0] => NeighborSample::Known(mesh_input.neg_x[idx]),
                [0, 1, 0] => NeighborSample::Known(mesh_input.pos_y[idx]),
                [0, -1, 0] => NeighborSample::Known(mesh_input.neg_y[idx]),
                [0, 0, 1] => NeighborSample::Known(mesh_input.pos_z[idx]),
                [0, 0, -1] => NeighborSample::Known(mesh_input.neg_z[idx]),
                _ => NeighborSample::Unknown,
            };
            if is_face_occluded(id, neighbor) {
                return false;
            }
        }
    }
    true
}

fn add_chunk_face_quad(
    chunk_world_min: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    color: [u8; 4],
    dir: [i32; 3],
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let side = CHUNK_SIZE_VOXELS as f32 * VOXEL_SIZE;
    let (quad, shade) = match dir {
        [1, 0, 0] => (
            [
                [side, 0.0, 0.0],
                [side, side, 0.0],
                [side, side, side],
                [side, 0.0, side],
            ],
            0.88,
        ),
        [-1, 0, 0] => (
            [
                [0.0, 0.0, side],
                [0.0, side, side],
                [0.0, side, 0.0],
                [0.0, 0.0, 0.0],
            ],
            0.73,
        ),
        [0, 1, 0] => (
            [
                [0.0, side, side],
                [side, side, side],
                [side, side, 0.0],
                [0.0, side, 0.0],
            ],
            1.0,
        ),
        [0, -1, 0] => (
            [
                [0.0, 0.0, 0.0],
                [side, 0.0, 0.0],
                [side, 0.0, side],
                [0.0, 0.0, side],
            ],
            0.58,
        ),
        [0, 0, 1] => (
            [
                [side, 0.0, side],
                [side, side, side],
                [0.0, side, side],
                [0.0, 0.0, side],
            ],
            0.8,
        ),
        _ => (
            [
                [0.0, 0.0, 0.0],
                [0.0, side, 0.0],
                [side, side, 0.0],
                [side, 0.0, 0.0],
            ],
            0.66,
        ),
    };
    let b = verts.len() as u32;
    let shaded = shade_color(turf_face_color(id, dir, color), shade);
    for v in quad {
        verts.push(Vertex {
            pos: [
                (chunk_world_min.x - origin_voxel.x) as f32 * VOXEL_SIZE + v[0],
                (chunk_world_min.y - origin_voxel.y) as f32 * VOXEL_SIZE + v[1],
                (chunk_world_min.z - origin_voxel.z) as f32 * VOXEL_SIZE + v[2],
            ],
            color: shaded,
        });
    }
    inds.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
}

fn add_uniform_chunk_shell(
    mesh_input: &ChunkMeshingInput<'_>,
    chunk_world_min: VoxelCoord,
    origin_voxel: VoxelCoord,
    id: MaterialId,
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u32>,
) {
    let color = material(id).color;
    for z in 0..CHUNK_SIZE_VOXELS {
        for y in 0..CHUNK_SIZE_VOXELS {
            let world_voxel = VoxelCoord {
                x: chunk_world_min.x,
                y: chunk_world_min.y + y,
                z: chunk_world_min.z + z,
            };
            add_voxel_faces(
                mesh_input,
                0,
                y as usize,
                z as usize,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );

            let world_voxel = VoxelCoord {
                x: chunk_world_min.x + CHUNK_SIZE_VOXELS - 1,
                y: chunk_world_min.y + y,
                z: chunk_world_min.z + z,
            };
            add_voxel_faces(
                mesh_input,
                CHUNK_SIDE - 1,
                y as usize,
                z as usize,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );
        }
    }

    for z in 0..CHUNK_SIZE_VOXELS {
        for x in 1..(CHUNK_SIZE_VOXELS - 1) {
            let world_voxel = VoxelCoord {
                x: chunk_world_min.x + x,
                y: chunk_world_min.y,
                z: chunk_world_min.z + z,
            };
            add_voxel_faces(
                mesh_input,
                x as usize,
                0,
                z as usize,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );

            let world_voxel = VoxelCoord {
                x: chunk_world_min.x + x,
                y: chunk_world_min.y + CHUNK_SIZE_VOXELS - 1,
                z: chunk_world_min.z + z,
            };
            add_voxel_faces(
                mesh_input,
                x as usize,
                CHUNK_SIDE - 1,
                z as usize,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );
        }
    }

    for y in 1..(CHUNK_SIZE_VOXELS - 1) {
        for x in 1..(CHUNK_SIZE_VOXELS - 1) {
            let world_voxel = VoxelCoord {
                x: chunk_world_min.x + x,
                y: chunk_world_min.y + y,
                z: chunk_world_min.z,
            };
            add_voxel_faces(
                mesh_input,
                x as usize,
                y as usize,
                0,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );

            let world_voxel = VoxelCoord {
                x: chunk_world_min.x + x,
                y: chunk_world_min.y + y,
                z: chunk_world_min.z + CHUNK_SIZE_VOXELS - 1,
            };
            add_voxel_faces(
                mesh_input,
                x as usize,
                y as usize,
                CHUNK_SIDE - 1,
                world_voxel,
                origin_voxel,
                id,
                color,
                verts,
                inds,
            );
        }
    }
}

fn is_occluding_material(id: MaterialId) -> bool {
    if id == EMPTY || matches!(id, BUSH_ID | GRASS_ID) {
        return false;
    }
    let info = material(id);
    info.color[3] == 255 && matches!(info.phase, Phase::Solid | Phase::Powder)
}

fn is_transparent_material(id: MaterialId) -> bool {
    if id == EMPTY || matches!(id, BUSH_ID | GRASS_ID) {
        return false;
    }
    material(id).color[3] < 255
}

fn is_occluding_sample(sample: NeighborSample) -> bool {
    match sample {
        NeighborSample::Unknown => true,
        NeighborSample::Known(id) => is_occluding_material(id),
    }
}

fn is_face_occluded(self_id: MaterialId, neighbor: NeighborSample) -> bool {
    let NeighborSample::Known(neighbor_id) = neighbor else {
        return true;
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{Chunk, NeighborDirtyPolicy};

    fn coord() -> ChunkCoord {
        ChunkCoord { x: 0, y: 0, z: 0 }
    }

    fn chunk_with_voxel(x: usize, y: usize, z: usize, id: MaterialId) -> Chunk {
        let mut c = Chunk::new_empty();
        c.set(x, y, z, id);
        c
    }

    #[test]
    fn grass_not_culled_by_transparent_non_occluding_above() {
        let mut store = ChunkStore::new();
        let mut center = Chunk::new_empty();
        center.set(1, 1, 1, GRASS_ID);
        center.set(1, 2, 1, 5);
        store.insert_chunk_with_policy(coord(), center, false, NeighborDirtyPolicy::None);

        let mesh = mesh_chunk(&store, coord(), VoxelCoord { x: 0, y: 0, z: 0 });
        assert!(!mesh.opaque_verts.is_empty());
    }

    #[test]
    fn unknown_neighbor_not_treated_as_known_empty_for_boundary_faces() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            coord(),
            chunk_with_voxel(CHUNK_SIDE - 1, 2, 2, 1),
            false,
            NeighborDirtyPolicy::None,
        );

        let no_neighbor = mesh_chunk(&store, coord(), VoxelCoord { x: 0, y: 0, z: 0 });
        assert!(no_neighbor.opaque_verts.is_empty());

        store.insert_chunk_with_policy(
            ChunkCoord { x: 1, y: 0, z: 0 },
            Chunk::new_empty(),
            false,
            NeighborDirtyPolicy::None,
        );
        let with_known_empty = mesh_chunk(&store, coord(), VoxelCoord { x: 0, y: 0, z: 0 });
        assert!(!with_known_empty.opaque_verts.is_empty());
    }

    #[test]
    fn water_is_routed_to_transparent_output() {
        let mut store = ChunkStore::new();
        store.insert_chunk_with_policy(
            coord(),
            chunk_with_voxel(1, 1, 1, 5),
            false,
            NeighborDirtyPolicy::None,
        );

        let mesh = mesh_chunk(&store, coord(), VoxelCoord { x: 0, y: 0, z: 0 });
        assert!(mesh.opaque_inds.is_empty());
        assert!(!mesh.transparent_inds.is_empty());
    }

    #[test]
    fn uniform_solid_chunk_uses_merged_face_quads_when_fully_exposed() {
        let mut store = ChunkStore::new();
        let mut center = Chunk::new_empty();
        for z in 0..CHUNK_SIDE {
            for y in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    center.set(x, y, z, 1);
                }
            }
        }

        let neighbors = [
            ChunkCoord { x: -1, y: 0, z: 0 },
            ChunkCoord { x: 1, y: 0, z: 0 },
            ChunkCoord { x: 0, y: -1, z: 0 },
            ChunkCoord { x: 0, y: 1, z: 0 },
            ChunkCoord { x: 0, y: 0, z: -1 },
            ChunkCoord { x: 0, y: 0, z: 1 },
        ];
        for c in neighbors {
            store.insert_chunk_with_policy(c, Chunk::new_empty(), false, NeighborDirtyPolicy::None);
        }
        store.insert_chunk_with_policy(coord(), center, false, NeighborDirtyPolicy::None);

        let mesh = mesh_chunk(&store, coord(), VoxelCoord { x: 0, y: 0, z: 0 });
        assert_eq!(mesh.opaque_verts.len(), 24);
        assert_eq!(mesh.opaque_inds.len(), 36);
    }
}
