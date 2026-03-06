const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

const EMPTY: u32 = 0u;

const FACE_MASK_POS_X: u32 = 1u << 0u;
const FACE_MASK_NEG_X: u32 = 1u << 1u;
const FACE_MASK_POS_Y: u32 = 1u << 2u;
const FACE_MASK_NEG_Y: u32 = 1u << 3u;
const FACE_MASK_POS_Z: u32 = 1u << 4u;
const FACE_MASK_NEG_Z: u32 = 1u << 5u;

const VOXEL_SIZE: f32 = 0.5;

// Provide a default; override from Rust if you want.
override GPU_MAX_FACES_PER_PAGE: u32 = 262144u;

struct FrameParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    simulation_tick: u32,
    state_index: u32,
    edit_count: u32,
    active_tile_budget: u32,
    jacobi_iterations: u32,
    jacobi_iteration: u32,
};

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

// NOTE: vertex_offset / index_offset are in ELEMENTS (not bytes):
// - vertex_offset: vertex index in the global vertex buffer
// - index_offset: index index in the global index buffer
//
// OPTIONAL but strongly recommended: include chunk_origin_world so GPU output lands correctly.
struct ChunkMeshMeta {
    slot_index: u32,
    vertex_offset: u32,
    index_offset: u32,
    _pad0: u32,
    chunk_origin_world: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> face_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> face_offset: array<u32>;

// Raw dwords for packed Vertex (16 bytes = 4 u32) to avoid WGSL struct stride/padding surprises.
@group(0) @binding(3) var<storage, read_write> chunk_vertex_words: array<u32>;

@group(0) @binding(4) var<storage, read_write> chunk_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_indirect_buffer: array<DrawIndexedIndirectArgs>;
@group(0) @binding(6) var<storage, read> frame_params: array<FrameParams>;
@group(0) @binding(7) var<storage, read_write> face_count: array<u32>;
@group(0) @binding(8) var<storage, read> mesh_meta_buffer: array<ChunkMeshMeta>;

fn atlas_state_offset(page: u32, state: u32) -> u32 {
    return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME;
}

fn unpack(idx: u32) -> vec3<u32> {
    let z = idx / (CHUNK_SIDE * CHUNK_SIDE);
    let rem = idx - z * (CHUNK_SIDE * CHUNK_SIDE);
    let y = rem / CHUNK_SIDE;
    let x = rem - y * CHUNK_SIDE;
    return vec3<u32>(x, y, z);
}

fn pack(p: vec3<u32>) -> u32 {
    return p.x + p.y * CHUNK_SIDE + p.z * CHUNK_SIDE * CHUNK_SIDE;
}

fn in_bounds(p: vec3<i32>) -> bool {
    return all(p >= vec3<i32>(0)) && all(p < vec3<i32>(i32(CHUNK_SIDE)));
}

fn neighbor_dir(i: u32) -> vec3<i32> {
    switch i {
        case 0u: { return vec3<i32>( 1, 0, 0); }
        case 1u: { return vec3<i32>(-1, 0, 0); }
        case 2u: { return vec3<i32>( 0, 1, 0); }
        case 3u: { return vec3<i32>( 0,-1, 0); }
        case 4u: { return vec3<i32>( 0, 0, 1); }
        default: { return vec3<i32>( 0, 0,-1); }
    }
}

fn voxel_at(base_off: u32, p: vec3<i32>) -> u32 {
    if (!in_bounds(p)) { return EMPTY; }
    return atlas_voxels[base_off + pack(vec3<u32>(p))];
}

fn pack_rgba8(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

// Cheap stable debug color from material id (so GPU meshes are visibly non-zero).
fn color_for_material(id: u32) -> u32 {
    // LCG-ish hash
    let h = id * 1664525u + 1013904223u;
    let r = (h >>  0u) & 255u;
    let g = (h >>  8u) & 255u;
    let b = (h >> 16u) & 255u;
    return pack_rgba8(r, g, b, 255u);
}

// Write one packed Vertex at vertex_index (in vertices).
fn write_vertex(vertex_index: u32, pos_world: vec3<f32>, rgba8: u32) {
    // 4 u32 words per vertex = 16 bytes (matches Rust Vertex stride 16).
    let w = vertex_index * 4u;
    chunk_vertex_words[w + 0u] = bitcast<u32>(pos_world.x);
    chunk_vertex_words[w + 1u] = bitcast<u32>(pos_world.y);
    chunk_vertex_words[w + 2u] = bitcast<u32>(pos_world.z);
    chunk_vertex_words[w + 3u] = rgba8;
}

// IMPORTANT: indices written are LOCAL to base_vertex.
// - vertices are written to (vertex_base + local)
// - index buffer stores local indices (0..)
// - indirect sets base_vertex = vertex_base
fn write_face_quad(
    dir: u32,
    base_local_vox: vec3<f32>,
    chunk_origin_world: vec3<f32>,
    rgba8: u32,
    vertex_base: u32,
    index_base: u32,
    local_vbase: u32,
) {
    var c0: vec3<f32>;
    var c1: vec3<f32>;
    var c2: vec3<f32>;
    var c3: vec3<f32>;

    switch dir {
        case 0u: { c0 = base_local_vox + vec3<f32>(1,0,0); c1 = base_local_vox + vec3<f32>(1,1,0); c2 = base_local_vox + vec3<f32>(1,1,1); c3 = base_local_vox + vec3<f32>(1,0,1); }
        case 1u: { c0 = base_local_vox + vec3<f32>(0,0,1); c1 = base_local_vox + vec3<f32>(0,1,1); c2 = base_local_vox + vec3<f32>(0,1,0); c3 = base_local_vox + vec3<f32>(0,0,0); }
        case 2u: { c0 = base_local_vox + vec3<f32>(0,1,0); c1 = base_local_vox + vec3<f32>(0,1,1); c2 = base_local_vox + vec3<f32>(1,1,1); c3 = base_local_vox + vec3<f32>(1,1,0); }
        case 3u: { c0 = base_local_vox + vec3<f32>(0,0,1); c1 = base_local_vox + vec3<f32>(0,0,0); c2 = base_local_vox + vec3<f32>(1,0,0); c3 = base_local_vox + vec3<f32>(1,0,1); }
        case 4u: { c0 = base_local_vox + vec3<f32>(1,0,1); c1 = base_local_vox + vec3<f32>(1,1,1); c2 = base_local_vox + vec3<f32>(0,1,1); c3 = base_local_vox + vec3<f32>(0,0,1); }
        default:{ c0 = base_local_vox + vec3<f32>(0,0,0); c1 = base_local_vox + vec3<f32>(0,1,0); c2 = base_local_vox + vec3<f32>(1,1,0); c3 = base_local_vox + vec3<f32>(1,0,0); }
    }

    // Convert to world meters
    let p0 = chunk_origin_world + c0 * VOXEL_SIZE;
    let p1 = chunk_origin_world + c1 * VOXEL_SIZE;
    let p2 = chunk_origin_world + c2 * VOXEL_SIZE;
    let p3 = chunk_origin_world + c3 * VOXEL_SIZE;

    // Write vertices at (vertex_base + local)
    write_vertex(vertex_base + local_vbase + 0u, p0, rgba8);
    write_vertex(vertex_base + local_vbase + 1u, p1, rgba8);
    write_vertex(vertex_base + local_vbase + 2u, p2, rgba8);
    write_vertex(vertex_base + local_vbase + 3u, p3, rgba8);

    // Indices are LOCAL (0..), base_vertex in indirect adds vertex_base
    chunk_index_buffer[index_base + 0u] = local_vbase + 0u;
    chunk_index_buffer[index_base + 1u] = local_vbase + 1u;
    chunk_index_buffer[index_base + 2u] = local_vbase + 2u;
    chunk_index_buffer[index_base + 3u] = local_vbase + 0u;
    chunk_index_buffer[index_base + 4u] = local_vbase + 2u;
    chunk_index_buffer[index_base + 5u] = local_vbase + 3u;
}

@compute @workgroup_size(128)
fn detect_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let src_off = atlas_state_offset(page, params.state_index);

    if (voxel_idx >= params.voxel_count) {
        face_mask[mask_base + voxel_idx] = 0u;
        return;
    }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) {
        face_mask[mask_base + voxel_idx] = 0u;
        return;
    }

    let p = vec3<i32>(unpack(voxel_idx));

    var mask = 0u;

    if (voxel_at(src_off, p + neighbor_dir(0u)) == EMPTY) { mask |= FACE_MASK_POS_X; }
    if (voxel_at(src_off, p + neighbor_dir(1u)) == EMPTY) { mask |= FACE_MASK_NEG_X; }
    if (voxel_at(src_off, p + neighbor_dir(2u)) == EMPTY) { mask |= FACE_MASK_POS_Y; }
    if (voxel_at(src_off, p + neighbor_dir(3u)) == EMPTY) { mask |= FACE_MASK_NEG_Y; }
    if (voxel_at(src_off, p + neighbor_dir(4u)) == EMPTY) { mask |= FACE_MASK_POS_Z; }
    if (voxel_at(src_off, p + neighbor_dir(5u)) == EMPTY) { mask |= FACE_MASK_NEG_Z; }

    face_mask[mask_base + voxel_idx] = mask;
}

@compute @workgroup_size(1)
fn prefix_scan() {
    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let offset_base = chunk_slot * CHUNK_VOLUME;

    var running: u32 = 0u;

    for (var i: u32 = 0u; i < CHUNK_VOLUME; i = i + 1u) {
        face_offset[offset_base + i] = running;
        running = running + countOneBits(face_mask[mask_base + i]);
    }

    face_count[page] = min(running, GPU_MAX_FACES_PER_PAGE);
}

@compute @workgroup_size(128)
fn emit_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let offset_base = chunk_slot * CHUNK_VOLUME;

    let vertex_base = mesh_meta.vertex_offset; // vertices
    let index_base0 = mesh_meta.index_offset;  // indices

    let total_faces = face_count[page];

    // Write indirect ONCE (no races).
    if (gid.x == 0u) {
        draw_indirect_buffer[chunk_slot].index_count = total_faces * 6u;
        draw_indirect_buffer[chunk_slot].instance_count = 1u;
        draw_indirect_buffer[chunk_slot].first_index = index_base0;
        draw_indirect_buffer[chunk_slot].base_vertex = i32(vertex_base);
        draw_indirect_buffer[chunk_slot].first_instance = 0u;
    }

    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let src_off = atlas_state_offset(page, params.state_index);

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let mask = face_mask[mask_base + voxel_idx];
    if (mask == 0u) { return; }

    var write_face = face_offset[offset_base + voxel_idx];
    if (write_face >= total_faces) { return; }

    let base_local = vec3<f32>(unpack(voxel_idx));
    let rgba8 = color_for_material(id);

    for (var dir: u32 = 0u; dir < 6u; dir = dir + 1u) {
        if ((mask & (1u << dir)) == 0u) { continue; }
        if (write_face >= total_faces) { break; }

        let local_vo = write_face * 4u; // local vertex base (0..)
        let local_io = write_face * 6u;

        let index_write = index_base0 + local_io;

        write_face_quad(
            dir,
            base_local,
            mesh_meta.chunk_origin_world,
            rgba8,
            vertex_base,
            index_write,
            local_vo,
        );

        write_face = write_face + 1u;
    }
}