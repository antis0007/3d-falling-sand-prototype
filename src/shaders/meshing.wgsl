const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

const EMPTY: u32 = 0u;

const FACE_MASK_POS_X: u32 = 1u << 0u;
const FACE_MASK_NEG_X: u32 = 1u << 1u;
const FACE_MASK_POS_Y: u32 = 1u << 2u;
const FACE_MASK_NEG_Y: u32 = 1u << 3u;
const FACE_MASK_POS_Z: u32 = 1u << 4u;
const FACE_MASK_NEG_Z: u32 = 1u << 5u;

const MAX_FACES_PER_PAGE: u32 = GPU_MAX_FACES_PER_PAGE;

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

struct GpuVertex {
    position: vec3<f32>,
    material_id: u32,
};

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct ChunkMeshMeta {
    slot_index: u32,
    vertex_offset: u32,
    index_offset: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> face_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> face_offset: array<u32>;
@group(0) @binding(3) var<storage, read_write> chunk_vertex_buffer: array<GpuVertex>;
@group(0) @binding(4) var<storage, read_write> chunk_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_indirect_buffer: array<DrawIndexedIndirectArgs>;
@group(0) @binding(6) var<storage, read> frame_params: array<FrameParams>;
@group(0) @binding(7) var<storage, read_write> face_count: array<u32>;
@group(0) @binding(8) var<storage, read_write> mesh_meta_buffer: array<ChunkMeshMeta>;
@group(0) @binding(9) var<storage, read> chunk_origin_buffer: array<vec4<f32>>;

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
        case 0u: { return vec3<i32>(1,0,0); }
        case 1u: { return vec3<i32>(-1,0,0); }
        case 2u: { return vec3<i32>(0,1,0); }
        case 3u: { return vec3<i32>(0,-1,0); }
        case 4u: { return vec3<i32>(0,0,1); }
        default: { return vec3<i32>(0,0,-1); }
    }
}

fn voxel_at(base_off: u32, p: vec3<i32>) -> u32 {
    if (!in_bounds(p)) { return EMPTY; }
    return atlas_voxels[base_off + pack(vec3<u32>(p))];
}

fn write_face_quad(
    dir: u32,
    base: vec3<f32>,
    chunk_origin: vec3<f32>,
    color: u32,
    global_vertex_offset: u32,
    global_index_offset: u32,
) {
    var c0: vec3<f32>;
    var c1: vec3<f32>;
    var c2: vec3<f32>;
    var c3: vec3<f32>;

    switch dir {
        case 0u: { c0 = base + vec3<f32>(1,0,0); c1 = base + vec3<f32>(1,1,0); c2 = base + vec3<f32>(1,1,1); c3 = base + vec3<f32>(1,0,1); }
        case 1u: { c0 = base + vec3<f32>(0,0,1); c1 = base + vec3<f32>(0,1,1); c2 = base + vec3<f32>(0,1,0); c3 = base + vec3<f32>(0,0,0); }
        case 2u: { c0 = base + vec3<f32>(0,1,0); c1 = base + vec3<f32>(0,1,1); c2 = base + vec3<f32>(1,1,1); c3 = base + vec3<f32>(1,1,0); }
        case 3u: { c0 = base + vec3<f32>(0,0,1); c1 = base + vec3<f32>(0,0,0); c2 = base + vec3<f32>(1,0,0); c3 = base + vec3<f32>(1,0,1); }
        case 4u: { c0 = base + vec3<f32>(1,0,1); c1 = base + vec3<f32>(1,1,1); c2 = base + vec3<f32>(0,1,1); c3 = base + vec3<f32>(0,0,1); }
        default:{ c0 = base + vec3<f32>(0,0,0); c1 = base + vec3<f32>(0,1,0); c2 = base + vec3<f32>(1,1,0); c3 = base + vec3<f32>(1,0,0); }
    }

    let v0 = global_vertex_offset;
    let v1 = global_vertex_offset + 1u;
    let v2 = global_vertex_offset + 2u;
    let v3 = global_vertex_offset + 3u;

    chunk_vertex_buffer[v0] = GpuVertex(chunk_origin + c0, color);
    chunk_vertex_buffer[v1] = GpuVertex(chunk_origin + c1, color);
    chunk_vertex_buffer[v2] = GpuVertex(chunk_origin + c2, color);
    chunk_vertex_buffer[v3] = GpuVertex(chunk_origin + c3, color);

    chunk_index_buffer[global_index_offset + 0u] = v0;
    chunk_index_buffer[global_index_offset + 1u] = v1;
    chunk_index_buffer[global_index_offset + 2u] = v2;
    chunk_index_buffer[global_index_offset + 3u] = v0;
    chunk_index_buffer[global_index_offset + 4u] = v2;
    chunk_index_buffer[global_index_offset + 5u] = v3;
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

    face_count[page] = min(running, MAX_FACES_PER_PAGE);
}

@compute @workgroup_size(128)
fn emit_mesh(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {

    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let offset_base = chunk_slot * CHUNK_VOLUME;

    let vertex_base = mesh_meta.vertex_offset;
    let index_base = mesh_meta.index_offset;

    let total_faces = face_count[page];
    let chunk_origin = chunk_origin_buffer[page].xyz;

    // --- always write draw command ---
    if (lid.x == 0u) {

        draw_indirect_buffer[chunk_slot].index_count = total_faces * 6u;
        draw_indirect_buffer[chunk_slot].instance_count = 1u;
        draw_indirect_buffer[chunk_slot].first_index = index_base;
        draw_indirect_buffer[chunk_slot].base_vertex = i32(vertex_base);
        draw_indirect_buffer[chunk_slot].first_instance = 0u;
    }

    // --- ensure all threads see face_count ---
    workgroupBarrier();
    storageBarrier();

    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let src_off = atlas_state_offset(page, params.state_index);

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let mask = face_mask[mask_base + voxel_idx];
    if (mask == 0u) { return; }

    var write_face = face_offset[offset_base + voxel_idx];
    if (write_face >= total_faces) { return; }

    let base = vec3<f32>(vec3<u32>(unpack(voxel_idx)));

    for (var dir: u32 = 0u; dir < 6u; dir = dir + 1u) {

        if ((mask & (1u << dir)) == 0u) { continue; }
        if (write_face >= total_faces) { break; }

        let vo = write_face * 4u;
        let io = write_face * 6u;

        write_face_quad(dir, base, chunk_origin, id, vertex_base + vo, index_base + io);

        write_face = write_face + 1u;
    }
}