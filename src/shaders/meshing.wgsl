const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const EMPTY: u32 = 0u;
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u32 = CHUNK_VOLUME * 24u;
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u32 = CHUNK_VOLUME * 36u;
const FACE_MASK_POS_X: u32 = 1u << 0u;
const FACE_MASK_NEG_X: u32 = 1u << 1u;
const FACE_MASK_POS_Y: u32 = 1u << 2u;
const FACE_MASK_NEG_Y: u32 = 1u << 3u;
const FACE_MASK_POS_Z: u32 = 1u << 4u;
const FACE_MASK_NEG_Z: u32 = 1u << 5u;
const MAX_FACES_PER_PAGE: u32 = min(GPU_MESH_VERTEX_CAPACITY_PER_PAGE / 4u, GPU_MESH_INDEX_CAPACITY_PER_PAGE / 6u);
const SCAN_WORKGROUP_SIZE: u32 = 128u;
const SCAN_VOXELS_PER_THREAD: u32 = CHUNK_VOLUME / SCAN_WORKGROUP_SIZE;

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
    page_index: u32,
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

var<workgroup> scan_chunk_offsets: array<u32, SCAN_WORKGROUP_SIZE>;

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

fn pack(p: vec3<u32>) -> u32 { return p.x + p.y * CHUNK_SIDE + p.z * CHUNK_SIDE * CHUNK_SIDE; }
fn in_bounds(p: vec3<i32>) -> bool { return all(p >= vec3<i32>(0)) && all(p < vec3<i32>(i32(CHUNK_SIDE))); }
fn neighbor_dir(i: u32) -> vec3<i32> {
    switch i {
        case 0u: { return vec3<i32>(1, 0, 0); }
        case 1u: { return vec3<i32>(-1, 0, 0); }
        case 2u: { return vec3<i32>(0, 1, 0); }
        case 3u: { return vec3<i32>(0, -1, 0); }
        case 4u: { return vec3<i32>(0, 0, 1); }
        default: { return vec3<i32>(0, 0, -1); }
    }
}

fn voxel_at(base_off: u32, p: vec3<i32>) -> u32 {
    if (!in_bounds(p)) { return EMPTY; }
    return atlas_voxels[base_off + pack(vec3<u32>(p))];
}

fn write_face_quad(
    dir: u32,
    base: vec3<f32>,
    material_id: u32,
    global_vertex_offset: u32,
    global_index_offset: u32,
    local_vertex_offset: u32,
) {
    var c0 = vec3<f32>(0.0, 0.0, 0.0);
    var c1 = vec3<f32>(0.0, 0.0, 0.0);
    var c2 = vec3<f32>(0.0, 0.0, 0.0);
    var c3 = vec3<f32>(0.0, 0.0, 0.0);
    switch dir {
        case 0u: {
            c0 = base + vec3<f32>(1.0, 0.0, 0.0);
            c1 = base + vec3<f32>(1.0, 1.0, 0.0);
            c2 = base + vec3<f32>(1.0, 1.0, 1.0);
            c3 = base + vec3<f32>(1.0, 0.0, 1.0);
        }
        case 1u: {
            c0 = base + vec3<f32>(0.0, 0.0, 1.0);
            c1 = base + vec3<f32>(0.0, 1.0, 1.0);
            c2 = base + vec3<f32>(0.0, 1.0, 0.0);
            c3 = base + vec3<f32>(0.0, 0.0, 0.0);
        }
        case 2u: {
            c0 = base + vec3<f32>(0.0, 1.0, 0.0);
            c1 = base + vec3<f32>(0.0, 1.0, 1.0);
            c2 = base + vec3<f32>(1.0, 1.0, 1.0);
            c3 = base + vec3<f32>(1.0, 1.0, 0.0);
        }
        case 3u: {
            c0 = base + vec3<f32>(0.0, 0.0, 1.0);
            c1 = base + vec3<f32>(0.0, 0.0, 0.0);
            c2 = base + vec3<f32>(1.0, 0.0, 0.0);
            c3 = base + vec3<f32>(1.0, 0.0, 1.0);
        }
        case 4u: {
            c0 = base + vec3<f32>(1.0, 0.0, 1.0);
            c1 = base + vec3<f32>(1.0, 1.0, 1.0);
            c2 = base + vec3<f32>(0.0, 1.0, 1.0);
            c3 = base + vec3<f32>(0.0, 0.0, 1.0);
        }
        default: {
            c0 = base + vec3<f32>(0.0, 0.0, 0.0);
            c1 = base + vec3<f32>(0.0, 1.0, 0.0);
            c2 = base + vec3<f32>(1.0, 1.0, 0.0);
            c3 = base + vec3<f32>(1.0, 0.0, 0.0);
        }
    }

    let v0 = global_vertex_offset;
    let v1 = global_vertex_offset + 1u;
    let v2 = global_vertex_offset + 2u;
    let v3 = global_vertex_offset + 3u;
    chunk_vertex_buffer[v0] = GpuVertex(c0, material_id);
    chunk_vertex_buffer[v1] = GpuVertex(c1, material_id);
    chunk_vertex_buffer[v2] = GpuVertex(c2, material_id);
    chunk_vertex_buffer[v3] = GpuVertex(c3, material_id);

    chunk_index_buffer[global_index_offset + 0u] = local_vertex_offset + 0u;
    chunk_index_buffer[global_index_offset + 1u] = local_vertex_offset + 1u;
    chunk_index_buffer[global_index_offset + 2u] = local_vertex_offset + 2u;
    chunk_index_buffer[global_index_offset + 3u] = local_vertex_offset + 0u;
    chunk_index_buffer[global_index_offset + 4u] = local_vertex_offset + 2u;
    chunk_index_buffer[global_index_offset + 5u] = local_vertex_offset + 3u;
}

@compute @workgroup_size(128)
fn detect_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let params = frame_params[0u];
    let src_off = atlas_state_offset(params.page_index, (params.state_index + 1u) & 1u);
    if (voxel_idx >= params.voxel_count) {
        face_mask[voxel_idx] = 0u;
        return;
    }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) {
        face_mask[voxel_idx] = 0u;
        return;
    }

    let p = vec3<i32>(unpack(voxel_idx));
    var mask = 0u;
    if (voxel_at(src_off, p + neighbor_dir(0u)) == EMPTY) { mask = mask | FACE_MASK_POS_X; }
    if (voxel_at(src_off, p + neighbor_dir(1u)) == EMPTY) { mask = mask | FACE_MASK_NEG_X; }
    if (voxel_at(src_off, p + neighbor_dir(2u)) == EMPTY) { mask = mask | FACE_MASK_POS_Y; }
    if (voxel_at(src_off, p + neighbor_dir(3u)) == EMPTY) { mask = mask | FACE_MASK_NEG_Y; }
    if (voxel_at(src_off, p + neighbor_dir(4u)) == EMPTY) { mask = mask | FACE_MASK_POS_Z; }
    if (voxel_at(src_off, p + neighbor_dir(5u)) == EMPTY) { mask = mask | FACE_MASK_NEG_Z; }
    face_mask[voxel_idx] = mask;
}

@compute @workgroup_size(128)
fn prefix_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let chunk_start = tid * SCAN_VOXELS_PER_THREAD;
    let chunk_end = chunk_start + SCAN_VOXELS_PER_THREAD;

    var local_sum = 0u;
    for (var i = chunk_start; i < chunk_end; i = i + 1u) {
        local_sum = local_sum + countOneBits(face_mask[i]);
    }
    scan_chunk_offsets[tid] = local_sum;
    workgroupBarrier();

    if (tid == 0u) {
        var running = 0u;
        for (var t = 0u; t < SCAN_WORKGROUP_SIZE; t = t + 1u) {
            let chunk_faces = scan_chunk_offsets[t];
            scan_chunk_offsets[t] = running;
            running = running + chunk_faces;
        }
        face_count[0u] = min(running, MAX_FACES_PER_PAGE);
    }
    workgroupBarrier();

    var write_offset = scan_chunk_offsets[tid];
    for (var i = chunk_start; i < chunk_end; i = i + 1u) {
        face_offset[i] = write_offset;
        write_offset = write_offset + countOneBits(face_mask[i]);
    }
}

@compute @workgroup_size(128)
fn emit_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let params = frame_params[0u];
    if (voxel_idx >= params.voxel_count) { return; }

    let page = params.page_index;
    let src_off = atlas_state_offset(page, (params.state_index + 1u) & 1u);
    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let mask = face_mask[voxel_idx];
    if (mask == 0u) { return; }

    let total_faces = face_count[0u];
    let vertex_base = page * GPU_MESH_VERTEX_CAPACITY_PER_PAGE;
    let index_base = page * GPU_MESH_INDEX_CAPACITY_PER_PAGE;
    let p = vec3<i32>(unpack(voxel_idx));
    let base = vec3<f32>(vec3<u32>(p));

    var write_face = face_offset[voxel_idx];
    for (var d = 0u; d < 6u; d = d + 1u) {
        let face_bit = 1u << d;
        if ((mask & face_bit) == 0u) {
            continue;
        }
        if (write_face >= total_faces) {
            break;
        }

        let local_vertex_offset = write_face * 4u;
        let local_index_offset = write_face * 6u;
        write_face_quad(
            d,
            base,
            id,
            vertex_base + local_vertex_offset,
            index_base + local_index_offset,
            local_vertex_offset,
        );
        write_face = write_face + 1u;
    }

    if (voxel_idx == 0u) {
        draw_indirect_buffer[page].index_count = total_faces * 6u;
        draw_indirect_buffer[page].instance_count = 1u;
        draw_indirect_buffer[page].first_index = index_base;
        draw_indirect_buffer[page].base_vertex = i32(vertex_base);
        draw_indirect_buffer[page].first_instance = 0u;
        mesh_meta_buffer[page] = ChunkMeshMeta(page, vertex_base, index_base, 0u);
    }
}
