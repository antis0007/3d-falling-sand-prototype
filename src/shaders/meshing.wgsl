const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const EMPTY: u32 = 0u;
const GPU_MESH_VERTEX_CAPACITY_PER_PAGE: u32 = CHUNK_VOLUME;
const GPU_MESH_INDEX_CAPACITY_PER_PAGE: u32 = CHUNK_VOLUME * 6u;

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

struct DrawIndirectArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct GpuVertex {
    position: vec3<f32>,
    material_id: u32,
};

struct ChunkMeshMeta {
    page_index: u32,
    vertex_offset: u32,
    index_offset: u32,
    _pad: u32,
};

struct DrawIndexedIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;
@group(0) @binding(9) var<storage, read_write> page_indirect: array<DrawIndirectArgs>;
@group(0) @binding(10) var<storage, read_write> dirty_page_indices: array<u32>;
@group(0) @binding(11) var<storage, read_write> dirty_page_counter: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> chunk_vertex_buffer: array<GpuVertex>;
@group(0) @binding(14) var<storage, read_write> chunk_index_buffer: array<u32>;
@group(0) @binding(15) var<storage, read_write> draw_indirect_buffer: array<DrawIndexedIndirectArgs>;
@group(0) @binding(16) var<storage, read_write> mesh_meta_buffer: array<ChunkMeshMeta>;
@group(0) @binding(17) var<storage, read_write> vertex_counter: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> index_counter: array<atomic<u32>>;

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

@compute @workgroup_size(64)
fn meshing_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];

    if (i >= params.frontier_len) { return; }
    let src_off = atlas_state_offset(params.page_index, (params.state_index + 1u) & 1u);
    let voxel_idx = active_tiles[i];
    if (voxel_idx >= params.voxel_count) { return; }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let p = vec3<i32>(unpack(voxel_idx));
    var faces: u32 = 0u;
    for (var d = 0u; d < 6u; d = d + 1u) {
        if (voxel_at(src_off, p + neighbor_dir(d)) == EMPTY) {
            faces = faces + 1u;
        }
    }
    if (faces == 0u) { return; }

    let page = params.page_index;
    let vertex_base = page * GPU_MESH_VERTEX_CAPACITY_PER_PAGE;
    let index_base = page * GPU_MESH_INDEX_CAPACITY_PER_PAGE;

    let local_vertex_index = atomicAdd(&vertex_counter[page], 1u);
    if (local_vertex_index >= GPU_MESH_VERTEX_CAPACITY_PER_PAGE) {
        return;
    }

    let local_index_offset = atomicAdd(&index_counter[page], 6u);
    if (local_index_offset + 5u >= GPU_MESH_INDEX_CAPACITY_PER_PAGE) {
        return;
    }

    let vtx = vertex_base + local_vertex_index;
    chunk_vertex_buffer[vtx] = GpuVertex(
        vec3<f32>(vec3<u32>(p)) + vec3<f32>(0.5, 0.5, 0.5),
        id,
    );

    for (var k = 0u; k < 6u; k = k + 1u) {
        chunk_index_buffer[index_base + local_index_offset + k] = local_vertex_index;
    }

    mesh_meta_buffer[page] = ChunkMeshMeta(page, vertex_base, index_base, 0u);

    let dirty_prev = atomicAdd(&diagnostics[0u], 1u);
    if (dirty_prev == 0u) {
        let dirty_idx = atomicAdd(&dirty_page_counter[0u], 1u);
        if (dirty_idx < arrayLength(&dirty_page_indices)) {
            dirty_page_indices[dirty_idx] = page;
        }
        page_indirect[page].instance_count = 1u;
    }

    draw_indirect_buffer[page].vertex_count = atomicLoad(&vertex_counter[page]);
    draw_indirect_buffer[page].instance_count = 1u;
    draw_indirect_buffer[page].first_vertex = 0u;
    draw_indirect_buffer[page].first_instance = 0u;

    atomicAdd(&page_indirect[page].index_count, 6u);
}
