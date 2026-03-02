const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const EMPTY: u32 = 0u;
const STONE: u32 = 1u;
const SAND: u32 = 3u;
const SNOW: u32 = 4u;
const WATER: u32 = 5u;
const LAVA: u32 = 6u;
const ACID: u32 = 7u;
const SMOKE: u32 = 8u;
const STEAM: u32 = 9u;
const FIRE_GAS: u32 = 11u;
const DEAD_LEAF: u32 = 24u;

struct FrameParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
    active_tile_budget: u32,
    camera_region_meta0: u32,
    camera_region_meta1: u32,
};

struct EditCommand {
    voxel_index: u32,
    material_id: u32,
    flags: u32,
    _pad: u32,
};

struct DrawIndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(4) var<storage, read_write> active_tile_counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> edit_commands: array<EditCommand>;
@group(0) @binding(6) var<storage, read> frame_params: array<FrameParams>;
@group(0) @binding(7) var<storage, read_write> page_indirect: array<DrawIndirectArgs>;
@group(0) @binding(8) var<storage, read_write> dirty_chunk_ids: array<u32>;
@group(0) @binding(9) var<storage, read_write> dirty_chunk_counter: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> diagnostics: array<atomic<u32>>;

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
fn is_light_fluid(v: u32) -> bool { return v == WATER || v == ACID || v == SMOKE || v == STEAM || v == FIRE_GAS; }
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
fn lateral_xoff(i: u32, parity: u32) -> i32 { if (((i + parity) & 1u) == 0u) { return -1; } return 1; }

fn apply_rule(id: u32, idx: u32, src_off: u32) -> u32 {
    if (id == EMPTY) { return EMPTY; }
    let p = unpack(idx);
    if (id == LAVA || id == WATER) {
        for (var i = 0u; i < 6u; i = i + 1u) {
            let np = vec3<i32>(p) + neighbor_dir(i);
            if (!in_bounds(np)) { continue; }
            let nid = atlas_voxels[src_off + pack(vec3<u32>(np))];
            if ((id == LAVA && nid == WATER) || (id == WATER && nid == LAVA)) {
                if (id == LAVA) { return STONE; }
                return STEAM;
            }
        }
    }
    if (id == SAND || id == SNOW || id == DEAD_LEAF || id == WATER || id == ACID || id == LAVA) {
        if (p.y > 0u) {
            let below_idx = idx - CHUNK_SIDE;
            let below = atlas_voxels[src_off + below_idx];
            if (below == EMPTY || ((id == SAND || id == SNOW || id == DEAD_LEAF) && is_light_fluid(below))) {
                return EMPTY;
            }
        }
        if (id == WATER || id == ACID || id == LAVA) {
            let parity = (idx ^ (idx >> 3u)) & 1u;
            for (var i = 0u; i < 2u; i = i + 1u) {
                let nx = i32(p.x) + lateral_xoff(i, parity);
                if (nx < 0 || nx >= i32(CHUNK_SIDE)) { continue; }
                let nidx = pack(vec3<u32>(u32(nx), p.y, p.z));
                if (atlas_voxels[src_off + nidx] == EMPTY) { return EMPTY; }
            }
        }
    }
    return id;
}

fn voxel_at(base_off: u32, p: vec3<i32>) -> u32 {
    if (!in_bounds(p)) { return EMPTY; }
    return atlas_voxels[base_off + pack(vec3<u32>(p))];
}

@compute @workgroup_size(64)
fn simulation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (params.voxel_count == 0u) { return; }

    let src_state = params.state_index & 1u;
    let dst_state = (src_state + 1u) & 1u;
    let src_off = atlas_state_offset(params.page_index, src_state);
    let dst_off = atlas_state_offset(params.page_index, dst_state);

    if (i == 0u) {
        atomicStore(&active_tile_counter[0u], 0u);
        atomicStore(&diagnostics[0u], 0u);
        atomicStore(&diagnostics[1u], 0u);
    }

    if (i < params.voxel_count) {
        atlas_voxels[dst_off + i] = atlas_voxels[src_off + i];
    }

    if (i < params.edit_count) {
        let cmd = edit_commands[i];
        if (cmd.voxel_index < params.voxel_count) {
            atlas_voxels[src_off + cmd.voxel_index] = cmd.material_id;
            atlas_voxels[dst_off + cmd.voxel_index] = cmd.material_id;
        }
    }

    if (i >= params.frontier_len) { return; }
    let voxel_idx = active_tiles[i];
    if (voxel_idx >= params.voxel_count) { return; }

    // placeholder in-place fluid steps on persistent resources
    pressure[src_off + voxel_idx] = pressure[src_off + voxel_idx] * 0.98;
    velocity[src_off + voxel_idx] = velocity[src_off + voxel_idx] * vec4<f32>(0.99, 0.99, 0.99, 1.0);

    let id = atlas_voxels[src_off + voxel_idx];
    let next = apply_rule(id, voxel_idx, src_off);
    atlas_voxels[dst_off + voxel_idx] = next;
    if (id != next) {
        let out_idx = atomicAdd(&active_tile_counter[0u], 1u);
        if (out_idx < params.active_tile_budget) {
            active_tiles[out_idx] = voxel_idx;
        }
        atomicAdd(&diagnostics[1u], 1u);
    }
}

@compute @workgroup_size(64)
fn meshing_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (i == 0u) {
        atomicStore(&page_indirect[params.page_index].vertex_count, 0u);
        page_indirect[params.page_index].instance_count = 1u;
        page_indirect[params.page_index].first_vertex = 0u;
        page_indirect[params.page_index].first_instance = 0u;
        atomicStore(&diagnostics[0u], 0u);
    }

    if (i >= params.frontier_len) { return; }
    let src_off = atlas_state_offset(params.page_index, (params.state_index + 1u) & 1u);
    let voxel_idx = active_tiles[i];
    if (voxel_idx >= params.voxel_count) { return; }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let p = vec3<i32>(unpack(voxel_idx));
    var faces: u32 = 0u;
    for (var d = 0u; d < 6u; d = d + 1u) {
        if (voxel_at(src_off, p + neighbor_dir(d)) == EMPTY) { faces = faces + 1u; }
    }
    if (faces > 0u) {
        atomicAdd(&page_indirect[params.page_index].vertex_count, faces * 6u);
        atomicAdd(&diagnostics[0u], faces);
    }

    if (i == 0u && atomicLoad(&diagnostics[1u]) > 0u) {
        let dirty_idx = atomicAdd(&dirty_chunk_counter[0u], 1u);
        dirty_chunk_ids[dirty_idx] = params.page_index;
    }
}
