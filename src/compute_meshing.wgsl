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

struct PageParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    state_index: u32,
};

struct DrawIndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> input_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(2) var<storage, read> active_frontier: array<u32>;
@group(0) @binding(3) var<storage, read> page_params: array<PageParams>;
@group(0) @binding(4) var<storage, read_write> page_indirect: array<DrawIndirectArgs>;
@group(0) @binding(5) var<storage, read_write> diagnostics: array<atomic<u32>>;

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

fn is_light_fluid(v: u32) -> bool {
    return v == WATER || v == ACID || v == SMOKE || v == STEAM || v == FIRE_GAS;
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

fn lateral_xoff(i: u32, parity: u32) -> i32 {
    switch ((i + parity) & 1u) {
        case 0u: { return -1; }
        default: { return 1; }
    }
}
fn apply_rule(id: u32, idx: u32, src_off: u32) -> u32 {
    if (id == EMPTY) {
        return EMPTY;
    }

    let p = unpack(idx);

    if (id == LAVA || id == WATER) {
        for (var i = 0u; i < 6u; i = i + 1u) {
            let np = vec3<i32>(p) + neighbor_dir(i);
            if (!in_bounds(np)) {
                continue;
            }
            let nid = atlas_voxels[src_off + pack(vec3<u32>(np))];
            if ((id == LAVA && nid == WATER) || (id == WATER && nid == LAVA)) {
                if (id == LAVA) {
                    return STONE;
                }
                return STEAM;
            }
        }
    }

    if (id == SAND || id == SNOW || id == DEAD_LEAF || id == WATER || id == ACID || id == LAVA) {
        if (p.y > 0u) {
            let below_idx = idx - CHUNK_SIDE;
            let below = atlas_voxels[src_off + below_idx];
            if (below == EMPTY) {
                return EMPTY;
            }
            if ((id == SAND || id == SNOW || id == DEAD_LEAF) && is_light_fluid(below)) {
                return EMPTY;
            }
        }

        if (id == WATER || id == ACID || id == LAVA) {
            let parity = (idx ^ (idx >> 3u)) & 1u;
            for (var i = 0u; i < 2u; i = i + 1u) {
                let nx = i32(p.x) + lateral_xoff(i, parity);
                if (nx < 0 || nx >= i32(CHUNK_SIDE)) {
                    continue;
                }
                let nidx = pack(vec3<u32>(u32(nx), p.y, p.z));
                if (atlas_voxels[src_off + nidx] == EMPTY) {
                    return EMPTY;
                }
            }
        }
    }

    return id;
}

@compute @workgroup_size(64)
fn simulation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = page_params[0u];
    if (i >= params.frontier_len || params.voxel_count == 0u) {
        return;
    }

    let src_state = params.state_index & 1u;
    let dst_state = (src_state + 1u) & 1u;
    let src_off = atlas_state_offset(params.page_index, src_state);
    let dst_off = atlas_state_offset(params.page_index, dst_state);

    let voxel_idx = active_frontier[i];
    if (voxel_idx >= params.voxel_count) {
        return;
    }

    if (i < params.voxel_count) {
        atlas_voxels[dst_off + i] = atlas_voxels[src_off + i];
        if (src_state == 0u) {
            atlas_voxels[src_off + i] = input_voxels[i];
            atlas_voxels[dst_off + i] = input_voxels[i];
        }
    }

    let id = atlas_voxels[src_off + voxel_idx];
    let next = apply_rule(id, voxel_idx, src_off);
    atlas_voxels[dst_off + voxel_idx] = next;

    if (id != EMPTY && next == EMPTY) {
        let p = unpack(voxel_idx);
        if (p.y > 0u) {
            atlas_voxels[dst_off + (voxel_idx - CHUNK_SIDE)] = id;
        }
    }
}


fn voxel_at(base_off: u32, p: vec3<i32>) -> u32 {
    if (!in_bounds(p)) {
        return EMPTY;
    }
    return atlas_voxels[base_off + pack(vec3<u32>(p))];
}

@compute @workgroup_size(64)
fn meshing_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = page_params[0u];
    if (i == 0u) {
        atomicStore(&page_indirect[params.page_index].vertex_count, 0u);
        page_indirect[params.page_index].instance_count = 1u;
        page_indirect[params.page_index].first_vertex = 0u;
        page_indirect[params.page_index].first_instance = 0u;
        atomicStore(&diagnostics[0u], 0u);
    }

    if (i >= params.frontier_len) {
        return;
    }

    let src_off = atlas_state_offset(params.page_index, (params.state_index + 1u) & 1u);
    let voxel_idx = active_frontier[i];
    if (voxel_idx >= params.voxel_count) {
        return;
    }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) {
        return;
    }

    let p = vec3<i32>(unpack(voxel_idx));
    var faces: u32 = 0u;
    for (var d = 0u; d < 6u; d = d + 1u) {
        if (voxel_at(src_off, p + neighbor_dir(d)) == EMPTY) {
            faces = faces + 1u;
        }
    }
    if (faces > 0u) {
        atomicAdd(&page_indirect[params.page_index].vertex_count, faces * 6u);
        atomicAdd(&diagnostics[0u], faces);
    }
}
