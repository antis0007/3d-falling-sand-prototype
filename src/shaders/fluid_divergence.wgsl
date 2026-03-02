const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

struct FrameParams {
    page_index: u32,
    voxel_count: u32,
    frontier_len: u32,
    state_index: u32,
    edit_count: u32,
    active_tile_budget: u32,
    jacobi_iterations: u32,
    jacobi_iteration: u32,
};

@group(0) @binding(1) var<storage, read_write> velocity_mac: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;

fn atlas_state_offset(page: u32, state: u32) -> u32 {
    return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (i >= params.frontier_len) { return; }

    let idx = active_tiles[i];
    if (idx >= params.voxel_count) { return; }

    let state = (params.state_index + 1u) & 1u;
    let off = atlas_state_offset(params.page_index, state);
    let v = velocity_mac[off + idx].xyz;
    divergence[off + idx] = (v.x + v.y + v.z) * 0.3333;
    pressure[off + idx] = 0.0;
}
