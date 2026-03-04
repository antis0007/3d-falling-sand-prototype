const CHUNK_SIDE: u32 = 32u;
const CHUNK_SLICE: u32 = CHUNK_SIDE * CHUNK_SIDE;
const CHUNK_VOLUME: u32 = CHUNK_SLICE * CHUNK_SIDE;
const INVALID_PAGE: u32 = 0xffffffffu;
const MAC_U_COUNT: u32 = (CHUNK_SIDE + 1u) * CHUNK_SIDE * CHUNK_SIDE;
const MAC_V_COUNT: u32 = CHUNK_SIDE * (CHUNK_SIDE + 1u) * CHUNK_SIDE;
const MAC_W_COUNT: u32 = CHUNK_SIDE * CHUNK_SIDE * (CHUNK_SIDE + 1u);
const MAC_TOTAL_COUNT: u32 = MAC_U_COUNT + MAC_V_COUNT + MAC_W_COUNT;

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
    cell_size: f32,
    max_velocity: f32,
    velocity_damping: f32,
    viscosity: f32,
    neighbor_pages: array<u32, 6>,
    _pad: array<u32, 2>,
};

@group(0) @binding(1) var<storage, read_write> velocity_mac: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;

fn scalar_state_offset(page: u32, state: u32) -> u32 { return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME; }
fn mac_state_offset(page: u32, state: u32) -> u32 { return page * (MAC_TOTAL_COUNT * 2u) + state * MAC_TOTAL_COUNT; }

fn unpack_coord(index: u32) -> vec3<u32> {
    let z = index / CHUNK_SLICE;
    let rem = index - z * CHUNK_SLICE;
    let y = rem / CHUNK_SIDE;
    let x = rem - y * CHUNK_SIDE;
    return vec3<u32>(x, y, z);
}

fn u_index(c: vec3<u32>) -> u32 { return c.x + c.y * (CHUNK_SIDE + 1u) + c.z * ((CHUNK_SIDE + 1u) * CHUNK_SIDE); }
fn v_index(c: vec3<u32>) -> u32 { return c.x + c.y * CHUNK_SIDE + c.z * (CHUNK_SIDE * (CHUNK_SIDE + 1u)); }
fn w_index(c: vec3<u32>) -> u32 { return c.x + c.y * CHUNK_SIDE + c.z * CHUNK_SLICE; }

fn remap_page(local: vec3<i32>, params: FrameParams) -> vec4<u32> {
    var p = local;
    var page = params.page_index;
    if (p.x < 0) { page = params.neighbor_pages[0u]; p.x = i32(CHUNK_SIDE); }
    else if (p.x > i32(CHUNK_SIDE)) { page = params.neighbor_pages[1u]; p.x = 0; }
    if (p.y < 0) { page = params.neighbor_pages[2u]; p.y = i32(CHUNK_SIDE); }
    else if (p.y > i32(CHUNK_SIDE)) { page = params.neighbor_pages[3u]; p.y = 0; }
    if (p.z < 0) { page = params.neighbor_pages[4u]; p.z = i32(CHUNK_SIDE); }
    else if (p.z > i32(CHUNK_SIDE)) { page = params.neighbor_pages[5u]; p.z = 0; }
    return vec4<u32>(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z), page);
}

fn sample_u(params: FrameParams, state: u32, c: vec3<i32>) -> f32 {
    let mapped = remap_page(c, params);
    if (mapped.w == INVALID_PAGE) { return 0.0; }
    let off = mac_state_offset(mapped.w, state);
    return velocity_mac[off + u_index(vec3<u32>(mapped.x, clamp(mapped.y, 0u, CHUNK_SIDE - 1u), clamp(mapped.z, 0u, CHUNK_SIDE - 1u)))];
}
fn sample_v(params: FrameParams, state: u32, c: vec3<i32>) -> f32 {
    let mapped = remap_page(c, params);
    if (mapped.w == INVALID_PAGE) { return 0.0; }
    let off = mac_state_offset(mapped.w, state) + MAC_U_COUNT;
    return velocity_mac[off + v_index(vec3<u32>(clamp(mapped.x, 0u, CHUNK_SIDE - 1u), mapped.y, clamp(mapped.z, 0u, CHUNK_SIDE - 1u)))];
}
fn sample_w(params: FrameParams, state: u32, c: vec3<i32>) -> f32 {
    let mapped = remap_page(c, params);
    if (mapped.w == INVALID_PAGE) { return 0.0; }
    let off = mac_state_offset(mapped.w, state) + MAC_U_COUNT + MAC_V_COUNT;
    return velocity_mac[off + w_index(vec3<u32>(clamp(mapped.x, 0u, CHUNK_SIDE - 1u), clamp(mapped.y, 0u, CHUNK_SIDE - 1u), mapped.z))];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (i >= params.frontier_len) { return; }
    let idx = active_tiles[i];
    if (idx >= params.voxel_count) { return; }

    let coord = unpack_coord(idx);
    let state = (params.state_index + 1u) & 1u;
    let div_off = scalar_state_offset(params.page_index, state);
    let h = max(params.cell_size, 1e-4);

    let u_r = sample_u(params, state, vec3<i32>(i32(coord.x) + 1, i32(coord.y), i32(coord.z)));
    let u_l = sample_u(params, state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z)));
    let v_t = sample_v(params, state, vec3<i32>(i32(coord.x), i32(coord.y) + 1, i32(coord.z)));
    let v_b = sample_v(params, state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z)));
    let w_u = sample_w(params, state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z) + 1));
    let w_d = sample_w(params, state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z)));

    divergence[div_off + idx] = (u_r - u_l + v_t - v_b + w_u - w_d) / h;
    pressure[div_off + idx] = 0.0;
}
