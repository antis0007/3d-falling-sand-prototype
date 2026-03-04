const CHUNK_SIDE: u32 = 32u;
const CHUNK_SLICE: u32 = CHUNK_SIDE * CHUNK_SIDE;
const CHUNK_VOLUME: u32 = CHUNK_SLICE * CHUNK_SIDE;
const INVALID_PAGE: u32 = 0xffffffffu;

struct FrameParams {
    page_index: u32, voxel_count: u32, frontier_len: u32, simulation_tick: u32,
    state_index: u32, edit_count: u32, active_tile_budget: u32, jacobi_iterations: u32,
    jacobi_iteration: u32, cell_size: f32, max_velocity: f32, velocity_damping: f32, viscosity: f32,
    neighbor_pages: array<u32, 6>, _pad: array<u32, 2>,
};
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;
fn scalar_state_offset(page: u32, state: u32) -> u32 { return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME; }
fn unpack_coord(index: u32) -> vec3<u32> { let z=index/CHUNK_SLICE; let r=index-z*CHUNK_SLICE; let y=r/CHUNK_SIDE; return vec3<u32>(r-y*CHUNK_SIDE,y,z); }
fn pack_coord(c: vec3<u32>) -> u32 { return c.x + c.y * CHUNK_SIDE + c.z * CHUNK_SLICE; }
fn sample_p(params: FrameParams, src_state: u32, c: vec3<i32>) -> f32 {
    var cc = c; var page = params.page_index;
    if (cc.x < 0) { page = params.neighbor_pages[0u]; cc.x = i32(CHUNK_SIDE) - 1; }
    else if (cc.x >= i32(CHUNK_SIDE)) { page = params.neighbor_pages[1u]; cc.x = 0; }
    if (cc.y < 0) { page = params.neighbor_pages[2u]; cc.y = i32(CHUNK_SIDE) - 1; }
    else if (cc.y >= i32(CHUNK_SIDE)) { page = params.neighbor_pages[3u]; cc.y = 0; }
    if (cc.z < 0) { page = params.neighbor_pages[4u]; cc.z = i32(CHUNK_SIDE) - 1; }
    else if (cc.z >= i32(CHUNK_SIDE)) { page = params.neighbor_pages[5u]; cc.z = 0; }
    if (page == INVALID_PAGE) { return 0.0; }
    return pressure[scalar_state_offset(page, src_state) + pack_coord(vec3<u32>(cc))];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x; let params=frame_params[0u]; if(i>=params.frontier_len){return;} let idx=active_tiles[i]; if(idx>=params.voxel_count){return;}
    let coord = unpack_coord(idx);
    let src_state = (params.state_index + (params.jacobi_iteration & 1u)) & 1u;
    let dst_state = (src_state + 1u) & 1u;
    let div_off = scalar_state_offset(params.page_index, (params.state_index + 1u) & 1u);
    let h2 = params.cell_size * params.cell_size;
    let p_l = sample_p(params, src_state, vec3<i32>(i32(coord.x)-1, i32(coord.y), i32(coord.z)));
    let p_r = sample_p(params, src_state, vec3<i32>(i32(coord.x)+1, i32(coord.y), i32(coord.z)));
    let p_b = sample_p(params, src_state, vec3<i32>(i32(coord.x), i32(coord.y)-1, i32(coord.z)));
    let p_t = sample_p(params, src_state, vec3<i32>(i32(coord.x), i32(coord.y)+1, i32(coord.z)));
    let p_d = sample_p(params, src_state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z)-1));
    let p_u = sample_p(params, src_state, vec3<i32>(i32(coord.x), i32(coord.y), i32(coord.z)+1));
    pressure[scalar_state_offset(params.page_index, dst_state) + idx] = (p_l + p_r + p_b + p_t + p_d + p_u - h2 * divergence[div_off + idx]) / 6.0;
}
