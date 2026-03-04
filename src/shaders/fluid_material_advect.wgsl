const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const CHUNK_SLICE: u32 = CHUNK_SIDE * CHUNK_SIDE;
const DT: f32 = 0.1;
const DENSITY_EPSILON: f32 = 1e-4;
const DELTA_EPSILON: f32 = 1e-4;
const STATIC_VELOCITY_EPSILON: f32 = 2.5e-3;
const STATIC_GRADIENT_EPSILON: f32 = 2.5e-3;
const MAC_U_COUNT:u32=(CHUNK_SIDE+1u)*CHUNK_SIDE*CHUNK_SIDE; const MAC_V_COUNT:u32=CHUNK_SIDE*(CHUNK_SIDE+1u)*CHUNK_SIDE; const MAC_W_COUNT:u32=CHUNK_SIDE*CHUNK_SIDE*(CHUNK_SIDE+1u); const MAC_TOTAL_COUNT:u32=MAC_U_COUNT+MAC_V_COUNT+MAC_W_COUNT;

struct FrameParams { page_index:u32, voxel_count:u32, frontier_len:u32, simulation_tick:u32, state_index:u32, edit_count:u32, active_tile_budget:u32, jacobi_iterations:u32, jacobi_iteration:u32, cell_size:f32, max_velocity:f32, velocity_damping:f32, viscosity:f32, neighbor_pages:array<u32,6>, _pad:array<u32,2>, };
@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> velocity_mac: array<f32>;
@group(0) @binding(4) var<storage, read_write> material_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(6) var<storage, read_write> active_tile_counter: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;

fn atlas_state_offset(page:u32,state:u32)->u32{return page*(CHUNK_VOLUME*2u)+state*CHUNK_VOLUME;} fn mac_state_offset(page:u32,state:u32)->u32{return page*(MAC_TOTAL_COUNT*2u)+state*MAC_TOTAL_COUNT;}
fn unpack_coord(index:u32)->vec3<u32>{let z=index/CHUNK_SLICE;let rem=index-z*CHUNK_SLICE;let y=rem/CHUNK_SIDE;return vec3<u32>(rem-y*CHUNK_SIDE,y,z);} fn pack_coord(p:vec3<u32>)->u32{return p.x+p.y*CHUNK_SIDE+p.z*CHUNK_SLICE;}
fn in_bounds(p: vec3<i32>) -> bool { return all(p >= vec3<i32>(0)) && all(p < vec3<i32>(i32(CHUNK_SIDE))); }
fn u_index(c:vec3<u32>)->u32{ return c.x + c.y*(CHUNK_SIDE+1u) + c.z*((CHUNK_SIDE+1u)*CHUNK_SIDE);} fn v_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*(CHUNK_SIDE*(CHUNK_SIDE+1u)); } fn w_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*CHUNK_SLICE; }
fn sample_density(off: u32, p: vec3<i32>, self_density: f32) -> f32 { if (!in_bounds(p)) { return self_density; } return material_density[off + pack_coord(vec3<u32>(p))]; }
fn is_transport_open(material_id: u32) -> bool { return material_id == EMPTY || material_id == WATER; }
fn transport_coeff(material_id: u32) -> f32 { if (material_id == WATER) { return 1.0; } return 0.0; }
fn sample_center_velocity(v_off:u32, c:vec3<u32>) -> vec3<f32>{
 let ux = 0.5*(velocity_mac[v_off+u_index(vec3<u32>(c.x,c.y,c.z))] + velocity_mac[v_off+u_index(vec3<u32>(c.x+1u,c.y,c.z))]);
 let vy = 0.5*(velocity_mac[v_off+MAC_U_COUNT+v_index(vec3<u32>(c.x,c.y,c.z))] + velocity_mac[v_off+MAC_U_COUNT+v_index(vec3<u32>(c.x,c.y+1u,c.z))]);
 let wz = 0.5*(velocity_mac[v_off+MAC_U_COUNT+MAC_V_COUNT+w_index(vec3<u32>(c.x,c.y,c.z))] + velocity_mac[v_off+MAC_U_COUNT+MAC_V_COUNT+w_index(vec3<u32>(c.x,c.y,c.z+1u))]);
 return vec3<f32>(ux,vy,wz);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (i == 0u) { atomicStore(&diagnostics[1u], 0u); atomicStore(&diagnostics[2u], 0u); atomicStore(&diagnostics[3u], 0u); }
    if (i >= params.frontier_len) { return; }
    let idx = active_tiles[i];
    if (idx >= params.voxel_count) { atomicAdd(&diagnostics[2u], 1u); return; }
    let state = (params.state_index + 1u) & 1u;
    let off = atlas_state_offset(params.page_index, state);
    let v_off = mac_state_offset(params.page_index, state);
    let coord = unpack_coord(idx);
    let prev_density = material_density[off + idx];
    let self_material = atlas_voxels[off + idx];
    let self_velocity = sample_center_velocity(v_off, coord);
    let grad_x = sample_density(off, vec3<i32>(coord) + vec3<i32>(1, 0, 0), prev_density) - sample_density(off, vec3<i32>(coord) + vec3<i32>(-1, 0, 0), prev_density);
    let grad_y = sample_density(off, vec3<i32>(coord) + vec3<i32>(0, 1, 0), prev_density) - sample_density(off, vec3<i32>(coord) + vec3<i32>(0, -1, 0), prev_density);
    let grad_z = sample_density(off, vec3<i32>(coord) + vec3<i32>(0, 0, 1), prev_density) - sample_density(off, vec3<i32>(coord) + vec3<i32>(0, 0, -1), prev_density);
    let gradient_mag = length(vec3<f32>(grad_x, grad_y, grad_z)) * 0.5;
    if (length(self_velocity) < STATIC_VELOCITY_EPSILON && gradient_mag < STATIC_GRADIENT_EPSILON) { return; }
    var outflow = 0.0; var inflow = 0.0;
    for (var d = 0u; d < 6u; d = d + 1u) {
        var dir = vec3<i32>(0);
        if (d == 0u) { dir = vec3<i32>(1, 0, 0); } if (d == 1u) { dir = vec3<i32>(-1, 0, 0); } if (d == 2u) { dir = vec3<i32>(0, 1, 0); } if (d == 3u) { dir = vec3<i32>(0, -1, 0); } if (d == 4u) { dir = vec3<i32>(0, 0, 1); } if (d == 5u) { dir = vec3<i32>(0, 0, -1); }
        let neighbor_coord = vec3<i32>(coord) + dir;
        if (!in_bounds(neighbor_coord)) { atomicAdd(&diagnostics[2u], 1u); continue; }
        let neighbor_idx = pack_coord(vec3<u32>(neighbor_coord)); let neighbor_material = atlas_voxels[off + neighbor_idx];
        if (!is_transport_open(neighbor_material) || !is_transport_open(self_material)) { continue; }
        let neighbor_velocity = sample_center_velocity(v_off, vec3<u32>(neighbor_coord));
        let face_speed = dot((self_velocity + neighbor_velocity) * 0.5, vec3<f32>(dir)); let neighbor_density = material_density[off + neighbor_idx];
        if (face_speed > 0.0) { outflow = outflow + face_speed * transport_coeff(self_material) * prev_density * DT; }
        else if (face_speed < 0.0) { inflow = inflow + (-face_speed) * transport_coeff(neighbor_material) * neighbor_density * DT; }
    }
    let bounded_outflow = min(outflow, prev_density); let next_density = max(0.0, prev_density - bounded_outflow + inflow); let delta = next_density - prev_density;
    material_density[off + idx] = next_density;
    let next_material = select(EMPTY, WATER, next_density > DENSITY_EPSILON); let material_changed = atlas_voxels[off + idx] != next_material; if (material_changed) { atlas_voxels[off + idx] = next_material; }
    if (material_changed || abs(delta) > DELTA_EPSILON) { let out_idx = atomicAdd(&active_tile_counter[0u], 1u); if (out_idx < params.active_tile_budget) { active_tiles[out_idx] = idx; } else { atomicAdd(&diagnostics[3u], 1u); } atomicAdd(&diagnostics[1u], 1u); }
}
