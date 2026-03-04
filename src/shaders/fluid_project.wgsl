const CHUNK_SIDE:u32=32u; const CHUNK_SLICE:u32=CHUNK_SIDE*CHUNK_SIDE; const CHUNK_VOLUME:u32=CHUNK_SLICE*CHUNK_SIDE;
const MAC_U_COUNT:u32=(CHUNK_SIDE+1u)*CHUNK_SIDE*CHUNK_SIDE; const MAC_V_COUNT:u32=CHUNK_SIDE*(CHUNK_SIDE+1u)*CHUNK_SIDE; const MAC_W_COUNT:u32=CHUNK_SIDE*CHUNK_SIDE*(CHUNK_SIDE+1u); const MAC_TOTAL_COUNT:u32=MAC_U_COUNT+MAC_V_COUNT+MAC_W_COUNT;
struct FrameParams { page_index:u32, voxel_count:u32, frontier_len:u32, simulation_tick:u32, state_index:u32, edit_count:u32, active_tile_budget:u32, jacobi_iterations:u32, jacobi_iteration:u32, cell_size:f32, max_velocity:f32, velocity_damping:f32, viscosity:f32, neighbor_pages:array<u32,6>, _pad:array<u32,2>, };
@group(0) @binding(1) var<storage, read_write> velocity_mac: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;
fn scalar_state_offset(page:u32,state:u32)->u32{ return page*(CHUNK_VOLUME*2u)+state*CHUNK_VOLUME; }
fn mac_state_offset(page:u32,state:u32)->u32{ return page*(MAC_TOTAL_COUNT*2u)+state*MAC_TOTAL_COUNT; }
fn unpack_coord(index:u32)->vec3<u32>{ let z=index/CHUNK_SLICE; let r=index-z*CHUNK_SLICE; let y=r/CHUNK_SIDE; return vec3<u32>(r-y*CHUNK_SIDE,y,z);} 
fn u_index(c:vec3<u32>)->u32{ return c.x + c.y*(CHUNK_SIDE+1u) + c.z*((CHUNK_SIDE+1u)*CHUNK_SIDE);} fn v_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*(CHUNK_SIDE*(CHUNK_SIDE+1u)); } fn w_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*CHUNK_SLICE; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x; let params=frame_params[0u]; if(i>=params.frontier_len){return;} let idx=active_tiles[i]; if(idx>=params.voxel_count){return;}
 let coord=unpack_coord(idx); let vel_state=(params.state_index+1u)&1u; let p_state=(params.state_index + (params.jacobi_iterations & 1u)) & 1u;
 let p_off=scalar_state_offset(params.page_index,p_state); let v_off=mac_state_offset(params.page_index,vel_state); let h=max(params.cell_size,1e-4);
 let p_c=pressure[p_off+idx];
 if(coord.x < CHUNK_SIDE){ let p_r = select(p_c, pressure[p_off+idx+1u], coord.x + 1u < CHUNK_SIDE); let ui = v_off + u_index(vec3<u32>(coord.x+1u,coord.y,coord.z)); velocity_mac[ui] = velocity_mac[ui] - (p_r - p_c)/h; }
 if(coord.y < CHUNK_SIDE){ let p_t = select(p_c, pressure[p_off+idx+CHUNK_SIDE], coord.y + 1u < CHUNK_SIDE); let vi = v_off + MAC_U_COUNT + v_index(vec3<u32>(coord.x,coord.y+1u,coord.z)); velocity_mac[vi] = velocity_mac[vi] - (p_t - p_c)/h; }
 if(coord.z < CHUNK_SIDE){ let p_u = select(p_c, pressure[p_off+idx+CHUNK_SLICE], coord.z + 1u < CHUNK_SIDE); let wi = v_off + MAC_U_COUNT + MAC_V_COUNT + w_index(vec3<u32>(coord.x,coord.y,coord.z+1u)); velocity_mac[wi] = velocity_mac[wi] - (p_u - p_c)/h; }
}
