const CHUNK_SIDE:u32=32u; const CHUNK_SLICE:u32=CHUNK_SIDE*CHUNK_SIDE; const CHUNK_VOLUME:u32=CHUNK_SLICE*CHUNK_SIDE;
const MAC_U_COUNT:u32=(CHUNK_SIDE+1u)*CHUNK_SIDE*CHUNK_SIDE; const MAC_V_COUNT:u32=CHUNK_SIDE*(CHUNK_SIDE+1u)*CHUNK_SIDE; const MAC_W_COUNT:u32=CHUNK_SIDE*CHUNK_SIDE*(CHUNK_SIDE+1u); const MAC_TOTAL_COUNT:u32=MAC_U_COUNT+MAC_V_COUNT+MAC_W_COUNT;
struct FrameParams { page_index:u32, voxel_count:u32, frontier_len:u32, simulation_tick:u32, state_index:u32, edit_count:u32, active_tile_budget:u32, jacobi_iterations:u32, jacobi_iteration:u32, cell_size:f32, max_velocity:f32, velocity_damping:f32, viscosity:f32, neighbor_pages:array<u32,6>, _pad:array<u32,2>, };
struct EditCommand { voxel_index:u32, material_id:u32, flags:u32, _pad:u32, };
@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> velocity_mac: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(4) var<storage, read_write> material_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(6) var<storage, read_write> active_tile_counter: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> edit_commands: array<EditCommand>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;
fn scalar_state_offset(page:u32,state:u32)->u32{return page*(CHUNK_VOLUME*2u)+state*CHUNK_VOLUME;} fn mac_state_offset(page:u32,state:u32)->u32{return page*(MAC_TOTAL_COUNT*2u)+state*MAC_TOTAL_COUNT;}
fn unpack_coord(index:u32)->vec3<u32>{ let z=index/CHUNK_SLICE; let r=index-z*CHUNK_SLICE; let y=r/CHUNK_SIDE; return vec3<u32>(r-y*CHUNK_SIDE,y,z);} 
fn u_index(c:vec3<u32>)->u32{ return c.x + c.y*(CHUNK_SIDE+1u) + c.z*((CHUNK_SIDE+1u)*CHUNK_SIDE);} fn v_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*(CHUNK_SIDE*(CHUNK_SIDE+1u)); } fn w_index(c:vec3<u32>)->u32{ return c.x + c.y*CHUNK_SIDE + c.z*CHUNK_SLICE; }
fn damp_and_clamp(v:f32, params:FrameParams)->f32{ return clamp(v * params.velocity_damping, -params.max_velocity, params.max_velocity); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x; let params=frame_params[0u]; if(params.voxel_count==0u){return;} let src_state=params.state_index&1u; let dst_state=(src_state+1u)&1u;
 let src_off=scalar_state_offset(params.page_index,src_state); let dst_off=scalar_state_offset(params.page_index,dst_state);
 if(i==0u){ atomicStore(&active_tile_counter[0u],0u); }
 if(i<params.voxel_count){ atlas_voxels[dst_off+i]=atlas_voxels[src_off+i]; material_density[dst_off+i]=material_density[src_off+i]; pressure[dst_off+i]=0.0; divergence[dst_off+i]=0.0; }
 if(i<params.edit_count){ let cmd=edit_commands[i]; if(cmd.voxel_index<params.voxel_count){ atlas_voxels[src_off+cmd.voxel_index]=cmd.material_id; atlas_voxels[dst_off+cmd.voxel_index]=cmd.material_id; material_density[src_off+cmd.voxel_index]=select(0.0,1.0,cmd.material_id==WATER); material_density[dst_off+cmd.voxel_index]=material_density[src_off+cmd.voxel_index]; }}
 if(i>=params.frontier_len){ return; }
 let voxel_idx=active_tiles[i]; if(voxel_idx>=params.voxel_count){return;} let c=unpack_coord(voxel_idx);
 let vsrc=mac_state_offset(params.page_index,src_state); let vdst=mac_state_offset(params.page_index,dst_state);
 let g=-0.15;
 let ui=vdst+u_index(vec3<u32>(c.x,c.y,c.z)); velocity_mac[ui]=damp_and_clamp(velocity_mac[vsrc+u_index(vec3<u32>(c.x,c.y,c.z))]*(1.0-params.viscosity)+g*0.25,params);
 let vi=vdst+MAC_U_COUNT+v_index(vec3<u32>(c.x,c.y,c.z)); velocity_mac[vi]=damp_and_clamp(velocity_mac[vsrc+MAC_U_COUNT+v_index(vec3<u32>(c.x,c.y,c.z))]*(1.0-params.viscosity)+g,params);
 let wi=vdst+MAC_U_COUNT+MAC_V_COUNT+w_index(vec3<u32>(c.x,c.y,c.z)); velocity_mac[wi]=damp_and_clamp(velocity_mac[vsrc+MAC_U_COUNT+MAC_V_COUNT+w_index(vec3<u32>(c.x,c.y,c.z))]*(1.0-params.viscosity)+g*0.25,params);
}
