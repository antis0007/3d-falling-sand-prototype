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

struct EditCommand {
    voxel_index: u32,
    material_id: u32,
    flags: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> velocity_mac: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(4) var<storage, read_write> material_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(6) var<storage, read_write> active_tile_counter: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> edit_commands: array<EditCommand>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;

fn atlas_state_offset(page: u32, state: u32) -> u32 {
    return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (params.voxel_count == 0u) { return; }

    let src_state = params.state_index & 1u;
    let dst_state = (src_state + 1u) & 1u;
    let src_off = atlas_state_offset(params.page_index, src_state);
    let dst_off = atlas_state_offset(params.page_index, dst_state);

    if (i == 0u) {
        atomicStore(&active_tile_counter[0u], 0u);
    }

    if (i < params.voxel_count) {
        atlas_voxels[dst_off + i] = atlas_voxels[src_off + i];
        material_density[dst_off + i] = material_density[src_off + i];
    }

    if (i < params.edit_count) {
        let cmd = edit_commands[i];
        if (cmd.voxel_index < params.voxel_count) {
            atlas_voxels[src_off + cmd.voxel_index] = cmd.material_id;
            atlas_voxels[dst_off + cmd.voxel_index] = cmd.material_id;
            material_density[src_off + cmd.voxel_index] = select(0.0, 1.0, cmd.material_id == WATER);
            material_density[dst_off + cmd.voxel_index] = material_density[src_off + cmd.voxel_index];
        }
    }

    if (i >= params.frontier_len) { return; }
    let voxel_idx = active_tiles[i];
    if (voxel_idx >= params.voxel_count) { return; }

    let g = vec3<f32>(0.0, -0.15, 0.0);
    let prev_v = velocity_mac[src_off + voxel_idx].xyz;
    velocity_mac[dst_off + voxel_idx] = vec4<f32>(prev_v + g, 0.0);
    pressure[dst_off + voxel_idx] = pressure[src_off + voxel_idx];
    divergence[dst_off + voxel_idx] = 0.0;
}
