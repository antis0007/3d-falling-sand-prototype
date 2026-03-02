const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const EMPTY: u32 = 0u;
const WATER: u32 = 5u;

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

@group(0) @binding(0) var<storage, read_write> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> velocity_mac: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> material_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_tiles: array<u32>;
@group(0) @binding(6) var<storage, read_write> active_tile_counter: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> frame_params: array<FrameParams>;

fn atlas_state_offset(page: u32, state: u32) -> u32 {
    return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let params = frame_params[0u];
    if (i == 0u) {
        atomicStore(&diagnostics[1u], 0u);
    }
    if (i >= params.frontier_len) { return; }

    let idx = active_tiles[i];
    if (idx >= params.voxel_count) { return; }

    let state = (params.state_index + 1u) & 1u;
    let off = atlas_state_offset(params.page_index, state);
    let prev_density = material_density[off + idx];
    let speed = length(velocity_mac[off + idx].xyz);
    let next_density = clamp(prev_density + speed * 0.01 - 0.005, 0.0, 1.0);
    material_density[off + idx] = next_density;

    let next_material = select(EMPTY, WATER, next_density > 0.2);
    if (atlas_voxels[off + idx] != next_material) {
        atlas_voxels[off + idx] = next_material;
        let out_idx = atomicAdd(&active_tile_counter[0u], 1u);
        if (out_idx < params.active_tile_budget) {
            active_tiles[out_idx] = idx;
        }
        atomicAdd(&diagnostics[1u], 1u);
    }
}
