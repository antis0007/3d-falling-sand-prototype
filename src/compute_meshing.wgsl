struct PageParams {
    page_index: u32,
    voxel_count: u32,
    _pad0: u32,
    _pad1: u32,
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

@compute @workgroup_size(64)
fn simulation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let params = page_params[0u];
    if (idx >= params.voxel_count || active_frontier[0u] != params.page_index) {
        return;
    }

    let dst = params.page_index * params.voxel_count + idx;
    atlas_voxels[dst] = input_voxels[idx];
}

@group(0) @binding(0) var<storage, read> meshing_atlas: array<u32>;
@group(0) @binding(1) var<storage, read> meshing_frontier: array<u32>;
@group(0) @binding(2) var<storage, read_write> page_indirect: array<DrawIndirectArgs>;
@group(0) @binding(3) var<storage, read_write> diagnostics: array<u32>;

@compute @workgroup_size(64)
fn meshing_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let page = meshing_frontier[0u];
    if (idx == 0u) {
        atomicStore(&page_indirect[page].vertex_count, 0u);
        page_indirect[page].instance_count = 1u;
        page_indirect[page].first_vertex = 0u;
        page_indirect[page].first_instance = 0u;
        diagnostics[0u] = 1u;
    }

    if (idx >= arrayLength(&meshing_atlas)) {
        return;
    }

    if (meshing_atlas[idx] != 0u) {
        atomicAdd(&page_indirect[page].vertex_count, 6u);
    }
}
