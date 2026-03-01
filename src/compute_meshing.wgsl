struct ChunkParams {
    world_min: vec4<i32>,
};

struct DrawIndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> input_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> material_field: array<u32>;
@group(0) @binding(2) var<uniform> params: ChunkParams;

@compute @workgroup_size(64)
fn density_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_voxels)) {
        return;
    }
    let mat = input_voxels[idx];
    material_field[idx] = mat;
}

@group(0) @binding(0) var<storage, read> compact_source: array<u32>;
@group(0) @binding(1) var<storage, read_write> compacted_voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect_args: DrawIndirectArgs;

@compute @workgroup_size(64)
fn compact_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&compact_source)) {
        return;
    }
    if (idx == 0u) {
        atomicStore(&indirect_args.vertex_count, 0u);
        indirect_args.instance_count = 1u;
        indirect_args.first_vertex = 0u;
        indirect_args.first_instance = 0u;
    }
    let mat = compact_source[idx];
    if (mat == 0u) {
        return;
    }
    let out_idx = atomicAdd(&indirect_args.vertex_count, 6u) / 6u;
    if (out_idx < arrayLength(&compacted_voxels)) {
        compacted_voxels[out_idx] = idx;
    }
}
