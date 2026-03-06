struct FluidSimParams {
    cell_count: u32,
    jacobi_iterations: u32,
    edit_count: u32,
    _pad: u32,
};

struct GpuEdit {
    index: u32,
    material: u32,
};

@group(0) @binding(0) var<storage, read_write> velocity_u: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(4) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> material_state: array<u32>;
@group(0) @binding(6) var<uniform> params: FluidSimParams;
@group(0) @binding(7) var<storage, read> edits: array<GpuEdit>;


fn in_range(i: u32) -> bool {
    return i < params.cell_count;
}

@compute @workgroup_size(64)
fn force_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (!in_range(i)) { return; }

    if (i < params.edit_count) {
        let e = edits[i];
        if (e.index < params.cell_count) {
            material_state[e.index] = e.material;
        }
    }

    let is_fluid = select(0.0, 1.0, material_state[i] != EMPTY);
    velocity_u[i] = clamp((velocity_u[i] + -0.02 * is_fluid) * 0.995, -1.0, 1.0);
    velocity_v[i] = clamp((velocity_v[i] + -0.08 * is_fluid) * 0.995, -1.0, 1.0);
    velocity_w[i] = clamp((velocity_w[i] + -0.02 * is_fluid) * 0.995, -1.0, 1.0);
}

@compute @workgroup_size(64)
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (!in_range(i)) { return; }
    divergence[i] = velocity_u[i] + velocity_v[i] + velocity_w[i];
}

@compute @workgroup_size(64)
fn jacobi_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (!in_range(i)) { return; }
    var p = pressure[i];
    for (var iter = 0u; iter < params.jacobi_iterations; iter = iter + 1u) {
        p = (p - divergence[i]) * 0.25;
    }
    pressure[i] = p;
}

@compute @workgroup_size(64)
fn project_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (!in_range(i)) { return; }
    let p = pressure[i];
    velocity_u[i] = velocity_u[i] - p;
    velocity_v[i] = velocity_v[i] - p;
    velocity_w[i] = velocity_w[i] - p;
}

@compute @workgroup_size(64)
fn advect_material(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (!in_range(i)) { return; }
    let speed = abs(velocity_u[i]) + abs(velocity_v[i]) + abs(velocity_w[i]);
    if (material_state[i] != EMPTY && speed < 0.0001) {
        material_state[i] = EMPTY;
    }
}
