struct Camera {
    vp: mat4x4<f32>,
    world_origin_offset: vec3<f32>,
    padding: f32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;

    let world_pos = input.pos;
    let render_pos = world_pos - camera.world_origin_offset;

    out.position = camera.vp * vec4<f32>(render_pos, 1.0);
    out.color = input.color;

    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    return input.color;
}