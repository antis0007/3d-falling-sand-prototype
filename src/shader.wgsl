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
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;

    let world_pos = input.pos;
    let render_pos = world_pos - camera.world_origin_offset;

    out.position = camera.vp * vec4<f32>(render_pos, 1.0);
    out.color = input.color;
    out.world_pos = world_pos;

    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    let dx = dpdx(input.world_pos);
    let dy = dpdy(input.world_pos);
    let n = normalize(cross(dx, dy));

    // Slightly elevated sun direction for depth cues.
    let light_dir = normalize(vec3<f32>(0.45, 0.85, 0.30));
    let ndotl = max(dot(n, light_dir), 0.0);

    let ambient = 0.42;
    let diffuse = 0.58 * ndotl;
    let lit = clamp(ambient + diffuse, 0.0, 1.0);

    return vec4<f32>(input.color.rgb * lit, input.color.a);
}
