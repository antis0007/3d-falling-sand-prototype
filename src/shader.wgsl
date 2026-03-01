struct Camera {
  vp: mat4x4<f32>,
  origin_offset: vec3<f32>,
  _pad: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(i: VsIn) -> VsOut {
  var o: VsOut;
  let render_pos = i.pos - camera.origin_offset;
  o.pos = camera.vp * vec4<f32>(render_pos, 1.0);
  o.color = i.color;
  return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
  return i.color;
}
