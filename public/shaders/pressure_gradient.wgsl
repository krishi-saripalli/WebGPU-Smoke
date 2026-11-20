@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var pressureIn: texture_3d<f32>;
@group(1) @binding(2) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let pos = vec3i(id) + 1;
  let pressure = textureLoad(pressureIn, pos, 0).x;
  let dp_dx = (textureLoad(pressureIn, pos + X, 0).x - pressure) / params.dx;
  let dp_dy = (textureLoad(pressureIn, pos + Y, 0).x - pressure) / params.dx;
  let dp_dz = (textureLoad(pressureIn, pos + Z, 0).x - pressure) / params.dx;

  // subtract pressure gradient to make velocity divergence-free (equation 5)
  let velocity = textureLoad(velocityIn, pos, 0).xyz;
  let pressureGradient = vec3f(dp_dx, dp_dy, dp_dz);
  textureStore(velocityOut, pos, vec4f(wallVelocity(pos, velocity - pressureGradient), 0));
}
