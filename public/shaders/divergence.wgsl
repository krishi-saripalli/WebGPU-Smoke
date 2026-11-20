@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var divergenceOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let pos = vec3i(id) + 1;
  let velocity = textureLoad(velocityIn, pos, 0).xyz;
  let dv_dx = velocity.x - textureLoad(velocityIn, pos - X, 0).x;
  let dv_dy = velocity.y - textureLoad(velocityIn, pos - Y, 0).y;
  let dv_dz = velocity.z - textureLoad(velocityIn, pos - Z, 0).z;

  let div = (dv_dx + dv_dy + dv_dz) / params.dx;
  textureStore(divergenceOut, pos, vec4f(div, 0.0, 0.0, 0.0));
}
