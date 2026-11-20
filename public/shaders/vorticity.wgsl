// Import common definitions
@import "common.wgsl";

// --- Resource declarations for vorticity calculation ---
@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityOut: texture_storage_3d<rgba16float, write>;

fn centerVelocity(p: vec3i) -> vec3f {
  let c = cell(p);
  let incoming = vec3f(textureLoad(velocityIn, c-X, 0).x,
                       textureLoad(velocityIn, c-Y, 0).y,
                       textureLoad(velocityIn, c-Z, 0).z);
  return 0.5 * (textureLoad(velocityIn, c, 0).xyz + incoming);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let pos = vec3i(id) + 1;

  // vorticity = curl(velocity) (Eq. 9)
  let vp_y1 = centerVelocity(pos + Y);
  let vn_y1 = centerVelocity(pos - Y);
  let vp_z1 = centerVelocity(pos + Z);
  let vn_z1 = centerVelocity(pos - Z);
  let vp_x1 = centerVelocity(pos + X);
  let vn_x1 = centerVelocity(pos - X);

  let dvz_dy = vp_y1.z - vn_y1.z;
  let dvy_dz = vp_z1.y - vn_z1.y;
  let dvx_dz = vp_z1.x - vn_z1.x;
  let dvz_dx = vp_x1.z - vn_x1.z;
  let dvy_dx = vp_x1.y - vn_x1.y;
  let dvx_dy = vp_y1.x - vn_y1.x;

  let vorticity = vec3f(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy) / (2.0 * params.dx);

  textureStore(vorticityOut, pos, vec4f(vorticity, length(vorticity)));
}
