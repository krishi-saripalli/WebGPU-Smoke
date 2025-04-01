// Import common definitions
@import "common.wgsl";

// --- Resource declarations for vorticity calculation ---
@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // vorticity = curl(velocity) (Eq. 9)
  let vp_y1 = textureLoad(velocityIn, pos + vec3<i32>(0,1,0), 0).xyz;
  let vn_y1 = textureLoad(velocityIn, pos - vec3<i32>(0,1,0), 0).xyz;
  let vp_z1 = textureLoad(velocityIn, pos + vec3<i32>(0,0,1), 0).xyz;
  let vn_z1 = textureLoad(velocityIn, pos - vec3<i32>(0,0,1), 0).xyz;
  let vp_x1 = textureLoad(velocityIn, pos + vec3<i32>(1,0,0), 0).xyz;
  let vn_x1 = textureLoad(velocityIn, pos - vec3<i32>(1,0,0), 0).xyz;

  let dvz_dy = vp_y1.z - vn_y1.z;
  let dvy_dz = vp_z1.y - vn_z1.y;
  let dvx_dz = vp_z1.x - vn_z1.x;
  let dvz_dx = vp_x1.z - vn_x1.z;
  let dvy_dx = vp_x1.y - vn_x1.y;
  let dvx_dy = vp_y1.x - vn_y1.x;

  let vort = vec3<f32>(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy) / (2.0 * params.dx);

  textureStore(vorticityOut, id, vec4f(vort, 0.0));
} 