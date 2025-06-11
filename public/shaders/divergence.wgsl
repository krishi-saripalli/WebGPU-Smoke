@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<min16float>;
@group(1) @binding(1) var divergenceOut: texture_storage_3d<min16float_storage, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let pos = vec3<i32>(id);

  let vp_x1 = textureLoad(velocityIn, pos + vec3<i32>(1,0,0), 0).xyz;
  let vn_x1 = textureLoad(velocityIn, pos - vec3<i32>(1,0,0), 0).xyz;
  let vp_y1 = textureLoad(velocityIn, pos + vec3<i32>(0,1,0), 0).xyz;
  let vn_y1 = textureLoad(velocityIn, pos - vec3<i32>(0,1,0), 0).xyz;
  let vp_z1 = textureLoad(velocityIn, pos + vec3<i32>(0,0,1), 0).xyz;
  let vn_z1 = textureLoad(velocityIn, pos - vec3<i32>(0,0,1), 0).xyz;

  let dv_dx = vp_x1.x - vn_x1.x;
  let dv_dy = vp_y1.y - vn_y1.y;
  let dv_dz = vp_z1.z - vn_z1.z;

  let div = (dv_dx + dv_dy + dv_dz) / (2.0 * params.dx);

  textureStore(divergenceOut, id, vec4f(div, 0.0, 0.0, 0.0));
} 