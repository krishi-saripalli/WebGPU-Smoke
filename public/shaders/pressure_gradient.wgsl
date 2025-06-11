@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<min16float>;
@group(1) @binding(1) var pressureIn: texture_3d<min16float>;
@group(1) @binding(2) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let pos = vec3<i32>(id);

  let dp_dx = (textureLoad(pressureIn, pos + vec3<i32>(1,0,0), 0).x -
               textureLoad(pressureIn, pos - vec3<i32>(1,0,0), 0).x) / (2.0 * params.dx);

  let dp_dy = (textureLoad(pressureIn, pos + vec3<i32>(0,1,0), 0).x -
               textureLoad(pressureIn, pos - vec3<i32>(0,1,0), 0).x) / (2.0 * params.dx);

  let dp_dz = (textureLoad(pressureIn, pos + vec3<i32>(0,0,1), 0).x -
               textureLoad(pressureIn, pos - vec3<i32>(0,0,1), 0).x) / (2.0 * params.dx);

  // subtract pressure gradient to make velocity divergence-free (equation 5)
  let velocity = textureLoad(velocityIn, id, 0).xyz;
  let pressureGradient = vec3f(dp_dx, dp_dy, dp_dz);

  textureStore(velocityOut, id, vec4f(velocity - pressureGradient, 0.0));
} 