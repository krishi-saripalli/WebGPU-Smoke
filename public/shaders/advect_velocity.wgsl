@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var texSampler: sampler;
@group(1) @binding(2) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let v_at_id = textureLoad(velocityIn, id, 0).xyz;

  let coord_grid = vec3<f32>(id) - params.dt * v_at_id * params.dx;

  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 2) - Advect velocity field by sampling at the backtraced coordinate
  textureStore(velocityOut, id, vec4f(textureSampleLevel(velocityIn, texSampler, coord_normalized, 0.0).xyz, 0.0));
} 