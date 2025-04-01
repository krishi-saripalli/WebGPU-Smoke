// Import common definitions
@import "common.wgsl";

// --- Resource declarations for velocity advection ---
@group(1) @binding(0) var velocityIn: texture_3d<f32>; // Used for sampling coord
@group(1) @binding(1) var texSampler: sampler;
@group(1) @binding(2) var velocityToAdvect: texture_3d<f32>; // The field being advected
@group(1) @binding(3) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Get velocity vector at current location to calculate backtrace coord
  let v_at_id = textureLoad(velocityIn, id, 0).xyz;

  // Calculate the coordinate to sample from (in grid space)
  let coord_grid = vec3<f32>(id) - params.dt * v_at_id * params.dx;

  // Get texture dimensions (including halo) for normalization
  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);
  // Convert source coordinate to normalized [0.0, 1.0] space
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 2) - Advect velocity field by sampling at the backtraced coordinate
  textureStore(velocityOut, id, vec4f(textureSampleLevel(velocityToAdvect, texSampler, coord_normalized, 0.0).xyz, 0.0));
} 