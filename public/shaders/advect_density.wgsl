// Import common definitions
@import "common.wgsl";

// --- Resource declarations for density advection ---
@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var densityIn: texture_3d<f32>;
@group(1) @binding(2) var texSampler: sampler;
@group(1) @binding(3) var densityOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let v = textureLoad(velocityIn, id, 0).xyz;

  // Calculate the coordinate to sample from
  let coord_grid = vec3<f32>(id) - params.dt * v * params.dx;

  // Get texture dimensions (including halo)
  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);

  // Convert source coordinate to normalized [0.0, 1.0] space
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 7) - Advect density
  textureStore(densityOut, id, textureSampleLevel(densityIn, texSampler, coord_normalized, 0.0));
} 