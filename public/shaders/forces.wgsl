// Import common definitions
@import "common.wgsl";

// --- Resource declarations for external forces step ---
@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var temperatureIn: texture_3d<f32>;
@group(1) @binding(2) var densityIn: texture_3d<f32>;
@group(1) @binding(3) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let velocity = textureLoad(velocityIn, id, 0).xyz;
  let temp = textureLoad(temperatureIn, id, 0).x;
  let density = textureLoad(densityIn, id, 0).x;

  // (Eq. 8)
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let buoyancy = -1.0 * params.buoyancyAlpha * density * up + 
                 params.buoyancyBeta * (temp - params.ambientTemperature) * up;

  textureStore(velocityOut, id, vec4f(velocity + buoyancy, 0.0));
} 