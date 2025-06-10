@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var densityIn: texture_3d<f32>;
@group(1) @binding(2) var temperatureIn: texture_3d<f32>;
@group(1) @binding(3) var texSampler: sampler;
@group(1) @binding(4) var velocityOut: texture_storage_3d<rgba16float,write>;
@group(1) @binding(5) var densityOut: texture_storage_3d<r32float,write>;
@group(1) @binding(6) var temperatureOut: texture_storage_3d<r32float,write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let velocity = textureLoad(velocityIn, id, 0).xyz;
  
  let coord = vec3<f32>(id) - params.dt * velocity * params.dx;

  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);
  let coord_normalized = (coord + 0.5) / tex_dims;

  let velocity_adv = textureSampleLevel(velocityIn, texSampler, coord_normalized, 0.0).xyz;
  let density = textureSampleLevel(densityIn, texSampler, coord_normalized, 0.0).x;
  let temperature = textureSampleLevel(temperatureIn, texSampler, coord_normalized, 0.0).x;

  let dissipation = 0.999;
  textureStore(velocityOut, id, vec4f(velocity_adv * dissipation, 0.0));
  textureStore(densityOut, id, vec4f(density * dissipation, 0.0, 0.0, 0.0));
  textureStore(temperatureOut, id, vec4f(temperature * dissipation, 0.0, 0.0, 0.0));
} 