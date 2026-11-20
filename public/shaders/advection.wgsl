// textureSampleLevel instead of textureSample bc its not allowed in compute shaders
// https://github.com/gpuweb/gpuweb/discussions/4814
@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var linearSampler: sampler;
@group(1) @binding(2) var velocityOut: texture_storage_3d<rgba16float, write>;
@group(1) @binding(3) var smokeIn: texture_3d<f32>;
@group(1) @binding(4) var smokeOut: texture_storage_3d<rgba16float, write>;

fn sampleVelocity(p: vec3f) -> vec3f {
  return vec3f(
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(p, 0.5 * vec3f(X)), 0).x,
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(p, 0.5 * vec3f(Y)), 0).y,
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(p, 0.5 * vec3f(Z)), 0).z
  );
}

@compute @workgroup_size(4, 4, 4)
fn advectVelocity(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let p = vec3f(id) + 1.0;
  let x = 0.5 * vec3f(X);
  let y = 0.5 * vec3f(Y);
  let z = 0.5 * vec3f(Z);
  let coord_x = p + x - params.dt * sampleVelocity(p + x) / params.dx;
  let coord_y = p + y - params.dt * sampleVelocity(p + y) / params.dx;
  let coord_z = p + z - params.dt * sampleVelocity(p + z) / params.dx;
  let velocity_adv = vec3f(
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(coord_x, x), 0).x,
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(coord_y, y), 0).y,
    textureSampleLevel(velocityIn, linearSampler, sampleCoord(coord_z, z), 0).z
  );
  textureStore(velocityOut, vec3i(p), vec4f(wallVelocity(vec3i(p), velocity_adv), 0));
}

@compute @workgroup_size(4, 4, 4)
fn advectSmoke(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let p = vec3i(id) + 1;
  let velocity = sampleVelocity(vec3f(p));
  let coord = clamp(vec3f(p) - params.dt * velocity / params.dx, vec3f(1), vec3f(uniforms.gridSize));
  let tex_dims = vec3f(uniforms.gridSize + 2u);
  let coord_normalized = (coord + 0.5) / tex_dims;
  let state = textureSampleLevel(smokeIn, linearSampler, coord_normalized, 0).xy;
  let height = (f32(id.y) + 0.5) / f32(uniforms.gridSize.y);
  let decayRate = params.dissipation * mix(1.0, 15.0, smoothstep(0.8, 0.95, height));
  let dissipation = exp(-decayRate * params.dt);
  var density = state.x * dissipation;
  var temperature = mix(params.ambientTemperature, state.y, dissipation);

  let offset = abs(vec2i(id.xz) - vec2i(uniforms.gridSize.xz / 2u));
  if (id.y <= 3u && all(offset <= vec2i(7))) {
    density = 1.0;
    temperature = params.ambientTemperature + 100.0;
  }
  textureStore(smokeOut, p, vec4f(density, temperature, 0, 0));
}
