/////////////////////////////////////////////////////////////////////////
// Common definitions for all shaders
/////////////////////////////////////////////////////////////////////////

struct Uniforms {
  modelViewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  gridSize: vec3u, // This is the INTERNAL grid size (e.g., 100x100x100)
  cameraForward: vec3f,
  absorption: f32, // coefficient for light absorption by smoke
  lightPosition: vec3f,
  scattering: f32, // scattering coefficient
  lightIntensity: vec3f, // color and strength of the light
  brushStroke: vec4f,
  brushStrength: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct SimulationParams {
  dt: f32, // time step
  dx: f32, // grid cell size
  vorticityStrength: f32, // epsilon in the paper
  buoyancyAlpha: f32, // alpha in buoyancy equation
  buoyancyBeta: f32, // beta in buoyancy equation
  ambientTemperature: f32, // T_amb in buoyancy equation
  dissipation: f32,
}
@group(0) @binding(1) var<uniform> params: SimulationParams;

const X = vec3i(1, 0, 0);
const Y = vec3i(0, 1, 0);
const Z = vec3i(0, 0, 1);

fn cell(p: vec3i) -> vec3i {
  return clamp(p, vec3i(1), vec3i(uniforms.gridSize));
}

fn wallVelocity(p: vec3i, velocity: vec3f) -> vec3f {
  return select(velocity, vec3f(0), p == vec3i(uniforms.gridSize));
}

fn sampleCoord(p: vec3f, offset: vec3f) -> vec3f {
  let index = clamp(p - offset, vec3f(1) - 2.0 * offset, vec3f(uniforms.gridSize));
  return (index + 0.5) / vec3f(uniforms.gridSize + 2u);
}
