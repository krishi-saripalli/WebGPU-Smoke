@import "common.wgsl";

struct VertexInput {
  @location(0) position: vec3f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) boxPosition: vec3f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let ratio = vec3f(uniforms.gridSize) / f32(uniforms.gridSize.y);
  output.boxPosition = input.position * ratio;
  let viewPosition = uniforms.modelViewMatrix * vec4f(output.boxPosition, 1.0);
  output.position = uniforms.projectionMatrix * viewPosition; // clip space
  return output;
}

@group(1) @binding(0) var densityIn: texture_3d<f32>;
@group(1) @binding(1) var densitySampler: sampler;
@group(1) @binding(2) var lightOut: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(3) var light: texture_3d<f32>;

@fragment
fn fragmentWireframe() -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 1.0, 1.0);
}

// returns the ray intersection points for an axis aligned  box
// https://en.wikipedia.org/wiki/Slab_method
fn intersect(rayOrigin: vec3f, rayDirection: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
  let moving = abs(rayDirection) > vec3f(1e-7);
  let invDir = 1.0 / select(vec3f(1), rayDirection, moving);
  let t0s = (boxMin - rayOrigin) * invDir;
  let t1s = (boxMax - rayOrigin) * invDir;
  let tmin = select(vec3f(-1e20), min(t0s, t1s), moving);
  let tmax = select(vec3f(1e20), max(t0s, t1s), moving);
  return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// Henyey-Greenstien phase function
fn phase(cosTheta: f32, g: f32) -> f32 {
  let pi = 3.14159265358979323846;
  let denom = 4.0 * pi * pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
  return (1.0 - g * g) / denom;
}

//returns the light attenuation coefficient by sampling point between the primary sample and light position
fn inScattering(currentPosition: vec3f, lightPos: vec3f, ratio: vec3f) -> f32 {
  let numSteps = 16u;
  let delta = (lightPos - currentPosition) / f32(numSteps);
  let stepSize = length(delta);
  let gridSize = vec3f(uniforms.gridSize);
  var totalDensity = 0.0;
  for (var i = 0u; i < numSteps; i++) {
    let secondaryPosition = currentPosition + (f32(i) + 0.5) * delta;
    let texCoord = (secondaryPosition / ratio + 1.0) * 0.5;
    let coord = clamp(texCoord * gridSize + 0.5, vec3f(1), gridSize);
    totalDensity += textureSampleLevel(densityIn, densitySampler, (coord + 0.5) / (gridSize + 2.0), 0).x;
  }
  let extinction = uniforms.absorption + uniforms.scattering;
  return exp(-totalDensity * stepSize * extinction);
}

@compute @workgroup_size(4, 4, 4)
fn illuminate(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(lightOut);
  if (any(id >= dims)) { return; }
  let ratio = vec3f(uniforms.gridSize) / f32(uniforms.gridSize.y);
  let position = ((vec3f(id) + 0.5) / vec3f(dims) * 2.0 - 1.0) * ratio;
  let attenuation = inScattering(position, uniforms.lightPosition, ratio);
  textureStore(lightOut, id, vec4f(attenuation, 0, 0, 1));
}

@fragment
fn fragmentVolume(vertexOut: VertexOutput) -> @location(0) vec4f {
  let gridSize = vec3f(uniforms.gridSize);
  let boxMax = gridSize / gridSize.y;
  let rayDirection = uniforms.cameraForward;
  let intersection = intersect(vertexOut.boxPosition, rayDirection, -boxMax, boxMax);
  let rayLength = intersection.y - intersection.x;
  let numSteps = 50u;
  let stepSize = rayLength / f32(numSteps);
  let entryPoint = vertexOut.boxPosition + rayDirection * intersection.x;
  let extinction = uniforms.absorption + uniforms.scattering;
  var finalColor = vec3f(0.0);
  var transmission = 1.0;

  for (var i = 0u; i < numSteps; i++) {
    let t = (f32(i) + 0.5) * stepSize;
    let currentPosition = entryPoint + rayDirection * t;
    let texCoord = (currentPosition / boxMax + 1.0) * 0.5; // assuming positions in [-1,1] -> [0,1]
    let coord = clamp(texCoord * gridSize + 0.5, vec3f(1), gridSize);
    let density = textureSampleLevel(densityIn, densitySampler, (coord + 0.5) / (gridSize + 2.0), 0).x;
    let opacity = 1.0 - exp(-density * stepSize * extinction);
    if (density > 0.0) {
      let positionToLight = uniforms.lightPosition - currentPosition;
      let cosTheta = dot(rayDirection, positionToLight / max(length(positionToLight), 1e-8));
      let attenuation = textureSampleLevel(light, densitySampler, texCoord, 0).x;
      let lighting = vec3f(0.38) + uniforms.lightIntensity * attenuation * phase(cosTheta, 0.01);
      finalColor += lighting * (uniforms.scattering / extinction) * transmission * opacity;
    }
    transmission *= 1.0 - opacity;
    if (transmission < 0.01) { break; }
  }
  return vec4f(finalColor, 1.0 - transmission);
}
