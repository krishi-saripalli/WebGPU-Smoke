@import "common.wgsl";

struct VertexInput {
  @location(0) position: vec3f, 
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec3f,
  @location(1) worldPosition: vec3f       
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let worldPosition = vec4f(input.position, 1.0); // no model matrix because we want our world space to be [-1,1]
  let viewPosition = uniforms.viewMatrix * worldPosition;

  output.texCoord = (input.position  + vec3f(1.0, 1.0, 1.0)) * 0.5; // assuming positions in [-1,1] -> [0,1]
  output.position = uniforms.projectionMatrix * viewPosition; // clip space
  output.worldPosition = worldPosition.xyz;

  return output;
}

@group(1) @binding(0)
var densityIn: texture_3d<min16float>;

@group(1) @binding(1)
var densitySampler: sampler;

@fragment
fn fragmentWireframe(vertexOut: VertexOutput) -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 1.0, 1.0); 
}

// returns the ray intersection points for an axis aligned  box
// https://en.wikipedia.org/wiki/Slab_method
fn intersect(rayOrigin: vec3f, rayDirection: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
    let invDir = 1.0 / rayDirection;
    let t0s = (boxMin - rayOrigin) * invDir;
    let t1s = (boxMax - rayOrigin) * invDir;

    let tmin = min(t0s, t1s);
    let tmax = max(t0s, t1s);

    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return vec2f(t_enter, t_exit);
}

// Henyey-Greenstien phase function
fn phase(cosTheta: f32, g: f32) -> f32 {
  let pi = 3.14159265358979323846;
  let denom = 4.0 * f32(pi) * (1.0 + g * g - 2.0 * g * cosTheta);
  return (1.0 - g * g) / denom;
}

//returns the light attenuation coefficient by sampling point between the primary sample and light position
fn inScattering(currentPosition : vec3f, lightPos: vec3f) -> f32 {
  let directionToLight = normalize(lightPos - currentPosition);
  let numSteps = 16u;
  let rayLength = length(lightPos - currentPosition);
  let stepSize = rayLength / f32(numSteps);
  var totalDensity = 0.0;


  for (var i : u32 = 1; i < numSteps; i++) {
    let t = f32(i) * stepSize;
    let secondaryPosition = currentPosition + directionToLight * t;
    let texCoord = (secondaryPosition + vec3f(1.0)) * 0.5;
    totalDensity += textureSampleLevel(densityIn, densitySampler, texCoord, 0.0).x;
  }
  let extinction = uniforms.absorption + uniforms.scattering;
  return exp(-totalDensity * stepSize * extinction);
}

fn radiance(currentPosition: vec3f, rayDirection: vec3f, density: f32, stepSize: f32, transmission: f32) -> vec3f {
  var radiance = vec3f(0.0);
  
  if (density > 0.01) {
    let positionToLight = uniforms.lightPosition - currentPosition;
    let attenuation = inScattering(currentPosition, uniforms.lightPosition);
    let cosTheta = dot(normalize(-rayDirection), normalize(positionToLight));
    
    let scatteringContribution = uniforms.scattering * density;
    let phase = phase(cosTheta, 0.01);
    radiance += uniforms.lightIntensity * attenuation * transmission * stepSize * scatteringContribution * phase;
    
  }
  return radiance;
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {

  let rayOrigin = uniforms.cameraPos;
  let rayDirection = normalize(vertexOut.worldPosition - rayOrigin);

  let boxMin = vec3f(-1.0);
  let boxMax = vec3f(1.0);
  let intersection = intersect(rayOrigin, rayDirection, boxMin, boxMax);
  let tmin = intersection.x;
  let tmax = intersection.y;

  if (tmin >= tmax) {
    discard;
  }
  
  let numSteps = 50u;
  let rayLength = tmax - tmin;
  let stepSize = rayLength / f32(numSteps);
  
  let entryPoint = rayOrigin + rayDirection * tmin;

  var finalColor = vec3f(0.0);
  var transmission = 1.0;
  var weightedDistance = 0.0;

  for (var i : u32 = 0; i < numSteps; i++) {
    let t = f32(i) * stepSize; 
    let currentPosition = entryPoint + rayDirection * t;
    let texCoord = (currentPosition + vec3f(1.0)) * 0.5;
    let density = textureSampleLevel(densityIn, densitySampler, texCoord, 0.0).x;

    if (transmission < 0.01) {
      break;
    }

    if (density > 0.01) {
      let radiance = radiance(currentPosition, rayDirection, density, stepSize, transmission);
      finalColor += radiance;
    }

    let extinction = uniforms.absorption + uniforms.scattering;
    transmission *= exp(-density * stepSize * extinction);
      
  }
    return vec4(finalColor, 1.0 - transmission);


}

 
 
