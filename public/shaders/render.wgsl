@import "common.wgsl";

//TODO: get rid of wireframe and shade outside of the box
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
  let worldPos = vec4f(input.position, 1.0);
  let viewPos = uniforms.viewMatrix * worldPos;
  output.texCoord = (input.position  + vec3f(1.0, 1.0, 1.0)) * 0.5; // assuming positions in [-1,1] -> [0,1]
  output.position = uniforms.projectionMatrix * viewPos;
  output.worldPosition = worldPos.xyz;
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

// Intersects a ray with an axis-aligned bounding box.
// Returns a vec2f containing the near (t_min) and far (t_max) intersection distances.
// If the ray misses the box, t_max will be less than t_min.
fn rayBoxIntersect(rayOrigin: vec3f, rayDirection: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
    let invDir = 1.0 / rayDirection;
    let t0s = (boxMin - rayOrigin) * invDir;
    let t1s = (boxMax - rayOrigin) * invDir;

    let tmin = min(t0s, t1s);
    let tmax = max(t0s, t1s);

    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return vec2f(t_enter, t_exit);
}

fn phaseHG(cosTheta: f32, g: f32) -> f32 {
  let pi = 3.14159265358979323846;
  let denom = 4.0 * f32(pi) * (1.0 + g * g - 2.0 * g * cosTheta);
  return (1.0 - g * g) / denom;
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {
  //color the floor slightly grey
  // if (vertexOut.texCoord.y < 0.01) {
  //   return vec4f(0.1, 0.1, 0.1, 1.0);
  // }

  let rayOrigin = uniforms.cameraPos;
  let rayDirection = normalize(vertexOut.worldPosition - rayOrigin);

  let boxMin = vec3f(-1.0);
  let boxMax = vec3f(1.0);
  let intersect = rayBoxIntersect(rayOrigin, rayDirection, boxMin, boxMax);
  let tmin = intersect.x;
  let tmax = intersect.y;

  if (tmin >= tmax) {
    // Ray misses the box, or we're inside and looking away.
    discard;
  }
  
  let numSteps = 64u;
  let jitter = fract(sin(vertexOut.position.x * 1024.0 + vertexOut.position.y * 2048.0) * 43758.5453);
  let rayLength = tmax - tmin;
  let stepSize = rayLength / f32(numSteps);
  
  // Start marching from the front of the box
  let entryPoint = rayOrigin + rayDirection * tmin;

  var transmittance = 1.0;
  var finalColor = vec3f(0.0);

  for (var i : u32 = 0; i < numSteps; i++) {
    // Add jitter to the first step to reduce banding artifacts
    let currentT = f32(i) + (select(0.0, jitter, i == 0u));
    let currentPosition = entryPoint + rayDirection * currentT * stepSize;
    let texCoord = (currentPosition + vec3f(1.0)) * 0.5;

    // We no longer need the explicit boundary check because we are marching 
    // from a calculated entry point to a calculated exit point.
    // However, a safety check can prevent sampling outside the texture due to precision errors.
    if (all(currentPosition > boxMin) && all(currentPosition < boxMax)) {
        let density = textureSampleLevel(densityIn, densitySampler, texCoord, 0.0).x;
      
        if (density > 0.01) {
          var lightTransmittance = 1.0;
          let lightDirection = normalize(uniforms.lightPosition - currentPosition);
          let numShadowSteps = 3u;
          for (var j : u32 = 1; j < numShadowSteps; j++) {
            let shadowPosition = currentPosition + lightDirection * f32(j) * stepSize;
            let shadowTexCoord = (shadowPosition + vec3f(1.0)) * 0.5;

            if (all(shadowTexCoord >= vec3f(0.0)) && all(shadowTexCoord <= vec3f(1.0))) {
                 lightTransmittance *= exp(-textureSampleLevel(densityIn, densitySampler, shadowTexCoord, 0.0).x * stepSize * uniforms.absorption);
            }
          }

          //scattering
          let cosTheta = dot(-lightDirection, rayDirection);
          let g = 0.5;
          let phase = phaseHG(cosTheta, g);
          let scattering = density * phase * lightTransmittance;
          finalColor += scattering * transmittance * uniforms.lightIntensity;

          transmittance *= exp(-density * stepSize * uniforms.absorption);
        }
    }
  }

  return vec4(finalColor, 1.0 - transmittance);
} 