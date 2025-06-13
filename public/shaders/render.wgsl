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

fn phaseHG(cosTheta: f32, g: f32) -> f32 {
  let pi = 3.14159265358979323846;
  let denom = 4.0 * f32(pi) * (1.0 + g * g - 2.0 * g * cosTheta);
  return (1.0 - g * g) / denom;
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {
  //color the floor slightly grey
  if (vertexOut.texCoord.y < 0.01) {
    return vec4f(0.1, 0.1, 0.1, 1.0);
  }
  //}
  let rayDirection = normalize(vertexOut.worldPosition - uniforms.cameraPos);
  let numSteps = 10u; // More steps for better quality. The previous value of 10 was too low.

  var transmittance = 1.0;
  let maxDistance = length(vec3f(uniforms.ratio));
  let stepSize = maxDistance / f32(numSteps);
  var finalColor = vec3f(0.0);

  
  // vertexOut.worldPosition is the entry point on the front face of the bounding cube.
  for (var i : u32 = 0; i < numSteps; i++) {
    let currentPosition = vertexOut.worldPosition + rayDirection * f32(i) * stepSize;


    // convert the world position to texture coordinates.
    let texCoord = (currentPosition + vec3f(1.0)) * 0.5;

    if (all(texCoord >= vec3f(0.0)) && all(texCoord <= vec3f(1.0))) {
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

  
  
  //return vec4f(density.x * grey.x, density.x * grey.y, density.x * grey.z, density.x); // premultiplied alpha.
} 