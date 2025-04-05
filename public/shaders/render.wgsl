@import "common.wgsl";

//TODO: get rid of wireframe and shade outside of the box
struct VertexInput {
  @location(0) position: vec3f, 
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec3f,       
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = vec4f(input.position, 1.0);
  let viewPos = uniforms.viewMatrix * worldPos;
  output.texCoord = (input.position  + vec3f(1.0, 1.0, 1.0)) * 0.5; // assuming positions in [-1,1] -> [0,1]
  output.position = uniforms.projectionMatrix * viewPos;
  return output;
}

@group(1) @binding(0)
var densityView: texture_3d<f32>;

@group(1) @binding(1)
var densitySampler: sampler;

@fragment
fn fragmentMain(vertexOut: VertexOutput) -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 1.0, 1.0);
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {
  let density = textureSample(densityView, densitySampler, vertexOut.texCoord);
  var outer_wall = vec4f(0.0, 0.0, 0.0, 0.0);
  if (vertexOut.texCoord.x == 0.0 || vertexOut.texCoord.x == 1.0 || vertexOut.texCoord.z == 0.0) {
    outer_wall = vec4f(0.1, 0.1, 0.1, 0.0);
  }
  let grey = vec4f(0.7,0.7,0.7,1.0);
  return vec4f(density.x * grey.x, density.x * grey.y, density.x * grey.z, density.x) + outer_wall; // premultiplied alpha.
} 