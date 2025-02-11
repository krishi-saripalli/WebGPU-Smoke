export const shader = `
struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
}

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3f,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  var worldPos = vec4f(input.position, 1.0);
  var viewPos = uniforms.viewMatrix * worldPos;
  output.position = uniforms.projectionMatrix * viewPos;
  return output;
}

@fragment
fn fragmentMain(vertexOut: VertexOutput) -> @location(0) vec4f {
  return vec4f(1.0, 0.0, 0.0, 1.0);
}
`;
