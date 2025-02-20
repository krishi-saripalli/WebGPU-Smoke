struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
}


@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(0) var<storage, read_write> density: array<f32>;
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


    
@compute @workgroup_size(4,4,4)
fn computeMain() {
    density[0] = 1.0;
}

