struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  gridSize: u32,
  numGridSlices: u32,
  
}

@group(0) @binding(0)  var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3f, //Object space position
};

struct VertexOutput {
  @builtin(position) position: vec4f, //Clip space position
  @location(0) texCoord: vec3f, //Texture coordinates
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  var worldPos = vec4f(input.position, 1.0);
  var viewPos = uniforms.viewMatrix * worldPos;

  output.texCoord = input.position * 0.5 + 0.5; // map [-1,1] to [0,1]
  output.position = uniforms.projectionMatrix * viewPos;
  return output;
}

@group(0) @binding(1) var srcDensity: texture_3d<f32>;
@group(0)@binding(2) var dstDensity: texture_storage_3d<rgba16float, write>;

@group(0) @binding(3) var densityView: texture_3d<f32>;
@group(0) @binding(4) var densitySampler: sampler;

@fragment
fn fragmentMain(vertexOut: VertexOutput) -> @location(0) vec4f {
  let density = textureSample(densityView, densitySampler, vertexOut.texCoord);
  return vec4f(1.0, 0.0, 0.0, density.x); // red smoke
}

@compute @workgroup_size(4,4,4)
fn computeMain(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.gridSize || 
        id.y >= uniforms.gridSize || 
        id.z >= uniforms.gridSize) {
        return;
    }

    let gridCenter = vec3<u32>(uniforms.gridSize / 2u);

    //let input = textureLoad(srcDensity, id, 0);
    
    if (id.x == gridCenter.x && 
        id.y == gridCenter.y && 
        id.z == gridCenter.z) {
        textureStore(dstDensity, id, vec4f(1.0));
    } else {
        textureStore(dstDensity, id, vec4f(0.0));
    }
}

