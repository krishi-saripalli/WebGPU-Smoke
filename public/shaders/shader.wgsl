/////////////////////////////////////////////////////////////////////////
// @group(0) will have only the uniforms for both vertex/fragment/compute stages.  
//@group(1) will hold either (src, dst) for the compute pipeline or (texture, sampler) for the render pipeline.
/////////////////////////////////////////////////////////////////////////

// ----- Uniforms (group 0) -----
  struct Uniforms {
      viewMatrix      : mat4x4<f32>,
      projectionMatrix: mat4x4<f32>,
      gridSize        : vec3<u32>,
      _pad1           : u32,          // forces 16-byte alignment after vec3<u32>
      cameraForward   : vec3<f32>,
      _pad2           : f32            // forces 16-byte alignment after vec3<f32>
  };
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3f, // Object space position
};

struct VertexOutput {
  @builtin(position) position: vec4f, // Clip space position
  @location(0) texCoord: vec3f,        // Texture coordinates for slice rendering
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = vec4f(input.position, 1.0);
  let viewPos = uniforms.viewMatrix * worldPos;
  output.texCoord = (input.position  + vec3f(1.0, 1.0, 1.0)) * 0.5; // assuming positions in [-1,1]
  output.position = uniforms.projectionMatrix * viewPos;
  return output;
}

// ----- Fragment Textures (group 1) -----
@group(1) @binding(0)
var densityView: texture_3d<f32>;

@group(1) @binding(1)
var densitySampler: sampler;

@fragment
fn fragmentMain(vertexOut: VertexOutput) -> @location(0) vec4f {
  return vec4f(1.0, 0.0, 0.0, 1.0);
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {
  
  let density = textureSample(densityView, densitySampler, vertexOut.texCoord);
  return vec4f(density.x * 1.0, density.x * 0.0, density.x * 0.0, density.x); // premultiplied alpha.
}

// ----- Compute Shader  ----- 

// ----- Compute Textures (group 1) ----- 
@group(1) @binding(0)
var srcDensity: texture_3d<f32>;

@group(1) @binding(1)
var dstDensity: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn computeMain(@builtin(global_invocation_id) id: vec3<u32>) {
  // Check bounds
  if (id.x >= uniforms.gridSize.x ||
      id.y >= uniforms.gridSize.y ||
      id.z >= uniforms.gridSize.z) {
      return;
  }

  // convert to signed integers for position calculations
  let pos = vec3<i32>(id);
  let center = vec3<i32>(uniforms.gridSize) / 2;
  
  let dx = abs(pos.x - center.x);
  let dy = abs(pos.y - center.y);
  let dz = abs(pos.z - center.z);
  
  if (dx <= 1 && dy <= 1 && dz <= 1) {
      textureStore(dstDensity, id, vec4f(1.0));
  } else {
      textureStore(dstDensity, id, vec4f(0.0));
  }
}


