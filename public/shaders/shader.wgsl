/////////////////////////////////////////////////////////////////////////
// @group(0) will have only the uniforms for vertex/fragment/compute stages.  
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
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }


  // convert to signed integers for position calculations
  let pos = vec3<i32>(id);
  let center = vec3<i32>(uniforms.gridSize) / 2;
  
  let dx = abs(pos.x - center.x);
  let dy = abs(pos.y - center.y);
  let dz = abs(pos.z - center.z);

  let dist = sqrt(f32(dx*dx + dy*dy + dz*dz));
  
  if (dist <= 10.0) {
    let gaussian = exp(-dist*dist/(10.0));
    textureStore(dstDensity, id, vec4f(gaussian, gaussian, gaussian, 1.0));
  } else {
      textureStore(dstDensity, id, vec4f(0.0));
  }
}


////////////////////////////////////////////////////////////
///////////////////// Compute Shaders
////////////////////////////////////////////////////////////

// TODO: Add halo cells (n+2) in the CPU allocation code to avoid out of bounds access.


// ----- Additional Compute Resources -----

// Current state
@group(1) @binding(0) var srcVelocityX: texture_3d<f32>;
@group(1) @binding(1) var srcVelocityY: texture_3d<f32>;
@group(1) @binding(2) var srcVelocityZ: texture_3d<f32>;
@group(1) @binding(3) var srcDensity: texture_3d<f32>;
@group(1) @binding(4) var srcTemperature: texture_3d<f32>;

// Next state
@group(1) @binding(5) var dstVelocityX: texture_storage_3d<r32float, write>;
@group(1) @binding(6) var dstVelocityY: texture_storage_3d<r32float, write>;
@group(1) @binding(7) var dstVelocityZ: texture_storage_3d<r32float, write>;
@group(1) @binding(8) var dstDensity: texture_storage_3d<r32float, write>;
@group(1) @binding(9) var dstTemperature: texture_storage_3d<r32float, write>;

// Temporary fields for calculations
@group(1) @binding(10) var pressure: texture_storage_3d<r32float, write>;
@group(1) @binding(11) var divergence: texture_storage_3d<r32float, write>;
@group(1) @binding(12) var vorticity: texture_storage_3d<rgba16float, write>; // 3D vector
@group(1) @binding(13) var vorticityForce: texture_storage_3d<rgba16float, write>; // 3D vector
@group(1) @binding(14) var texSampler: sampler;



// Simulation parameters
struct SimulationParams {
    dt: f32,              // time step
    dx: f32,              // grid cell size
    vorticityStrength: f32, // epsilon in the paper
    buoyancyAlpha: f32,   // alpha in buoyancy equation
    buoyancyBeta: f32,    // beta in buoyancy equation
    ambientTemperature: f32, // T_amb in buoyancy equation
    iterations: u32,      // pressure solver iterations
    dissipation: f32      // optional dissipation factor
}
@group(0) @binding(1) var<uniform> params: SimulationParams;


@compute @workgroup_size(4,4,4)
fn applyExternalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }


  let velocity_x = textureLoad(srcVelocityX, id, 0).x;
  let velocity_y = textureLoad(srcVelocityY, id, 0).x;
  let velocity_z = textureLoad(srcVelocityZ, id, 0).x;


  let temp = textureLoad(srcTemperature, id, 0);
  let density = textureLoad(srcDensity, id, 0);

  //(Eq. 8)
  let up = vec3<f32>(0.0,1.0,0.0);
  let buoyancy = -1.0 * params.buoyancyAlpha * density * up + params.buoyancyBeta * (temp - params.ambientTemperature) * up;

  textureStore(dstVelocityX, id, vec4f(velocity_x + buoyancy.x,0,0,0));
  textureStore(dstVelocityY, id, vec4f(velocity_y + buoyancy.y,0,0,0));
  textureStore(dstVelocityZ, id, vec4f(velocity_z + buoyancy.z,0,0,0));

}

@compute @workgroup_size(4,4,4)
fn computeVorticity(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }


  let velocity_x = textureLoad(srcVelocityX, id, 0).x;
  let velocity_y = textureLoad(srcVelocityY, id, 0).x;
  let velocity_z = textureLoad(srcVelocityZ, id, 0).x;

  // vorticity = curl(velocity) (Eq. 9)
  let dvz_dy = (textureLoad(srcVelocityZ, id + vec3<i32>(0,1,0), 0).x) - (textureLoad(srcVelocityZ, id - vec3<i32>(0,1,0), 0).x);
  let dvy_dz = (textureLoad(srcVelocityY, id + vec3<i32>(0,0,1), 0).x) - (textureLoad(srcVelocityY, id - vec3<i32>(0,0,1), 0).x);
  let dvx_dz = (textureLoad(srcVelocityX, id + vec3<i32>(0,0,1), 0).x) - (textureLoad(srcVelocityX, id - vec3<i32>(0,0,1), 0).x);
  let dvy_dx = (textureLoad(srcVelocityY, id + vec3<i32>(1,0,0), 0).x) - (textureLoad(srcVelocityY, id - vec3<i32>(1,0,0), 0).x);
  let dvz_dx = (textureLoad(srcVelocityZ, id + vec3<i32>(1,0,0), 0).x) - (textureLoad(srcVelocityZ, id - vec3<i32>(1,0,0), 0).x);
  let dvx_dy = (textureLoad(srcVelocityX, id + vec3<i32>(0,1,0), 0).x) - (textureLoad(srcVelocityX, id - vec3<i32>(0,1,0), 0).x);

  let vort = vec3<f32>(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy) / (2.0 * params.dx);

  textureStore(vorticity, id, vec4f(vort, 0.0));

}

@compute @workgroup_size(4,4,4)
fn computeVorticityConfinement(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }


  // gradient of the *magnitude* of vorticity (Eq. 10)
  let dvort_mag_dx = length(textureLoad(vorticity, id + vec3<i32>(1,0,0), 0).xyz) - length(textureLoad(vorticity, id - vec3<i32>(1,0,0), 0).xyz);
  let dvort_mag_dy = length(textureLoad(vorticity, id + vec3<i32>(0,1,0), 0).xyz) - length(textureLoad(vorticity, id - vec3<i32>(0,1,0), 0).xyz);
  let dvort_mag_dz = length(textureLoad(vorticity, id + vec3<i32>(0,0,1), 0).xyz) - length(textureLoad(vorticity, id - vec3<i32>(0,0,1), 0).xyz);
  var grad = vec3<f32>(dvort_mag_dx, dvort_mag_dy, dvort_mag_dz)  / (2.0 * params.dx);

  let gradLength = length(grad);
  
  if (gradLength > 1e-6) {
    grad = grad / gradLength;
  } else {
    grad = vec3<f32>(0.0);
  }

  let vort = textureLoad(vorticity, id, 0).xyz;
  let confinement = params.vorticityStrength * params.dx * cross(grad, vort);


  textureStore(vorticityForce, id, vec4f(confinement, 0.0));
  
}


fn advect(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  vec3 coord = vec3<f32>(id) - params.dt * textureLoad(srcVelocityX, id, 0).xyz; // TODO: do I need texel size here?


  // (Eq. 2)
  textureStore(dstVelocityX, id, textureSample(srcVelocityX, texSampler, coord)); 
  textureStore(dstVelocityY, id, textureSample(srcVelocityY, texSampler, coord)); 
  textureStore(dstVelocityZ, id, textureSample(srcVelocityZ, texSampler, coord)); 

  // (Eq. 6)
  textureStore(dstTemperature, id, textureSample(srcTemperature, texSampler, coord)); 
 
  // (Eq. 7)
  textureStore(dstDensity, id, textureSample(srcDensity, texSampler, coord)); 

}

fn computePressure(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  


}



