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

////////////////////////////////////////////////////////////
///////////////////// Compute Shaders
////////////////////////////////////////////////////////////

@group(1) @binding(0) var srcVelocity: texture_3d<f32>; // Consolidated velocity (xyz in rgb)
@group(1) @binding(1) var srcDensity: texture_3d<f32>;
@group(1) @binding(2) var srcTemperature: texture_3d<f32>;
@group(1) @binding(3) var srcPressure: texture_3d<f32>;
@group(1) @binding(4) var srcDivergence: texture_3d<f32>;
@group(1) @binding(5) var srcVorticity: texture_3d<f32>; // 3D vector
@group(1) @binding(6) var srcVorticityForce: texture_3d<f32>; // 3D vector

@group(1) @binding(7) var dstVelocity: texture_storage_3d<rgba16float, write>; // Consolidated velocity (xyz in rgb)
@group(1) @binding(8) var dstDensity: texture_storage_3d<r32float, write>;
@group(1) @binding(9) var dstTemperature: texture_storage_3d<r32float, write>;
@group(1) @binding(10) var dstPressure: texture_storage_3d<r32float, write>;
@group(1) @binding(11) var dstDivergence: texture_storage_3d<r32float, write>;
@group(1) @binding(12) var dstVorticity: texture_storage_3d<rgba16float, write>; // 3D vector
@group(1) @binding(13) var dstVorticityForce: texture_storage_3d<rgba16float, write>; // 3D vector

@group(1) @binding(14) var texSampler: sampler;

// Simulation parameters
struct SimulationParams {
    dt: f32,              // time step
    dx: f32,              // grid cell size
    vorticityStrength: f32, // epsilon in the paper
    buoyancyAlpha: f32,   // alpha in buoyancy equation
    buoyancyBeta: f32,    // beta in buoyancy equation
    ambientTemperature: f32, // T_amb in buoyancy equation
}
@group(0) @binding(1) var<uniform> params: SimulationParams;

@compute @workgroup_size(4,4,4)
fn computeMain(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  let density = textureLoad(srcDensity, id, 0).x;
  
  textureStore(dstDensity, id, vec4f(density, 0.0, 0.0, 0.0));
}

// TODO: Add halo cells (n+2) in the CPU allocation code to avoid out of bounds access.

@compute @workgroup_size(4,4,4)
fn applyExternalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  let velocity = textureLoad(srcVelocity, id, 0).xyz;
  let temp = textureLoad(srcTemperature, id, 0).x;
  let density = textureLoad(srcDensity, id, 0).x;

  //(Eq. 8)
  let up = vec3<f32>(0.0,1.0,0.0);
  let buoyancy = -1.0 * params.buoyancyAlpha * density * up + params.buoyancyBeta * (temp - params.ambientTemperature) * up;

  textureStore(dstVelocity, id, vec4f(velocity + buoyancy, 0.0));
}

@compute @workgroup_size(4,4,4)
fn computeVorticity(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // vorticity = curl(velocity) (Eq. 9)
  let vp_y1 = textureLoad(srcVelocity, pos + vec3<i32>(0,1,0), 0).xyz;
  let vn_y1 = textureLoad(srcVelocity, pos - vec3<i32>(0,1,0), 0).xyz;
  let vp_z1 = textureLoad(srcVelocity, pos + vec3<i32>(0,0,1), 0).xyz;
  let vn_z1 = textureLoad(srcVelocity, pos - vec3<i32>(0,0,1), 0).xyz;
  let vp_x1 = textureLoad(srcVelocity, pos + vec3<i32>(1,0,0), 0).xyz;
  let vn_x1 = textureLoad(srcVelocity, pos - vec3<i32>(1,0,0), 0).xyz;

  let dvz_dy = vp_y1.z - vn_y1.z;
  let dvy_dz = vp_z1.y - vn_z1.y;
  let dvx_dz = vp_z1.x - vn_z1.x;
  let dvz_dx = vp_x1.z - vn_x1.z;
  let dvy_dx = vp_x1.y - vn_x1.y;
  let dvx_dy = vp_y1.x - vn_y1.x;

  let vort = vec3<f32>(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy) / (2.0 * params.dx);

  textureStore(dstVorticity, id, vec4f(vort, 0.0));
}

@compute @workgroup_size(4,4,4)
fn computeVorticityConfinement(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // gradient of the *magnitude* of vorticity (Eq. 10)
  let dvort_mag_dx = length(textureLoad(srcVorticity, pos + vec3<i32>(1,0,0), 0).xyz) - length(textureLoad(srcVorticity, pos - vec3<i32>(1,0,0), 0).xyz);
  let dvort_mag_dy = length(textureLoad(srcVorticity, pos + vec3<i32>(0,1,0), 0).xyz) - length(textureLoad(srcVorticity, pos - vec3<i32>(0,1,0), 0).xyz);
  let dvort_mag_dz = length(textureLoad(srcVorticity, pos + vec3<i32>(0,0,1), 0).xyz) - length(textureLoad(srcVorticity, pos - vec3<i32>(0,0,1), 0).xyz);
  var grad = vec3<f32>(dvort_mag_dx, dvort_mag_dy, dvort_mag_dz) / (2.0 * params.dx);

  let gradLength = length(grad);
  
  if (gradLength > 1e-6) {
    grad = grad / gradLength;
  } else {
    grad = vec3<f32>(0.0);
  }

  let vort = textureLoad(srcVorticity, id, 0).xyz;
  let confinement = params.vorticityStrength * params.dx * cross(grad, vort);

  textureStore(dstVorticityForce, id, vec4f(confinement, 0.0));
}

// Advect velocity only
@group(1) @binding(0) var srcVelocityForAdvectV: texture_3d<f32>;
@group(1) @binding(1) var samplerForAdvectV: sampler;
@group(1) @binding(2) var dstVelocityForAdvectV: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn advectVelocity(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Get velocity vector
  let v = textureLoad(srcVelocityForAdvectV, id, 0).xyz;

  // Calculate the coordinate to sample from
  let coord = vec3<f32>(id) - params.dt * v * params.dx;

  // (Eq. 2) - Advect velocity
  textureStore(dstVelocityForAdvectV, id, vec4f(textureSampleLevel(srcVelocityForAdvectV, samplerForAdvectV, coord, 0.0).xyz, 0.0));
}

// Advect temperature only
@group(1) @binding(0) var srcVelocityForAdvectT: texture_3d<f32>;
@group(1) @binding(1) var srcTemperatureForAdvectT: texture_3d<f32>;
@group(1) @binding(2) var samplerForAdvectT: sampler;
@group(1) @binding(3) var dstTemperatureForAdvectT: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4,4,4)
fn advectTemperature(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Get velocity vector
  let v = textureLoad(srcVelocityForAdvectT, id, 0).xyz;

  // Calculate the coordinate to sample from
  let coord = vec3<f32>(id) - params.dt * v * params.dx;
  
  // (Eq. 6) - Advect temperature
  textureStore(dstTemperatureForAdvectT, id, textureSampleLevel(srcTemperatureForAdvectT, samplerForAdvectT, coord, 0.0));
}

// Advect density only
@group(1) @binding(0) var srcVelocityForAdvectD: texture_3d<f32>;
@group(1) @binding(1) var srcDensityForAdvectD: texture_3d<f32>;
@group(1) @binding(2) var samplerForAdvectD: sampler;
@group(1) @binding(3) var dstDensityForAdvectD: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4,4,4)
fn advectDensity(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Get velocity vector
  let v = textureLoad(srcVelocityForAdvectD, id, 0).xyz;

  // Calculate the coordinate to sample from
  let coord = vec3<f32>(id) - params.dt * v * params.dx;
  
  // (Eq. 7) - Advect density
  textureStore(dstDensityForAdvectD, id, textureSampleLevel(srcDensityForAdvectD, samplerForAdvectD, coord, 0.0));
}

@compute @workgroup_size(4,4,4)
fn computePressure(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // Use Laplacian and then isolate pressure; Laplacian can be found here (https://scicomp.stackexchange.com/questions/35920/3d-laplacian-operator)
  // p(i,j,k) = (p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k) + p(i,j,k+1) + p(i,j,k-1) - h²·div(i,j,k))/6
  let h_squared = params.dx * params.dx;
  
  let p = (
    textureLoad(srcPressure, pos + vec3<i32>(1,0,0), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(1,0,0), 0).x +
    textureLoad(srcPressure, pos + vec3<i32>(0,1,0), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(0,1,0), 0).x +
    textureLoad(srcPressure, pos + vec3<i32>(0,0,1), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(0,0,1), 0).x -
    h_squared * textureLoad(srcDivergence, id, 0).x
  ) / 6.0;
  
  textureStore(dstPressure, id, vec4f(p, 0.0, 0.0, 0.0));
  return;
}

@compute @workgroup_size(4,4,4)
fn computeDivergence(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  let vp_x1 = textureLoad(srcVelocity, pos + vec3<i32>(1,0,0), 0).xyz;
  let vn_x1 = textureLoad(srcVelocity, pos - vec3<i32>(1,0,0), 0).xyz;
  let vp_y1 = textureLoad(srcVelocity, pos + vec3<i32>(0,1,0), 0).xyz;
  let vn_y1 = textureLoad(srcVelocity, pos - vec3<i32>(0,1,0), 0).xyz;
  let vp_z1 = textureLoad(srcVelocity, pos + vec3<i32>(0,0,1), 0).xyz;
  let vn_z1 = textureLoad(srcVelocity, pos - vec3<i32>(0,0,1), 0).xyz;

  let dv_dx = vp_x1.x - vn_x1.x;
  let dv_dy = vp_y1.y - vn_y1.y;
  let dv_dz = vp_z1.z - vn_z1.z;

  let div = (dv_dx + dv_dy + dv_dz) / (2.0 * params.dx);

  textureStore(dstDivergence, id, vec4f(div, 0.0, 0.0, 0.0));
  return;
}

@compute @workgroup_size(4,4,4)
fn applyPressureGradient(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }
  
  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);
  
  let dp_dx = (textureLoad(srcPressure, pos + vec3<i32>(1,0,0), 0).x - 
               textureLoad(srcPressure, pos - vec3<i32>(1,0,0), 0).x) / (2.0 * params.dx);
               
  let dp_dy = (textureLoad(srcPressure, pos + vec3<i32>(0,1,0), 0).x - 
               textureLoad(srcPressure, pos - vec3<i32>(0,1,0), 0).x) / (2.0 * params.dx);
               
  let dp_dz = (textureLoad(srcPressure, pos + vec3<i32>(0,0,1), 0).x - 
               textureLoad(srcPressure, pos - vec3<i32>(0,0,1), 0).x) / (2.0 * params.dx);
  
  // Subtract pressure gradient to make velocity divergence-free (equation 5)
  let velocity = textureLoad(srcVelocity, id, 0).xyz;
  let pressureGradient = vec3f(dp_dx, dp_dy, dp_dz);
  
  textureStore(dstVelocity, id, vec4f(velocity - pressureGradient, 0.0));
}

@compute @workgroup_size(4,4,4)
fn applyVorticityForce(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }
  
  let velocity = textureLoad(srcVelocity, id, 0).xyz;
  let force = textureLoad(srcVorticityForce, id, 0).xyz;
  
  textureStore(dstVelocity, id, vec4f(velocity + force * params.dt, 0.0));
}

@compute @workgroup_size(4,4,4)
fn solvePressureJacobi(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 1 || id.x >= uniforms.gridSize.x - 1 ||
      id.y < 1 || id.y >= uniforms.gridSize.y - 1 ||
      id.z < 1 || id.z >= uniforms.gridSize.z - 1) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // Use Laplacian and then isolate pressure; Laplacian can be found here (https://scicomp.stackexchange.com/questions/35920/3d-laplacian-operator)
  // p(i,j,k) = (p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k) + p(i,j,k+1) + p(i,j,k-1) - h²·div(i,j,k))/6
  let h_squared = params.dx * params.dx;
  
  let p = (
    textureLoad(srcPressure, pos + vec3<i32>(1,0,0), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(1,0,0), 0).x +
    textureLoad(srcPressure, pos + vec3<i32>(0,1,0), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(0,1,0), 0).x +
    textureLoad(srcPressure, pos + vec3<i32>(0,0,1), 0).x +
    textureLoad(srcPressure, pos - vec3<i32>(0,0,1), 0).x -
    h_squared * textureLoad(srcDivergence, id, 0).x
  ) / 6.0;
  
  textureStore(dstPressure, id, vec4f(p, 0.0, 0.0, 0.0));
}
