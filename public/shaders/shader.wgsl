/////////////////////////////////////////////////////////////////////////
// @group(0) will have only the uniforms for vertex/fragment/compute stages.  
//@group(1) will hold either (src, dst) for the compute pipeline or (texture, sampler) for the render pipeline.
/////////////////////////////////////////////////////////////////////////

// ----- Uniforms (group 0) -----
  struct Uniforms {
      viewMatrix      : mat4x4<f32>,
      projectionMatrix: mat4x4<f32>,
      gridSize        : vec3<u32>, // This is the INTERNAL grid size (e.g., 100x100x100)
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
  return vec4f(1.0, 1.0, 1.0, 1.0);
}

@fragment
fn fragmentSlices(vertexOut: VertexOutput) -> @location(0) vec4f {
  
  let density = textureSample(densityView, densitySampler, vertexOut.texCoord);
  return vec4f(density.x * 0.5, density.x * 0.5, density.x * 0.5, density.x); // premultiplied alpha.
}

////////////////////////////////////////////////////////////
///////////////////// Compute Shaders
////////////////////////////////////////////////////////////

// --- REMOVED OLD GLOBAL BINDINGS ---

// --- Resource declarations for densityCopy ---
@group(1) @binding(0) var densityCopy_densityIn: texture_3d<f32>;
@group(1) @binding(1) var densityCopy_densityOut: texture_storage_3d<r32float, write>;

// --- Resource declarations for externalForcesStep ---
@group(1) @binding(0) var externalForcesStep_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var externalForcesStep_temperatureIn: texture_3d<f32>;
@group(1) @binding(2) var externalForcesStep_densityIn: texture_3d<f32>;
@group(1) @binding(3) var externalForcesStep_velocityOut: texture_storage_3d<rgba16float, write>;

// --- Resource declarations for vorticityCalculation ---
@group(1) @binding(0) var vorticityCalculation_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityCalculation_vorticityOut: texture_storage_3d<rgba16float, write>;

// --- Resource declarations for vorticityConfinementForce ---
@group(1) @binding(0) var vorticityConfinementForce_vorticityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityConfinementForce_forceOut: texture_storage_3d<rgba16float, write>;

// --- Resource declarations for vorticityForceApplication ---
@group(1) @binding(0) var vorticityForceApplication_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityForceApplication_forceIn: texture_3d<f32>;
@group(1) @binding(2) var vorticityForceApplication_velocityOut: texture_storage_3d<rgba16float, write>;

// --- Resource declarations for velocityAdvection ---
@group(1) @binding(0) var velocityAdvection_velocityIn: texture_3d<f32>; // Used for sampling coord
@group(1) @binding(1) var velocityAdvection_sampler: sampler;
@group(1) @binding(2) var velocityAdvection_velocityToAdvect: texture_3d<f32>; // The field being advected
@group(1) @binding(3) var velocityAdvection_velocityOut: texture_storage_3d<rgba16float, write>;

// --- Resource declarations for temperatureAdvection ---
@group(1) @binding(0) var temperatureAdvection_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var temperatureAdvection_temperatureIn: texture_3d<f32>;
@group(1) @binding(2) var temperatureAdvection_sampler: sampler;
@group(1) @binding(3) var temperatureAdvection_temperatureOut: texture_storage_3d<r32float, write>;

// --- Resource declarations for densityAdvection ---
@group(1) @binding(0) var densityAdvection_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var densityAdvection_densityIn: texture_3d<f32>;
@group(1) @binding(2) var densityAdvection_sampler: sampler;
@group(1) @binding(3) var densityAdvection_densityOut: texture_storage_3d<r32float, write>;

// --- Resource declarations for divergenceCalculation ---
@group(1) @binding(0) var divergenceCalculation_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var divergenceCalculation_divergenceOut: texture_storage_3d<r32float, write>;

// --- Resource declarations for pressureIteration (Jacobi) ---
@group(1) @binding(0) var pressureIteration_pressureIn: texture_3d<f32>;
@group(1) @binding(1) var pressureIteration_divergenceIn: texture_3d<f32>;
@group(1) @binding(2) var pressureIteration_pressureOut: texture_storage_3d<r32float, write>;

// --- Resource declarations for pressureGradientSubtraction ---
@group(1) @binding(0) var pressureGradientSubtraction_velocityIn: texture_3d<f32>;
@group(1) @binding(1) var pressureGradientSubtraction_pressureIn: texture_3d<f32>;
@group(1) @binding(2) var pressureGradientSubtraction_velocityOut: texture_storage_3d<rgba16float, write>;


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

// Helper constant for boundary checks
const HALO_SIZE: u32 = 1u;

@compute @workgroup_size(4,4,4)
fn densityCopy(@builtin(global_invocation_id) id: vec3<u32>) { // Renamed from computeMain
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let density = textureLoad(densityCopy_densityIn, id, 0).x; // Use new var

  textureStore(densityCopy_densityOut, id, vec4f(density, 0.0, 0.0, 0.0)); // Use new var
}


@compute @workgroup_size(4,4,4)
fn externalForcesStep(@builtin(global_invocation_id) id: vec3<u32>) { // Renamed from applyExternalForces
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let velocity = textureLoad(externalForcesStep_velocityIn, id, 0).xyz; // Use new var
  let temp = textureLoad(externalForcesStep_temperatureIn, id, 0).x; // Use new var
  let density = textureLoad(externalForcesStep_densityIn, id, 0).x; // Use new var

  //(Eq. 8)
  let up = vec3<f32>(0.0,1.0,0.0);
  let buoyancy = -1.0 * params.buoyancyAlpha * density * up + params.buoyancyBeta * (temp - params.ambientTemperature) * up;

  textureStore(externalForcesStep_velocityOut, id, vec4f(velocity + buoyancy, 0.0)); // Use new var
}

@compute @workgroup_size(4,4,4)
fn vorticityCalculation(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from computeVorticity
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // vorticity = curl(velocity) (Eq. 9)
  let vp_y1 = textureLoad(vorticityCalculation_velocityIn, pos + vec3<i32>(0,1,0), 0).xyz; // Use new var
  let vn_y1 = textureLoad(vorticityCalculation_velocityIn, pos - vec3<i32>(0,1,0), 0).xyz; // Use new var
  let vp_z1 = textureLoad(vorticityCalculation_velocityIn, pos + vec3<i32>(0,0,1), 0).xyz; // Use new var
  let vn_z1 = textureLoad(vorticityCalculation_velocityIn, pos - vec3<i32>(0,0,1), 0).xyz; // Use new var
  let vp_x1 = textureLoad(vorticityCalculation_velocityIn, pos + vec3<i32>(1,0,0), 0).xyz; // Use new var
  let vn_x1 = textureLoad(vorticityCalculation_velocityIn, pos - vec3<i32>(1,0,0), 0).xyz; // Use new var

  let dvz_dy = vp_y1.z - vn_y1.z;
  let dvy_dz = vp_z1.y - vn_z1.y;
  let dvx_dz = vp_z1.x - vn_z1.x;
  let dvz_dx = vp_x1.z - vn_x1.z;
  let dvy_dx = vp_x1.y - vn_x1.y;
  let dvx_dy = vp_y1.x - vn_y1.x;

  let vort = vec3<f32>(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy) / (2.0 * params.dx);

  textureStore(vorticityCalculation_vorticityOut, id, vec4f(vort, 0.0)); // Use new var
}

@compute @workgroup_size(4,4,4)
fn vorticityConfinementForce(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from computeVorticityConfinement
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // gradient of the *magnitude* of vorticity (Eq. 10)
  let dvort_mag_dx = length(textureLoad(vorticityConfinementForce_vorticityIn, pos + vec3<i32>(1,0,0), 0).xyz) - length(textureLoad(vorticityConfinementForce_vorticityIn, pos - vec3<i32>(1,0,0), 0).xyz); // Use new var
  let dvort_mag_dy = length(textureLoad(vorticityConfinementForce_vorticityIn, pos + vec3<i32>(0,1,0), 0).xyz) - length(textureLoad(vorticityConfinementForce_vorticityIn, pos - vec3<i32>(0,1,0), 0).xyz); // Use new var
  let dvort_mag_dz = length(textureLoad(vorticityConfinementForce_vorticityIn, pos + vec3<i32>(0,0,1), 0).xyz) - length(textureLoad(vorticityConfinementForce_vorticityIn, pos - vec3<i32>(0,0,1), 0).xyz); // Use new var
  var grad = vec3<f32>(dvort_mag_dx, dvort_mag_dy, dvort_mag_dz) / (2.0 * params.dx);

  let gradLength = length(grad);
  
  if (gradLength > 1e-6) {
    grad = grad / gradLength;
  } else {
    grad = vec3<f32>(0.0);
  }

  let vort = textureLoad(vorticityConfinementForce_vorticityIn, id, 0).xyz; // Use new var
  let confinement = params.vorticityStrength * params.dx * cross(grad, vort);

  textureStore(vorticityConfinementForce_forceOut, id, vec4f(confinement, 0.0)); // Use new var
}


@compute @workgroup_size(4,4,4)
fn velocityAdvection(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from advectVelocity
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Get velocity vector at current location to calculate backtrace coord
  let v_at_id = textureLoad(velocityAdvection_velocityIn, id, 0).xyz; // Use new var

  // Calculate the coordinate to sample from (in grid space)
  let coord_grid = vec3<f32>(id) - params.dt * v_at_id * params.dx; // Use v_at_id

  // Get texture dimensions (including halo) for normalization
  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);
  // Convert source coordinate to normalized [0.0, 1.0] space
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 2) - Advect velocity field by sampling at the backtraced coordinate
  textureStore(velocityAdvection_velocityOut, id, vec4f(textureSampleLevel(velocityAdvection_velocityToAdvect, velocityAdvection_sampler, coord_normalized, 0.0).xyz, 0.0)); // Use new vars
}


@compute @workgroup_size(4,4,4)
fn temperatureAdvection(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from advectTemperature
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Get velocity vector
  let v = textureLoad(temperatureAdvection_velocityIn, id, 0).xyz; // Use new var

  // Calculate the coordinate to sample from
  let coord_grid = vec3<f32>(id) - params.dt * v * params.dx;
  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 6) - Advect temperature
  textureStore(temperatureAdvection_temperatureOut, id, textureSampleLevel(temperatureAdvection_temperatureIn, temperatureAdvection_sampler, coord_normalized, 0.0)); // Use new vars
}


@compute @workgroup_size(4,4,4)
fn densityAdvection(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from advectDensity
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let v = textureLoad(densityAdvection_velocityIn, id, 0).xyz; // Use new var

  // Calculate the coordinate to sample from
  let coord_grid = vec3<f32>(id) - params.dt * v * params.dx;

  // Get texture dimensions (including halo)
  let tex_dims = vec3<f32>(uniforms.gridSize + 2u * HALO_SIZE);

  // Convert source coordinate to normalized [0.0, 1.0] space
  let coord_normalized = (coord_grid + 0.5) / tex_dims;

  // (Eq. 7) - Advect density
  textureStore(densityAdvection_densityOut, id, textureSampleLevel(densityAdvection_densityIn, densityAdvection_sampler, coord_normalized, 0.0)); // Use new vars
}

// REMOVED computePressure - replaced by pressureIteration (Jacobi)

@compute @workgroup_size(4,4,4)
fn divergenceCalculation(@builtin(global_invocation_id) id : vec3<u32>) { // Renamed from computeDivergence
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  let vp_x1 = textureLoad(divergenceCalculation_velocityIn, pos + vec3<i32>(1,0,0), 0).xyz; // Use new var
  let vn_x1 = textureLoad(divergenceCalculation_velocityIn, pos - vec3<i32>(1,0,0), 0).xyz; // Use new var
  let vp_y1 = textureLoad(divergenceCalculation_velocityIn, pos + vec3<i32>(0,1,0), 0).xyz; // Use new var
  let vn_y1 = textureLoad(divergenceCalculation_velocityIn, pos - vec3<i32>(0,1,0), 0).xyz; // Use new var
  let vp_z1 = textureLoad(divergenceCalculation_velocityIn, pos + vec3<i32>(0,0,1), 0).xyz; // Use new var
  let vn_z1 = textureLoad(divergenceCalculation_velocityIn, pos - vec3<i32>(0,0,1), 0).xyz; // Use new var

  let dv_dx = vp_x1.x - vn_x1.x;
  let dv_dy = vp_y1.y - vn_y1.y;
  let dv_dz = vp_z1.z - vn_z1.z;

  let div = (dv_dx + dv_dy + dv_dz) / (2.0 * params.dx);

  textureStore(divergenceCalculation_divergenceOut, id, vec4f(div, 0.0, 0.0, 0.0)); // Use new var
  return;
}

@compute @workgroup_size(4,4,4)
fn pressureGradientSubtraction(@builtin(global_invocation_id) id: vec3<u32>) { // Renamed from applyPressureGradient
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  let dp_dx = (textureLoad(pressureGradientSubtraction_pressureIn, pos + vec3<i32>(1,0,0), 0).x - // Use new var
               textureLoad(pressureGradientSubtraction_pressureIn, pos - vec3<i32>(1,0,0), 0).x) / (2.0 * params.dx); // Use new var

  let dp_dy = (textureLoad(pressureGradientSubtraction_pressureIn, pos + vec3<i32>(0,1,0), 0).x - // Use new var
               textureLoad(pressureGradientSubtraction_pressureIn, pos - vec3<i32>(0,1,0), 0).x) / (2.0 * params.dx); // Use new var

  let dp_dz = (textureLoad(pressureGradientSubtraction_pressureIn, pos + vec3<i32>(0,0,1), 0).x - // Use new var
               textureLoad(pressureGradientSubtraction_pressureIn, pos - vec3<i32>(0,0,1), 0).x) / (2.0 * params.dx); // Use new var

  // Subtract pressure gradient to make velocity divergence-free (equation 5)
  let velocity = textureLoad(pressureGradientSubtraction_velocityIn, id, 0).xyz; // Use new var
  let pressureGradient = vec3f(dp_dx, dp_dy, dp_dz);

  textureStore(pressureGradientSubtraction_velocityOut, id, vec4f(velocity - pressureGradient, 0.0)); // Use new var
}

@compute @workgroup_size(4,4,4)
fn vorticityForceApplication(@builtin(global_invocation_id) id: vec3<u32>) { // Renamed from applyVorticityForce
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let velocity = textureLoad(vorticityForceApplication_velocityIn, id, 0).xyz; // Use new var
  let force = textureLoad(vorticityForceApplication_forceIn, id, 0).xyz; // Use new var

  textureStore(vorticityForceApplication_velocityOut, id, vec4f(velocity + force * params.dt, 0.0)); // Use new var
}

@compute @workgroup_size(4,4,4)
fn pressureIteration(@builtin(global_invocation_id) id: vec3<u32>) { // Renamed from solvePressureJacobi
  // Boundary check: Skip halo cells
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  // Cast id to vec3<i32> for consistent type operations
  let pos = vec3<i32>(id);

  // Use Laplacian and then isolate pressure; Laplacian can be found here (https://scicomp.stackexchange.com/questions/35920/3d-laplacian-operator)
  // p(i,j,k) = (p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k) + p(i,j,k+1) + p(i,j,k-1) - h²·div(i,j,k))/6
  let h_squared = params.dx * params.dx;

  let p = (
    textureLoad(pressureIteration_pressureIn, pos + vec3<i32>(1,0,0), 0).x + // Use new var
    textureLoad(pressureIteration_pressureIn, pos - vec3<i32>(1,0,0), 0).x + // Use new var
    textureLoad(pressureIteration_pressureIn, pos + vec3<i32>(0,1,0), 0).x + // Use new var
    textureLoad(pressureIteration_pressureIn, pos - vec3<i32>(0,1,0), 0).x + // Use new var
    textureLoad(pressureIteration_pressureIn, pos + vec3<i32>(0,0,1), 0).x + // Use new var
    textureLoad(pressureIteration_pressureIn, pos - vec3<i32>(0,0,1), 0).x - // Use new var
    h_squared * textureLoad(pressureIteration_divergenceIn, id, 0).x // Use new var
  ) / 6.0;

  textureStore(pressureIteration_pressureOut, id, vec4f(p, 0.0, 0.0, 0.0)); // Use new var
}
