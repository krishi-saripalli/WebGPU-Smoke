/////////////////////////////////////////////////////////////////////////
// Common definitions for all shaders
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