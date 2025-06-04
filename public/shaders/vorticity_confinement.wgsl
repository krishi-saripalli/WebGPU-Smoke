@import "common.wgsl";

@group(1) @binding(0) var vorticityIn: texture_3d<f32>;
@group(1) @binding(1) var forceOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let pos = vec3<i32>(id);

  // gradient of the *magnitude* of vorticity (Eq. 10)
  let dvort_mag_dx = length(textureLoad(vorticityIn, pos + vec3<i32>(1,0,0), 0).xyz) - 
                    length(textureLoad(vorticityIn, pos - vec3<i32>(1,0,0), 0).xyz);
  let dvort_mag_dy = length(textureLoad(vorticityIn, pos + vec3<i32>(0,1,0), 0).xyz) - 
                    length(textureLoad(vorticityIn, pos - vec3<i32>(0,1,0), 0).xyz);
  let dvort_mag_dz = length(textureLoad(vorticityIn, pos + vec3<i32>(0,0,1), 0).xyz) - 
                    length(textureLoad(vorticityIn, pos - vec3<i32>(0,0,1), 0).xyz);
  var grad = vec3<f32>(dvort_mag_dx, dvort_mag_dy, dvort_mag_dz) / (2.0 * params.dx);

  let gradLength = length(grad);
  
  if (gradLength > 1e-6) {
    grad = grad / gradLength;
  } else {
    grad = vec3<f32>(0.0);
  }

  let vorticity = textureLoad(vorticityIn, id, 0).xyz;
  let confinement = params.dx * cross(grad, vorticity);

  textureStore(forceOut, id, vec4f(confinement, 0.0));
} 