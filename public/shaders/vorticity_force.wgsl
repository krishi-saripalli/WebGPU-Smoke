@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var vorticityIn: texture_3d<f32>;
@group(1) @binding(2) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let velocity = textureLoad(velocityIn, id, 0).xyz;
  let vorticity = textureLoad(vorticityIn, id, 0).xyz;
  textureStore(velocityOut, id, vec4f(velocity +  vorticity * params.dt, 0.0));
} 