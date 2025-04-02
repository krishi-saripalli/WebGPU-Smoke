@import "common.wgsl";

@group(1) @binding(0) var densityIn: texture_3d<f32>;
@group(1) @binding(1) var densityOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) {
    return;
  }

  let density = textureLoad(densityIn, id, 0).x;
  textureStore(densityOut, id, vec4f(density, 0.0, 0.0, 0.0));
} 