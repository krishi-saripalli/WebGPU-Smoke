//re-initializes a heat and density source in the exact same way as initialization.ts
@import "common.wgsl";

@group(1) @binding(0) var temperatureOut: texture_storage_3d<r32float, write>;
@group(1) @binding(1) var densityOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

    if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE)
    {
        return;
    }

    let centerX = uniforms.gridSize.x / 2;
    let centerZ = uniforms.gridSize.z / 2;
    let radius = 3u;
    let heatHeight = uniforms.gridSize.y;
    let densityHeight = 3u;
    if (
      id.y <= heatHeight &&
      id.x >= centerX - radius &&
      id.x <= centerX + radius &&
      id.z >= centerZ - radius &&
      id.z <= centerZ + radius
    ) {
      textureStore(temperatureOut, id, vec4f(params.ambientTemperature + 90.0, 1.0, 0.0, 0.0)); // hot!!;

      //density source
      if (id.y <= densityHeight) {
        textureStore(densityOut, id, vec4f(0.2, 0.0, 0.0, 0.0));
      }
      return;
    }


}
