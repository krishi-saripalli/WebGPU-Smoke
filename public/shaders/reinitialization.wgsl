//re-initializes a heat and density source in the exact same way as initialization.ts
@import "common.wgsl";

@group(1) @binding(0) var temperatureIn: texture_3d<f32>;     
@group(1) @binding(1) var temperatureOut: texture_storage_3d<r32float, write>;
@group(1) @binding(2) var densityIn: texture_3d<f32>;
@group(1) @binding(3) var densityOut: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var velocityIn: texture_3d<f32>;
@group(1) @binding(5) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

    let is_halo_cell = id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
                         id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
                         id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE;

    var temperature = textureLoad(temperatureIn, id, 0).x;
    var density = textureLoad(densityIn, id, 0).x;
    var velocity = textureLoad(velocityIn, id, 0).xyz;

    if (is_halo_cell) {

        if (id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE) { // Bottom or Top
            velocity.y = 0.0;
        }
        if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE) { // Left or Right
            velocity.x = 0.0;
        }
        if (id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) { // Front or Back
            velocity.z = 0.0;
        }
        temperature = 0.0;
        density = 0.0;
    }

    let centerX = (uniforms.gridSize.x + 2*HALO_SIZE) / 2 ;
    let centerZ = (uniforms.gridSize.z + 2*HALO_SIZE) / 2;
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
      temperature = params.ambientTemperature + 100.0;

      if (id.y <= densityHeight) { 
        density = 1.0;
      }
    }
    textureStore(temperatureOut, id, vec4f(temperature, 0.0, 0.0, 0.0));
    textureStore(densityOut, id, vec4f(density, 0.0, 0.0, 0.0));
    textureStore(velocityOut, id, vec4f(velocity, 0.0));
}

   
    
