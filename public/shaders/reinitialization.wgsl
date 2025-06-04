//re-initializes a heat and density source in the exact same way as initialization.ts
@import "common.wgsl";

@group(1) @binding(0) var temperatureOut: texture_storage_3d<r32float, write>;
@group(1) @binding(1) var densityOut: texture_storage_3d<r32float, write>;
@group(1) @binding(2) var velocityIn: texture_3d<f32>;
@group(1) @binding(3) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

    let is_halo_cell = id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
                         id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
                         id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE;

    if (is_halo_cell) {
        var v_bc = textureLoad(velocityIn, id, 0).xyz; // Load current velocity state

        if (id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE) { // Bottom or Top
            v_bc.y = 0.0;
        }
        if (id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE) { // Left or Right
            v_bc.x = 0.0;
        }
        if (id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE) { // Front or Back
            v_bc.z = 0.0;
        }
        textureStore(velocityOut, id, vec4f(v_bc, 0.0));
        textureStore(temperatureOut, id, vec4f(params.ambientTemperature, 0.0, 0.0, 0.0));
        textureStore(densityOut, id, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }

    //var currentVelocity = textureLoad(velocityIn, id, 0).xyz;


    let centerX = uniforms.gridSize.x / 2;
    let centerZ = uniforms.gridSize.z / 2;
    let radius = 3u;
    let heatHeight = uniforms.gridSize.y; 
    let densityHeight = 3u;        

    if (
      id.y < heatHeight && 
      id.x >= centerX - radius &&
      id.x <= centerX + radius &&
      id.z >= centerZ - radius &&
      id.z <= centerZ + radius
    ) {
      textureStore(temperatureOut, id, vec4f(params.ambientTemperature + 50.0, 0.0, 0.0, 0.0));

      if (id.y <= densityHeight) { 
        textureStore(densityOut, id, vec4f(1.0, 0.0, 0.0, 0.0));
      }
    }

   
    
