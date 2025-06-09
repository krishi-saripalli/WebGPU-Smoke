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
        let interior_neighbor_coord = vec3<u32>(
              clamp(id.x, HALO_SIZE, uniforms.gridSize.x + HALO_SIZE - 1u),
              clamp(id.y, HALO_SIZE, uniforms.gridSize.y + HALO_SIZE - 1u),
              clamp(id.z, HALO_SIZE, uniforms.gridSize.z + HALO_SIZE - 1u)
          );
        velocity = textureLoad(velocityIn, interior_neighbor_coord, 0).xyz;
       
        let top_bottom = id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE;
        let left_right = id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE;
        let front_back = id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE;
        var boundary_mask: u32 = 0u;
        if (top_bottom) { boundary_mask = boundary_mask | 1u; }
        if (left_right) { boundary_mask = boundary_mask | 2u; }
        if (front_back) { boundary_mask = boundary_mask | 4u; }

        switch (boundary_mask) {
            case 1u: { 
                velocity.y = 0.0;
                break;
            }
            case 2u: {
                velocity.x = 0.0;
                break;
            }
            case 4u: { 
                velocity.z = 0.0;
                break;
            }
            default: {
                break;
            }
        }
    }
    
    else {
      let internalX = id.x - HALO_SIZE;
      let internalY = id.y - HALO_SIZE;
      let internalZ = id.z - HALO_SIZE;

      let source_centerX = uniforms.gridSize.x / 2;
      let source_centerZ = uniforms.gridSize.z / 2;
      let radius = 6u;
      let heatHeight = uniforms.gridSize.y / 4; 
      let densityHeight = 3u;        

      if (
        internalY <= heatHeight &&
        internalX >= source_centerX - radius &&
        internalX <= source_centerX + radius &&
        internalZ >= source_centerZ - radius &&
        internalZ <= source_centerZ + radius
      ) {
        temperature = params.ambientTemperature + 400.0;

        if (internalY <= densityHeight) { 
          density = 0.7;
        }
      }
    }

    textureStore(temperatureOut, id, vec4f(temperature , 0.0, 0.0, 0.0));
    textureStore(densityOut, id, vec4f(density , 0.0, 0.0, 0.0));
    textureStore(velocityOut, id, vec4f(velocity , 0.0));
}

   
    
