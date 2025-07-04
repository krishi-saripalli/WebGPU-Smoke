@import "common.wgsl";

@group(1) @binding(0) var pressureIn: texture_3d<min16float>;
@group(1) @binding(1) var divergenceIn: texture_3d<min16float>;
@group(1) @binding(2) var pressureOut: texture_storage_3d<min16float_storage, write>;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

  // Neumann boundary condition dp/dn = 0
  let is_halo_cell = id.x < HALO_SIZE || id.x >= uniforms.gridSize.x + HALO_SIZE ||
      id.y < HALO_SIZE || id.y >= uniforms.gridSize.y + HALO_SIZE ||
      id.z < HALO_SIZE || id.z >= uniforms.gridSize.z + HALO_SIZE;
  if (is_halo_cell) {

    let interior_neighbor_coord = vec3<u32>(
            clamp(id.x, HALO_SIZE, uniforms.gridSize.x + HALO_SIZE - 1u),
            clamp(id.y, HALO_SIZE, uniforms.gridSize.y + HALO_SIZE - 1u),
            clamp(id.z, HALO_SIZE, uniforms.gridSize.z + HALO_SIZE - 1u)
        );
    let interior_neighbor_value = textureLoad(pressureIn, interior_neighbor_coord, 0).x;
    textureStore(pressureOut, id, vec4f(interior_neighbor_value, 0.0, 0.0, 0.0));
    return;
  }

  let pos = vec3<i32>(id);

  // Use Laplacian and then isolate pressure; 3D discrete Laplacian can be found here (https://scicomp.stackexchange.com/questions/35920/3d-laplacian-operator)
  // p(i,j,k) = (p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k) + p(i,j,k+1) + p(i,j,k-1) - h²·div(i,j,k))/6
  let h_squared = params.dx * params.dx;

  let p = (
    textureLoad(pressureIn, pos + vec3<i32>(1,0,0), 0).x +
    textureLoad(pressureIn, pos - vec3<i32>(1,0,0), 0).x +
    textureLoad(pressureIn, pos + vec3<i32>(0,1,0), 0).x +
    textureLoad(pressureIn, pos - vec3<i32>(0,1,0), 0).x +
    textureLoad(pressureIn, pos + vec3<i32>(0,0,1), 0).x +
    textureLoad(pressureIn, pos - vec3<i32>(0,0,1), 0).x -
    h_squared * textureLoad(divergenceIn, id, 0).x
  ) / 6.0;

  textureStore(pressureOut, id, vec4f(p, 0.0, 0.0, 0.0));
} 