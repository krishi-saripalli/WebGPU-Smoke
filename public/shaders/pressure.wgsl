@import "common.wgsl";

@group(1) @binding(0) var pressureIn: texture_3d<f32>;
@group(1) @binding(1) var divergence: texture_3d<f32>;
@group(1) @binding(2) var pressureOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let p = vec3i(id) + 1;
  // Neumann boundary condition dp/dn = 0
  // Use Laplacian and then isolate pressure; 3D discrete Laplacian can be found here (https://scicomp.stackexchange.com/questions/35920/3d-laplacian-operator)
  // p(i,j,k) = (p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k) + p(i,j,k+1) + p(i,j,k-1) - h²·div(i,j,k))/6
  let h_squared = params.dx * params.dx;

  let pressure = (
    textureLoad(pressureIn, cell(p + X), 0).x +
    textureLoad(pressureIn, cell(p - X), 0).x +
    textureLoad(pressureIn, cell(p + Y), 0).x +
    textureLoad(pressureIn, cell(p - Y), 0).x +
    textureLoad(pressureIn, cell(p + Z), 0).x +
    textureLoad(pressureIn, cell(p - Z), 0).x -
    h_squared * textureLoad(divergence, p, 0).x
  ) / 6.0;
  textureStore(pressureOut, p, vec4f(mix(textureLoad(pressureIn, p, 0).x, pressure, 2.0 / 3.0), 0, 0, 0));
}

@group(1) @binding(3) var coarseRhs: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var coarseInitial: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn restrictResidual(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize / 2u)) { return; }
  var residual = 0.0;
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let p = vec3i(id * 2u) + 1 + vec3i(x, y, z);
        let neighbors =
          textureLoad(pressureIn, cell(p + X), 0).x + textureLoad(pressureIn, cell(p - X), 0).x +
          textureLoad(pressureIn, cell(p + Y), 0).x + textureLoad(pressureIn, cell(p - Y), 0).x +
          textureLoad(pressureIn, cell(p + Z), 0).x + textureLoad(pressureIn, cell(p - Z), 0).x;
        let laplacian = (neighbors - 6.0 * textureLoad(pressureIn, p, 0).x) / (params.dx * params.dx);
        residual += textureLoad(divergence, p, 0).x - laplacian;
      }
    }
  }
  let c = vec3i(id) + 1;
  textureStore(coarseRhs, c, vec4f(residual * 0.125, 0, 0, 0));
  textureStore(coarseInitial, c, vec4f(0));
}

@group(1) @binding(5) var correction: texture_3d<f32>;

fn coarseValue(p: vec3i) -> f32 {
  return textureLoad(correction, clamp(p, vec3i(1), vec3i(uniforms.gridSize / 2u)), 0).x;
}

@compute @workgroup_size(4, 4, 4)
fn prolongate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }

  let coord = vec3f(id) * 0.5 + 0.75;
  let c = vec3i(floor(coord));
  let t = fract(coord);
  let lo = mix(mix(coarseValue(c), coarseValue(c+X), t.x),
               mix(coarseValue(c+Y), coarseValue(c+X+Y), t.x), t.y);
  let hi = mix(mix(coarseValue(c+Z), coarseValue(c+X+Z), t.x),
               mix(coarseValue(c+Y+Z), coarseValue(c+X+Y+Z), t.x), t.y);
  let p = vec3i(id) + 1;
  textureStore(pressureOut, p, vec4f(textureLoad(pressureIn, p, 0).x + mix(lo, hi, t.z), 0, 0, 0));
}
