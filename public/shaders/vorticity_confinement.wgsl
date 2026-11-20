@import "common.wgsl";

@group(1) @binding(0) var vorticityIn: texture_3d<f32>;
@group(1) @binding(1) var forceOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let pos = vec3i(id) + 1;

  // gradient of the *magnitude* of vorticity (Eq. 10)
  let dvort_mag_dx = textureLoad(vorticityIn, cell(pos + X), 0).w -
                    textureLoad(vorticityIn, cell(pos - X), 0).w;
  let dvort_mag_dy = textureLoad(vorticityIn, cell(pos + Y), 0).w -
                    textureLoad(vorticityIn, cell(pos - Y), 0).w;
  let dvort_mag_dz = textureLoad(vorticityIn, cell(pos + Z), 0).w -
                    textureLoad(vorticityIn, cell(pos - Z), 0).w;
  var grad = vec3f(dvort_mag_dx, dvort_mag_dy, dvort_mag_dz) / (2.0 * params.dx);

  let gradLength = length(grad);

  if (gradLength > 1e-6) {
    grad = grad / gradLength;
  } else {
    grad = vec3f(0.0);
  }

  let vorticity = textureLoad(vorticityIn, pos, 0).xyz;
  let vorticity_conf = params.dx * params.vorticityStrength * cross(grad, vorticity);

  var force = vorticity_conf;
  if (uniforms.brushStrength > 0.0) {
    let center = (vec3f(id) + 0.5 - 0.5 * vec3f(uniforms.gridSize)) * (2.0 * params.dx);
    let viewPosition = (uniforms.modelViewMatrix * vec4f(center, 1)).xy;
    let start = uniforms.brushStroke.xy;
    let delta = uniforms.brushStroke.zw - start;
    let t = clamp(dot(viewPosition - start, delta) / max(dot(delta, delta), 1e-8), 0.0, 1.0);
    let distance = length(viewPosition - (start + t * delta));
    let weight = 1.0 - smoothstep(0.0, 0.22, distance);
    let direction = (vec4f(delta, 0, 0) * uniforms.modelViewMatrix).xyz;
    force += weight * uniforms.brushStrength * direction / params.dt;
  }
  textureStore(forceOut, pos, vec4f(force, 0));
}
