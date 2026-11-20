@import "common.wgsl";

@group(1) @binding(0) var velocityIn: texture_3d<f32>;
@group(1) @binding(1) var force: texture_3d<f32>;
@group(1) @binding(2) var smoke: texture_3d<f32>;
@group(1) @binding(3) var velocityOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= uniforms.gridSize)) { return; }
  let p = vec3i(id) + 1;
  let adjacent = vec3f(
    textureLoad(force, p + X, 0).x,
    textureLoad(force, p + Y, 0).y,
    textureLoad(force, p + Z, 0).z
  );
  let faceForce = 0.5 * (textureLoad(force, p, 0).xyz + adjacent);
  let state = 0.5 * (textureLoad(smoke, p, 0).xy + textureLoad(smoke, p + Y, 0).xy);
  let density = state.x;
  let temp = state.y;
  let up = vec3f(0.0, 1.0, 0.0);
  let buoyancy = -params.buoyancyAlpha * density * up +
                 params.buoyancyBeta * (temp - params.ambientTemperature) * up;
  let velocity = textureLoad(velocityIn, p, 0).xyz + (faceForce + buoyancy) * params.dt;
  textureStore(velocityOut, p, vec4f(wallVelocity(p, velocity), 0));
}
