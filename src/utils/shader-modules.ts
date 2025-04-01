// This file helps manage shader modules and their entry points

// Define the shader module paths
export const SHADER_PATHS = {
  common: '/shaders/common.wgsl',
  render: '/shaders/render.wgsl',
  density: '/shaders/density.wgsl',
  forces: '/shaders/forces.wgsl',
  vorticity: '/shaders/vorticity.wgsl',
  vortConfinement: '/shaders/vorticity_confinement.wgsl',
  vortForce: '/shaders/vorticity_force.wgsl',
  velAdvect: '/shaders/advect_velocity.wgsl',
  tempAdvect: '/shaders/advect_temperature.wgsl',
  densAdvect: '/shaders/advect_density.wgsl',
  divergence: '/shaders/divergence.wgsl',
  pressure: '/shaders/pressure.wgsl',
  pressureGrad: '/shaders/pressure_gradient.wgsl',
};

// Map compute entry points to their modules
export const COMPUTE_ENTRY_POINTS = {
  densityCopy: {
    module: 'density',
    entryPoint: 'main',
  },
  externalForcesStep: {
    module: 'forces',
    entryPoint: 'main',
  },
  vorticityCalculation: {
    module: 'vorticity',
    entryPoint: 'main',
  },
  vorticityConfinementForce: {
    module: 'vortConfinement',
    entryPoint: 'main',
  },
  vorticityForceApplication: {
    module: 'vortForce',
    entryPoint: 'main',
  },
  velocityAdvection: {
    module: 'velAdvect',
    entryPoint: 'main',
  },
  temperatureAdvection: {
    module: 'tempAdvect',
    entryPoint: 'main',
  },
  densityAdvection: {
    module: 'densAdvect',
    entryPoint: 'main',
  },
  divergenceCalculation: {
    module: 'divergence',
    entryPoint: 'main',
  },
  pressureIteration: {
    module: 'pressure',
    entryPoint: 'main',
  },
  pressureGradientSubtraction: {
    module: 'pressureGrad',
    entryPoint: 'main',
  },
};

// Map render entry points to their modules
export const RENDER_ENTRY_POINTS = {
  vertex: {
    module: 'render',
    entryPoint: 'vertexMain',
  },
  fragment: {
    module: 'render',
    entryPoint: 'fragmentMain',
  },
  fragmentSlices: {
    module: 'render',
    entryPoint: 'fragmentSlices',
  },
};

export function createComputePipeline(
  device: GPUDevice,
  shaderModules: Record<string, GPUShaderModule>,
  pipelineName: keyof typeof COMPUTE_ENTRY_POINTS,
  pipelineLayout: GPUPipelineLayout,
  label: string
): GPUComputePipeline {
  const { module, entryPoint } = COMPUTE_ENTRY_POINTS[pipelineName];
  const shaderModule = shaderModules[module];

  if (!shaderModule) {
    throw new Error(`Shader module '${module}' not found for pipeline '${pipelineName}'`);
  }

  return device.createComputePipeline({
    label,
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint,
    },
  });
}
