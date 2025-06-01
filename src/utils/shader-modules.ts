export const SHADER_PATHS = {
  common: '/shaders/common.wgsl',
  render: '/shaders/render.wgsl',
  density: '/shaders/density.wgsl',
  forces: '/shaders/forces.wgsl',
  vorticity: '/shaders/vorticity.wgsl',
  vorticityConfinement: '/shaders/vorticity_confinement.wgsl',
  vorticityForce: '/shaders/vorticity_force.wgsl',
  velocityAdvection: '/shaders/advect_velocity.wgsl',
  temperatureAdvection: '/shaders/advect_temperature.wgsl',
  densityAdvection: '/shaders/advect_density.wgsl',
  divergence: '/shaders/divergence.wgsl',
  pressure: '/shaders/pressure.wgsl',
  pressureGradient: '/shaders/pressure_gradient.wgsl',
  reinitialization: '/shaders/reinitialization.wgsl',
};

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
    module: 'vorticityConfinement',
    entryPoint: 'main',
  },
  vorticityForceApplication: {
    module: 'vorticityForce',
    entryPoint: 'main',
  },
  velocityAdvection: {
    module: 'velocityAdvection',
    entryPoint: 'main',
  },
  temperatureAdvection: {
    module: 'temperatureAdvection',
    entryPoint: 'main',
  },
  densityAdvection: {
    module: 'densityAdvection',
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
    module: 'pressureGradient',
    entryPoint: 'main',
  },
  reinitialization: {
    module: 'reinitialization',
    entryPoint: 'main',
  },
};

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
