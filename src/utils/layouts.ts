// bind group layouts for rendering and compute operations

// ----- UNIFORM BIND GROUP LAYOUT -----
export const createUniformBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Uniform Bind Group Layout',
    entries: [
      // Uniform buffer with camera and grid data (@binding(0))
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
      // Simulation parameters (@binding(1))
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });
};

// ----- RENDER BIND GROUP LAYOUT -----
export const createRenderBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Render Bind Group Layout',
    entries: [
      // Density texture for visualization (@binding(0))
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Sampler (@binding(1))
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' },
      },
    ],
  });
};

// ----- COMPUTE BIND GROUP LAYOUTS -----

// Layout for densityCopy (replaces computeMain)
export const createDensityCopyBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Density Copy Bind Group Layout',
    entries: [
      // densityCopy_densityIn (@binding(0))
      {
        binding: 0, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // densityCopy_densityOut (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for externalForcesStep (replaces applyExternalForces)
export const createExternalForcesStepBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'External Forces Step Bind Group Layout',
    entries: [
      // externalForcesStep_velocityIn (@binding(0))
      {
        binding: 0, // Keep binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // externalForcesStep_temperatureIn (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // externalForcesStep_densityIn (@binding(2))
      {
        binding: 2, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // externalForcesStep_velocityOut (@binding(3))
      {
        binding: 3, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for vorticityCalculation (replaces computeVorticity)
export const createVorticityCalculationBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Vorticity Calculation Bind Group Layout',
    entries: [
      // vorticityCalculation_velocityIn (@binding(0))
      {
        binding: 0, // Keep binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // vorticityCalculation_vorticityOut (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for vorticityConfinementForce (replaces computeVorticityConfinement)
export const createVorticityConfinementBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Vorticity Confinement Force Bind Group Layout',
    entries: [
      // vorticityConfinementForce_vorticityIn (@binding(0))
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // vorticityConfinementForce_velocityIn (@binding(1))
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // vorticityConfinementForce_velocityOut (@binding(2))
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

export const createAdvectionBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Advection Bind Group Layout',
    entries: [
      // advection_velocityIn (@binding(0))
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // advection_densityIn (@binding(1))
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // advection_temperatureIn (@binding(2))
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // sampler (@binding(3))
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        sampler: { type: 'filtering' },
      },
      // advection_velocityOut (@binding(4))
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
      // advection_densityOut (@binding(5))
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
      // advection_temperatureOut (@binding(6))
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for divergenceCalculation (replaces computeDivergence)
export const createDivergenceCalculationBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Divergence Calculation Bind Group Layout',
    entries: [
      // divergenceCalculation_velocityIn (@binding(0))
      {
        binding: 0, // Keep binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // divergenceCalculation_divergenceOut (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for pressureIteration (replaces solvePressureJacobi)
export const createPressureIterationBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Pressure Iteration Bind Group Layout',
    entries: [
      // pressureIteration_pressureIn (@binding(0))
      {
        binding: 0, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // pressureIteration_divergenceIn (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // pressureIteration_pressureOut (@binding(2))
      {
        binding: 2, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for pressureGradientSubtraction (replaces applyPressureGradient)
export const createPressureGradientSubtractionBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Pressure Gradient Subtraction Bind Group Layout',
    entries: [
      // pressureGradientSubtraction_velocityIn (@binding(0))
      {
        binding: 0, // Keep binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // pressureGradientSubtraction_pressureIn (@binding(1))
      {
        binding: 1, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // pressureGradientSubtraction_velocityOut (@binding(2))
      {
        binding: 2, // Updated binding
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

export const createReinitializationBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    label: 'Reinitialization Bind Group Layout',
    entries: [
      // temperatureIn (@binding(0)))
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // temperatureOut (@binding(1))
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
      // densityIn (@binding(2))
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // densityOut (@binding(3))
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
      // velocityIn (@binding(4))
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // velocityOut (@binding(5))
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};
