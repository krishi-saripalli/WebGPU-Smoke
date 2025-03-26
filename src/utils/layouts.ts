// This file contains bind group layouts for rendering and compute operations
// Each compute stage has its own layout tailored to its specific texture needs
// to stay within the WebGPU limit of 4 storage textures per bind group

// ----- UNIFORM BIND GROUP LAYOUT -----
export const createUniformBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Uniform buffer with camera and grid data
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
      // Simulation parameters
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
    entries: [
      // Density texture for visualization
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Sampler
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' },
      },
    ],
  });
};

// ----- COMPUTE BIND GROUP LAYOUTS -----

// Layout for the computeMain function (basic density copy operation)
export const createComputeMainBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source density texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Destination density texture (storage)
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for applyExternalForces
export const createApplyExternalForcesBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Source temperature texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Source density texture
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Destination velocity texture (storage)
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for computeVorticity
export const createComputeVorticityBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Destination vorticity texture (storage)
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for computeVorticityConfinement
export const createVorticityConfinementBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source vorticity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Destination vorticity force texture (storage)
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for applyVorticityForce
export const createApplyVorticityForceBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Source vorticity force texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Destination velocity texture (storage)
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for advectVelocity
export const createAdvectVelocityBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Sampler
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        sampler: { type: 'filtering' },
      },
      // Destination velocity texture (storage)
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for advectTemperature
export const createAdvectTemperatureBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Source temperature texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Sampler
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        sampler: { type: 'filtering' },
      },
      // Destination temperature texture (storage)
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for advectDensity
export const createAdvectDensityBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Source density texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Sampler
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        sampler: { type: 'filtering' },
      },
      // Destination density texture (storage)
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for computeDivergence
export const createComputeDivergenceBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Destination divergence texture (storage)
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for solvePressureJacobi
export const createSolvePressureJacobiBindGroupLayout = (device: GPUDevice): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source pressure texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Source divergence texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Destination pressure texture (storage)
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' },
      },
    ],
  });
};

// Layout for applyPressureGradient
export const createApplyPressureGradientBindGroupLayout = (
  device: GPUDevice
): GPUBindGroupLayout => {
  return device.createBindGroupLayout({
    entries: [
      // Source velocity texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '3d' },
      },
      // Source pressure texture
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float', viewDimension: '3d' },
      },
      // Destination velocity texture (storage)
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
      },
    ],
  });
};
