export function createUniformBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  simulationParamsBuffer: GPUBuffer
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: simulationParamsBuffer } },
    ],
  });
}

export interface TexturePair {
  texture: GPUTexture;
  view: GPUTextureView;
}

export function createAdvectionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTextureView,
  densityIn: GPUTextureView,
  temperatureIn: GPUTextureView,
  sampler: GPUSampler,
  velocityOut: GPUTextureView,
  densityOut: GPUTextureView,
  temperatureOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn },
      { binding: 1, resource: densityIn },
      { binding: 2, resource: temperatureIn },
      { binding: 3, resource: sampler },
      { binding: 4, resource: velocityOut },
      { binding: 5, resource: densityOut },
      { binding: 6, resource: temperatureOut },
    ],
  });
}

export function createForcesBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTextureView,
  temperatureIn: GPUTextureView,
  densityIn: GPUTextureView,
  velocityOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn },
      { binding: 1, resource: temperatureIn },
      { binding: 2, resource: densityIn },
      { binding: 3, resource: velocityOut },
    ],
  });
}

export function createVorticityCalculationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTextureView,
  vorticityOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn },
      { binding: 1, resource: vorticityOut },
    ],
  });
}

export function createVorticityConfinementBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTextureView,
  vorticityIn: GPUTextureView,
  velocityOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: vorticityIn },
      { binding: 1, resource: velocityIn },
      { binding: 2, resource: velocityOut },
    ],
  });
}

export function createDivergenceCalculationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTextureView,
  divergenceOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn },
      { binding: 1, resource: divergenceOut },
    ],
  });
}

export function createPressureIterationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  divergenceIn: GPUTextureView,
  pressureIn: GPUTextureView,
  pressureOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: pressureIn },
      { binding: 1, resource: divergenceIn },
      { binding: 2, resource: pressureOut },
    ],
  });
}

export function createPressureGradientSubtractionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  pressureIn: GPUTextureView,
  velocityIn: GPUTextureView,
  velocityOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn },
      { binding: 1, resource: pressureIn },
      { binding: 2, resource: velocityOut },
    ],
  });
}

export function createReinitializationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  temperatureIn: GPUTextureView,
  temperatureOut: GPUTextureView,
  densityIn: GPUTextureView,
  densityOut: GPUTextureView,
  velocityIn: GPUTextureView,
  velocityOut: GPUTextureView
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: temperatureIn },
      { binding: 1, resource: temperatureOut },
      { binding: 2, resource: densityIn },
      { binding: 3, resource: densityOut },
      { binding: 4, resource: velocityIn },
      { binding: 5, resource: velocityOut },
    ],
  });
}

export function createRenderBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  densityIn: GPUTextureView,
  sampler: GPUSampler
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: densityIn },
      { binding: 1, resource: sampler },
    ],
  });
}
