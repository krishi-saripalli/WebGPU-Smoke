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
export function createVelocityAdvectionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  sampler: GPUSampler,
  velocityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: velocityOut.createView() },
    ],
  });
}

export function createTemperatureAdvectionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  temperatureIn: GPUTexture,
  sampler: GPUSampler,
  temperatureOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: temperatureIn.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: temperatureOut.createView() },
    ],
  });
}

export function createDensityAdvectionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  densityIn: GPUTexture,
  sampler: GPUSampler,
  densityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: densityIn.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: densityOut.createView() },
    ],
  });
}

export function createForcesBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  temperatureIn: GPUTexture,
  densityIn: GPUTexture,
  velocityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: temperatureIn.createView() },
      { binding: 2, resource: densityIn.createView() },
      { binding: 3, resource: velocityOut.createView() },
    ],
  });
}

export function createVorticityCalculationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  vorticityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: vorticityOut.createView() },
    ],
  });
}

export function createVorticityConfinementBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  vorticityIn: GPUTexture,
  velocityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: vorticityIn.createView() },
      { binding: 1, resource: velocityIn.createView() },
      { binding: 2, resource: velocityOut.createView() },
    ],
  });
}

export function createDivergenceCalculationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  velocityIn: GPUTexture,
  divergenceOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: divergenceOut.createView() },
    ],
  });
}

export function createPressureIterationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  divergenceIn: GPUTexture,
  pressureIn: GPUTexture,
  pressureOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: pressureIn.createView() },
      { binding: 1, resource: divergenceIn.createView() },
      { binding: 2, resource: pressureOut.createView() },
    ],
  });
}

export function createPressureGradientSubtractionBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  pressureIn: GPUTexture,
  velocityIn: GPUTexture,
  velocityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: pressureIn.createView() },
      { binding: 2, resource: velocityOut.createView() },
    ],
  });
}

export function createReinitializationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  temperatureIn: GPUTexture,
  temperatureOut: GPUTexture,
  densityIn: GPUTexture,
  densityOut: GPUTexture,
  velocityIn: GPUTexture,
  velocityOut: GPUTexture
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: temperatureIn.createView() },
      { binding: 1, resource: temperatureOut.createView() },
      { binding: 2, resource: densityIn.createView() },
      { binding: 3, resource: densityOut.createView() },
      { binding: 4, resource: velocityIn.createView() },
      { binding: 5, resource: velocityOut.createView() },
    ],
  });
}

export function createRenderBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  densityIn: GPUTexture,
  sampler: GPUSampler
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: densityIn.createView() },
      { binding: 1, resource: sampler },
    ],
  });
}
