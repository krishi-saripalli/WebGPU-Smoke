import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';
import { loadShader, loadShaderModules } from '@/utils/shader-loader';
import { generateBox, generateWireframe } from '@/utils/geometry';
import { initializeSimulationData } from '@/utils/initializion';
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';
import * as layouts from '@/utils/layouts';
import { SHADER_PATHS, RENDER_ENTRY_POINTS, createComputePipeline } from '@/utils/shader-modules';

export interface RenderPipelineResources {
  slicesPipeline: GPURenderPipeline;
  wireframePipeline: GPURenderPipeline;
  externalForcesStepPipeline: GPUComputePipeline;
  vorticityCalculationPipeline: GPUComputePipeline;
  vorticityConfinementPipeline: GPUComputePipeline;
  advectionPipeline: GPUComputePipeline;
  divergenceCalculationPipeline: GPUComputePipeline;
  pressureIterationPipeline: GPUComputePipeline;
  pressureGradientSubtractionPipeline: GPUComputePipeline;
  reinitializationPipeline: GPUComputePipeline;
  slicesVertexBuffer: GPUBuffer;
  slicesIndexBuffer: GPUBuffer;
  wireframeVertexBuffer: GPUBuffer;
  wireframeIndexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  simulationParamsBuffer: GPUBuffer;
  multisampleTexture: GPUTexture;
  slicesIndexCount: number;
  wireframeIndexCount: number;
  camera: Camera;
  gridSize: number;
  totalGridSize: number;
  halosSize: number;
  sampler: GPUSampler;
  densityTextureA: GPUTexture;
  densityTextureB: GPUTexture;
  velocityTextureA: GPUTexture;
  velocityTextureB: GPUTexture;
  temperatureTextureA: GPUTexture;
  temperatureTextureB: GPUTexture;
  pressureTextureA: GPUTexture;
  pressureTextureB: GPUTexture;
  divergenceTextureA: GPUTexture;
  divergenceTextureB: GPUTexture;
  vorticityTextureA: GPUTexture;
  vorticityTextureB: GPUTexture;
  vorticityForceTextureA: GPUTexture;
  vorticityForceTextureB: GPUTexture;
  densityTextureAView: GPUTextureView;
  densityTextureBView: GPUTextureView;
  velocityTextureAView: GPUTextureView;
  velocityTextureBView: GPUTextureView;
  temperatureTextureAView: GPUTextureView;
  temperatureTextureBView: GPUTextureView;
  pressureTextureAView: GPUTextureView;
  pressureTextureBView: GPUTextureView;
  divergenceTextureAView: GPUTextureView;
  divergenceTextureBView: GPUTextureView;
  vorticityTextureAView: GPUTextureView;
  vorticityTextureBView: GPUTextureView;
  vorticityForceTextureAView: GPUTextureView;
  vorticityForceTextureBView: GPUTextureView;
  uniformsView: any;
  uniformConstants: {
    gridSize: [number, number, number];
    lightPosition: [number, number, number];
    lightIntensity: [number, number, number];
    ratio: [number, number, number];
    absorption: number;
    scattering: number;
  };
  bindGroupLayouts: {
    uniform: GPUBindGroupLayout;
    advection: GPUBindGroupLayout;
    externalForces: GPUBindGroupLayout;
    vorticityCalculation: GPUBindGroupLayout;
    vorticityConfinement: GPUBindGroupLayout;
    divergenceCalculation: GPUBindGroupLayout;
    pressureIteration: GPUBindGroupLayout;
    pressureGradientSubtraction: GPUBindGroupLayout;
    reinitialization: GPUBindGroupLayout;
    render: GPUBindGroupLayout;
  };
}

const commonShaderCode = await loadShader(SHADER_PATHS.common);
if (!commonShaderCode || commonShaderCode.trim().length === 0) {
  throw new Error('Invalid common shader code: shader code is empty');
}
export const shaderDefs = makeShaderDataDefinitions(commonShaderCode);

export const useRenderResources = (
  webGPUState: WebGPUState | null,
  shaderHeader: string,
  min16float: string,
  min16floatStorage: string
) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initResources() {
      try {
        if (!webGPUState) return;
        const { device, canvasFormat } = webGPUState;

        const shaderModules = await loadShaderModules(
          device,
          SHADER_PATHS,
          shaderHeader,
          min16floatStorage
        );

        const gridSize = 100;

        const shaderDefs = makeShaderDataDefinitions(commonShaderCode);

        const internalGridSize = gridSize;
        const halosSize = 1;
        const totalGridSize = internalGridSize + 2 * halosSize;

        const uniformsView = makeStructuredView(shaderDefs.uniforms.uniforms);

        const uniformBuffer = device.createBuffer({
          size: uniformsView.arrayBuffer.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const camera = new Camera({
          position: new Vec3([-1.5, 0, 1.5]),
          forward: new Vec3([1, 0, -1]),
          up: new Vec3([0, 1, 0]),
          heightAngle: Math.PI / 2,
          near: 0.1,
          far: 100,
          aspect: 1,
        });
        const viewMatrix = camera.getViewMatrix() as Float32Array;
        const projectionMatrix = camera.getProjectionMatrix() as Float32Array;
        const cameraForward = camera.getForward() as Float32Array;
        const cameraPos = camera.getPosition() as Float32Array;
        const lightPosition: [number, number, number] = [-0.5, 0.8, 0.0];
        const lightIntensity: [number, number, number] = [3 * 4.0, 3 * 3.5, 3 * 3.0];
        const ratio: [number, number, number] = [1.0, 1.0, 1.0];
        const absorption: number = 4.0;
        const scattering: number = 5.0;

        uniformsView.set({
          viewMatrix: viewMatrix,
          projectionMatrix: projectionMatrix,
          gridSize: [internalGridSize, internalGridSize, internalGridSize],
          cameraForward: cameraForward,
          cameraPos: cameraPos,
          lightPosition: lightPosition,
          lightIntensity: lightIntensity,
          ratio: ratio,
          absorption: absorption,
          scattering: scattering,
        });

        device.queue.writeBuffer(uniformBuffer, 0, uniformsView.arrayBuffer);

        const simulationParamsView = makeStructuredView(shaderDefs.uniforms.params);

        const simulationParamsBuffer = device.createBuffer({
          size: simulationParamsView.arrayBuffer.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        simulationParamsView.set({
          dt: 0.01,
          dx: 1.0 / internalGridSize,
          vorticityStrength: 8.0,
          buoyancyAlpha: 19.8,
          buoyancyBeta: 33.0,
          ambientTemperature: 0.0,
        });

        device.queue.writeBuffer(simulationParamsBuffer, 0, simulationParamsView.arrayBuffer);

        const velocityTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const densityTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC,
        });

        const temperatureTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const velocityTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const densityTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC,
        });

        const temperatureTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const pressureTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const pressureTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const divergenceTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const divergenceTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: min16floatStorage as GPUTextureFormat,
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const vorticityTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const vorticityTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const vorticityForceTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const vorticityForceTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const {
          densityData: initDensityData,
          temperatureData: initTemperatureData,
          velocityData: initVelocityData,
          pressureData: initPressureData,
        } = initializeSimulationData(totalGridSize, halosSize, internalGridSize);

        device.queue.writeTexture(
          { texture: densityTextureA },
          initDensityData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        //TODO: do these bytes (4 and 8 become 2 and 4) need to de dynamic based on the min16float type?
        // or can I write in full precision from the javascript side and still re-interpret the values as f16 if enabled?
        device.queue.writeTexture(
          { texture: temperatureTextureA },
          initTemperatureData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: velocityTextureA },
          initVelocityData,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: pressureTextureA },
          initPressureData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: densityTextureB },
          initDensityData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: temperatureTextureB },
          initTemperatureData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: velocityTextureB },
          initVelocityData,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        device.queue.writeTexture(
          { texture: pressureTextureB },
          initPressureData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        const zeroDataRGBA16F = new Float32Array(totalGridSize * totalGridSize * totalGridSize * 4);
        const zeroDataR32F = new Float32Array(totalGridSize * totalGridSize * totalGridSize);

        device.queue.writeTexture(
          { texture: divergenceTextureA },
          zeroDataR32F,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );
        device.queue.writeTexture(
          { texture: divergenceTextureB },
          zeroDataR32F,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );
        device.queue.writeTexture(
          { texture: vorticityTextureA },
          zeroDataRGBA16F,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );
        device.queue.writeTexture(
          { texture: vorticityTextureB },
          zeroDataRGBA16F,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );
        device.queue.writeTexture(
          { texture: vorticityForceTextureA },
          zeroDataRGBA16F,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );
        device.queue.writeTexture(
          { texture: vorticityForceTextureB },
          zeroDataRGBA16F,
          { bytesPerRow: totalGridSize * 8, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        const densityTextureAView = densityTextureA.createView();
        const densityTextureBView = densityTextureB.createView();
        const velocityTextureAView = velocityTextureA.createView();
        const velocityTextureBView = velocityTextureB.createView();
        const temperatureTextureAView = temperatureTextureA.createView();
        const temperatureTextureBView = temperatureTextureB.createView();
        const pressureTextureAView = pressureTextureA.createView();
        const pressureTextureBView = pressureTextureB.createView();
        const divergenceTextureAView = divergenceTextureA.createView();
        const divergenceTextureBView = divergenceTextureB.createView();
        const vorticityTextureAView = vorticityTextureA.createView();
        const vorticityTextureBView = vorticityTextureB.createView();
        const vorticityForceTextureAView = vorticityForceTextureA.createView();
        const vorticityForceTextureBView = vorticityForceTextureB.createView();

        const uniformBindGroupLayout = layouts.createUniformBindGroupLayout(device);
        const renderBindGroupLayout = layouts.createRenderBindGroupLayout(device);

        const advectionBindGroupLayout = layouts.createAdvectionBindGroupLayout(
          device,
          min16floatStorage
        );
        const externalForcesBindGroupLayout =
          layouts.createExternalForcesStepBindGroupLayout(device);
        const vorticityCalculationBindGroupLayout =
          layouts.createVorticityCalculationBindGroupLayout(device);
        const vorticityConfinementBindGroupLayout =
          layouts.createVorticityConfinementBindGroupLayout(device);
        const divergenceCalculationBindGroupLayout =
          layouts.createDivergenceCalculationBindGroupLayout(device, min16floatStorage);
        const pressureIterationBindGroupLayout = layouts.createPressureIterationBindGroupLayout(
          device,
          min16floatStorage
        );
        const pressureGradientSubtractionBindGroupLayout =
          layouts.createPressureGradientSubtractionBindGroupLayout(device);
        const reinitializationBindGroupLayout = layouts.createReinitializationBindGroupLayout(
          device,
          min16floatStorage
        );

        const sampler = device.createSampler({
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
          addressModeW: 'clamp-to-edge',
          magFilter: 'linear',
          minFilter: 'linear',
          mipmapFilter: 'nearest',
        });

        const { vertexPositions: slicesVertexPositions, indicesList: slicesIndicesList } =
          generateBox();

        const slicesVertices = new Float32Array(slicesVertexPositions);
        const slicesIndices = new Uint32Array(slicesIndicesList);

        const slicesVertexBuffer = device.createBuffer({
          size: slicesVertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(slicesVertexBuffer, 0, slicesVertices);

        const slicesIndexBuffer = device.createBuffer({
          size: slicesIndices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(slicesIndexBuffer, 0, slicesIndices);

        const { vertexPositions: wireframeVertexPositions, indicesList: wireframeIndicesList } =
          generateWireframe();

        const wireframeVertices = new Float32Array(wireframeVertexPositions);
        const wireframeIndices = new Uint32Array(wireframeIndicesList);

        const wireframeVertexBuffer = device.createBuffer({
          size: wireframeVertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(wireframeVertexBuffer, 0, wireframeVertices);

        const wireframeIndexBuffer = device.createBuffer({
          size: wireframeIndices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(wireframeIndexBuffer, 0, wireframeIndices);
        const baseRenderPipelineDescriptor: Omit<
          GPURenderPipelineDescriptor,
          'label' | 'fragment' | 'depthStencil'
        > = {
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, renderBindGroupLayout],
          }),
          vertex: {
            module: shaderModules[RENDER_ENTRY_POINTS.vertex.module],
            entryPoint: RENDER_ENTRY_POINTS.vertex.entryPoint,
            buffers: [
              {
                arrayStride: 12,
                attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
              },
            ],
          },
          multisample: { count: 4 },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
        };

        const slicesPipeline = device.createRenderPipeline({
          ...baseRenderPipelineDescriptor,
          label: 'Slices Rendering',
          fragment: {
            module: shaderModules[RENDER_ENTRY_POINTS.fragmentSlices.module],
            entryPoint: RENDER_ENTRY_POINTS.fragmentSlices.entryPoint,
            targets: [
              {
                format: canvasFormat,
                blend: {
                  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                },
              },
            ],
          },
        });

        const wireframePipeline = device.createRenderPipeline({
          ...baseRenderPipelineDescriptor,
          label: 'Wireframe Rendering',
          primitive: { topology: 'line-list' }, // This is the key change!
          fragment: {
            module: shaderModules[RENDER_ENTRY_POINTS.fragmentWireframe.module],
            entryPoint: RENDER_ENTRY_POINTS.fragmentWireframe.entryPoint,
            targets: [
              {
                format: canvasFormat,
              },
            ],
          },
        });

        const externalForcesStepPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, externalForcesBindGroupLayout],
        });
        const externalForcesStepPipeline = createComputePipeline(
          device,
          shaderModules,
          'externalForcesStep',
          externalForcesStepPipelineLayout,
          'External Forces Step'
        );

        const vorticityCalculationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, vorticityCalculationBindGroupLayout],
        });
        const vorticityCalculationPipeline = createComputePipeline(
          device,
          shaderModules,
          'vorticityCalculation',
          vorticityCalculationPipelineLayout,
          'Vorticity Calculation'
        );

        const vorticityConfinementPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, vorticityConfinementBindGroupLayout],
        });
        const vorticityConfinementPipeline = createComputePipeline(
          device,
          shaderModules,
          'vorticityConfinement',
          vorticityConfinementPipelineLayout,
          'Vorticity Confinement'
        );

        const advectionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, advectionBindGroupLayout],
        });
        const advectionPipeline = createComputePipeline(
          device,
          shaderModules,
          'advection',
          advectionPipelineLayout,
          'Advection'
        );

        const divergenceCalculationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, divergenceCalculationBindGroupLayout],
        });
        const divergenceCalculationPipeline = createComputePipeline(
          device,
          shaderModules,
          'divergenceCalculation',
          divergenceCalculationPipelineLayout,
          'Divergence Calculation'
        );

        const pressureIterationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, pressureIterationBindGroupLayout],
        });
        const pressureIterationPipeline = createComputePipeline(
          device,
          shaderModules,
          'pressureIteration',
          pressureIterationPipelineLayout,
          'Pressure Iteration (Jacobi)'
        );

        const pressureGradientSubtractionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, pressureGradientSubtractionBindGroupLayout],
        });
        const pressureGradientSubtractionPipeline = createComputePipeline(
          device,
          shaderModules,
          'pressureGradientSubtraction',
          pressureGradientSubtractionPipelineLayout,
          'Pressure Gradient Subtraction'
        );

        const reinitializationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [uniformBindGroupLayout, reinitializationBindGroupLayout],
        });

        const reinitializationPipeline = createComputePipeline(
          device,
          shaderModules,
          'reinitialization',
          reinitializationPipelineLayout,
          'Reinitialization'
        );

        const multisampleTexture = device.createTexture({
          format: canvasFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
          size: [
            webGPUState.context.getCurrentTexture().width,
            webGPUState.context.getCurrentTexture().height,
          ],
          sampleCount: 4,
        });

        setResources({
          slicesPipeline,
          wireframePipeline,
          externalForcesStepPipeline,
          vorticityCalculationPipeline,
          vorticityConfinementPipeline,
          advectionPipeline,
          divergenceCalculationPipeline,
          pressureIterationPipeline,
          pressureGradientSubtractionPipeline,
          reinitializationPipeline,
          slicesVertexBuffer,
          slicesIndexBuffer,
          uniformBuffer,
          wireframeVertexBuffer,
          wireframeIndexBuffer,
          wireframeIndexCount: wireframeIndices.length,
          simulationParamsBuffer,
          multisampleTexture,
          slicesIndexCount: slicesIndices.length,
          camera,
          gridSize: internalGridSize,
          totalGridSize,
          halosSize,
          sampler,
          densityTextureA,
          densityTextureB,
          velocityTextureA,
          velocityTextureB,
          temperatureTextureA,
          temperatureTextureB,
          pressureTextureA,
          pressureTextureB,
          divergenceTextureA,
          divergenceTextureB,
          vorticityTextureA,
          vorticityTextureB,
          vorticityForceTextureA,
          vorticityForceTextureB,
          densityTextureAView,
          densityTextureBView,
          velocityTextureAView,
          velocityTextureBView,
          temperatureTextureAView,
          temperatureTextureBView,
          pressureTextureAView,
          pressureTextureBView,
          divergenceTextureAView,
          divergenceTextureBView,
          vorticityTextureAView,
          vorticityTextureBView,
          vorticityForceTextureAView,
          vorticityForceTextureBView,
          uniformsView,
          uniformConstants: {
            gridSize: [internalGridSize, internalGridSize, internalGridSize],
            lightPosition: lightPosition,
            lightIntensity: lightIntensity,
            ratio: ratio,
            absorption: absorption,
            scattering: scattering,
          },
          bindGroupLayouts: {
            uniform: uniformBindGroupLayout,
            advection: advectionBindGroupLayout,
            externalForces: externalForcesBindGroupLayout,
            vorticityCalculation: vorticityCalculationBindGroupLayout,
            vorticityConfinement: vorticityConfinementBindGroupLayout,
            divergenceCalculation: divergenceCalculationBindGroupLayout,
            pressureIteration: pressureIterationBindGroupLayout,
            pressureGradientSubtraction: pressureGradientSubtractionBindGroupLayout,
            reinitialization: reinitializationBindGroupLayout,
            render: renderBindGroupLayout,
          },
        });
      } catch (e) {
        const error = e instanceof Error ? e : new Error(String(e));
        console.error('Failed to initialize render resources:', error);
        setError(error);
        setResources(null);
      }
    }

    initResources().catch((error) => {
      console.error('Unhandled error in initResources:', error);
      setError(error instanceof Error ? error : new Error(String(error)));
      setResources(null);
    });
  }, [webGPUState, min16float, min16floatStorage, shaderHeader]);

  if (error) {
    console.warn('Render resources initialization failed:', error);
  }

  return resources;
};
