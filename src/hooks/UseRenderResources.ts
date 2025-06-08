import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';
import { loadShader, loadShaderModules } from '@/utils/shader-loader';
import { generateSlices } from '@/utils/generate-slices';
import { initializeSimulationData } from '@/utils/initializion';
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';
import * as layouts from '@/utils/layouts';
import { SHADER_PATHS, RENDER_ENTRY_POINTS, createComputePipeline } from '@/utils/shader-modules';

export interface RenderPipelineResources {
  slicesPipeline: GPURenderPipeline;
  externalForcesStepPipeline: GPUComputePipeline;
  vorticityCalculationPipeline: GPUComputePipeline;
  vorticityConfinementPipeline: GPUComputePipeline;
  velocityAdvectionPipeline: GPUComputePipeline;
  temperatureAdvectionPipeline: GPUComputePipeline;
  densityAdvectionPipeline: GPUComputePipeline;
  divergenceCalculationPipeline: GPUComputePipeline;
  pressureIterationPipeline: GPUComputePipeline;
  pressureGradientSubtractionPipeline: GPUComputePipeline;
  reinitializationPipeline: GPUComputePipeline;
  slicesVertexBuffer: GPUBuffer;
  slicesIndexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  simulationParamsBuffer: GPUBuffer;
  multisampleTexture: GPUTexture;
  slicesIndexCount: number;
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
}

export const useRenderResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initResources() {
      try {
        if (!webGPUState) return;
        const { device, canvasFormat } = webGPUState;

        const commonShaderCode = await loadShader(SHADER_PATHS.common);
        if (!commonShaderCode || commonShaderCode.trim().length === 0) {
          throw new Error('Invalid common shader code: shader code is empty');
        }

        const shaderModules = await loadShaderModules(device, SHADER_PATHS);

        const gridSize = 50;

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
          position: new Vec3([0, 0, 2]),
          forward: new Vec3([0, 0, -1]),
          up: new Vec3([0, 1, 0]),
          heightAngle: Math.PI / 2,
          near: 0.1,
          far: 100,
          aspect: 1,
        });

        uniformsView.set({
          viewMatrix: camera.getViewMatrix() as unknown as Float32Array,
          projectionMatrix: camera.getProjectionMatrix() as unknown as Float32Array,
          gridSize: [internalGridSize, internalGridSize, internalGridSize],
          cameraForward: camera.getForward(),
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
          vorticityStrength: 1.0,
          buoyancyAlpha: 12.0,
          buoyancyBeta: 18.0,
          ambientTemperature: 1.0,
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
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC,
        });

        const temperatureTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
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
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC,
        });

        const temperatureTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const pressureTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const pressureTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const divergenceTextureA = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const divergenceTextureB = device.createTexture({
          size: [totalGridSize, totalGridSize, totalGridSize],
          dimension: '3d',
          format: 'r32float',
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

        const uniformBindGroupLayout = layouts.createUniformBindGroupLayout(device);
        const renderTexturesBindGroupLayout = layouts.createRenderBindGroupLayout(device);
        const densityCopyBindGroupLayout = layouts.createDensityCopyBindGroupLayout(device);
        const externalForcesStepBindGroupLayout =
          layouts.createExternalForcesStepBindGroupLayout(device);
        const vorticityCalculationBindGroupLayout =
          layouts.createVorticityCalculationBindGroupLayout(device);
        const vorticityConfinementBindGroupLayout =
          layouts.createVorticityConfinementBindGroupLayout(device);
        const velocityAdvectionBindGroupLayout =
          layouts.createVelocityAdvectionBindGroupLayout(device);
        const temperatureAdvectionBindGroupLayout =
          layouts.createTemperatureAdvectionBindGroupLayout(device);
        const densityAdvectionBindGroupLayout =
          layouts.createDensityAdvectionBindGroupLayout(device);
        const divergenceCalculationBindGroupLayout =
          layouts.createDivergenceCalculationBindGroupLayout(device);
        const pressureIterationBindGroupLayout =
          layouts.createPressureIterationBindGroupLayout(device);
        const pressureGradientSubtractionBindGroupLayout =
          layouts.createPressureGradientSubtractionBindGroupLayout(device);
        const reinitializationBindGroupLayout =
          layouts.createReinitializationBindGroupLayout(device);

        const sampler = device.createSampler({
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
          addressModeW: 'clamp-to-edge',
          magFilter: 'linear',
          minFilter: 'linear',
          mipmapFilter: 'nearest',
        });

        const { vertexPositions: slicesVertexPositions, indicesList: slicesIndicesList } =
          generateSlices(internalGridSize);

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

        let shaderModule: GPUShaderModule;
        try {
          shaderModule = device.createShaderModule({ code: commonShaderCode });
        } catch (e) {
          throw new Error(
            `Failed to create shader module: ${e instanceof Error ? e.message : String(e)}`
          );
        }

        const baseRenderPipelineDescriptor: Omit<
          GPURenderPipelineDescriptor,
          'label' | 'fragment' | 'depthStencil'
        > = {
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, renderTexturesBindGroupLayout],
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
          primitive: { topology: 'triangle-list' },
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

        const externalForcesStepPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createExternalForcesStepBindGroupLayout(device),
          ],
        });
        const externalForcesStepPipeline = createComputePipeline(
          device,
          shaderModules,
          'externalForcesStep',
          externalForcesStepPipelineLayout,
          'External Forces Step'
        );

        const vorticityCalculationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createVorticityCalculationBindGroupLayout(device),
          ],
        });
        const vorticityCalculationPipeline = createComputePipeline(
          device,
          shaderModules,
          'vorticityCalculation',
          vorticityCalculationPipelineLayout,
          'Vorticity Calculation'
        );

        const vorticityConfinementPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createVorticityConfinementBindGroupLayout(device),
          ],
        });
        const vorticityConfinementPipeline = createComputePipeline(
          device,
          shaderModules,
          'vorticityConfinement',
          vorticityConfinementPipelineLayout,
          'Vorticity Confinement'
        );

        const velocityAdvectionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createVelocityAdvectionBindGroupLayout(device),
          ],
        });
        const velocityAdvectionPipeline = createComputePipeline(
          device,
          shaderModules,
          'velocityAdvection',
          velocityAdvectionPipelineLayout,
          'Velocity Advection'
        );

        const temperatureAdvectionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createTemperatureAdvectionBindGroupLayout(device),
          ],
        });
        const temperatureAdvectionPipeline = createComputePipeline(
          device,
          shaderModules,
          'temperatureAdvection',
          temperatureAdvectionPipelineLayout,
          'Temperature Advection'
        );

        const densityAdvectionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createDensityAdvectionBindGroupLayout(device),
          ],
        });
        const densityAdvectionPipeline = createComputePipeline(
          device,
          shaderModules,
          'densityAdvection',
          densityAdvectionPipelineLayout,
          'Density Advection'
        );

        const divergenceCalculationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createDivergenceCalculationBindGroupLayout(device),
          ],
        });
        const divergenceCalculationPipeline = createComputePipeline(
          device,
          shaderModules,
          'divergenceCalculation',
          divergenceCalculationPipelineLayout,
          'Divergence Calculation'
        );

        const pressureIterationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createPressureIterationBindGroupLayout(device),
          ],
        });
        const pressureIterationPipeline = createComputePipeline(
          device,
          shaderModules,
          'pressureIteration',
          pressureIterationPipelineLayout,
          'Pressure Iteration (Jacobi)'
        );

        const pressureGradientSubtractionPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createPressureGradientSubtractionBindGroupLayout(device),
          ],
        });
        const pressureGradientSubtractionPipeline = createComputePipeline(
          device,
          shaderModules,
          'pressureGradientSubtraction',
          pressureGradientSubtractionPipelineLayout,
          'Pressure Gradient Subtraction'
        );

        const reinitializationPipelineLayout = device.createPipelineLayout({
          bindGroupLayouts: [
            uniformBindGroupLayout,
            layouts.createReinitializationBindGroupLayout(device),
          ],
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
          externalForcesStepPipeline,
          vorticityCalculationPipeline,
          vorticityConfinementPipeline,
          velocityAdvectionPipeline,
          temperatureAdvectionPipeline,
          densityAdvectionPipeline,
          divergenceCalculationPipeline,
          pressureIterationPipeline,
          pressureGradientSubtractionPipeline,
          reinitializationPipeline,
          slicesVertexBuffer,
          slicesIndexBuffer,
          uniformBuffer,
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
  }, [webGPUState]);

  if (error) {
    console.warn('Render resources initialization failed:', error);
  }

  return resources;
};
