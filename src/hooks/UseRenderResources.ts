import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';
import { loadShader } from '@/utils/shader-loader';
import { generateWireframe } from '@/utils/generate-wireframe';
import { generateSlices } from '@/utils/generate-slices';
import {
  initializeDensity,
  initializeTemperature,
  initializeSimulationData,
} from '@/utils/initializion';
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';
import * as layouts from '@/utils/layouts';

export interface RenderPipelineResources {
  wireframePipeline: GPURenderPipeline;
  slicesPipeline: GPURenderPipeline;
  computePipeline: GPUComputePipeline;
  computeMainPipeline: GPUComputePipeline;
  applyExternalForcesPipeline: GPUComputePipeline;
  computeVorticityPipeline: GPUComputePipeline;
  computeVorticityConfinementPipeline: GPUComputePipeline;
  applyVorticityForcePipeline: GPUComputePipeline;
  advectVelocityPipeline: GPUComputePipeline;
  advectTemperaturePipeline: GPUComputePipeline;
  advectDensityPipeline: GPUComputePipeline;
  computeDivergencePipeline: GPUComputePipeline;
  solvePressureJacobiPipeline: GPUComputePipeline;
  applyPressureGradientPipeline: GPUComputePipeline;
  wireframeVertexBuffer: GPUBuffer;
  wireframeIndexBuffer: GPUBuffer;
  slicesVertexBuffer: GPUBuffer;
  slicesIndexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  simulationParamsBuffer: GPUBuffer;
  multisampleTexture: GPUTexture;
  computeMainBindGroupA: GPUBindGroup;
  computeMainBindGroupB: GPUBindGroup;
  applyExternalForcesBindGroupA: GPUBindGroup;
  applyExternalForcesBindGroupB: GPUBindGroup;
  computeVorticityBindGroupA: GPUBindGroup;
  computeVorticityBindGroupB: GPUBindGroup;
  vorticityConfinementBindGroupA: GPUBindGroup;
  vorticityConfinementBindGroupB: GPUBindGroup;
  applyVorticityForceBindGroupA: GPUBindGroup;
  applyVorticityForceBindGroupB: GPUBindGroup;
  advectVelocityBindGroupA: GPUBindGroup;
  advectVelocityBindGroupB: GPUBindGroup;
  advectTemperatureBindGroupA: GPUBindGroup;
  advectTemperatureBindGroupB: GPUBindGroup;
  advectDensityBindGroupA: GPUBindGroup;
  advectDensityBindGroupB: GPUBindGroup;
  computeDivergenceBindGroupA: GPUBindGroup;
  computeDivergenceBindGroupB: GPUBindGroup;
  solvePressureJacobiBindGroupA: GPUBindGroup;
  solvePressureJacobiBindGroupB: GPUBindGroup;
  applyPressureGradientBindGroupA: GPUBindGroup;
  applyPressureGradientBindGroupB: GPUBindGroup;
  renderBindGroupA: GPUBindGroup;
  renderBindGroupB: GPUBindGroup;
  uniformBindGroup: GPUBindGroup;
  wireframeIndexCount: number;
  slicesIndexCount: number;
  camera: Camera;
  gridSize: number; // Internal grid size (for simulation)
  totalGridSize: number; // Total grid size including halos
  halosSize: number; // Size of the halo padding on each side
}

export const useRenderResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initResources() {
      try {
        if (!webGPUState) return;
        const { device, canvasFormat } = webGPUState;

        // Load and validate shader
        const shaderCode = await loadShader('/shaders/shader.wgsl');
        if (!shaderCode || shaderCode.trim().length === 0) {
          throw new Error('Invalid shader code: shader code is empty');
        }

        const gridSize = 100;

        // Parse shader definitions using webgpu-utils
        const shaderDefs = makeShaderDataDefinitions(shaderCode);

        /////////////////////////////////////////////////////////////////////////
        // Grid setup with halo cells
        /////////////////////////////////////////////////////////////////////////
        const internalGridSize = gridSize; // This is the internal/usable grid size
        const halosSize = 1; // We add 1 cell on each side as halo
        const totalGridSize = internalGridSize + 2 * halosSize; // Total grid size including halos

        /////////////////////////////////////////////////////////////////////////
        // Uniforms buffer (for camera and grid)
        /////////////////////////////////////////////////////////////////////////
        const uniformsView = makeStructuredView(shaderDefs.uniforms.uniforms);

        const uniformBuffer = device.createBuffer({
          size: uniformsView.arrayBuffer.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const camera = new Camera({
          position: new Vec3([0, 0, 4]),
          forward: new Vec3([0, 0, -1]),
          up: new Vec3([0, 1, 0]),
          heightAngle: Math.PI / 2,
          near: 0.1,
          far: 100,
          aspect: 1,
        });

        // Set uniform values using the structured view
        uniformsView.set({
          viewMatrix: camera.getViewMatrix() as unknown as Float32Array,
          projectionMatrix: camera.getProjectionMatrix() as unknown as Float32Array,
          gridSize: [internalGridSize, internalGridSize, internalGridSize],
          cameraForward: camera.getForward(),
        });

        // Upload the uniform buffer data
        device.queue.writeBuffer(uniformBuffer, 0, uniformsView.arrayBuffer);

        /////////////////////////////////////////////////////////////////////////
        // Simulation parameters buffer
        /////////////////////////////////////////////////////////////////////////
        const simulationParamsView = makeStructuredView(shaderDefs.uniforms.params);

        const simulationParamsBuffer = device.createBuffer({
          size: simulationParamsView.arrayBuffer.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Set simulation parameters
        simulationParamsView.set({
          dt: 0.033, // time step (30 FPS equivalent)
          dx: 0.1, // grid cell size
          vorticityStrength: 0.2, // vorticity confinement strength
          buoyancyAlpha: 0.1, // density influence on buoyancy
          buoyancyBeta: 0.2, // temperature influence on buoyancy
          ambientTemperature: 0.0, // ambient temperature
        });

        // Upload the simulation parameters buffer data
        device.queue.writeBuffer(simulationParamsBuffer, 0, simulationParamsView.arrayBuffer);

        /////////////////////////////////////////////////////////////////////////
        // Create all 3D textures needed for the simulation
        /////////////////////////////////////////////////////////////////////////
        // 1. Create textures for fluid state with ping-pong buffers (set A)
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
            GPUTextureUsage.COPY_DST,
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

        // 2. Create textures for fluid state with ping-pong buffers (set B)
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
            GPUTextureUsage.COPY_DST,
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

        // 3. Create auxiliary textures for simulation steps
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
          format: 'rgba16float', // Vector field needs all components
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

        /////////////////////////////////////////////////////////////////////////
        // Initialize simulation data with initial conditions
        /////////////////////////////////////////////////////////////////////////

        // Initialize simulation data using the utility function
        const {
          densityData: initDensityData,
          temperatureData: initTemperatureData,
          velocityData: initVelocityData,
          pressureData: initPressureData,
        } = initializeSimulationData(totalGridSize, halosSize, internalGridSize);

        // Upload initial conditions to the textures
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

        // Upload consolidated velocity data
        device.queue.writeTexture(
          { texture: velocityTextureA },
          initVelocityData,
          { bytesPerRow: totalGridSize * 16, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        // Upload pressure data
        device.queue.writeTexture(
          { texture: pressureTextureA },
          initPressureData,
          { bytesPerRow: totalGridSize * 4, rowsPerImage: totalGridSize },
          [totalGridSize, totalGridSize, totalGridSize]
        );

        /////////////////////////////////////////////////////////////////////////
        // Bind group layouts
        /////////////////////////////////////////////////////////////////////////
        // 1. Uniform-only layout
        const uniformBindGroupLayout = layouts.createUniformBindGroupLayout(device);

        // 2. Render layout
        const renderTexturesBindGroupLayout = layouts.createRenderBindGroupLayout(device);

        // 3. Compute layouts - one for each compute function
        const computeMainBindGroupLayout = layouts.createComputeMainBindGroupLayout(device);
        const applyExternalForcesBindGroupLayout =
          layouts.createApplyExternalForcesBindGroupLayout(device);
        const computeVorticityBindGroupLayout =
          layouts.createComputeVorticityBindGroupLayout(device);
        const vorticityConfinementBindGroupLayout =
          layouts.createVorticityConfinementBindGroupLayout(device);
        const applyVorticityForceBindGroupLayout =
          layouts.createApplyVorticityForceBindGroupLayout(device);
        // New bind group layouts for separate advection functions
        const advectVelocityBindGroupLayout = layouts.createAdvectVelocityBindGroupLayout(device);
        const advectTemperatureBindGroupLayout =
          layouts.createAdvectTemperatureBindGroupLayout(device);
        const advectDensityBindGroupLayout = layouts.createAdvectDensityBindGroupLayout(device);
        const computeDivergenceBindGroupLayout =
          layouts.createComputeDivergenceBindGroupLayout(device);
        const solvePressureJacobiBindGroupLayout =
          layouts.createSolvePressureJacobiBindGroupLayout(device);
        const applyPressureGradientBindGroupLayout =
          layouts.createApplyPressureGradientBindGroupLayout(device);

        /////////////////////////////////////////////////////////////////////////
        // Sampler
        /////////////////////////////////////////////////////////////////////////
        const sampler = device.createSampler({
          magFilter: 'linear',
          minFilter: 'linear',
          mipmapFilter: 'linear',
        });

        /////////////////////////////////////////////////////////////////////////
        // Bind groups
        /////////////////////////////////////////////////////////////////////////
        const uniformBindGroup = device.createBindGroup({
          layout: uniformBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: { buffer: uniformBuffer },
            },
            {
              binding: 1,
              resource: { buffer: simulationParamsBuffer },
            },
          ],
        });

        // For compute pass: one bind group for "A→B", another for "B→A"
        const computeMainBindGroupA = device.createBindGroup({
          layout: computeMainBindGroupLayout,
          entries: [
            // Source density texture
            { binding: 0, resource: densityTextureA.createView() },
            // Destination density texture
            { binding: 1, resource: densityTextureB.createView() },
          ],
        });

        const computeMainBindGroupB = device.createBindGroup({
          layout: computeMainBindGroupLayout,
          entries: [
            // Source density texture
            { binding: 0, resource: densityTextureB.createView() },
            // Destination density texture
            { binding: 1, resource: densityTextureA.createView() },
          ],
        });

        // For render pass: one bind group that samples from density textures
        const renderBindGroupA = device.createBindGroup({
          layout: renderTexturesBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: densityTextureA.createView(), // sample from A
            },
            {
              binding: 1,
              resource: sampler,
            },
          ],
        });
        const renderBindGroupB = device.createBindGroup({
          layout: renderTexturesBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: densityTextureB.createView(), // sample from B
            },
            {
              binding: 1,
              resource: sampler,
            },
          ],
        });

        const { vertexPositions: wireframeVertexPositions, indicesList: wireframeIndicesList } =
          generateWireframe(gridSize);

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

        const { vertexPositions: slicesVertexPositions, indicesList: slicesIndicesList } =
          generateSlices(gridSize);

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

        // create shader module and pipeline
        let shaderModule: GPUShaderModule;
        try {
          shaderModule = device.createShaderModule({ code: shaderCode });
        } catch (e) {
          throw new Error(
            `Failed to create shader module: ${e instanceof Error ? e.message : String(e)}`
          );
        }

        /////////////////////////////////////////////////////////////////////////
        // Render pipeline
        /////////////////////////////////////////////////////////////////////////
        const pipelineDescriptor: GPURenderPipelineDescriptor = {
          label: 'Wireframe',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, renderTexturesBindGroupLayout],
          }),
          vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [
              {
                arrayStride: 12, // only storing position, so 3 floats * 4 bytes
                stepMode: 'vertex',
                attributes: [
                  {
                    format: 'float32x3',
                    offset: 0,
                    shaderLocation: 0,
                  },
                ],
              },
            ],
          },
          fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [
              {
                format: canvasFormat,
              },
            ],
          },
          multisample: {
            count: 4,
          },
          primitive: {
            topology: 'triangle-list',
          },
        };

        const slicesPipeline = device.createRenderPipeline({
          ...pipelineDescriptor,
          label: 'Slices Rendering',
          fragment: {
            module: shaderModule,
            entryPoint: 'fragmentSlices',
            targets: [
              {
                format: canvasFormat,
                blend: {
                  color: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                  },
                  alpha: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                  },
                },
              },
            ],
          },
        });

        const wireframePipeline = device.createRenderPipeline({
          ...pipelineDescriptor,
          label: 'Wireframe Rendering',
        });

        /////////////////////////////////////////////////////////////////////////
        // Compute pipeline
        /////////////////////////////////////////////////////////////////////////
        const computeMainPipeline = device.createComputePipeline({
          label: 'Basic Compute',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, computeMainBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeMain',
          },
        });

        // External forces pipeline
        const applyExternalForcesPipeline = device.createComputePipeline({
          label: 'Apply External Forces',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, applyExternalForcesBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'applyExternalForces',
          },
        });

        // Create the bind groups for applyExternalForces
        const applyExternalForcesBindGroupA = device.createBindGroup({
          layout: applyExternalForcesBindGroupLayout,
          entries: [
            // Source velocity texture (vector field combined into one texture for reading)
            { binding: 0, resource: velocityTextureA.createView() },
            // Source temperature texture
            { binding: 1, resource: temperatureTextureA.createView() },
            // Source density texture
            { binding: 2, resource: densityTextureA.createView() },
            // Destination velocity texture (storage)
            { binding: 3, resource: velocityTextureB.createView() },
          ],
        });

        const applyExternalForcesBindGroupB = device.createBindGroup({
          layout: applyExternalForcesBindGroupLayout,
          entries: [
            // Source velocity texture (vector field combined into one texture for reading)
            { binding: 0, resource: velocityTextureB.createView() },
            // Source temperature texture
            { binding: 1, resource: temperatureTextureB.createView() },
            // Source density texture
            { binding: 2, resource: densityTextureB.createView() },
            // Destination velocity texture (storage)
            { binding: 3, resource: velocityTextureA.createView() },
          ],
        });

        // Vorticity computation pipeline
        const computeVorticityPipeline = device.createComputePipeline({
          label: 'Compute Vorticity',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, computeVorticityBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeVorticity',
          },
        });

        // Create bind groups for computeVorticity
        const computeVorticityBindGroupA = device.createBindGroup({
          layout: computeVorticityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Destination vorticity texture
            { binding: 1, resource: vorticityTextureB.createView() },
          ],
        });

        const computeVorticityBindGroupB = device.createBindGroup({
          layout: computeVorticityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Destination vorticity texture
            { binding: 1, resource: vorticityTextureA.createView() },
          ],
        });

        // Vorticity confinement pipeline
        const computeVorticityConfinementPipeline = device.createComputePipeline({
          label: 'Compute Vorticity Confinement',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, vorticityConfinementBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeVorticityConfinement',
          },
        });

        // Create bind groups for vorticityConfinement
        const vorticityConfinementBindGroupA = device.createBindGroup({
          layout: vorticityConfinementBindGroupLayout,
          entries: [
            // Source vorticity texture
            { binding: 0, resource: vorticityTextureA.createView() },
            // Destination vorticity force texture
            { binding: 1, resource: vorticityForceTextureB.createView() },
          ],
        });

        const vorticityConfinementBindGroupB = device.createBindGroup({
          layout: vorticityConfinementBindGroupLayout,
          entries: [
            // Source vorticity texture
            { binding: 0, resource: vorticityTextureB.createView() },
            // Destination vorticity force texture
            { binding: 1, resource: vorticityForceTextureA.createView() },
          ],
        });

        // Apply vorticity force pipeline
        const applyVorticityForcePipeline = device.createComputePipeline({
          label: 'Apply Vorticity Force',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, applyVorticityForceBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'applyVorticityForce',
          },
        });

        // Create bind groups for applyVorticityForce
        const applyVorticityForceBindGroupA = device.createBindGroup({
          layout: applyVorticityForceBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Source vorticity force texture
            { binding: 1, resource: vorticityForceTextureA.createView() },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureB.createView() },
          ],
        });

        const applyVorticityForceBindGroupB = device.createBindGroup({
          layout: applyVorticityForceBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Source vorticity force texture
            { binding: 1, resource: vorticityForceTextureB.createView() },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureA.createView() },
          ],
        });

        // New specialized advection pipelines
        // Velocity advection pipeline
        const advectVelocityPipeline = device.createComputePipeline({
          label: 'Advect Velocity',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, advectVelocityBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'advectVelocity',
          },
        });

        // Temperature advection pipeline
        const advectTemperaturePipeline = device.createComputePipeline({
          label: 'Advect Temperature',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, advectTemperatureBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'advectTemperature',
          },
        });

        // Density advection pipeline
        const advectDensityPipeline = device.createComputePipeline({
          label: 'Advect Density',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, advectDensityBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'advectDensity',
          },
        });

        // Create bind groups for each specialized advection

        // Velocity advection bind groups
        const advectVelocityBindGroupA = device.createBindGroup({
          layout: advectVelocityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Sampler
            { binding: 1, resource: sampler },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureB.createView() },
          ],
        });

        const advectVelocityBindGroupB = device.createBindGroup({
          layout: advectVelocityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Sampler
            { binding: 1, resource: sampler },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureA.createView() },
          ],
        });

        // Temperature advection bind groups
        const advectTemperatureBindGroupA = device.createBindGroup({
          layout: advectTemperatureBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Source temperature texture
            { binding: 1, resource: temperatureTextureA.createView() },
            // Sampler
            { binding: 2, resource: sampler },
            // Destination temperature texture
            { binding: 3, resource: temperatureTextureB.createView() },
          ],
        });

        const advectTemperatureBindGroupB = device.createBindGroup({
          layout: advectTemperatureBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Source temperature texture
            { binding: 1, resource: temperatureTextureB.createView() },
            // Sampler
            { binding: 2, resource: sampler },
            // Destination temperature texture
            { binding: 3, resource: temperatureTextureA.createView() },
          ],
        });

        // Density advection bind groups
        const advectDensityBindGroupA = device.createBindGroup({
          layout: advectDensityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Source density texture
            { binding: 1, resource: densityTextureA.createView() },
            // Sampler
            { binding: 2, resource: sampler },
            // Destination density texture
            { binding: 3, resource: densityTextureB.createView() },
          ],
        });

        const advectDensityBindGroupB = device.createBindGroup({
          layout: advectDensityBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Source density texture
            { binding: 1, resource: densityTextureB.createView() },
            // Sampler
            { binding: 2, resource: sampler },
            // Destination density texture
            { binding: 3, resource: densityTextureA.createView() },
          ],
        });

        // Divergence pipeline
        const computeDivergencePipeline = device.createComputePipeline({
          label: 'Compute Divergence',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, computeDivergenceBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeDivergence',
          },
        });

        // Create bind groups for computeDivergence
        const computeDivergenceBindGroupA = device.createBindGroup({
          layout: computeDivergenceBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Destination divergence texture
            { binding: 1, resource: divergenceTextureB.createView() },
          ],
        });

        const computeDivergenceBindGroupB = device.createBindGroup({
          layout: computeDivergenceBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Destination divergence texture
            { binding: 1, resource: divergenceTextureA.createView() },
          ],
        });

        // Pressure solver (Jacobi) pipeline
        const solvePressureJacobiPipeline = device.createComputePipeline({
          label: 'Solve Pressure Jacobi',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, solvePressureJacobiBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'solvePressureJacobi',
          },
        });

        // Create bind groups for solvePressureJacobi
        const solvePressureJacobiBindGroupA = device.createBindGroup({
          layout: solvePressureJacobiBindGroupLayout,
          entries: [
            // Source pressure texture
            { binding: 0, resource: pressureTextureA.createView() },
            // Source divergence texture
            { binding: 1, resource: divergenceTextureA.createView() },
            // Destination pressure texture
            { binding: 2, resource: pressureTextureB.createView() },
          ],
        });

        const solvePressureJacobiBindGroupB = device.createBindGroup({
          layout: solvePressureJacobiBindGroupLayout,
          entries: [
            // Source pressure texture
            { binding: 0, resource: pressureTextureB.createView() },
            // Source divergence texture
            { binding: 1, resource: divergenceTextureB.createView() },
            // Destination pressure texture
            { binding: 2, resource: pressureTextureA.createView() },
          ],
        });

        // Apply pressure gradient pipeline
        const applyPressureGradientPipeline = device.createComputePipeline({
          label: 'Apply Pressure Gradient',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, applyPressureGradientBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'applyPressureGradient',
          },
        });

        // Create bind groups for applyPressureGradient
        const applyPressureGradientBindGroupA = device.createBindGroup({
          layout: applyPressureGradientBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureA.createView() },
            // Source pressure texture
            { binding: 1, resource: pressureTextureA.createView() },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureB.createView() },
          ],
        });

        const applyPressureGradientBindGroupB = device.createBindGroup({
          layout: applyPressureGradientBindGroupLayout,
          entries: [
            // Source velocity texture
            { binding: 0, resource: velocityTextureB.createView() },
            // Source pressure texture
            { binding: 1, resource: pressureTextureB.createView() },
            // Destination velocity texture
            { binding: 2, resource: velocityTextureA.createView() },
          ],
        });

        /////////////////////////////////////////////////////////////////////////
        // Multisample texture
        /////////////////////////////////////////////////////////////////////////
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
          wireframePipeline,
          slicesPipeline,
          computePipeline: computeMainPipeline,
          computeMainPipeline,
          applyExternalForcesPipeline,
          computeVorticityPipeline,
          computeVorticityConfinementPipeline,
          applyVorticityForcePipeline,
          advectVelocityPipeline,
          advectTemperaturePipeline,
          advectDensityPipeline,
          computeDivergencePipeline,
          solvePressureJacobiPipeline,
          applyPressureGradientPipeline,
          wireframeVertexBuffer,
          wireframeIndexBuffer,
          slicesVertexBuffer,
          slicesIndexBuffer,
          uniformBuffer,
          simulationParamsBuffer,
          multisampleTexture,
          computeMainBindGroupA,
          computeMainBindGroupB,
          applyExternalForcesBindGroupA,
          applyExternalForcesBindGroupB,
          computeVorticityBindGroupA,
          computeVorticityBindGroupB,
          vorticityConfinementBindGroupA,
          vorticityConfinementBindGroupB,
          applyVorticityForceBindGroupA,
          applyVorticityForceBindGroupB,
          advectVelocityBindGroupA,
          advectVelocityBindGroupB,
          advectTemperatureBindGroupA,
          advectTemperatureBindGroupB,
          advectDensityBindGroupA,
          advectDensityBindGroupB,
          computeDivergenceBindGroupA,
          computeDivergenceBindGroupB,
          solvePressureJacobiBindGroupA,
          solvePressureJacobiBindGroupB,
          applyPressureGradientBindGroupA,
          applyPressureGradientBindGroupB,
          renderBindGroupA,
          renderBindGroupB,
          uniformBindGroup,
          wireframeIndexCount: wireframeIndices.length,
          slicesIndexCount: slicesIndices.length,
          camera,
          gridSize: internalGridSize,
          totalGridSize,
          halosSize,
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
