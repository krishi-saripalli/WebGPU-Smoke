import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';
import { loadShader } from '@/utils/shaderLoader';
import { generateWireframe } from '@/utils/generateWireframe';
import { generateSlices } from '@/utils/generateSlices';

export interface RenderPipelineResources {
  renderPipeline: GPURenderPipeline;
  computePipeline: GPUComputePipeline;
  wireframeVertexBuffer: GPUBuffer;
  wireframeIndexBuffer: GPUBuffer;
  slicesVertexBuffer: GPUBuffer;
  slicesIndexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  multisampleTexture: GPUTexture;
  computeBindGroupA: GPUBindGroup;
  computeBindGroupB: GPUBindGroup;
  renderBindGroupA: GPUBindGroup;
  renderBindGroupB: GPUBindGroup;
  uniformBindGroup: GPUBindGroup;
  wireframeIndexCount: number;
  slicesIndexCount: number;
  camera: Camera;
  gridSize: number;
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

        const gridSize = 8;

        /////////////////////////////////////////////////////////////////////////
        // Uniform buffer
        /////////////////////////////////////////////////////////////////////////
        // Size is 2 matrices (view and projection) * 16 floats per matrix * 4 bytes per float
        // + 2 Vec3s * 4 bytes per float
        // prettier-ignore
        const PADDING = 8;
        const uniformBufferSize = 2 * 16 * 4 + 2 * 3 * 4 + PADDING;
        const uniformBuffer = device.createBuffer({
          size: uniformBufferSize,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const camera = new Camera({
          position: new Vec3([0, 1, 4]),
          forward: new Vec3([0, 0, -1]),
          up: new Vec3([0, 1, 0]),
          heightAngle: Math.PI / 2,
          near: 0.1,
          far: 100,
          aspect: 1,
        });

        const viewMatrix = camera.getViewMatrix();
        const projectionMatrix = camera.getProjectionMatrix();
        let offset = 0;
        device.queue.writeBuffer(uniformBuffer, offset, viewMatrix as Float32Array);
        offset += 16 * 4;
        device.queue.writeBuffer(uniformBuffer, offset, projectionMatrix as Float32Array);
        offset += 16 * 4;
        device.queue.writeBuffer(
          uniformBuffer,
          offset,
          new Uint32Array([gridSize, gridSize, gridSize])
        );
        offset += 3 * 4;
        device.queue.writeBuffer(uniformBuffer, offset, new Float32Array([...camera.getForward()]));

        /////////////////////////////////////////////////////////////////////////
        // Density texture A
        /////////////////////////////////////////////////////////////////////////
        const densityTextureA = device.createTexture({
          size: [gridSize, gridSize, gridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        const densityTextureB = device.createTexture({
          size: [gridSize, gridSize, gridSize],
          dimension: '3d',
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST,
        });

        /////////////////////////////////////////////////////////////////////////
        // Bind group layouts
        /////////////////////////////////////////////////////////////////////////
        // 1. Uniform-only layout
        const uniformBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' },
            },
          ],
        });

        // 2. Compute layout (reads one 3D texture and writes another)
        const computeTexturesBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: 'float', viewDimension: '3d' },
            },
            {
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              storageTexture: {
                access: 'write-only',
                format: 'rgba16float',
                viewDimension: '3d',
              },
            },
          ],
        });

        // 3. Render layout (samples from a 3D texture)
        const renderTexturesBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.FRAGMENT,
              texture: {
                sampleType: 'float',
                viewDimension: '3d',
              },
            },
            {
              binding: 1,
              visibility: GPUShaderStage.FRAGMENT,
              sampler: { type: 'filtering' },
            },
          ],
        });

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
          ],
        });

        // For compute pass: one bind group for "A→B", another for "B→A"
        const computeBindGroupA = device.createBindGroup({
          layout: computeTexturesBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: densityTextureA.createView(), // read from A
            },
            {
              binding: 1,
              resource: densityTextureB.createView(), // write to B
            },
          ],
        });
        const computeBindGroupB = device.createBindGroup({
          layout: computeTexturesBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: densityTextureB.createView(), // read from B
            },
            {
              binding: 1,
              resource: densityTextureA.createView(), // write to A
            },
          ],
        });

        // For render pass: one bind group that samples from A, another from B
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
        const renderPipeline = device.createRenderPipeline({
          label: 'Wireframe',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, renderTexturesBindGroupLayout],
          }),
          vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [
              {
                arrayStride: 12,
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
                blend: {
                  color: {
                    srcFactor: 'src-alpha',
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
          multisample: {
            count: 4,
          },
          primitive: {
            topology: 'triangle-list',
          },
        });

        /////////////////////////////////////////////////////////////////////////
        // Compute pipeline
        /////////////////////////////////////////////////////////////////////////
        const computePipeline = device.createComputePipeline({
          label: 'Smoke Simulation',
          layout: device.createPipelineLayout({
            bindGroupLayouts: [uniformBindGroupLayout, computeTexturesBindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeMain',
          },
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
          renderPipeline,
          computePipeline,
          wireframeVertexBuffer,
          wireframeIndexBuffer,
          slicesVertexBuffer,
          slicesIndexBuffer,
          uniformBuffer,
          multisampleTexture,
          computeBindGroupA,
          computeBindGroupB,
          renderBindGroupA,
          renderBindGroupB,
          uniformBindGroup,
          wireframeIndexCount: wireframeIndices.length,
          slicesIndexCount: slicesIndices.length,
          camera,
          gridSize,
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
