import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';
import { loadShader } from '@/utils/shaderLoader';
import { generateWireframe } from '@/utils/generateWireframe';

export interface RenderPipelineResources {
  renderPipeline: GPURenderPipeline;
  computePipeline: GPUComputePipeline;
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  indexCount: number;
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
        // Size is 2 4x4 matrices (view and projection) * 16 floats per matrix * 4 bytes per float
        const uniformBuffer = device.createBuffer({
          size: 2 * 16 * 4,
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

        device.queue.writeBuffer(uniformBuffer, 0, viewMatrix as Float32Array);
        const offset = 16 * 4; // offset for projection matrix (after view matrix)
        device.queue.writeBuffer(uniformBuffer, offset, projectionMatrix as Float32Array);

        /////////////////////////////////////////////////////////////////////////
        // Density buffer
        /////////////////////////////////////////////////////////////////////////
        const densityBuffer = device.createBuffer({
          size: gridSize * gridSize * gridSize * 4, // 4 bytes per float
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: 'uniform' },
            },
            {
              binding: 1,
              visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
              buffer: { type: 'storage' },
            },
          ],
        });

        const bindGroup = device.createBindGroup({
          layout: bindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: { buffer: uniformBuffer },
            },
            {
              binding: 1,
              resource: { buffer: densityBuffer },
            },
          ],
        });

        const { vertexPositions, indicesList } = generateWireframe(gridSize);

        const vertices = new Float32Array(vertexPositions);
        const indices = new Uint32Array(indicesList);

        console.log('Vertices count:', vertices.length / 3);
        console.log('Quads count:', indices.length / 6);

        const vertexBuffer = device.createBuffer({
          size: vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, 0, vertices);

        const indexBuffer = device.createBuffer({
          size: indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(indexBuffer, 0, indices);

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
            bindGroupLayouts: [bindGroupLayout],
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
            targets: [{ format: canvasFormat }],
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
            bindGroupLayouts: [bindGroupLayout],
          }),
          compute: {
            module: shaderModule,
            entryPoint: 'computeMain',
          },
        });

        setResources({
          renderPipeline,
          computePipeline,
          vertexBuffer,
          indexBuffer,
          uniformBuffer,
          bindGroup,
          indexCount: indices.length,
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
