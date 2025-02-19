import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { loadShader } from '@/utils/shaderLoader';

export interface ComputePipelineResources {
  computePipeline: GPUComputePipeline;
  densityBuffer: GPUBuffer;
  computeBindGroup: GPUBindGroup;
  gridSize: number;
}

export const useComputeResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<ComputePipelineResources | null>(null);

  useEffect(() => {
    async function initResources() {
      if (!webGPUState) return;
      const { device } = webGPUState;

      // Load compute shader
      const shaderCode = await loadShader('/shaders/compute.wgsl');
      const gridSize = 8;

      // Create density buffer for compute shader
      const densityBuffer = device.createBuffer({
        size: gridSize * gridSize * gridSize * 4, // 4 bytes per float
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Create compute bind group layout
      const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' },
          },
        ],
      });

      // Create compute bind group
      const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: { buffer: densityBuffer },
          },
        ],
      });

      // Create compute shader module and pipeline
      const computeShaderModule = device.createShaderModule({ code: shaderCode });
      const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [computeBindGroupLayout],
        }),
        compute: {
          module: computeShaderModule,
          entryPoint: 'computeMain',
        },
      });

      setResources({
        computePipeline,
        densityBuffer,
        computeBindGroup,
        gridSize,
      });
    }

    initResources().catch(console.error);
  }, [webGPUState]);

  return resources;
};
