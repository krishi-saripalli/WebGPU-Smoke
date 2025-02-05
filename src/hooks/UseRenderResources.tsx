import { useEffect, useState } from "react";
import { WebGPUState } from "./useWebGPU";
import { shader } from "@/shaders/shader";

export interface RenderPipelineResources {
    pipeline: GPURenderPipeline;
    vertexBuffer: GPUBuffer;
  }


export const useRenderResources = (webGPUState: WebGPUState | null) => {
    const [resources, setResources] = useState<RenderPipelineResources | null>(
      null
    );
  
    useEffect(() => {
      if (!webGPUState) return;
      const { device, canvasFormat } = webGPUState;
  
      // Create vertex buffer
      const vertices = new Float32Array([
        0.0,
        0.5, // top
        -0.5,
        -0.5, // bottom left
        0.5,
        -0.5, // bottom right
      ]);
      const vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(vertexBuffer, 0, vertices);
  
      // Create shader module and pipeline
      const shaderModule = device.createShaderModule({ code: shader });
      const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: shaderModule,
          entryPoint: "vertexMain",
          buffers: [
            {
              arrayStride: 8,
              attributes: [
                {
                  format: "float32x2",
                  offset: 0,
                  shaderLocation: 0,
                },
              ],
            },
          ],
        },
        fragment: {
          module: shaderModule,
          entryPoint: "fragmentMain",
          targets: [{ format: canvasFormat }],
        },
        primitive: {
          topology: "point-list",
        },
      });
  
      setResources({ pipeline, vertexBuffer });
    }, [webGPUState]);
  
    return resources;
  };