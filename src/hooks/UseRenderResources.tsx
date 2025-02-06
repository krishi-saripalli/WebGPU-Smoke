import { useEffect, useState } from "react";
import { WebGPUState } from "./useWebGPU";
import { shader } from "@/shaders/shader";

export interface RenderPipelineResources {
    pipeline: GPURenderPipeline; //  the VAO
    vertexBuffer: GPUBuffer; // the VBO
  }


export const useRenderResources = (webGPUState: WebGPUState | null) => {
    const [resources, setResources] = useState<RenderPipelineResources | null>(
      null
    );
  
    useEffect(() => {
      if (!webGPUState) return;
      const { device, canvasFormat } = webGPUState;
  
      // Create vertex buffer for a single quad
      const points = new Float32Array([
        0.0,0.5, // top
        -0.5,-0.5, // bottom left
        0.5,-0.5, // bottom right
      ]);
      const vertexBuffer = device.createBuffer({
        size: points.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(vertexBuffer, 0, points);
  
      // Create shader module and pipeline
      const shaderModule = device.createShaderModule({ code: shader });
      const pipeline = device.createRenderPipeline({
        label: "Wireframe Points",
        layout: "auto",
        vertex: {
          module: shaderModule,
          entryPoint: "vertexMain",
          buffers: [
            {
              arrayStride: 8, // (2 floats * 4 bytes per float)
              stepMode: 'instance',
              attributes: [
                {
                  format: "float32x2", //vec2f
                  offset: 0,
                  shaderLocation: 0, //location(0) position: vec2f
                }
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
          topology: "triangle-list",
        },
      });
  
      setResources({ pipeline, vertexBuffer });
    }, [webGPUState]);
  
    return resources;
  };