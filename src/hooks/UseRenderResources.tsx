import { useEffect, useState } from "react";
import { WebGPUState } from "./useWebGPU";
import { shader } from "@/shaders/shader";

export interface RenderPipelineResources {
  pipeline: GPURenderPipeline; //  the VAO
  vertexBuffer: GPUBuffer; // the VBO
  indexBuffer: GPUBuffer;
}

export const useRenderResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(
    null
  );

  useEffect(() => {
    if (!webGPUState) return;
    const { device, canvasFormat } = webGPUState;

    const gridSize = 10;
    const lineWidth = 0.005;

    // each line will be represented by a rectangle, which is 2 triangles (4 vertices)
    //TODO: DO the same for the y and z axes
    const vertexCount = gridSize * gridSize * 4;
    const vertices = new Float32Array(vertexCount * 3); // x, y, z
    let i = 0;
    for (let y = 0; y < gridSize; y++) {
      for (let z = 0; z < gridSize; z++) {
        const xStart = -1.0;
        const xEnd = 1.0;
        const yPos = -1.0 + 2.0 * (y / (gridSize - 1));
        const zPos = z / (gridSize - 1);

        // four vertices for the line (rectangle)
        // top left
        vertices[i++] = xStart;
        vertices[i++] = yPos + lineWidth;
        vertices[i++] = zPos;
        
        // top right
        vertices[i++] = xEnd;
        vertices[i++] = yPos + lineWidth;
        vertices[i++] = zPos;
        
        // bottom left
        vertices[i++] = xStart;
        vertices[i++] = yPos - lineWidth;
        vertices[i++] = zPos;
        
        // bottom right
        vertices[i++] = xEnd;
        vertices[i++] = yPos - lineWidth;
        vertices[i++] = zPos;
    
        }
    }

    const indices = new Uint32Array(vertexCount * 6); // 0 - 5
    let j = 0;
    // map indices to the vertices (counterclockwise)
    // 0 -----.1
    // |   .   |
    // | .     |
    // 2 ------ 3
    for (let i = 0; i < vertexCount; i += 4) {
        const rectangleOffset = i * 4;
    
        // first triangle
        indices[j++] = rectangleOffset + 0;
        indices[j++] = rectangleOffset + 2;
        indices[j++] = rectangleOffset + 1;

        // second triangle
        indices[j++] = rectangleOffset + 2;
        indices[j++] = rectangleOffset + 3;
        indices[j++] = rectangleOffset + 1;
    }
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
    const shaderModule = device.createShaderModule({ code: shader });
    const pipeline = device.createRenderPipeline({
      label: "Wireframe",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [
          {
            arrayStride: 12, // (3 floats * 4 bytes per float)
            stepMode: "vertex",
            attributes: [
              {
                format: "float32x3", //vec3f
                offset: 0,
                shaderLocation: 0, //location(0) position: vec3f
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
        topology: "triangle-list",
      },
    });

    setResources({ pipeline, vertexBuffer, indexBuffer });
  }, [webGPUState]);

  return resources;
};
