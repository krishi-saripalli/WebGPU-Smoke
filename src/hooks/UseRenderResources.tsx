import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { shader } from '@/shaders/shader';

export interface RenderPipelineResources {
  pipeline: GPURenderPipeline; //  the VAO
  vertexBuffer: GPUBuffer; // the VBO
  indexBuffer: GPUBuffer;
}

export const useRenderResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);

  useEffect(() => {
    if (!webGPUState) return;
    const { device, canvasFormat } = webGPUState;

    const gridSize = 10;
    const lineWidth = 0.005;

    // each line on each axis of the grid will be represented by a thin quad, which is 2 triangles (4 vertices)
    //TODO: DO the same for the y and z axes
    const vertexCount = 3 * gridSize * gridSize * 4;
    const vertices = new Float32Array(vertexCount * 3); // x, y, z
    let i = 0;

    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        const zStart = 0.0;
        const zEnd = 1.0;
        const xPos = -1.0 + 2.0 * (x / (gridSize - 1));
        const yPos = -1.0 + 2.0 * (y / (gridSize - 1));

        // four vertices for the line (rectangle)
        // top left
        vertices[i++] = xPos;
        vertices[i++] = yPos + lineWidth;
        vertices[i++] = zEnd;

        // top right
        vertices[i++] = xPos;
        vertices[i++] = yPos + lineWidth;
        vertices[i++] = zStart;

        // bottom left
        vertices[i++] = xPos;
        vertices[i++] = yPos - lineWidth;
        vertices[i++] = zEnd;

        // bottom right
        vertices[i++] = xPos;
        vertices[i++] = yPos - lineWidth;
        vertices[i++] = zStart;
      }
    }

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

    for (let z = 0; z < gridSize; z++) {
      for (let x = 0; x < gridSize; x++) {
        const yStart = -1.0;
        const yEnd = 1.0;
        const xPos = -1.0 + 2.0 * (x / (gridSize - 1));
        const zPos = z / (gridSize - 1);

        // four vertices for the line (rectangle)
        // top left
        vertices[i++] = xPos;
        vertices[i++] = yStart;
        vertices[i++] = zPos;

        // top right
        vertices[i++] = xPos;
        vertices[i++] = yEnd;
        vertices[i++] = zPos;

        // bottom left
        vertices[i++] = xPos - lineWidth;
        vertices[i++] = yStart;
        vertices[i++] = zPos;

        // bottom right
        vertices[i++] = xPos - lineWidth;
        vertices[i++] = yEnd;
        vertices[i++] = zPos;
      }
    }

    console.log('Total vertices generated:', i);
    console.log('Expected vertices:', vertexCount * 3);
    console.log('Sample vertices (first quad):', vertices.slice(0, 12));

    // e have (3 * gridSize * gridSize) quads total
    // each quad needs 6 indices
    const indexCount = 3 * gridSize * gridSize * 6;
    const indices = new Uint32Array(indexCount); // 0 - 5
    let j = 0;
    // map indices to the vertices (counterclockwise)
    // 0 -----.1
    // |   .   |
    // | .     |
    // 2 ------ 3

    const quadCount = 3 * gridSize * gridSize;
    for (let i = 0; i < quadCount; i += 1) {
      const quadOffset = i * 4;

      // first triangle
      indices[j++] = quadOffset + 0;
      indices[j++] = quadOffset + 2;
      indices[j++] = quadOffset + 1;

      // second triangle
      indices[j++] = quadOffset + 2;
      indices[j++] = quadOffset + 3;
      indices[j++] = quadOffset + 1;
    }
    console.log('Total indices generated:', j);
    console.log('Expected indices:', indexCount);
    console.log('Sample indices (first quad):', indices.slice(0, 6));
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
      label: 'Wireframe',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vertexMain',
        buffers: [
          {
            arrayStride: 12, // (3 floats * 4 bytes per float)
            stepMode: 'vertex',
            attributes: [
              {
                format: 'float32x3', //vec3f
                offset: 0,
                shaderLocation: 0, //location(0) position: vec3f
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
      primitive: {
        topology: 'triangle-list',
      },
    });

    setResources({ pipeline, vertexBuffer, indexBuffer });
  }, [webGPUState]);

  return resources;
};
