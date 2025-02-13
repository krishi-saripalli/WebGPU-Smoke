import { useEffect, useState } from 'react';
import { WebGPUState } from './useWebGPU';
import { shader } from '@/shaders/shader';
import { Camera } from '@/modules/Camera';
import { Vec3 } from 'gl-matrix';

export interface RenderPipelineResources {
  pipeline: GPURenderPipeline; //  the VAO
  vertexBuffer: GPUBuffer; // the VBO
  indexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

export const useRenderResources = (webGPUState: WebGPUState | null) => {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);

  useEffect(() => {
    if (!webGPUState) return;
    const { device, canvasFormat } = webGPUState;

    // Size is 2 4x4 matrices (view and projection) * 16 floats per matrix * 4 bytes per float
    const uniformBuffer = device.createBuffer({
      size: 2 * 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const camera = new Camera({
      position: new Vec3([-2, 1.5, -4]),
      lookAt: new Vec3([0, 0, 0]),
      up: new Vec3([0, 1, 0]),
      heightAngle: Math.PI / 3,
      near: 0.1,
      far: 100,
      aspect: 1,
    });

    const viewMatrix = camera.getViewMatrix();
    const projectionMatrix = camera.getProjectionMatrix();

    device.queue.writeBuffer(uniformBuffer, 0, viewMatrix as Float32Array);
    const offset = 16 * 4; // offset for projection matrix (after view matrix)
    device.queue.writeBuffer(uniformBuffer, offset, projectionMatrix as Float32Array);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform' },
        },
      ],
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    const gridSize = 4;
    const lineWidth = 0.009;

    // each line on each axis of the grid will be represented by a thin quad, which is 2 triangles (4 vertices)

    const vertexCount = 3 * gridSize * gridSize * 4;
    const vertices = new Float32Array(vertexCount * 3); // x, y, z
    let i = 0;


    // z-aligned lines
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        const zStart = -1.0;
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

    // x-aligned lines
    for (let y = 0; y < gridSize; y++) {
      for (let z = 0; z < gridSize; z++) {
        const xStart = -1.0;
        const xEnd = 1.0;
        const yPos = -1.0 + 2.0 * (y / (gridSize - 1));
        const zPos = -1.0 + 2.0 * (z / (gridSize - 1));

        // four vertices for the line (quad)
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

    // y-aligned lines
    for (let z = 0; z < gridSize; z++) {
      for (let x = 0; x < gridSize; x++) {
        const yStart = -1.0;
        const yEnd = 1.0;
        const xPos = -1.0 + 2.0 * (x / (gridSize - 1));
        const zPos = -1.0 + 2.0 * (z / (gridSize - 1));

        // four vertices for the line (rectangle)
        // top left
        vertices[i++] = xPos + lineWidth;
        vertices[i++] = yStart;
        vertices[i++] = zPos;

        // top right
        vertices[i++] = xPos + lineWidth;
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
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
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
      multisample: {
        count: 4,
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    setResources({ pipeline, vertexBuffer, indexBuffer, uniformBuffer, bindGroup });
  }, [webGPUState]);

  return resources;
};
