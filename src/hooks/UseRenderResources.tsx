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
  indexCount: number;
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
      position: new Vec3([-0, 0, 4]),
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

    const gridSize = 8;
    const lineWidth = 0.009;

    /////////////////////////////////////////////////////////////////////////
    // Evenly space the points from -1 to +1 in (gridSize) steps
    /////////////////////////////////////////////////////////////////////////
    const positions: number[] = [];
    for (let idx = 0; idx < gridSize; idx++) {
      positions.push(-1.0 + (2.0 * idx) / (gridSize - 1));
    }

    /////////////////////////////////////////////////////////////////////////
    // Build vertices and indices
    // dynamically in arrays. Then convert them to typed arrays at the end.
    /////////////////////////////////////////////////////////////////////////
    const vertexPositions: number[] = [];
    const indicesList: number[] = [];

    // helper so we can add four vertices (quad) + six indices at once.
    function addQuad(
      p0: [number, number, number],
      p1: [number, number, number],
      p2: [number, number, number],
      p3: [number, number, number]
    ) {
      // "startIndex" = how many vertices are in vertexPositions so far
      const startIndex = vertexPositions.length / 3;

      // Push the 4 corner vertices of this quad
      vertexPositions.push(...p0, ...p1, ...p2, ...p3);

      // Add two triangles (6 indices) in CCW order:
      //   0--1
      //   | /|
      //   2--3
      indicesList.push(
        startIndex + 0,
        startIndex + 2,
        startIndex + 1,
        startIndex + 2,
        startIndex + 3,
        startIndex + 1
      );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // z-aligned lines: x,y vary, z goes from -1 to +1.
    // Only draw the quad if we are on the boundary plane (|x|=1 or |y|=1).
    ///////////////////////////////////////////////////////////////////////////////
    for (let x of positions) {
      for (let y of positions) {
        if (Math.abs(x) === 1 || Math.abs(y) === 1) {
          const zStart = -1.0;
          const zEnd = 1.0;

          // top-left, top-right, bottom-left, bottom-right
          const p0: [number, number, number] = [x, y + lineWidth, zEnd];
          const p1: [number, number, number] = [x, y + lineWidth, zStart];
          const p2: [number, number, number] = [x, y - lineWidth, zEnd];
          const p3: [number, number, number] = [x, y - lineWidth, zStart];

          addQuad(p0, p1, p2, p3);
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // x-aligned lines: y,z vary, x goes from -1 to +1.
    // Only draw if we're on the boundary plane (|y|=1 or |z|=1).
    ///////////////////////////////////////////////////////////////////////////////
    for (let y of positions) {
      for (let z of positions) {
        if (Math.abs(y) === 1 || Math.abs(z) === 1) {
          const xStart = -1.0;
          const xEnd = 1.0;

          const p0: [number, number, number] = [xStart, y + lineWidth, z];
          const p1: [number, number, number] = [xEnd, y + lineWidth, z];
          const p2: [number, number, number] = [xStart, y - lineWidth, z];
          const p3: [number, number, number] = [xEnd, y - lineWidth, z];

          addQuad(p0, p1, p2, p3);
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // y-aligned lines: x,z vary, y goes from -1 to +1.
    // Only draw if we're on the boundary plane (|x|=1 or |z|=1).
    ///////////////////////////////////////////////////////////////////////////////
    for (let z of positions) {
      for (let x of positions) {
        if (Math.abs(z) === 1 || Math.abs(x) === 1) {
          const yStart = -1.0;
          const yEnd = 1.0;

          const p0: [number, number, number] = [x + lineWidth, yStart, z];
          const p1: [number, number, number] = [x + lineWidth, yEnd, z];
          const p2: [number, number, number] = [x - lineWidth, yStart, z];
          const p3: [number, number, number] = [x - lineWidth, yEnd, z];

          addQuad(p0, p1, p2, p3);
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // convert the vertex & index arrays into typed arrays
    ////////////////////////////////////////////////////////////////////////////////
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

    setResources({
      pipeline,
      vertexBuffer,
      indexBuffer,
      uniformBuffer,
      bindGroup,
      indexCount: indices.length,
    });
  }, [webGPUState]);

  return resources;
};
