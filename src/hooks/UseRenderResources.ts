import { useEffect, useState } from 'react';
import { Mat4 } from 'gl-matrix';
import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { generateBox } from '@/utils/geometry';
import { loadShader, loadShaderModule } from '@/utils/shader-loader';
import type { WebGPUState } from './useWebGPU';

export const STEP_SECONDS = 1 / 60;
type GridSize = [number, number, number];
const GRID_SIZE: GridSize = [64, 160, 64];
const COARSE_SIZE: GridSize = [GRID_SIZE[0] / 2, GRID_SIZE[1] / 2, GRID_SIZE[2] / 2];
export type Stroke = { start: [number, number]; end: [number, number] };

async function createSimulation(
  { device, context, canvasFormat }: WebGPUState,
  signal: AbortSignal
) {
  const common = await loadShader('common');
  const names = [
    'vorticity',
    'vorticity_confinement',
    'forces',
    'advection',
    'divergence',
    'pressure',
    'pressure_gradient',
    'render',
  ] as const;
  const modules = Object.fromEntries(
    await Promise.all(
      names.map(async (name) => [name, await loadShaderModule(device, name, common)] as const)
    )
  ) as Record<(typeof names)[number], GPUShaderModule>;

  function compute(name: (typeof names)[number], entryPoint = 'main') {
    return device.createComputePipelineAsync({
      label: name + '/' + entryPoint,
      layout: 'auto',
      compute: { module: modules[name], entryPoint },
    });
  }
  const vertex: GPUVertexState = {
    module: modules.render,
    entryPoint: 'vertexMain',
    buffers: [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
    ],
  };
  const [
    curl,
    confinement,
    forces,
    advectVelocity,
    divergence,
    pressure,
    restrict,
    prolongate,
    illuminate,
    project,
    advectSmoke,
    render,
    wireframe,
  ] = await Promise.all([
    compute('vorticity'),
    compute('vorticity_confinement'),
    compute('forces'),
    compute('advection', 'advectVelocity'),
    compute('divergence'),
    compute('pressure'),
    compute('pressure', 'restrictResidual'),
    compute('pressure', 'prolongate'),
    compute('render', 'illuminate'),
    compute('pressure_gradient'),
    compute('advection', 'advectSmoke'),
    device.createRenderPipelineAsync({
      label: 'Smoke volume',
      layout: 'auto',
      vertex,
      primitive: { cullMode: 'back' },
      fragment: {
        module: modules.render,
        entryPoint: 'fragmentVolume',
        targets: [
          {
            format: canvasFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
    }),
    device.createRenderPipelineAsync({
      label: 'Grid boundary',
      layout: 'auto',
      vertex,
      primitive: { topology: 'line-list' },
      fragment: {
        module: modules.render,
        entryPoint: 'fragmentWireframe',
        targets: [{ format: canvasFormat }],
      },
    }),
  ]);
  signal.throwIfAborted();
  const allocations: (GPUTexture | GPUBuffer)[] = [];
  signal.addEventListener('abort', () => allocations.forEach((resource) => resource.destroy()), {
    once: true,
  });
  function volume(label: string, format: GPUTextureFormat, grid = GRID_SIZE, halo = 1) {
    const texture = device.createTexture({
      label,
      format,
      dimension: '3d',
      size: grid.map((n) => n + 2 * halo),
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    allocations.push(texture);
    return texture.createView();
  }
  function buffer(label: string, size: number, usage: GPUBufferUsageFlags) {
    const result = device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    allocations.push(result);
    return result;
  }
  const fields = {
    velocity: [volume('Velocity A', 'rgba16float'), volume('Velocity B', 'rgba16float')],
    smoke: [
      volume('Density and temperature A', 'rgba16float'),
      volume('Density and temperature B', 'rgba16float'),
    ],
    pressure: [volume('Pressure A', 'r32float'), volume('Pressure B', 'r32float')],
    coarsePressure: [
      volume('Coarse pressure A', 'r32float', COARSE_SIZE),
      volume('Coarse pressure B', 'r32float', COARSE_SIZE),
    ],
    coarseRhs: volume('Coarse residual', 'r32float', COARSE_SIZE),
    light: volume('Light transmittance', 'rgba8unorm', COARSE_SIZE, 0),
    curl: volume('Curl and magnitude', 'rgba16float'),
    force: volume('Cell-centered force', 'rgba16float'),
    divergence: volume('Divergence', 'r32float'),
  };
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const defs = makeShaderDataDefinitions(common);
  const uniforms = makeStructuredView(defs.uniforms.uniforms);
  const params = makeStructuredView(defs.uniforms.params);
  const uniformBuffer = buffer(
    'Scene uniforms',
    uniforms.arrayBuffer.byteLength,
    GPUBufferUsage.UNIFORM
  );
  const paramsBuffer = buffer(
    'Simulation parameters',
    params.arrayBuffer.byteLength,
    GPUBufferUsage.UNIFORM
  );
  params.set({
    dt: STEP_SECONDS,
    dx: 1 / GRID_SIZE[1],
    vorticityStrength: 8,
    buoyancyAlpha: 0.02,
    buoyancyBeta: 0.012,
    ambientTemperature: 0,
    dissipation: 0.12,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params.arrayBuffer);
  uniforms.set({
    gridSize: GRID_SIZE,
    absorption: 1.4,
    scattering: 6,
    lightPosition: [-0.2, 0.85, 0.16],
    lightIntensity: [12, 12, 12],
  });
  const view = Mat4.create();
  Mat4.lookAt(view, [0, 1.6, 5], [0, 0, 0], [0, 1, 0]);
  const projection = Mat4.create();
  const radius = Math.hypot(GRID_SIZE[0], GRID_SIZE[2]) / GRID_SIZE[1];
  const halfHeight = Math.abs(view[5]) + radius * Math.abs(view[9]);
  let viewHalfWidth = 0;
  let viewHalfHeight = 0;
  function resize(width: number, height: number) {
    const aspect = width / height;
    viewHalfHeight = 1.12 * Math.max(halfHeight, radius / aspect);
    viewHalfWidth = viewHalfHeight * aspect;
    Mat4.orthoZO(
      projection,
      -viewHalfWidth,
      viewHalfWidth,
      -viewHalfHeight,
      viewHalfHeight,
      0.1,
      10
    );
    uniforms.set({ projectionMatrix: projection });
    device.queue.writeBuffer(uniformBuffer, 0, uniforms.arrayBuffer);
  }
  resize(context.canvas.width, context.canvas.height);
  function screenToView(x: number, y: number): [number, number] {
    return [x * viewHalfWidth, y * viewHalfHeight];
  }
  const modelView = Mat4.create();
  let yaw = -Math.PI / 6;
  const coarseUniforms = makeStructuredView(defs.uniforms.uniforms);
  coarseUniforms.set({ gridSize: COARSE_SIZE });
  const coarseUniformBuffer = buffer(
    'Coarse grid',
    coarseUniforms.arrayBuffer.byteLength,
    GPUBufferUsage.UNIFORM
  );
  device.queue.writeBuffer(coarseUniformBuffer, 0, coarseUniforms.arrayBuffer);
  const coarseParams = makeStructuredView(defs.uniforms.params);
  coarseParams.set({ dx: 1 / COARSE_SIZE[1] });
  const coarseParamsBuffer = buffer(
    'Coarse cell width',
    coarseParams.arrayBuffer.byteLength,
    GPUBufferUsage.UNIFORM
  );
  device.queue.writeBuffer(coarseParamsBuffer, 0, coarseParams.arrayBuffer);
  const box = generateBox();
  const vertices = new Float32Array(box.vertexPositions);
  const indices = new Uint32Array(box.indicesList);
  const vertexBuffer = buffer('Volume vertices', vertices.byteLength, GPUBufferUsage.VERTEX);
  const indexBuffer = buffer('Volume indices', indices.byteLength, GPUBufferUsage.INDEX);
  device.queue.writeBuffer(vertexBuffer, 0, vertices);
  device.queue.writeBuffer(indexBuffer, 0, indices);

  const edges = new Uint32Array(box.edgeIndices);
  const edgeBuffer = buffer('Grid edges', edges.byteLength, GPUBufferUsage.INDEX);
  device.queue.writeBuffer(edgeBuffer, 0, edges);

  function group(
    pipeline: GPUComputePipeline | GPURenderPipeline,
    index: number,
    resources: (GPUBindingResource | null)[]
  ) {
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(index),
      entries: resources.flatMap((resource, binding) => (resource ? [{ binding, resource }] : [])),
    });
  }
  const fineGlobals = [{ buffer: uniformBuffer }, { buffer: paramsBuffer }];
  const coarseGlobals = [{ buffer: coarseUniformBuffer }, { buffer: coarseParamsBuffer }];
  const sceneGlobals = [{ buffer: uniformBuffer }];
  function stage(
    pipeline: GPUComputePipeline,
    resources: (GPUBindingResource | null)[],
    size = GRID_SIZE,
    bindings = fineGlobals
  ) {
    const globals = group(pipeline, 0, bindings);
    const [nx, ny, nz] = size.map((n) => Math.ceil(n / 4));
    const inputs = group(pipeline, 1, resources);
    return (pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, globals);
      pass.setBindGroup(1, inputs);
      pass.dispatchWorkgroups(nx, ny, nz);
    };
  }
  const jacobi = fields.pressure.map((p, i) =>
    stage(pressure, [p, fields.divergence, fields.pressure[1 - i]])
  );
  const coarseJacobi = fields.coarsePressure.map((p, i) =>
    stage(pressure, [p, fields.coarseRhs, fields.coarsePressure[1 - i]], COARSE_SIZE, coarseGlobals)
  );
  const restrictResidual = stage(
    restrict,
    [fields.pressure[0], fields.divergence, null, fields.coarseRhs, fields.coarsePressure[0]],
    COARSE_SIZE
  );
  const addCorrection = stage(
    prolongate,
    [fields.pressure[0], null, fields.pressure[1], null, null, fields.coarsePressure[0]],
    GRID_SIZE,
    sceneGlobals
  );

  function solvePressure(pass: GPUComputePassEncoder) {
    jacobi[0](pass);
    jacobi[1](pass);
    restrictResidual(pass);
    for (let i = 0; i < 14; i++) coarseJacobi[i % 2](pass);
    addCorrection(pass);
    for (let i = 0; i < 5; i++) jacobi[(i + 1) % 2](pass);
  }
  const renderUniforms = group(render, 0, [{ buffer: uniformBuffer }]);
  const wireUniforms = group(wireframe, 0, sceneGlobals);
  const frames = fields.velocity.map((u, i) => {
    const next = 1 - i;
    const v = fields.velocity[next];
    const smoke = fields.smoke[i];
    const nextSmoke = fields.smoke[next];
    const beforePressure = [
      stage(curl, [u, fields.curl]),
      stage(confinement, [fields.curl, fields.force]),
      stage(forces, [u, fields.force, smoke, v]),
      stage(advectVelocity, [v, sampler, u]),
      stage(divergence, [u, fields.divergence]),
    ];
    return {
      beforePressure,
      projectVelocity: stage(project, [u, fields.pressure[0], v]),
      advectSmoke: stage(advectSmoke, [v, sampler, null, smoke, nextSmoke]),
      illuminate: stage(illuminate, [smoke, sampler, fields.light], COARSE_SIZE, sceneGlobals),
      renderSmoke: group(render, 1, [smoke, sampler, null, fields.light]),
    };
  });

  let parity = 0;
  function step(steps: number, rotation: number, stroke: Stroke | null) {
    yaw = (yaw + rotation) % (2 * Math.PI);
    Mat4.rotateY(modelView, view, yaw);
    const length = stroke
      ? Math.hypot(stroke.end[0] - stroke.start[0], stroke.end[1] - stroke.start[1])
      : 0;
    uniforms.set({
      modelViewMatrix: modelView,
      cameraForward: [-modelView[2], -modelView[6], -modelView[10]],
      brushStroke: stroke ? [...stroke.start, ...stroke.end] : [0, 0, 0, 0],
      brushStrength: length && steps ? Math.min(2.5, 0.65 / length) / steps : 0,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniforms.arrayBuffer);
    const encoder = device.createCommandEncoder({ label: 'Smoke frame' });
    for (let n = 0; n < steps; n++) {
      const frame = frames[parity];
      const pass = encoder.beginComputePass();
      for (const dispatch of frame.beforePressure) dispatch(pass);
      solvePressure(pass);
      frame.projectVelocity(pass);
      frame.advectSmoke(pass);
      parity = 1 - parity;
      if (n === steps - 1) frames[parity].illuminate(pass);
      pass.end();
    }
    const draw = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 15 / 255, g: 23 / 255, b: 42 / 255, a: 1 },
        },
      ],
    });
    draw.setPipeline(render);
    draw.setBindGroup(0, renderUniforms);
    draw.setBindGroup(1, frames[parity].renderSmoke);
    draw.setVertexBuffer(0, vertexBuffer);
    draw.setIndexBuffer(indexBuffer, 'uint32');
    draw.drawIndexed(indices.length);
    draw.setPipeline(wireframe);
    draw.setBindGroup(0, wireUniforms);
    draw.setIndexBuffer(edgeBuffer, 'uint32');
    draw.drawIndexed(edges.length);
    draw.end();
    device.queue.submit([encoder.finish()]);
  }
  return { step, resize, screenToView };
}

type RenderPipelineResources = Awaited<ReturnType<typeof createSimulation>>;

export function useRenderResources(state: WebGPUState | null) {
  const [resources, setResources] = useState<RenderPipelineResources | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setResources(null);
    setError(null);
    if (state) {
      createSimulation(state, controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) setResources(result);
        })
        .catch((e) => {
          if (!controller.signal.aborted) {
            setError(String(e));
            controller.abort();
          }
        });
    }
    return () => controller.abort();
  }, [state]);
  return { resources, error };
}
