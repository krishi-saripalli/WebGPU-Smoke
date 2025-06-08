'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/camera-movement';
import { SimulationState } from '@/utils/types';
import {
  createUniformBindGroup,
  createVelocityAdvectionBindGroup,
  createTemperatureAdvectionBindGroup,
  createDensityAdvectionBindGroup,
  createForcesBindGroup,
  createVorticityCalculationBindGroup,
  createVorticityConfinementBindGroup,
  createDivergenceCalculationBindGroup,
  createPressureIterationBindGroup,
  createPressureGradientSubtractionBindGroup,
  createReinitializationBindGroup,
  createRenderBindGroup,
} from '@/utils/bind-group';
import {
  handleKeyDown,
  handleKeyUp,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
} from '@/utils/input-handler';
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';
import {
  createDensityAdvectionBindGroupLayout,
  createDivergenceCalculationBindGroupLayout,
  createExternalForcesStepBindGroupLayout,
  createPressureGradientSubtractionBindGroupLayout,
  createPressureIterationBindGroupLayout,
  createReinitializationBindGroupLayout,
  createRenderBindGroupLayout,
  createTemperatureAdvectionBindGroupLayout,
  createUniformBindGroupLayout,
  createVelocityAdvectionBindGroupLayout,
  createVorticityCalculationBindGroupLayout,
  createVorticityConfinementBindGroupLayout,
} from '@/utils/layouts';

const renderScene = (webGPUState: WebGPUState, renderResources: RenderPipelineResources): void => {
  const { device, context } = webGPUState;
  const {
    // Render Pipelines,
    slicesPipeline,
    // Compute Pipelines
    externalForcesStepPipeline,
    vorticityCalculationPipeline,
    vorticityConfinementPipeline,
    velocityAdvectionPipeline,
    temperatureAdvectionPipeline,
    densityAdvectionPipeline,
    divergenceCalculationPipeline,
    pressureIterationPipeline,
    pressureGradientSubtractionPipeline,
    reinitializationPipeline,
    // Buffers
    slicesVertexBuffer,
    slicesIndexBuffer,
    multisampleTexture,
    uniformBuffer,
    simulationParamsBuffer,
    //Sampler
    sampler,
    //Textures
    densityTextureA,
    densityTextureB,
    velocityTextureA,
    velocityTextureB,
    temperatureTextureA,
    temperatureTextureB,
    pressureTextureA,
    pressureTextureB,
    vorticityTextureA,
    vorticityTextureB,
    divergenceTextureA,
    divergenceTextureB,
    // Counts & Sizes
    slicesIndexCount,
    totalGridSize,
  } = renderResources;

  const simulationState: SimulationState = {
    velocity: { current: velocityTextureA, next: velocityTextureB },
    density: { current: densityTextureA, next: densityTextureB },
    temperature: { current: temperatureTextureA, next: temperatureTextureB },
    pressure: { current: pressureTextureA, next: pressureTextureB },
    divergence: { current: divergenceTextureA, next: divergenceTextureB },
    vorticity: { current: vorticityTextureA, next: vorticityTextureB },
  };

  const workgroupSize: [number, number, number] = [4, 4, 4];
  const numWorkgroups: [number, number, number] = [
    Math.ceil(totalGridSize / workgroupSize[0]),
    Math.ceil(totalGridSize / workgroupSize[1]),
    Math.ceil(totalGridSize / workgroupSize[2]),
  ];

  //TODO: Different behaviour for even and odd number of iterations?
  const JACOBI_ITERATIONS = 80;

  const swapTextures = (resource: keyof SimulationState) => {
    const temp = simulationState[resource].current;
    simulationState[resource].current = simulationState[resource].next;
    simulationState[resource].next = temp;
  };

  const uniformBindGroup = createUniformBindGroup(
    device,
    createUniformBindGroupLayout(device),
    uniformBuffer,
    simulationParamsBuffer
  );

  const advectVelEncoder = device.createCommandEncoder({ label: 'Velocity Advect Encoder' });
  const advectVelPass = advectVelEncoder.beginComputePass({ label: 'Velocity Advect Pass' });
  advectVelPass.setPipeline(velocityAdvectionPipeline);
  advectVelPass.setBindGroup(0, uniformBindGroup);
  const advectVelBindGroup = createVelocityAdvectionBindGroup(
    device,
    createVelocityAdvectionBindGroupLayout(device),
    simulationState.velocity.current,
    sampler,
    simulationState.velocity.next
  );
  advectVelPass.setBindGroup(0, uniformBindGroup);
  advectVelPass.setBindGroup(1, advectVelBindGroup);
  advectVelPass.dispatchWorkgroups(...numWorkgroups);
  advectVelPass.end();
  device.queue.submit([advectVelEncoder.finish()]);
  swapTextures('velocity');

  const tempAdvectEncoder = device.createCommandEncoder({ label: 'Temperature Advect Encoder' });
  const tempAdvectPass = tempAdvectEncoder.beginComputePass({ label: 'Temperature Advect Pass' });
  tempAdvectPass.setPipeline(temperatureAdvectionPipeline);
  tempAdvectPass.setBindGroup(0, uniformBindGroup);
  const tempAdvectBindGroup = createTemperatureAdvectionBindGroup(
    device,
    createTemperatureAdvectionBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.temperature.current,
    sampler,
    simulationState.temperature.next
  );
  tempAdvectPass.setBindGroup(0, uniformBindGroup);
  tempAdvectPass.setBindGroup(1, tempAdvectBindGroup);
  tempAdvectPass.dispatchWorkgroups(...numWorkgroups);
  tempAdvectPass.end();
  device.queue.submit([tempAdvectEncoder.finish()]);
  swapTextures('temperature');

  const densityAdvectEncoder = device.createCommandEncoder({ label: 'Density Advect Encoder' });
  const densityAdvectPass = densityAdvectEncoder.beginComputePass({ label: 'Density Advect Pass' });
  densityAdvectPass.setPipeline(densityAdvectionPipeline);
  const densityAdvectBindGroup = createDensityAdvectionBindGroup(
    device,
    createDensityAdvectionBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.density.current,
    sampler,
    simulationState.density.next
  );
  densityAdvectPass.setBindGroup(0, uniformBindGroup);
  densityAdvectPass.setBindGroup(1, densityAdvectBindGroup);
  densityAdvectPass.dispatchWorkgroups(...numWorkgroups);
  densityAdvectPass.end();
  device.queue.submit([densityAdvectEncoder.finish()]);
  swapTextures('density');

  const forcesEncoder = device.createCommandEncoder({ label: 'External Forces Encoder' });
  const forcesPass = forcesEncoder.beginComputePass({ label: 'External Forces Pass' });
  forcesPass.setPipeline(externalForcesStepPipeline);
  const forcesBindGroup = createForcesBindGroup(
    device,
    createExternalForcesStepBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.temperature.current,
    simulationState.density.current,
    simulationState.velocity.next
  );
  forcesPass.setBindGroup(0, uniformBindGroup);
  forcesPass.setBindGroup(1, forcesBindGroup);
  forcesPass.dispatchWorkgroups(...numWorkgroups);
  forcesPass.end();
  device.queue.submit([forcesEncoder.finish()]);
  swapTextures('velocity');

  const vorticityEncoder = device.createCommandEncoder({ label: 'Vorticity Calc Encoder' });
  const vorticityPass = vorticityEncoder.beginComputePass({ label: 'Vorticity Calc Pass' });
  vorticityPass.setPipeline(vorticityCalculationPipeline);
  const vorticityBindGroup = createVorticityCalculationBindGroup(
    device,
    createVorticityCalculationBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.vorticity.next
  );
  vorticityPass.setBindGroup(0, uniformBindGroup);
  vorticityPass.setBindGroup(1, vorticityBindGroup);
  vorticityPass.dispatchWorkgroups(...numWorkgroups);
  vorticityPass.end();
  device.queue.submit([vorticityEncoder.finish()]);
  swapTextures('vorticity');

  const confinementEncoder = device.createCommandEncoder({ label: 'Vorticity Conf Encoder' });
  const confinementPass = confinementEncoder.beginComputePass({ label: 'Vorticity Conf Pass' });
  confinementPass.setPipeline(vorticityConfinementPipeline);
  const confinementBindGroup = createVorticityConfinementBindGroup(
    device,
    createVorticityConfinementBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.vorticity.current,
    simulationState.velocity.next
  );
  confinementPass.setBindGroup(0, uniformBindGroup);
  confinementPass.setBindGroup(1, confinementBindGroup);
  confinementPass.dispatchWorkgroups(...numWorkgroups);
  confinementPass.end();
  device.queue.submit([confinementEncoder.finish()]);
  swapTextures('velocity');

  const divergenceEncoder = device.createCommandEncoder({ label: 'Divergence Encoder' });
  const divergencePass = divergenceEncoder.beginComputePass({ label: 'Divergence Pass' });
  divergencePass.setPipeline(divergenceCalculationPipeline);
  const divergenceBindGroup = createDivergenceCalculationBindGroup(
    device,
    createDivergenceCalculationBindGroupLayout(device),
    simulationState.velocity.current,
    simulationState.divergence.next
  );
  divergencePass.setBindGroup(0, uniformBindGroup);
  divergencePass.setBindGroup(1, divergenceBindGroup);
  divergencePass.dispatchWorkgroups(...numWorkgroups);
  divergencePass.end();
  device.queue.submit([divergenceEncoder.finish()]);
  swapTextures('divergence');

  const pressureEncoder = device.createCommandEncoder({ label: 'Pressure Solve Encoder' });
  const pressurePass = pressureEncoder.beginComputePass({ label: 'Pressure Solve Pass' });
  pressurePass.setPipeline(pressureIterationPipeline);
  pressurePass.setBindGroup(0, uniformBindGroup);

  for (let i = 0; i < JACOBI_ITERATIONS; i++) {
    const pressureBindGroup = createPressureIterationBindGroup(
      device,
      createPressureIterationBindGroupLayout(device),
      simulationState.divergence.current,
      simulationState.pressure.current,
      simulationState.pressure.next
    );
    pressurePass.setBindGroup(1, pressureBindGroup);
    pressurePass.dispatchWorkgroups(...numWorkgroups);
    swapTextures('pressure');
  }

  pressurePass.end();
  device.queue.submit([pressureEncoder.finish()]);

  const pressureGradEncoder = device.createCommandEncoder({ label: 'Pressure Grad Encoder' });
  const pressureGradPass = pressureGradEncoder.beginComputePass({ label: 'Pressure Grad Pass' });
  pressureGradPass.setPipeline(pressureGradientSubtractionPipeline);
  pressureGradPass.setBindGroup(0, uniformBindGroup);
  const pressureGradBindGroup = createPressureGradientSubtractionBindGroup(
    device,
    createPressureGradientSubtractionBindGroupLayout(device),
    simulationState.pressure.current,
    simulationState.velocity.current,
    simulationState.velocity.next
  );
  pressureGradPass.setBindGroup(1, pressureGradBindGroup);
  pressureGradPass.dispatchWorkgroups(...numWorkgroups);
  pressureGradPass.end();
  device.queue.submit([pressureGradEncoder.finish()]);
  swapTextures('velocity');

  const reinitializationEncoder = device.createCommandEncoder({
    label: 'Reinitialization Encoder',
  });
  const reinitializationPass = reinitializationEncoder.beginComputePass({
    label: 'Reinitialization Pass',
  });
  reinitializationPass.setPipeline(reinitializationPipeline);
  reinitializationPass.setBindGroup(0, uniformBindGroup);
  const reinitializationBindGroup = createReinitializationBindGroup(
    device,
    createReinitializationBindGroupLayout(device),
    simulationState.temperature.current,
    simulationState.temperature.next,
    simulationState.density.current,
    simulationState.density.next,
    simulationState.velocity.current,
    simulationState.velocity.next
  );
  reinitializationPass.setBindGroup(1, reinitializationBindGroup);
  reinitializationPass.dispatchWorkgroups(...numWorkgroups);
  reinitializationPass.end();
  device.queue.submit([reinitializationEncoder.finish()]);
  swapTextures('density');
  swapTextures('velocity');
  swapTextures('temperature');

  /////////////////////////////////////////////////////////////////////////
  // Render pass
  /////////////////////////////////////////////////////////////////////////
  const renderEncoder = device.createCommandEncoder({ label: 'Render Encoder' });
  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: multisampleTexture.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };
  const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor);

  renderPass.setBindGroup(0, uniformBindGroup);

  // transparent slices
  renderPass.setPipeline(slicesPipeline);
  const renderBindGroup = createRenderBindGroup(
    device,
    createRenderBindGroupLayout(device),
    simulationState.density.current,
    sampler
  );
  renderPass.setBindGroup(1, renderBindGroup);
  renderPass.setVertexBuffer(0, slicesVertexBuffer);
  renderPass.setIndexBuffer(slicesIndexBuffer, 'uint32');
  renderPass.drawIndexed(slicesIndexCount);

  renderPass.end();
  device.queue.submit([renderEncoder.finish()]);
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const pressedKeysRef = useRef(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    pressedKeysRef.current = pressedKeys;
  }, [pressedKeys]);

  const keyDownHandler = useCallback((e: KeyboardEvent) => {
    handleKeyDown(e, setPressedKeys);
  }, []);

  const keyUpHandler = useCallback((e: KeyboardEvent) => {
    handleKeyUp(e, setPressedKeys);
  }, []);

  //key handlers
  useEffect(() => {
    window.addEventListener('keydown', keyDownHandler);
    window.addEventListener('keyup', keyUpHandler);

    return () => {
      window.removeEventListener('keydown', keyDownHandler);
      window.removeEventListener('keyup', keyUpHandler);
    };
  }, [keyDownHandler, keyUpHandler]);

  useEffect(() => {
    if (!renderResources || !webGPUState?.device) return;

    let frameId: number;

    function animationLoop(currentTime: number) {
      if (!renderResources || !webGPUState?.device) {
        if (frameId) cancelAnimationFrame(frameId);
        return;
      }

      updateCameraPosition(renderResources.camera, pressedKeysRef.current);

      //TODO: use webgpu-utils
      const viewMatrix = renderResources.camera.getViewMatrix();
      const forward = renderResources.camera.getForward();
      // Assuming uniform buffer layout: viewMatrix (16*4), projMatrix(16*4), grid(3*4), pad1(4), cameraForward(3*4), pad2(4)
      const cameraForwardOffset = 16 * 4 + 16 * 4 + 3 * 4 + 4; // Byte offset

      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        0, // offset for view matrix
        viewMatrix as Float32Array
      );
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        cameraForwardOffset,
        forward as Float32Array
      );

      renderScene(webGPUState, renderResources);

      frameId = requestAnimationFrame(animationLoop);
    }

    frameId = requestAnimationFrame(animationLoop);

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
    // rerun effect if resources or webgpu state changes
  }, [renderResources, webGPUState]);

  const mouseDownHandler = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    handleMouseDown(e, setIsDragging, setPrevMousePos);
  }, []);

  const mouseUpHandler = useCallback(() => {
    handleMouseUp(setIsDragging, setPrevMousePos);
  }, []);

  const mouseMoveHandler = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDragging || !renderResources || !webGPUState?.device) return;

      const cameraUpdated = handleMouseMove(
        e,
        setPrevMousePos,
        isDragging,
        prevMousePos,
        canvasRef as React.RefObject<HTMLCanvasElement>,
        renderResources as RenderPipelineResources,
        updateCameraRotation
      );

      if (cameraUpdated) {
        // update uniform buffer if camera rotation changed
        const viewMatrix = renderResources.camera.getViewMatrix();
        const forward = renderResources.camera.getForward();
        const cameraForwardOffset = 16 * 4 + 16 * 4 + 3 * 4 + 4;

        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0,
          viewMatrix as Float32Array
        );
        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          cameraForwardOffset,
          forward as Float32Array
        );

        renderScene(webGPUState, renderResources);
      }
    },
    [isDragging, prevMousePos, renderResources, webGPUState]
  );

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={1028}
        height={1028}
        className="w-full h-auto"
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={mouseDownHandler}
        onMouseUp={mouseUpHandler}
        onMouseLeave={mouseUpHandler}
        onMouseMove={mouseMoveHandler}
      />
    </div>
  );
};

export default WebGPUCanvas;
