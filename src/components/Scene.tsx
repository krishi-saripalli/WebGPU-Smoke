'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import {
  useRenderResources,
  RenderPipelineResources,
  shaderDefs,
} from '@/hooks/UseRenderResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/camera';
import { SimulationState } from '@/utils/types';
import {
  createUniformBindGroup,
  createForcesBindGroup,
  createVorticityCalculationBindGroup,
  createVorticityConfinementBindGroup,
  createDivergenceCalculationBindGroup,
  createPressureIterationBindGroup,
  createPressureGradientSubtractionBindGroup,
  createReinitializationBindGroup,
  createRenderBindGroup,
  createAdvectionBindGroup,
} from '@/utils/bind-group';
import {
  handleKeyDown,
  handleKeyUp,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
} from '@/utils/input-handler';
import {
  createDivergenceCalculationBindGroupLayout,
  createExternalForcesStepBindGroupLayout,
  createPressureGradientSubtractionBindGroupLayout,
  createPressureIterationBindGroupLayout,
  createReinitializationBindGroupLayout,
  createRenderBindGroupLayout,
  createUniformBindGroupLayout,
  createVorticityCalculationBindGroupLayout,
  createVorticityConfinementBindGroupLayout,
  createAdvectionBindGroupLayout,
} from '@/utils/layouts';

const renderScene = (
  webGPUState: WebGPUState,
  renderResources: RenderPipelineResources,
  min16floatStorage: string
): void => {
  const { device, context } = webGPUState;
  const {
    // Render Pipelines,
    slicesPipeline,
    wireframePipeline,
    // Compute Pipelines
    externalForcesStepPipeline,
    vorticityCalculationPipeline,
    vorticityConfinementPipeline,
    advectionPipeline,
    divergenceCalculationPipeline,
    pressureIterationPipeline,
    pressureGradientSubtractionPipeline,
    reinitializationPipeline,
    // Buffers
    slicesVertexBuffer,
    slicesIndexBuffer,
    wireframeVertexBuffer,
    wireframeIndexBuffer,
    wireframeIndexCount,
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
  const JACOBI_ITERATIONS = 30;

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

  const advectEncoder = device.createCommandEncoder({ label: 'Advect Encoder' });
  const advectPass = advectEncoder.beginComputePass({ label: 'Advect Pass' });
  advectPass.setPipeline(advectionPipeline);
  const advectBindGroup = createAdvectionBindGroup(
    device,
    createAdvectionBindGroupLayout(device, min16floatStorage),
    simulationState.velocity.current,
    simulationState.density.current,
    simulationState.temperature.current,
    sampler,
    simulationState.velocity.next,
    simulationState.density.next,
    simulationState.temperature.next
  );
  advectPass.setBindGroup(0, uniformBindGroup);
  advectPass.setBindGroup(1, advectBindGroup);
  advectPass.dispatchWorkgroups(...numWorkgroups);
  advectPass.end();
  device.queue.submit([advectEncoder.finish()]);
  swapTextures('velocity');
  swapTextures('density');
  swapTextures('temperature');

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
    createDivergenceCalculationBindGroupLayout(device, min16floatStorage),
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
      createPressureIterationBindGroupLayout(device, min16floatStorage),
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
    createReinitializationBindGroupLayout(device, min16floatStorage),
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

  const renderEncoder = device.createCommandEncoder({ label: 'Render Encoder' });
  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: multisampleTexture.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        clearValue: { r: 0.8705, g: 0.7764, b: 0.5137, a: 1.0 },
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

  // // wireframe
  // renderPass.setPipeline(wireframePipeline);
  // renderPass.setBindGroup(0, uniformBindGroup);
  // renderPass.setVertexBuffer(0, wireframeVertexBuffer);
  // renderPass.setIndexBuffer(wireframeIndexBuffer, 'uint32');
  // renderPass.drawIndexed(wireframeIndexCount);

  renderPass.end();
  device.queue.submit([renderEncoder.finish()]);
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const {
    state: webGPUState,
    header,
    min16float,
    min16floatStorage,
  } = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState, header, min16float, min16floatStorage);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const pressedKeysRef = useRef(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);

  const [cameraChanged, setCameraChanged] = useState(true);
  const cameraChangedRef = useRef(true);

  useEffect(() => {
    pressedKeysRef.current = pressedKeys;
  }, [pressedKeys]);

  useEffect(() => {
    cameraChangedRef.current = cameraChanged;
  }, [cameraChanged]);

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
    let startTime: number | null = null;

    function animationLoop(currentTime: number) {
      if (!renderResources || !webGPUState?.device) {
        if (frameId) cancelAnimationFrame(frameId);
        return;
      }

      if (startTime === null) {
        startTime = currentTime;
      }

      const cameraPositionChanged = updateCameraPosition(
        renderResources.camera,
        pressedKeysRef.current
      );

      if (cameraChangedRef.current || cameraPositionChanged) {
        // Update the shared uniformsView with all uniform data
        renderResources.uniformsView.set({
          viewMatrix: renderResources.camera.getViewMatrix(),
          projectionMatrix: renderResources.camera.getProjectionMatrix(),
          gridSize: renderResources.uniformConstants.gridSize,
          cameraForward: renderResources.camera.getForward(),
          cameraPos: renderResources.camera.getPosition(),
          lightPosition: renderResources.uniformConstants.lightPosition,
          lightIntensity: renderResources.uniformConstants.lightIntensity,
          ratio: renderResources.uniformConstants.ratio,
          absorption: renderResources.uniformConstants.absorption,
          scattering: renderResources.uniformConstants.scattering,
        });

        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0,
          renderResources.uniformsView.arrayBuffer
        );

        setCameraChanged(false);
        cameraChangedRef.current = false;
      }

      renderScene(webGPUState, renderResources, min16floatStorage);

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
        // Set flag to update uniforms in the next animation frame
        setCameraChanged(true);
        cameraChangedRef.current = true;
      }
    },
    [isDragging, prevMousePos, renderResources, webGPUState?.device]
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
