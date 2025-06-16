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
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';
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

  /////////////////////////////////////////////////////////////////////////
  // Render pass
  /////////////////////////////////////////////////////////////////////////
  const renderEncoder = device.createCommandEncoder({ label: 'Render Encoder' });
  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: multisampleTexture.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 }, //dark brown
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

  const viewMatrixDef = makeShaderDataDefinitions(`
    struct ViewMatrixUpdate {
      viewMatrix: mat4x4<f32>,
    };
  `);
  const viewMatrixView = makeStructuredView(viewMatrixDef.structs.ViewMatrixUpdate);

  const cameraForwardDef = makeShaderDataDefinitions(`
    struct CameraForwardUpdate {
      cameraForward: vec3<f32>,
    };
  `);
  const cameraForwardView = makeStructuredView(cameraForwardDef.structs.CameraForwardUpdate);

  const lightPositionDef = makeShaderDataDefinitions(`
    struct LightPositionUpdate {
      lightPosition: vec3<f32>,
    };
  `);
  const lightPositionView = makeStructuredView(lightPositionDef.structs.LightPositionUpdate);

  const lightPosition2Def = makeShaderDataDefinitions(`
    struct LightPosition2Update {
      lightPosition2: vec3<f32>,
    };
  `);
  const lightPosition2View = makeStructuredView(lightPosition2Def.structs.LightPosition2Update);

  const cameraPosDef = makeShaderDataDefinitions(`
    struct CameraPosUpdate {
      cameraPos: vec3<f32>,
    };
  `);
  const cameraPosView = makeStructuredView(cameraPosDef.structs.CameraPosUpdate);

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
    let startTime: number | null = null;

    function animationLoop(currentTime: number) {
      if (!renderResources || !webGPUState?.device) {
        if (frameId) cancelAnimationFrame(frameId);
        return;
      }

      if (startTime === null) {
        startTime = currentTime;
      }

      const elapsedTime = (currentTime - startTime) / 1000.0;

      updateCameraPosition(renderResources.camera, pressedKeysRef.current);

      viewMatrixView.set({
        viewMatrix: renderResources.camera.getViewMatrix(),
      });
      cameraForwardView.set({
        cameraForward: renderResources.camera.getForward(),
      });

      const radius = 1.0;
      const rotationSpeed = 0.6;

      const t = elapsedTime * rotationSpeed;

      const lightX = radius * Math.cos(t);
      const lightY = 0.0;
      const lightZ = radius * Math.sin(t);

      lightPositionView.set({
        lightPosition: [lightX, lightY, lightZ],
      });

      const light2X = radius * Math.cos(-t);
      const light2Y = 0.0;
      const light2Z = radius * Math.sin(-t);

      lightPosition2View.set({
        lightPosition2: [light2X, light2Y, light2Z],
      });

      cameraPosView.set({
        cameraPos: renderResources.camera.getPosition(),
      });

      // viewMatrix is at offset 0, cameraForward is at offset 16*4 + 16*4 + 3*4 + 4 = 148 bytes
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        0, // viewMatrix offset
        viewMatrixView.arrayBuffer
      );
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        148, // cameraForward offset (after viewMatrix + projMatrix + gridSize + pad1)
        cameraForwardView.arrayBuffer
      );

      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        176, // lightPosition offset
        lightPositionView.arrayBuffer
      );

      // lightPosition2 offset calculation:
      // lightPosition(176) + vec3<f32>(12) + _pad3(4) + lightIntensity(12) + _pad4(4) + ratio(12) + _pad5(4) = 224
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        224, // lightPosition2 offset
        lightPosition2View.arrayBuffer
      );

      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        160, // cameraPos offset
        cameraPosView.arrayBuffer
      );

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
        // update uniform buffer if camera rotation changed using webgpu-utils
        viewMatrixView.set({
          viewMatrix: renderResources.camera.getViewMatrix(),
        });
        cameraForwardView.set({
          cameraForward: renderResources.camera.getForward(),
        });

        cameraPosView.set({
          cameraPos: renderResources.camera.getPosition(),
        });

        // Write to specific offsets in the uniform buffer
        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0, // viewMatrix offset
          viewMatrixView.arrayBuffer
        );
        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          148, // cameraForward offset
          cameraForwardView.arrayBuffer
        );

        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          160, // cameraPos offset
          cameraPosView.arrayBuffer
        );

        renderScene(webGPUState, renderResources, min16floatStorage);
      }
    },
    [
      isDragging,
      prevMousePos,
      renderResources,
      webGPUState,
      min16floatStorage,
      viewMatrixView,
      cameraForwardView,
      cameraPosView,
      lightPositionView,
      lightPosition2View,
    ]
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
