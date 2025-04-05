'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/camera-movement';
import {
  handleKeyDown,
  handleKeyUp,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
} from '@/utils/input-handler';
import { makeStructuredView, makeShaderDataDefinitions } from 'webgpu-utils';

const renderScene = (
  webGPUState: WebGPUState,
  renderResources: RenderPipelineResources,
  shouldSwapBindGroups: boolean // If true, first pass reads from B; if false, reads from A
): boolean => {
  const { device, context, canvasFormat } = webGPUState;
  const {
    // Render Pipelines,
    slicesPipeline,
    // Compute Pipelines
    densityCopyPipeline,
    externalForcesStepPipeline,
    vorticityCalculationPipeline,
    vorticityConfinementForcePipeline,
    vorticityForceApplicationPipeline,
    velocityAdvectionPipeline,
    temperatureAdvectionPipeline,
    densityAdvectionPipeline,
    divergenceCalculationPipeline,
    pressureIterationPipeline,
    pressureGradientSubtractionPipeline,
    // Buffers
    slicesVertexBuffer,
    slicesIndexBuffer,
    multisampleTexture,
    uniformBuffer,
    // Uniform Bind Group
    uniformBindGroup,
    // Compute Bind Groups
    externalForcesStepBindGroupA,
    externalForcesStepBindGroupB,
    vorticityCalculationBindGroupA,
    vorticityCalculationBindGroupB,
    vorticityConfinementForceBindGroupA,
    vorticityConfinementForceBindGroupB,
    vorticityForceApplicationBindGroupA,
    vorticityForceApplicationBindGroupB,
    velocityAdvectionBindGroupA,
    velocityAdvectionBindGroupB,
    temperatureAdvectionBindGroupA,
    temperatureAdvectionBindGroupB,
    densityAdvectionBindGroupA,
    densityAdvectionBindGroupB,
    divergenceCalculationBindGroupA,
    divergenceCalculationBindGroupB,
    pressureIterationBindGroupA,
    pressureIterationBindGroupB,
    pressureGradientSubtractionBindGroupA,
    pressureGradientSubtractionBindGroupB,
    // Render Bind Groups
    renderBindGroupA,
    renderBindGroupB,
    // Counts & Sizes
    slicesIndexCount,
    totalGridSize,
  } = renderResources;

  const workgroupSize = [4, 4, 4];
  const numWorkgroups = [
    Math.ceil(totalGridSize / workgroupSize[0]),
    Math.ceil(totalGridSize / workgroupSize[1]),
    Math.ceil(totalGridSize / workgroupSize[2]),
  ];

  //TODO: Jacobi solver introduces shimmering artifacts??
  const JACOBI_ITERATIONS = 0;

  // initially, shouldSwapBindGroups is false, so  data is in A
  let dataIsInA = !shouldSwapBindGroups;

  const selectBindGroup = (groupA: GPUBindGroup, groupB: GPUBindGroup) => {
    return dataIsInA ? groupA : groupB;
  };

  const forcesEncoder = device.createCommandEncoder({ label: 'External Forces Encoder' });
  const forcesPass = forcesEncoder.beginComputePass({ label: 'External Forces Pass' });
  forcesPass.setPipeline(externalForcesStepPipeline);
  forcesPass.setBindGroup(0, uniformBindGroup);
  forcesPass.setBindGroup(
    1,
    selectBindGroup(externalForcesStepBindGroupA, externalForcesStepBindGroupB)
  );
  forcesPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  forcesPass.end();
  device.queue.submit([forcesEncoder.finish()]);
  dataIsInA = !dataIsInA;

  const vorticityEncoder = device.createCommandEncoder({ label: 'Vorticity Calc Encoder' });
  const vorticityPass = vorticityEncoder.beginComputePass({ label: 'Vorticity Calc Pass' });
  vorticityPass.setPipeline(vorticityCalculationPipeline);
  vorticityPass.setBindGroup(0, uniformBindGroup);
  vorticityPass.setBindGroup(
    1,
    selectBindGroup(vorticityCalculationBindGroupA, vorticityCalculationBindGroupB)
  );
  vorticityPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  vorticityPass.end();
  device.queue.submit([vorticityEncoder.finish()]);

  const confinementEncoder = device.createCommandEncoder({ label: 'Vorticity Conf Encoder' });
  const confinementPass = confinementEncoder.beginComputePass({ label: 'Vorticity Conf Pass' });
  confinementPass.setPipeline(vorticityConfinementForcePipeline);
  confinementPass.setBindGroup(0, uniformBindGroup);
  confinementPass.setBindGroup(
    1,
    selectBindGroup(vorticityConfinementForceBindGroupA, vorticityConfinementForceBindGroupB)
  );
  confinementPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  confinementPass.end();
  device.queue.submit([confinementEncoder.finish()]);
  // not flipping state - vorticity confinement computes forces from vorticity

  const applyVortEncoder = device.createCommandEncoder({ label: 'Apply Vorticity Encoder' });
  const applyVortPass = applyVortEncoder.beginComputePass({ label: 'Apply Vorticity Pass' });
  applyVortPass.setPipeline(vorticityForceApplicationPipeline);
  applyVortPass.setBindGroup(0, uniformBindGroup);
  applyVortPass.setBindGroup(
    1,
    selectBindGroup(vorticityForceApplicationBindGroupA, vorticityForceApplicationBindGroupB)
  );
  applyVortPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  applyVortPass.end();
  device.queue.submit([applyVortEncoder.finish()]);
  dataIsInA = !dataIsInA;

  const advectVelEncoder = device.createCommandEncoder({ label: 'Velocity Advect Encoder' });
  const advectVelPass = advectVelEncoder.beginComputePass({ label: 'Velocity Advect Pass' });
  advectVelPass.setPipeline(velocityAdvectionPipeline);
  advectVelPass.setBindGroup(0, uniformBindGroup);
  advectVelPass.setBindGroup(
    1,
    selectBindGroup(velocityAdvectionBindGroupA, velocityAdvectionBindGroupB)
  );
  advectVelPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  advectVelPass.end();
  device.queue.submit([advectVelEncoder.finish()]);
  dataIsInA = !dataIsInA;

  const divergenceEncoder = device.createCommandEncoder({ label: 'Divergence Encoder' });
  const divergencePass = divergenceEncoder.beginComputePass({ label: 'Divergence Pass' });
  divergencePass.setPipeline(divergenceCalculationPipeline);
  divergencePass.setBindGroup(0, uniformBindGroup);
  divergencePass.setBindGroup(
    1,
    selectBindGroup(divergenceCalculationBindGroupA, divergenceCalculationBindGroupB)
  );
  divergencePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  divergencePass.end();
  device.queue.submit([divergenceEncoder.finish()]);

  let pressureIsInA = !dataIsInA; // initial pressure state is opposite of velocity state

  const pressureEncoder = device.createCommandEncoder({ label: 'Pressure Solve Encoder' });
  const pressurePass = pressureEncoder.beginComputePass({ label: 'Pressure Solve Pass' });
  pressurePass.setPipeline(pressureIterationPipeline);
  pressurePass.setBindGroup(0, uniformBindGroup);

  for (let i = 0; i < JACOBI_ITERATIONS; i++) {
    pressurePass.setBindGroup(
      1,
      pressureIsInA ? pressureIterationBindGroupA : pressureIterationBindGroupB
    );
    pressurePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
    pressureIsInA = !pressureIsInA;
  }

  pressurePass.end();
  device.queue.submit([pressureEncoder.finish()]);

  const pressureGradEncoder = device.createCommandEncoder({ label: 'Pressure Grad Encoder' });
  const pressureGradPass = pressureGradEncoder.beginComputePass({ label: 'Pressure Grad Pass' });
  pressureGradPass.setPipeline(pressureGradientSubtractionPipeline);
  pressureGradPass.setBindGroup(0, uniformBindGroup);
  pressureGradPass.setBindGroup(
    1,
    pressureIsInA ? pressureGradientSubtractionBindGroupA : pressureGradientSubtractionBindGroupB
  );
  pressureGradPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  pressureGradPass.end();
  device.queue.submit([pressureGradEncoder.finish()]);
  dataIsInA = !pressureIsInA;

  const tempAdvectEncoder = device.createCommandEncoder({ label: 'Temperature Advect Encoder' });
  const tempAdvectPass = tempAdvectEncoder.beginComputePass({ label: 'Temperature Advect Pass' });
  tempAdvectPass.setPipeline(temperatureAdvectionPipeline);
  tempAdvectPass.setBindGroup(0, uniformBindGroup);
  tempAdvectPass.setBindGroup(
    1,
    selectBindGroup(temperatureAdvectionBindGroupA, temperatureAdvectionBindGroupB)
  );
  tempAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  tempAdvectPass.end();
  device.queue.submit([tempAdvectEncoder.finish()]);

  const densityAdvectEncoder = device.createCommandEncoder({ label: 'Density Advect Encoder' });
  const densityAdvectPass = densityAdvectEncoder.beginComputePass({ label: 'Density Advect Pass' });
  densityAdvectPass.setPipeline(densityAdvectionPipeline);
  densityAdvectPass.setBindGroup(0, uniformBindGroup);
  densityAdvectPass.setBindGroup(
    1,
    selectBindGroup(densityAdvectionBindGroupA, densityAdvectionBindGroupB)
  );
  densityAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  densityAdvectPass.end();
  device.queue.submit([densityAdvectEncoder.finish()]);

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

  // use final density state for rendering
  const finalRenderBindGroup = dataIsInA ? renderBindGroupA : renderBindGroupB;

  // TODO: get rid of wireframe
  // renderPass.setPipeline(wireframePipeline);
  renderPass.setBindGroup(0, uniformBindGroup);
  // renderPass.setBindGroup(1, finalRenderBindGroup);
  // renderPass.setVertexBuffer(0, wireframeVertexBuffer);
  // renderPass.setIndexBuffer(wireframeIndexBuffer, 'uint32');
  // renderPass.drawIndexed(wireframeIndexCount);

  // transparent slices
  renderPass.setPipeline(slicesPipeline);
  renderPass.setBindGroup(1, finalRenderBindGroup);
  renderPass.setVertexBuffer(0, slicesVertexBuffer);
  renderPass.setIndexBuffer(slicesIndexBuffer, 'uint32');
  renderPass.drawIndexed(slicesIndexCount);

  renderPass.end();
  device.queue.submit([renderEncoder.finish()]);

  return !dataIsInA;
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const pressedKeysRef = useRef(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);
  const shouldSwapBindGroups = useRef(false); // start reading from A

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
    let lastUpdateTime = performance.now();

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

      const nextSwapState = renderScene(webGPUState, renderResources, shouldSwapBindGroups.current);
      shouldSwapBindGroups.current = nextSwapState;

      lastUpdateTime = currentTime;
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

        const nextSwapState = renderScene(
          webGPUState,
          renderResources,
          shouldSwapBindGroups.current
        );
        shouldSwapBindGroups.current = nextSwapState;
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
