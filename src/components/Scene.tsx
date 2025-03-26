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

const renderScene = (
  webGPUState: WebGPUState,
  renderResources: RenderPipelineResources,
  shouldSwapBindGroups: boolean
) => {
  const { device, context, canvasFormat } = webGPUState;
  const {
    wireframePipeline,
    slicesPipeline,
    computePipeline,
    applyExternalForcesPipeline,
    computeVorticityPipeline,
    computeVorticityConfinementPipeline,
    applyVorticityForcePipeline,
    advectVelocityPipeline,
    advectTemperaturePipeline,
    advectDensityPipeline,
    computeDivergencePipeline,
    solvePressureJacobiPipeline,
    applyPressureGradientPipeline,
    wireframeVertexBuffer,
    wireframeIndexBuffer,
    slicesVertexBuffer,
    slicesIndexBuffer,
    multisampleTexture,
    uniformBindGroup,
    applyExternalForcesBindGroupA,
    applyExternalForcesBindGroupB,
    computeVorticityBindGroupA,
    computeVorticityBindGroupB,
    vorticityConfinementBindGroupA,
    vorticityConfinementBindGroupB,
    applyVorticityForceBindGroupA,
    applyVorticityForceBindGroupB,
    advectVelocityBindGroupA,
    advectVelocityBindGroupB,
    advectTemperatureBindGroupA,
    advectTemperatureBindGroupB,
    advectDensityBindGroupA,
    advectDensityBindGroupB,
    computeDivergenceBindGroupA,
    computeDivergenceBindGroupB,
    solvePressureJacobiBindGroupA,
    solvePressureJacobiBindGroupB,
    applyPressureGradientBindGroupA,
    applyPressureGradientBindGroupB,
    renderBindGroupA,
    renderBindGroupB,
    wireframeIndexCount,
    slicesIndexCount,
    gridSize,
  } = renderResources;

  // Compute pass
  const workgroupSize = [4, 4, 4];
  // Total grid size including halos
  const totalGridSize = gridSize + 2; // Adding 1 cell padding on each side
  const numWorkgroups = [
    Math.ceil(totalGridSize / workgroupSize[0]),
    Math.ceil(totalGridSize / workgroupSize[1]),
    Math.ceil(totalGridSize / workgroupSize[2]),
  ];
  const JACOBI_ITERATIONS = 20;

  /////////////////////////////////////////////////////////////////////////
  // Simulation steps according to Fedkiw paper
  /////////////////////////////////////////////////////////////////////////

  // 1. Apply External Forces
  const applyForcesEncoder = device.createCommandEncoder();
  const applyForcesPass = applyForcesEncoder.beginComputePass();
  applyForcesPass.setPipeline(applyExternalForcesPipeline);
  applyForcesPass.setBindGroup(0, uniformBindGroup);
  applyForcesPass.setBindGroup(
    1,
    shouldSwapBindGroups ? applyExternalForcesBindGroupA : applyExternalForcesBindGroupB
  );
  applyForcesPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  applyForcesPass.end();
  device.queue.submit([applyForcesEncoder.finish()]);

  // 2a. Compute Vorticity (curl of velocity field)
  const vorticityEncoder = device.createCommandEncoder();
  const vorticityPass = vorticityEncoder.beginComputePass();
  vorticityPass.setPipeline(computeVorticityPipeline);
  vorticityPass.setBindGroup(0, uniformBindGroup);
  vorticityPass.setBindGroup(
    1,
    !shouldSwapBindGroups ? computeVorticityBindGroupA : computeVorticityBindGroupB
  );
  vorticityPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  vorticityPass.end();
  device.queue.submit([vorticityEncoder.finish()]);

  // 2b. Compute Vorticity Confinement forces
  const vorticityConfinementEncoder = device.createCommandEncoder();
  const vorticityConfinementPass = vorticityConfinementEncoder.beginComputePass();
  vorticityConfinementPass.setPipeline(computeVorticityConfinementPipeline);
  vorticityConfinementPass.setBindGroup(0, uniformBindGroup);
  vorticityConfinementPass.setBindGroup(
    1,
    shouldSwapBindGroups ? vorticityConfinementBindGroupA : vorticityConfinementBindGroupB
  );
  vorticityConfinementPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  vorticityConfinementPass.end();
  device.queue.submit([vorticityConfinementEncoder.finish()]);

  // 2c. Apply Vorticity Force
  const applyVorticityEncoder = device.createCommandEncoder();
  const applyVorticityPass = applyVorticityEncoder.beginComputePass();
  applyVorticityPass.setPipeline(applyVorticityForcePipeline);
  applyVorticityPass.setBindGroup(0, uniformBindGroup);
  applyVorticityPass.setBindGroup(
    1,
    !shouldSwapBindGroups ? applyVorticityForceBindGroupA : applyVorticityForceBindGroupB
  );
  applyVorticityPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  applyVorticityPass.end();
  device.queue.submit([applyVorticityEncoder.finish()]);

  // 3. Self-Advection (advect velocity field)
  const selfAdvectEncoder = device.createCommandEncoder();
  const selfAdvectPass = selfAdvectEncoder.beginComputePass();
  selfAdvectPass.setPipeline(advectVelocityPipeline);
  selfAdvectPass.setBindGroup(0, uniformBindGroup);
  selfAdvectPass.setBindGroup(
    1,
    shouldSwapBindGroups ? advectVelocityBindGroupA : advectVelocityBindGroupB
  );
  selfAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  selfAdvectPass.end();
  device.queue.submit([selfAdvectEncoder.finish()]);

  // 4a. Compute Divergence
  const divergenceEncoder = device.createCommandEncoder();
  const divergencePass = divergenceEncoder.beginComputePass();
  divergencePass.setPipeline(computeDivergencePipeline);
  divergencePass.setBindGroup(0, uniformBindGroup);
  divergencePass.setBindGroup(
    1,
    !shouldSwapBindGroups ? computeDivergenceBindGroupA : computeDivergenceBindGroupB
  );
  divergencePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  divergencePass.end();
  device.queue.submit([divergenceEncoder.finish()]);

  // 4b. Solve Pressure Poisson equation using Jacobi iteration
  let iterBindGroupSwap = !shouldSwapBindGroups;
  for (let i = 0; i < JACOBI_ITERATIONS; i++) {
    const pressureJacobiEncoder = device.createCommandEncoder();
    const pressureJacobiPass = pressureJacobiEncoder.beginComputePass();
    pressureJacobiPass.setPipeline(solvePressureJacobiPipeline);
    pressureJacobiPass.setBindGroup(0, uniformBindGroup);
    pressureJacobiPass.setBindGroup(
      1,
      iterBindGroupSwap ? solvePressureJacobiBindGroupA : solvePressureJacobiBindGroupB
    );
    pressureJacobiPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
    pressureJacobiPass.end();
    device.queue.submit([pressureJacobiEncoder.finish()]);

    // Swap bind groups for next iteration
    iterBindGroupSwap = !iterBindGroupSwap;
  }

  // 4c. Apply Pressure Gradient
  const pressureGradientEncoder = device.createCommandEncoder();
  const pressureGradientPass = pressureGradientEncoder.beginComputePass();
  pressureGradientPass.setPipeline(applyPressureGradientPipeline);
  pressureGradientPass.setBindGroup(0, uniformBindGroup);
  pressureGradientPass.setBindGroup(
    1,
    iterBindGroupSwap ? applyPressureGradientBindGroupA : applyPressureGradientBindGroupB
  );
  pressureGradientPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  pressureGradientPass.end();
  device.queue.submit([pressureGradientEncoder.finish()]);

  // 5. Scalar Field Advection (advect temperature and density)
  // 5a. Temperature advection
  const temperatureAdvectEncoder = device.createCommandEncoder();
  const temperatureAdvectPass = temperatureAdvectEncoder.beginComputePass();
  temperatureAdvectPass.setPipeline(advectTemperaturePipeline);
  temperatureAdvectPass.setBindGroup(0, uniformBindGroup);
  temperatureAdvectPass.setBindGroup(
    1,
    !iterBindGroupSwap ? advectTemperatureBindGroupA : advectTemperatureBindGroupB
  );
  temperatureAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  temperatureAdvectPass.end();
  device.queue.submit([temperatureAdvectEncoder.finish()]);

  // 5b. Density advection
  const densityAdvectEncoder = device.createCommandEncoder();
  const densityAdvectPass = densityAdvectEncoder.beginComputePass();
  densityAdvectPass.setPipeline(advectDensityPipeline);
  densityAdvectPass.setBindGroup(0, uniformBindGroup);
  densityAdvectPass.setBindGroup(
    1,
    !iterBindGroupSwap ? advectDensityBindGroupA : advectDensityBindGroupB
  );
  densityAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  densityAdvectPass.end();
  device.queue.submit([densityAdvectEncoder.finish()]);

  /////////////////////////////////////////////////////////////////////////
  // Render pass
  /////////////////////////////////////////////////////////////////////////
  const renderEncoder = device.createCommandEncoder();
  const renderPass = renderEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: multisampleTexture.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });

  renderPass.setPipeline(wireframePipeline);
  renderPass.setBindGroup(0, uniformBindGroup);
  renderPass.setBindGroup(1, !shouldSwapBindGroups ? renderBindGroupB : renderBindGroupA);

  // Draw wireframe
  renderPass.setVertexBuffer(0, wireframeVertexBuffer);
  renderPass.setIndexBuffer(wireframeIndexBuffer, 'uint32');
  renderPass.drawIndexed(wireframeIndexCount);

  // Draw slices
  renderPass.setPipeline(slicesPipeline); // TODO: optimize this out using render bundles?
  renderPass.setVertexBuffer(0, slicesVertexBuffer);
  renderPass.setIndexBuffer(slicesIndexBuffer, 'uint32');
  renderPass.drawIndexed(slicesIndexCount);

  renderPass.end();

  device.queue.submit([renderEncoder.finish()]);

  // After all these steps, we need to return whether we should swap bind groups for the next frame
  // We need to make sure this is consistent with the final state after all our operations
  return !iterBindGroupSwap;
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const pressedKeysRef = useRef(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);
  const shouldSwapBindGroups = useRef(true);

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

  //camera movement
  useEffect(() => {
    if (!renderResources || !webGPUState) return;

    let frameId: number;

    function moveCamera() {
      if (!renderResources || !webGPUState) return;
      updateCameraPosition(renderResources.camera, pressedKeysRef.current);

      if (webGPUState.device) {
        const viewMatrix = renderResources.camera.getViewMatrix();
        const forward = renderResources.camera.getForward();

        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0,
          viewMatrix as Float32Array
        );

        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          2 * 16 * 4 + 3 * 4 + 4,
          new Float32Array([...forward])
        );

        // Update shouldSwapBindGroups based on return value from renderScene
        shouldSwapBindGroups.current = renderScene(
          webGPUState,
          renderResources,
          shouldSwapBindGroups.current
        );
      }

      frameId = requestAnimationFrame(moveCamera);
    }

    frameId = requestAnimationFrame(moveCamera);
    return () => cancelAnimationFrame(frameId);
  }, [renderResources, webGPUState]);

  // initial render
  useEffect(() => {
    if (!canvasRef.current || !webGPUState || !renderResources) return;
    // Update shouldSwapBindGroups based on return value from renderScene
    shouldSwapBindGroups.current = renderScene(
      webGPUState,
      renderResources,
      shouldSwapBindGroups.current
    );
  }, [webGPUState, renderResources]);

  return (
    <canvas
      className="bg"
      ref={canvasRef}
      width={1028}
      height={1028}
      onMouseDown={(e) => handleMouseDown(e, setIsDragging, setPrevMousePos)}
      onMouseUp={() => handleMouseUp(setIsDragging, setPrevMousePos)}
      onMouseLeave={() => handleMouseUp(setIsDragging, setPrevMousePos)}
      onMouseMove={(e) => {
        const didRotate = handleMouseMove(
          e,
          setPrevMousePos,
          isDragging,
          prevMousePos,
          canvasRef as React.RefObject<HTMLCanvasElement>,
          renderResources as RenderPipelineResources,
          updateCameraRotation
        );
        if (renderResources && didRotate && webGPUState?.device) {
          const viewMatrix = renderResources.camera.getViewMatrix();
          const forward = renderResources.camera.getForward();

          webGPUState.device.queue.writeBuffer(
            renderResources.uniformBuffer,
            0,
            viewMatrix as Float32Array
          );

          // TODO: refactor this to use webgpu-utils
          webGPUState.device.queue.writeBuffer(
            renderResources.uniformBuffer,
            2 * 16 * 4 + 3 * 4 + 4,
            new Float32Array([...forward])
          );

          // Update shouldSwapBindGroups based on return value from renderScene
          shouldSwapBindGroups.current = renderScene(
            webGPUState,
            renderResources,
            shouldSwapBindGroups.current
          );
        }
      }}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      tabIndex={0}
    />
  );
};

export default WebGPUCanvas;
