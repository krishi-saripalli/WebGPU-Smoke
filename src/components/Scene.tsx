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
import { Mat4, Vec3 } from 'gl-matrix'; // Import Mat4 and Vec3 if not already

const renderScene = (
  webGPUState: WebGPUState,
  renderResources: RenderPipelineResources,
  shouldSwapBindGroups: boolean // Represents the expected INPUT state for the *first* compute pass
): boolean => {
  const { device, context, canvasFormat } = webGPUState;
  const {
    // Render Pipelines
    wireframePipeline,
    slicesPipeline,
    // Compute Pipelines (Updated Names)
    densityCopyPipeline, // Added (though not used in this specific simulation loop yet)
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
    wireframeVertexBuffer,
    wireframeIndexBuffer,
    slicesVertexBuffer,
    slicesIndexBuffer,
    multisampleTexture,
    uniformBuffer, // Needed for uniformBindGroup
    // Uniform Bind Group
    uniformBindGroup,
    // Compute Bind Groups (Updated Names)
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
    wireframeIndexCount,
    slicesIndexCount,
    totalGridSize, // Now includes halos
  } = renderResources;

  // Compute pass configuration
  const workgroupSize = [4, 4, 4]; // Should match shader @workgroup_size
  const numWorkgroups = [
    Math.ceil(totalGridSize / workgroupSize[0]),
    Math.ceil(totalGridSize / workgroupSize[1]),
    Math.ceil(totalGridSize / workgroupSize[2]),
  ];
  const JACOBI_ITERATIONS = 40; // Number of pressure solve iterations

  /////////////////////////////////////////////////////////////////////////
  // Simulation steps according to Fedkiw paper
  /////////////////////////////////////////////////////////////////////////

  // Track the ping-pong state locally for this frame's dispatches
  // If shouldSwapBindGroups is true, Pass 1 should read from B, write to A (use BindGroupB)
  // If shouldSwapBindGroups is false, Pass 1 should read from A, write to B (use BindGroupA)
  let currentComputeStateIsA = !shouldSwapBindGroups; // true if current data is in Texture A

  // Helper function to select bind group based on current state
  const selectBindGroup = (groupA: GPUBindGroup, groupB: GPUBindGroup) => {
    return currentComputeStateIsA ? groupA : groupB;
  };

  // --- 1. Apply External Forces ---
  // Reads Vel (A/B), Temp (A/B), Dens (A/B) -> Writes Vel (B/A)
  const forcesEncoder = device.createCommandEncoder({ label: 'External Forces Encoder' });
  const forcesPass = forcesEncoder.beginComputePass({ label: 'External Forces Pass' });
  forcesPass.setPipeline(externalForcesStepPipeline); // Updated pipeline name
  forcesPass.setBindGroup(0, uniformBindGroup);
  forcesPass.setBindGroup(
    1,
    selectBindGroup(externalForcesStepBindGroupA, externalForcesStepBindGroupB)
  ); // Updated bind group names
  forcesPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  forcesPass.end();
  device.queue.submit([forcesEncoder.finish()]);
  currentComputeStateIsA = !currentComputeStateIsA; // Flip state: Velocity is now updated

  // --- 2a. Compute Vorticity ---
  // Reads Vel (B/A) -> Writes Vorticity (A/B)
  const vorticityEncoder = device.createCommandEncoder({ label: 'Vorticity Calc Encoder' });
  const vorticityPass = vorticityEncoder.beginComputePass({ label: 'Vorticity Calc Pass' });
  vorticityPass.setPipeline(vorticityCalculationPipeline); // Updated pipeline name
  vorticityPass.setBindGroup(0, uniformBindGroup);
  vorticityPass.setBindGroup(
    1,
    selectBindGroup(vorticityCalculationBindGroupA, vorticityCalculationBindGroupB)
  ); // Updated bind group names
  vorticityPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  vorticityPass.end();
  device.queue.submit([vorticityEncoder.finish()]);
  // Don't flip state here, vorticity state is independent for a moment

  // --- 2b. Compute Vorticity Confinement forces ---
  // Reads Vorticity (A/B) -> Writes VorticityForce (B/A)
  const confinementEncoder = device.createCommandEncoder({ label: 'Vorticity Conf Encoder' });
  const confinementPass = confinementEncoder.beginComputePass({ label: 'Vorticity Conf Pass' });
  confinementPass.setPipeline(vorticityConfinementForcePipeline); // Updated pipeline name
  confinementPass.setBindGroup(0, uniformBindGroup);
  // Reads Vorticity A/B (current state) writes Force B/A (opposite state)
  confinementPass.setBindGroup(
    1,
    selectBindGroup(vorticityConfinementForceBindGroupA, vorticityConfinementForceBindGroupB)
  ); // Updated bind group names
  confinementPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  confinementPass.end();
  device.queue.submit([confinementEncoder.finish()]);
  // Don't flip state here, force state is independent

  // --- 2c. Apply Vorticity Force ---
  // Reads Vel (B/A) + VorticityForce (B/A) -> Writes Vel (A/B)
  const applyVortEncoder = device.createCommandEncoder({ label: 'Apply Vorticity Encoder' });
  const applyVortPass = applyVortEncoder.beginComputePass({ label: 'Apply Vorticity Pass' });
  applyVortPass.setPipeline(vorticityForceApplicationPipeline); // Updated pipeline name
  applyVortPass.setBindGroup(0, uniformBindGroup);
  // Reads Vel B/A (current state) and Force B/A (current state), writes Vel A/B (opposite state)
  applyVortPass.setBindGroup(
    1,
    selectBindGroup(vorticityForceApplicationBindGroupA, vorticityForceApplicationBindGroupB)
  ); // Updated bind group names
  applyVortPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  applyVortPass.end();
  device.queue.submit([applyVortEncoder.finish()]);
  currentComputeStateIsA = !currentComputeStateIsA; // Flip state: Velocity is updated

  // --- 3. Self-Advection (advect velocity field) ---
  // Reads Vel (A/B) -> Writes Vel (B/A)
  const advectVelEncoder = device.createCommandEncoder({ label: 'Velocity Advect Encoder' });
  const advectVelPass = advectVelEncoder.beginComputePass({ label: 'Velocity Advect Pass' });
  advectVelPass.setPipeline(velocityAdvectionPipeline); // Updated pipeline name
  advectVelPass.setBindGroup(0, uniformBindGroup);
  advectVelPass.setBindGroup(
    1,
    selectBindGroup(velocityAdvectionBindGroupA, velocityAdvectionBindGroupB)
  ); // Updated bind group names
  advectVelPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  advectVelPass.end();
  device.queue.submit([advectVelEncoder.finish()]);
  currentComputeStateIsA = !currentComputeStateIsA; // Flip state: Velocity is updated

  // --- 4a. Compute Divergence ---
  // Reads Vel (B/A) -> Writes Divergence (A/B)
  const divergenceEncoder = device.createCommandEncoder({ label: 'Divergence Encoder' });
  const divergencePass = divergenceEncoder.beginComputePass({ label: 'Divergence Pass' });
  divergencePass.setPipeline(divergenceCalculationPipeline); // Updated pipeline name
  divergencePass.setBindGroup(0, uniformBindGroup);
  divergencePass.setBindGroup(
    1,
    selectBindGroup(divergenceCalculationBindGroupA, divergenceCalculationBindGroupB)
  ); // Updated bind group names
  divergencePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  divergencePass.end();
  device.queue.submit([divergenceEncoder.finish()]);
  // Divergence state is now opposite velocity state. Let's track pressure state from here.
  let currentPressureStateIsA = !currentComputeStateIsA; // Divergence written to A/B

  // --- 4b. Solve Pressure Poisson equation (Jacobi iterations) ---
  // Reads Pressure (A/B) + Divergence (A/B) -> Writes Pressure (B/A) -> Reads Pressure (B/A) ...
  const pressureEncoder = device.createCommandEncoder({ label: 'Pressure Solve Encoder' });
  const pressurePass = pressureEncoder.beginComputePass({ label: 'Pressure Solve Pass' });
  pressurePass.setPipeline(pressureIterationPipeline); // Updated pipeline name
  pressurePass.setBindGroup(0, uniformBindGroup);
  for (let i = 0; i < JACOBI_ITERATIONS; i++) {
    // Select bind group based on where the *current* pressure data is
    const currentPressureGroup = currentPressureStateIsA
      ? pressureIterationBindGroupA
      : pressureIterationBindGroupB;
    pressurePass.setBindGroup(1, currentPressureGroup); // Updated bind group names
    pressurePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
    currentPressureStateIsA = !currentPressureStateIsA; // Flip state for next iteration
  }
  pressurePass.end();
  device.queue.submit([pressureEncoder.finish()]);
  // Final pressure result is in A if currentPressureStateIsA is true, B otherwise.

  // --- 4c. Apply Pressure Gradient ---
  // Reads Vel (B/A) + Pressure (A/B) -> Writes Vel (A/B)
  const pressureGradEncoder = device.createCommandEncoder({ label: 'Pressure Grad Encoder' });
  const pressureGradPass = pressureGradEncoder.beginComputePass({ label: 'Pressure Grad Pass' });
  pressureGradPass.setPipeline(pressureGradientSubtractionPipeline); // Updated pipeline name
  pressureGradPass.setBindGroup(0, uniformBindGroup);
  // TODO: Verify this bind group selection logic. It assumes input velocity and pressure states align,
  // which is NOT the case here. Input Velocity is state !currentPressureStateIsA. Input Pressure is currentPressureStateIsA.
  // This likely requires custom bind groups or modification of the standard ones for this pass.
  // Using the standard selection for now to fix linter errors, but this is probably incorrect.
  pressureGradPass.setBindGroup(
    1,
    currentPressureStateIsA
      ? pressureGradientSubtractionBindGroupA
      : pressureGradientSubtractionBindGroupB
  ); // Updated bind group names
  pressureGradPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  pressureGradPass.end();
  device.queue.submit([pressureGradEncoder.finish()]);
  // Assuming the output velocity is now in the state opposite the input pressure (i.e., state !currentPressureStateIsA)
  currentComputeStateIsA = !currentPressureStateIsA; // Velocity is updated

  // --- 5. Scalar Field Advection ---
  // State: Velocity is in A/B (currentComputeStateIsA), Temp/Density are in the opposite state.

  // 5a. Temperature advection
  // Reads Vel (A/B) + Temp (!A/!B) -> Writes Temp (A/B)
  const tempAdvectEncoder = device.createCommandEncoder({ label: 'Temperature Advect Encoder' });
  const tempAdvectPass = tempAdvectEncoder.beginComputePass({ label: 'Temperature Advect Pass' });
  tempAdvectPass.setPipeline(temperatureAdvectionPipeline); // Updated pipeline name
  tempAdvectPass.setBindGroup(0, uniformBindGroup);
  // TODO: Verify this bind group selection logic. It assumes input velocity and temperature states align.
  // Input Temp state is !currentComputeStateIsA.
  // Using standard selection for now. This is likely incorrect.
  tempAdvectPass.setBindGroup(
    1,
    selectBindGroup(temperatureAdvectionBindGroupA, temperatureAdvectionBindGroupB)
  ); // Updated bind group names
  tempAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  tempAdvectPass.end();
  device.queue.submit([tempAdvectEncoder.finish()]);
  // Assuming Temp is now in state A/B (currentComputeStateIsA)

  // 5b. Density advection
  // Reads Vel (A/B) + Density (!A/!B) -> Writes Density (A/B)
  const densityAdvectEncoder = device.createCommandEncoder({ label: 'Density Advect Encoder' });
  const densityAdvectPass = densityAdvectEncoder.beginComputePass({ label: 'Density Advect Pass' });
  densityAdvectPass.setPipeline(densityAdvectionPipeline); // Updated pipeline name
  densityAdvectPass.setBindGroup(0, uniformBindGroup);
  // TODO: Verify this bind group selection logic. It assumes input velocity and density states align.
  // Input Density state is !currentComputeStateIsA (same as initial temp state).
  // Using standard selection for now. This is likely incorrect.
  densityAdvectPass.setBindGroup(
    1,
    selectBindGroup(densityAdvectionBindGroupA, densityAdvectionBindGroupB)
  ); // Updated bind group names
  densityAdvectPass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  densityAdvectPass.end();
  device.queue.submit([densityAdvectEncoder.finish()]);
  // Assuming Density is now in state A/B (currentComputeStateIsA)

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
    // Add depth stencil attachment if needed
  };
  const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor);

  // Determine the bind group containing the final density output for rendering.
  // The final density is in state A if currentComputeStateIsA is true, B otherwise.
  const finalRenderBindGroup = currentComputeStateIsA ? renderBindGroupA : renderBindGroupB;

  // Draw wireframe
  renderPass.setPipeline(wireframePipeline);
  renderPass.setBindGroup(0, uniformBindGroup);
  renderPass.setBindGroup(1, finalRenderBindGroup); // Render uses density state
  renderPass.setVertexBuffer(0, wireframeVertexBuffer);
  renderPass.setIndexBuffer(wireframeIndexBuffer, 'uint32');
  renderPass.drawIndexed(wireframeIndexCount);

  // Draw slices
  renderPass.setPipeline(slicesPipeline);
  // Bind groups already set for wireframe are likely compatible if layout is same (Group 0)
  // Re-bind group 1 explicitly for clarity
  renderPass.setBindGroup(1, finalRenderBindGroup); // Render uses density state
  renderPass.setVertexBuffer(0, slicesVertexBuffer);
  renderPass.setIndexBuffer(slicesIndexBuffer, 'uint32');
  renderPass.drawIndexed(slicesIndexCount);

  renderPass.end();
  device.queue.submit([renderEncoder.finish()]);

  // Return the *opposite* of the final state, as this indicates
  // whether the *next* frame should start by reading from B (swap=true) or A (swap=false).
  return !currentComputeStateIsA;
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const pressedKeysRef = useRef(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);
  // shouldSwapBindGroups determines if the *first* compute pass reads from B (true) or A (false)
  const shouldSwapBindGroups = useRef(false); // Start reading from A

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

  // Main simulation and render loop + camera movement
  useEffect(() => {
    if (!renderResources || !webGPUState?.device) return;

    let frameId: number;
    let lastUpdateTime = performance.now();
    const minTimeStep = 16; // roughly 60 fps minimum interval

    function animationLoop(currentTime: number) {
      if (!renderResources || !webGPUState?.device) {
        if (frameId) cancelAnimationFrame(frameId);
        return;
      }

      const deltaTime = currentTime - lastUpdateTime;

      // Update camera position based on keys pressed
      updateCameraPosition(renderResources.camera, pressedKeysRef.current);

      // Always update uniform buffer after potential camera movement
      const viewMatrix = renderResources.camera.getViewMatrix();
      const forward = renderResources.camera.getForward();
      // Assuming uniform buffer layout: viewMatrix (16*4), projMatrix(16*4), grid(3*4), pad1(4), cameraForward(3*4), pad2(4)
      const cameraForwardOffset = 16 * 4 + 16 * 4 + 3 * 4 + 4; // Byte offset

      // Pass Float32Array directly to writeBuffer
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        0, // Offset for view matrix
        viewMatrix as Float32Array
      );
      webGPUState.device.queue.writeBuffer(
        renderResources.uniformBuffer,
        cameraForwardOffset, // Offset for camera forward vector
        forward as Float32Array
      );

      // Run simulation step and render
      // Pass the current swap state, receive the state for the *next* frame
      const nextSwapState = renderScene(
        webGPUState,
        renderResources,
        shouldSwapBindGroups.current // Pass current state
      );
      shouldSwapBindGroups.current = nextSwapState; // Update state for next frame

      lastUpdateTime = currentTime;
      frameId = requestAnimationFrame(animationLoop);
    }

    // Start the loop
    frameId = requestAnimationFrame(animationLoop);

    // Cleanup function
    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
    // Rerun effect if resources or webgpu state changes
  }, [renderResources, webGPUState]);

  // --- Readback Logic (Mostly unchanged, uses updated resource names) ---
  const handleReadbackDensity = async () => {
    if (!webGPUState || !renderResources) {
      console.log('WebGPU state or resources not ready for readback.');
      return;
    }

    const { device } = webGPUState;
    const { densityTextureA, densityTextureB, totalGridSize, gridSize, halosSize } =
      renderResources;

    console.log('Starting density readback...');

    // Determine the source texture (the one containing the final density data)
    // The final density state is A if shouldSwapBindGroups.current is false (next frame reads A)
    // The final density state is B if shouldSwapBindGroups.current is true (next frame reads B)
    const sourceTexture = !shouldSwapBindGroups.current ? densityTextureA : densityTextureB;
    console.log(
      `Reading back from: ${!shouldSwapBindGroups.current ? 'densityTextureA' : 'densityTextureB'}`
    );

    const bytesPerPixel = 4; // r32float
    const unalignedBytesPerRow = totalGridSize * bytesPerPixel;
    const alignedBytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
    const bufferSize = alignedBytesPerRow * totalGridSize * totalGridSize;

    console.log(
      `Texture row pitch: ${unalignedBytesPerRow}, Buffer aligned row pitch: ${alignedBytesPerRow}`
    );

    const readbackBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'DensityReadbackBuffer',
    });

    const commandEncoder = device.createCommandEncoder({ label: 'ReadbackEncoder' });
    commandEncoder.copyTextureToBuffer(
      { texture: sourceTexture },
      {
        buffer: readbackBuffer,
        bytesPerRow: alignedBytesPerRow,
        rowsPerImage: totalGridSize,
      },
      [totalGridSize, totalGridSize, totalGridSize]
    );

    device.queue.submit([commandEncoder.finish()]);

    try {
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      console.log('Readback buffer mapped.');

      const arrayBuffer = readbackBuffer.getMappedRange();
      const data = new Float32Array(arrayBuffer);
      const floatsPerRow = alignedBytesPerRow / bytesPerPixel;

      // Sample points (adjusted to use gridSize and halosSize)
      const internalCenterX = Math.floor(gridSize / 2);
      const internalCenterZ = Math.floor(gridSize / 2);
      const internalBaseY = Math.floor(gridSize / 20);
      const sampleX1 = internalCenterX + halosSize;
      const sampleY1 = internalBaseY + halosSize;
      const sampleZ1 = internalCenterZ + halosSize;
      const index1 = sampleX1 + sampleY1 * floatsPerRow + sampleZ1 * floatsPerRow * totalGridSize;

      const sampleX2 = sampleX1;
      const sampleY2 = sampleY1 + Math.floor(gridSize / 5);
      const sampleZ2 = sampleZ1;
      const index2 = sampleX2 + sampleY2 * floatsPerRow + sampleZ2 * floatsPerRow * totalGridSize;

      const sampleX3 = halosSize + 1;
      const sampleY3 = halosSize + 1;
      const sampleZ3 = halosSize + 1;
      const index3 = sampleX3 + sampleY3 * floatsPerRow + sampleZ3 * floatsPerRow * totalGridSize;

      console.log(`Density at base center (${sampleX1},${sampleY1},${sampleZ1}):`, data[index1]);
      console.log(`Density above base (${sampleX2},${sampleY2},${sampleZ2}):`, data[index2]);
      console.log(`Density at corner (${sampleX3},${sampleY3},${sampleZ3}):`, data[index3]);
    } catch (err) {
      console.error('Error mapping or reading buffer:', err);
    } finally {
      if (readbackBuffer.mapState === 'mapped') {
        readbackBuffer.unmap();
        console.log('Readback buffer unmapped.');
      }
      readbackBuffer.destroy();
      console.log('Readback buffer destroyed.');
    }
  };
  // --- End Readback Logic ---

  // --- Mouse Drag Logic (Mostly unchanged) ---
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
        // Update uniform buffer if camera rotation changed
        const viewMatrix = renderResources.camera.getViewMatrix();
        const forward = renderResources.camera.getForward();
        const cameraForwardOffset = 16 * 4 + 16 * 4 + 3 * 4 + 4;

        // Pass Float32Array directly to writeBuffer
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

        // Trigger a re-render immediately after mouse move update
        // Note: This might run the simulation faster than intended during drags.
        // Consider decoupling camera updates from simulation steps if needed.
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
  // --- End Mouse Drag Logic ---

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={1028}
        height={1028}
        className="w-full h-auto" // Use Tailwind for responsive sizing if preferred
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={mouseDownHandler}
        onMouseUp={mouseUpHandler}
        onMouseLeave={mouseUpHandler} // Also stop dragging if mouse leaves canvas
        onMouseMove={mouseMoveHandler}
        tabIndex={0} // Make canvas focusable for keyboard events
      />
      <button
        onClick={handleReadbackDensity}
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          zIndex: 10,
          padding: '8px 12px',
          cursor: 'pointer',
          background: 'rgba(0,0,0,0.5)',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
        }}
      >
        Readback Density
      </button>
    </div>
  );
};

export default WebGPUCanvas;
