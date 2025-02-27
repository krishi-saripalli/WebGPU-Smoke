'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/cameraMovement';
import {
  handleKeyDown,
  handleKeyUp,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
} from '@/utils/inputHandler';

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
    wireframeVertexBuffer,
    wireframeIndexBuffer,
    slicesVertexBuffer,
    slicesIndexBuffer,
    multisampleTexture,
    uniformBindGroup,
    computeBindGroupA,
    computeBindGroupB,
    renderBindGroupA,
    renderBindGroupB,
    wireframeIndexCount,
    slicesIndexCount,
    gridSize,
  } = renderResources;

  // Compute pass
  const workgroupSize = [4, 4, 4];
  const numWorkgroups = [
    Math.ceil(gridSize / workgroupSize[0]),
    Math.ceil(gridSize / workgroupSize[1]),
    Math.ceil(gridSize / workgroupSize[2]),
  ];

  /////////////////////////////////////////////////////////////////////////
  // Compute pass
  /////////////////////////////////////////////////////////////////////////
  const computeEncoder = device.createCommandEncoder();
  const computePass = computeEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, uniformBindGroup);
  computePass.setBindGroup(1, shouldSwapBindGroups ? computeBindGroupA : computeBindGroupB);
  computePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  computePass.end();

  device.queue.submit([computeEncoder.finish()]);

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
  renderPass.setBindGroup(1, shouldSwapBindGroups ? renderBindGroupB : renderBindGroupA);

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

        renderScene(webGPUState, renderResources, shouldSwapBindGroups.current);
        shouldSwapBindGroups.current = !shouldSwapBindGroups.current;
      }

      frameId = requestAnimationFrame(moveCamera);
    }

    frameId = requestAnimationFrame(moveCamera);
    return () => cancelAnimationFrame(frameId);
  }, [renderResources, webGPUState]);

  // initial render
  useEffect(() => {
    if (!canvasRef.current || !webGPUState || !renderResources) return;
    renderScene(webGPUState, renderResources, shouldSwapBindGroups.current);
    shouldSwapBindGroups.current = !shouldSwapBindGroups.current;
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

          webGPUState.device.queue.writeBuffer(
            renderResources.uniformBuffer,
            2 * 16 * 4 + 3 * 4 + 4,
            new Float32Array([...forward])
          );

          renderScene(webGPUState, renderResources, shouldSwapBindGroups.current);
          shouldSwapBindGroups.current = !shouldSwapBindGroups.current;
        }
      }}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      tabIndex={0}
    />
  );
};

export default WebGPUCanvas;
