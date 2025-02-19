'use client';
import { useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';
import { useComputeResources, ComputePipelineResources } from '@/hooks/useComputeResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/cameraMovement';
import { vec3 } from 'gl-matrix';

const renderScene = (
  webGPUState: WebGPUState,
  renderResources: RenderPipelineResources,
  computeResources: ComputePipelineResources
) => {
  const { device, context, canvasFormat } = webGPUState;
  const { pipeline, vertexBuffer, indexBuffer, renderBindGroup, indexCount } = renderResources;
  const { computePipeline, computeBindGroup, gridSize } = computeResources;

  const multisampleTexture = device.createTexture({
    format: canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    size: [context.getCurrentTexture().width, context.getCurrentTexture().height],
    sampleCount: 4,
  });

  const encoder = device.createCommandEncoder();

  // Compute pass
  const workgroupSize = [4, 4, 4];
  const numWorkgroups = [
    Math.ceil(gridSize / workgroupSize[0]),
    Math.ceil(gridSize / workgroupSize[1]),
    Math.ceil(gridSize / workgroupSize[2]),
  ];
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
  computePass.end();

  const renderPass = encoder.beginRenderPass({
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

  renderPass.setPipeline(pipeline);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setIndexBuffer(indexBuffer, 'uint32');
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.drawIndexed(indexCount);
  renderPass.end();

  device.queue.submit([encoder.finish()]);
};

export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const computeResources = useComputeResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      setPressedKeys((prev) => new Set(prev).add(e.key.toLowerCase()));
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      setPressedKeys((prev) => {
        const next = new Set(prev);
        next.delete(e.key.toLowerCase());
        return next;
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  useEffect(() => {
    if (!renderResources || !computeResources) return;

    const moveCamera = () => {
      const didMove = updateCameraPosition(renderResources.camera, pressedKeys);

      if (didMove && webGPUState?.device) {
        const viewMatrix = renderResources.camera.getViewMatrix();
        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0,
          viewMatrix as Float32Array
        );
        renderScene(webGPUState, renderResources, computeResources);
      }
    };

    let animationFrameId: number;
    function animate() {
      moveCamera();
      animationFrameId = requestAnimationFrame(animate);
    }

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [renderResources, computeResources, webGPUState, pressedKeys]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setPrevMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setPrevMousePos(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !prevMousePos || !canvasRef.current || !renderResources || !computeResources)
      return;

    const deltaX = e.clientX - prevMousePos.x;
    const deltaY = e.clientY - prevMousePos.y;
    setPrevMousePos({ x: e.clientX, y: e.clientY });

    updateCameraRotation(
      renderResources.camera,
      deltaX,
      deltaY,
      canvasRef.current.width,
      canvasRef.current.height
    );

    const device = webGPUState?.device;
    if (device) {
      const viewMatrix = renderResources.camera.getViewMatrix();
      device.queue.writeBuffer(renderResources.uniformBuffer, 0, viewMatrix as Float32Array);
      renderScene(webGPUState, renderResources, computeResources);
    }
  };

  useEffect(() => {
    if (!canvasRef.current || !webGPUState || !renderResources || !computeResources) return;
    renderScene(webGPUState, renderResources, computeResources);
  }, [webGPUState, renderResources, computeResources]);

  return (
    <canvas
      ref={canvasRef}
      width={1028}
      height={1028}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      tabIndex={0}
    />
  );
};

export default WebGPUCanvas;
