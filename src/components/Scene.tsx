'use client';
import { useEffect, useRef, useState } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';
import { updateCameraPosition, updateCameraRotation } from '@/utils/cameraMovement';

const renderPoints = (webGPUState: WebGPUState, resources: RenderPipelineResources) => {
  const { device, context, canvasFormat } = webGPUState;
  const { pipeline, vertexBuffer, indexBuffer, bindGroup, indexCount } = resources;

  const multisampleTexture = device.createTexture({
    format: canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    size: [context.getCurrentTexture().width, context.getCurrentTexture().height],
    sampleCount: 4,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
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

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setIndexBuffer(indexBuffer, 'uint32');
  pass.setBindGroup(0, bindGroup);
  pass.drawIndexed(indexCount);
  pass.end();

  device.queue.submit([encoder.finish()]);
};

// Main component
export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);
  const [pressedKeys, setPressedKeys] = useState(new Set<string>());
  const [isDragging, setIsDragging] = useState(false);
  const [prevMousePos, setPrevMousePos] = useState<{ x: number; y: number } | null>(null);

  // Handle keyboard input
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

  // Movement update loop
  useEffect(() => {
    if (!renderResources) return;

    const moveCamera = () => {
      const didMove = updateCameraPosition(renderResources.camera, pressedKeys);

      if (didMove && webGPUState?.device) {
        const viewMatrix = renderResources.camera.getViewMatrix();
        webGPUState.device.queue.writeBuffer(
          renderResources.uniformBuffer,
          0,
          viewMatrix as Float32Array
        );
        renderPoints(webGPUState, renderResources);
      }
    };

    let animationFrameId: number;
    function animate() {
      moveCamera();
      animationFrameId = requestAnimationFrame(animate);
    }

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [renderResources, webGPUState, pressedKeys]);

  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setPrevMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setPrevMousePos(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !prevMousePos || !canvasRef.current || !renderResources) return;

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
      renderPoints(webGPUState, renderResources);
    }
  };

  useEffect(() => {
    if (!canvasRef.current || !webGPUState || !renderResources) return;
    renderPoints(webGPUState, renderResources);
  }, [webGPUState, renderResources]);

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
