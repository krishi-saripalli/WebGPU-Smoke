'use client';
import { useEffect, useRef } from 'react';
import { useWebGPU, WebGPUState } from '@/hooks/useWebGPU';
import { useRenderResources, RenderPipelineResources } from '@/hooks/UseRenderResources';

const renderPoints = (webGPUState: WebGPUState, resources: RenderPipelineResources) => {
  const { device, context } = webGPUState;
  const { pipeline, vertexBuffer, indexBuffer, bindGroup } = resources; // Add indexBuffer

  const depthTexture = device.createTexture({
    size: {
      width: context.getCurrentTexture().width,
      height: context.getCurrentTexture().height,
      depthOrArrayLayers: 1,
    },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setIndexBuffer(indexBuffer, 'uint32');
  pass.setBindGroup(0, bindGroup);
  const gridSize = 4;
  const numIndices = 3 * gridSize * gridSize * 6;

  pass.drawIndexed(numIndices);
  pass.end();

  device.queue.submit([encoder.finish()]);
};

// Main component
export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const webGPUState = useWebGPU(canvasRef as React.RefObject<HTMLCanvasElement>);
  const renderResources = useRenderResources(webGPUState);

  // Render effect
  useEffect(() => {
    if (!canvasRef.current || !webGPUState || !renderResources) return;
    renderPoints(webGPUState, renderResources);
  }, [webGPUState, renderResources]);

  return <canvas ref={canvasRef} width={1028} height={1028} />;
};

export default WebGPUCanvas;
