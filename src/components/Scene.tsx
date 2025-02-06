"use client";
import { useEffect, useRef } from "react";
import { useWebGPU, WebGPUState } from "@/hooks/useWebGPU";
import {
  useRenderResources,
  RenderPipelineResources,
} from "@/hooks/UseRenderResources";

const renderPoints = (
  webGPUState: WebGPUState,
  resources: RenderPipelineResources
) => {
  const { device, context } = webGPUState;
  const { pipeline, vertexBuffer } = resources;

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  const numPoints = 3;
  pass.draw(6, numPoints); // since we're using instancing
  pass.end();

  device.queue.submit([encoder.finish()]);
};

// Main component
export const WebGPUCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  if (!canvasRef) {
    throw new Error("Canvas not found");
  }
  const webGPUState = useWebGPU(canvasRef);
  const renderResources = useRenderResources(webGPUState);

  // Render effect
  useEffect(() => {
    if (!webGPUState || !renderResources) return;
    renderPoints(webGPUState, renderResources);
  }, [webGPUState, renderResources]);

  return <canvas ref={canvasRef} width={512} height={512} />;
};

export default WebGPUCanvas;
