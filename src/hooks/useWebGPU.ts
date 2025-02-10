import { useEffect, useState } from 'react';

export interface WebGPUState {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

export const useWebGPU = (canvasRef: React.RefObject<HTMLCanvasElement>) => {
  const [state, setState] = useState<WebGPUState | null>(null);

  useEffect(() => {
    const initWebGPU = async () => {
      if (!canvasRef.current || !navigator.gpu) return null;

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;

      const device = await adapter.requestDevice();
      const context = canvasRef.current.getContext('webgpu');
      if (!context) return null;

      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device,
        format: canvasFormat,
      });

      setState({ device, context, canvasFormat });
    };

    initWebGPU();
  }, [canvasRef]);

  return state;
};
