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

      // Check if the adapter supports filterable float textures
      if (!adapter.features.has('float32-filterable')) {
        console.error(
          "Required WebGPU feature 'float32-filterable' is not supported on this browser/GPU."
        );

        return null;
      } else {
        console.log(
          'Required WebGPU feature "float32-filterable" is supported on this browser/GPU!'
        );
      }

      const device = await adapter.requestDevice({
        requiredFeatures: ['float32-filterable'],
      });
      const context = canvasRef.current.getContext('webgpu');
      if (!context) return null;

      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device,
        format: canvasFormat,
        alphaMode: 'premultiplied',
      });

      setState({ device, context, canvasFormat });
    };

    initWebGPU();
  }, [canvasRef]);

  return state;
};
