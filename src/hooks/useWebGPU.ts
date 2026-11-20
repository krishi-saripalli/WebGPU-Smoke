import { useEffect, useState } from 'react';

export interface WebGPUState {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

export const useWebGPU = (canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
  const [state, setState] = useState<WebGPUState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let device: GPUDevice | undefined;
    let context: GPUCanvasContext | null = null;
    async function initialize() {
      if (!navigator.gpu) throw new Error('This browser does not have WebGPU available.');
      const adapter = await navigator.gpu.requestAdapter();
      if (cancelled) return;
      if (!adapter) throw new Error('No WebGPU adapter found.');
      device = await adapter.requestDevice();
      if (cancelled) {
        device.destroy();
        return;
      }
      device.onuncapturederror = (event) => {
        if (!cancelled) {
          setError(event.error.message);
          setState(null);
        }
      };
      context = canvasRef.current?.getContext('webgpu') ?? null;
      if (!context) throw new Error('Unable to create a WebGPU canvas.');
      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
      device.lost.then((info) => {
        if (!cancelled) {
          setState(null);
          setError(`GPU device lost. ${info.message}`);
        }
      });
      setState({ device, context, canvasFormat });
    }
    initialize().catch((e) => {
      if (!cancelled) setError(String(e));
    });
    return () => {
      cancelled = true;
      context?.unconfigure();
      device?.destroy();
    };
  }, [canvasRef]);
  return { state, error };
};
