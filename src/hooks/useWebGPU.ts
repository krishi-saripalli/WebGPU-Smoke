import { useEffect, useState } from 'react';

export interface WebGPUState {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

export const useWebGPU = (canvasRef: React.RefObject<HTMLCanvasElement>) => {
  const [state, setState] = useState<WebGPUState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [header, setHeader] = useState<string>('');
  const [min16float, setMin16float] = useState<string>('');
  const [min16floatStorage, setMin16floatStorage] = useState<string>('');

  useEffect(() => {
    const initWebGPU = async () => {
      try {
        if (!canvasRef.current || !navigator.gpu) return null;

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error('No WebGPU adapter found');
        }

        const hasShaderF16 = adapter.features.has('shader-f16');

        const hasFloat32Filterable = adapter.features.has('float32-filterable');
        console.log('hasShaderF16', hasShaderF16);
        console.log('hasFloat32Filterable', hasFloat32Filterable);
        let requiredFeatures: GPUFeatureName[] = [];

        if (hasShaderF16) {
          console.log('Shader F16 supported but not pushing to requiredFeatures');
          // requiredFeatures.push('shader-f16');
          if (hasFloat32Filterable) {
            requiredFeatures.push('float32-filterable');
          }
        } else {
          if (hasFloat32Filterable) {
            requiredFeatures.push('float32-filterable');
          } else {
            console.log('No float32 filterable support');
            throw new Error(
              'No float32 filterable support. Please use a browser/device with float32 filterable support'
            );
          }
        }
        //if f16 is supported, use f16, otherwise use f32
        const min16float = hasShaderF16 ? 'f32' : 'f32';
        const min16floatStorage = hasShaderF16 ? 'r32float' : 'r32float';
        const header = hasShaderF16
          ? // need to add enable f16; if f16 is supported
            ` alias min16float = ${min16float};`
          : `alias min16float = ${min16float};`;

        const device = await adapter.requestDevice({
          requiredFeatures: requiredFeatures,
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
        setHeader(header);
        setMin16float(min16float);
        setMin16floatStorage(min16floatStorage);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      }
    };

    initWebGPU();
  }, [canvasRef]);

  return { state, error, header, min16float, min16floatStorage };
};
