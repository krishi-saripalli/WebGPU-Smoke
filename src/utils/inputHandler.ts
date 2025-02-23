import { RenderPipelineResources } from '@/hooks/UseRenderResources';
import { Camera } from '@/modules/Camera';

export const handleKeyDown = (
  e: KeyboardEvent,
  setPressedKeys: React.Dispatch<React.SetStateAction<Set<string>>>
) => {
  setPressedKeys((prev) => new Set(prev).add(e.code));
};

export const handleKeyUp = (
  e: KeyboardEvent,
  setPressedKeys: React.Dispatch<React.SetStateAction<Set<string>>>
) => {
  console.log('key up', e.code);
  setPressedKeys((prev) => {
    const next = new Set(prev);
    next.delete(e.code);
    return next;
  });
};

export const handleMouseDown = (
  e: React.MouseEvent<HTMLCanvasElement>,
  setIsDragging: React.Dispatch<React.SetStateAction<boolean>>,
  setPrevMousePos: React.Dispatch<React.SetStateAction<{ x: number; y: number } | null>>
) => {
  setIsDragging(true);
  setPrevMousePos({ x: e.clientX, y: e.clientY });
};

export const handleMouseUp = (
  setIsDragging: React.Dispatch<React.SetStateAction<boolean>>,
  setPrevMousePos: React.Dispatch<React.SetStateAction<{ x: number; y: number } | null>>
) => {
  setIsDragging(false);
  setPrevMousePos(null);
};

export const handleMouseMove = (
  e: React.MouseEvent<HTMLCanvasElement>,
  setPrevMousePos: React.Dispatch<React.SetStateAction<{ x: number; y: number } | null>>,
  isDragging: boolean,
  prevMousePos: { x: number; y: number } | null,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  renderResources: RenderPipelineResources,
  updateCameraRotation: (
    camera: Camera,
    deltaX: number,
    deltaY: number,
    canvasWidth: number,
    canvasHeight: number
  ) => void
) => {
  if (!isDragging || !prevMousePos || !canvasRef.current || !renderResources) return false;

  const deltaX = e.clientX - prevMousePos.x;
  const deltaY = e.clientY - prevMousePos.y;
  if (deltaX === 0 && deltaY === 0) return false;
  setPrevMousePos({ x: e.clientX, y: e.clientY });

  updateCameraRotation(
    renderResources.camera,
    deltaX,
    deltaY,
    canvasRef.current.width,
    canvasRef.current.height
  );
  return true;
};
