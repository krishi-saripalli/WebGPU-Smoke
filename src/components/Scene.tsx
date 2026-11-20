import { useEffect, useRef, useState } from 'react';
import { useWebGPU } from '@/hooks/useWebGPU';
import { useRenderResources, STEP_SECONDS, type Stroke } from '@/hooks/UseRenderResources';

const rotationKeys = new Map([
  ['KeyA', -1],
  ['KeyW', -1],
  ['ArrowLeft', -1],
  ['ArrowUp', -1],
  ['KeyD', 1],
  ['KeyS', 1],
  ['ArrowRight', 1],
  ['ArrowDown', 1],
]);

export const WebGPUCanvas = () => {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { state, error: deviceError } = useWebGPU(canvas);
  const { resources, error: simulationError } = useRenderResources(state);
  const keys = useRef(new Set<string>());
  const drag = useRef<{ id: number; position: [number, number] } | null>(null);
  const [dragging, setDragging] = useState(false);
  const stroke = useRef<Stroke | null>(null);

  useEffect(() => {
    const pressed = keys.current;
    const down = (e: KeyboardEvent) => {
      if (!rotationKeys.has(e.code) || e.metaKey || e.ctrlKey || e.altKey) return;
      e.preventDefault();
      pressed.add(e.code);
    };
    const up = (e: KeyboardEvent) => pressed.delete(e.code);
    const clear = () => pressed.clear();
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('blur', clear);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('blur', clear);
      clear();
    };
  }, []);

  useEffect(() => {
    if (!resources || !state) return;
    const { step } = resources;
    const { queue } = state.device;
    let frameId: number;
    let last = performance.now();
    let accumulator = STEP_SECONDS;
    let pending = 0;
    function animationLoop(now: number) {
      frameId = requestAnimationFrame(animationLoop);
      const elapsed = Math.min(Math.max((now - last) / 1000, 0), 0.1);
      last = now;
      accumulator = Math.min(2 * STEP_SECONDS, accumulator + elapsed);
      if (pending >= 2) return;
      let direction = 0;
      for (const key of keys.current) direction += rotationKeys.get(key) ?? 0;
      const rotation = Math.sign(direction) * elapsed * 1.3;
      const steps = Math.floor((accumulator + 1e-8) / STEP_SECONDS);
      if (steps || rotation) {
        accumulator = Math.max(0, accumulator - steps * STEP_SECONDS);
        step(steps, rotation, stroke.current);
        if (steps) stroke.current = null;
        pending++;
        queue.onSubmittedWorkDone().then(
          () => pending--,
          () => pending--
        );
      }
    }
    frameId = requestAnimationFrame(animationLoop);
    return () => cancelAnimationFrame(frameId);
  }, [resources, state]);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    if (!resources || !state) return;
    const resize = () => {
      const bounds = element.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const limit = state.device.limits.maxTextureDimension2D;
      const scale = Math.min(window.devicePixelRatio, limit / bounds.width, limit / bounds.height);
      const width = Math.max(1, Math.round(bounds.width * scale));
      const height = Math.max(1, Math.round(bounds.height * scale));
      if (element.width !== width || element.height !== height) {
        element.width = width;
        element.height = height;
      }
      resources.resize(bounds.width, bounds.height);
      stroke.current = null;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    window.addEventListener('resize', resize);
    resize();
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', resize);
    };
  }, [resources, state]);

  function pointerPosition(e: React.PointerEvent<HTMLCanvasElement>): [number, number] {
    const bounds = e.currentTarget.getBoundingClientRect();
    return resources!.screenToView(
      (2 * (e.clientX - bounds.left)) / bounds.width - 1,
      1 - (2 * (e.clientY - bounds.top)) / bounds.height
    );
  }

  const error = deviceError ?? simulationError;
  return (
    <div className="scene">
      {error && (
        <div role="alert" className="scene-error">
          {error}
        </div>
      )}
      <canvas
        ref={canvas}
        style={{ cursor: dragging ? 'grabbing' : 'grab', touchAction: 'none' }}
        aria-label="Smoke simulation. Use WASD or the arrow keys to rotate the box. Drag to stir the smoke."
        onPointerDown={(e) => {
          if (!resources || !e.isPrimary || e.button !== 0) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { id: e.pointerId, position: pointerPosition(e) };
          stroke.current = null;
          setDragging(true);
        }}
        onLostPointerCapture={(e) => {
          if (drag.current?.id !== e.pointerId) return;
          drag.current = null;
          setDragging(false);
        }}
        onPointerMove={(e) => {
          if (!resources || drag.current?.id !== e.pointerId) return;
          const end = pointerPosition(e);
          stroke.current = { start: stroke.current?.start ?? drag.current.position, end };
          drag.current.position = end;
        }}
      />
    </div>
  );
};

export default WebGPUCanvas;
