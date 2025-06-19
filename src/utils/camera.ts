import { vec3 } from 'gl-matrix';
import { Camera } from '@/modules/Camera';

export const MOVEMENT_SPEED = 0.1;

export function updateCameraPosition(camera: Camera, pressedKeys: Set<string>): boolean {
  if (pressedKeys.size === 0) return false;

  const position = camera.getPosition();
  const forward = camera.getForward();
  const right = camera.getRight();
  const newPosition = vec3.clone(position);

  let moved = false;

  // Forward/Backward
  if (pressedKeys.has('KeyW')) {
    vec3.scaleAndAdd(newPosition, newPosition, forward, MOVEMENT_SPEED);
    moved = true;
  }
  if (pressedKeys.has('KeyS')) {
    vec3.scaleAndAdd(newPosition, newPosition, forward, -MOVEMENT_SPEED);
    moved = true;
  }

  // Left/Right
  if (pressedKeys.has('KeyA')) {
    vec3.scaleAndAdd(newPosition, newPosition, right, -MOVEMENT_SPEED);
    moved = true;
  }
  if (pressedKeys.has('KeyD')) {
    vec3.scaleAndAdd(newPosition, newPosition, right, MOVEMENT_SPEED);
    moved = true;
  }

  if (moved) {
    // Update camera position
    vec3.copy(position, newPosition);
  }

  return moved;
}

export function updateCameraRotation(
  camera: Camera,
  deltaX: number,
  deltaY: number,
  canvasWidth: number,
  canvasHeight: number
): void {
  const angleX = (2.0 * deltaX) / canvasWidth;
  const angleY = (2.0 * deltaY) / canvasHeight;

  camera.rotateCamera(angleX, angleY);
}
