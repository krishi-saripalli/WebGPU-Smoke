import { vec3 } from 'gl-matrix';
import { Camera } from '@/modules/Camera';

export const MOVEMENT_SPEED = 0.1;

export function updateCameraPosition(camera: Camera, pressedKeys: Set<string>): void {
  if (pressedKeys.size === 0) return;

  const position = camera.getPosition();
  const forward = camera.getForward();
  const right = camera.getRight();
  const newPosition = vec3.clone(position);

  // Forward/Backward
  if (pressedKeys.has('KeyW')) {
    vec3.scaleAndAdd(newPosition, newPosition, forward, MOVEMENT_SPEED);
  }
  if (pressedKeys.has('KeyS')) {
    vec3.scaleAndAdd(newPosition, newPosition, forward, -MOVEMENT_SPEED);
  }

  // Left/Right
  if (pressedKeys.has('KeyA')) {
    vec3.scaleAndAdd(newPosition, newPosition, right, -MOVEMENT_SPEED);
  }
  if (pressedKeys.has('KeyD')) {
    vec3.scaleAndAdd(newPosition, newPosition, right, MOVEMENT_SPEED);
  }

  console.log(newPosition);

  // Update camera position
  vec3.copy(position, newPosition);
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
