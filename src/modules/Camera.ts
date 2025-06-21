import { Vec3, Mat4, vec3 } from 'gl-matrix';

export interface CameraData {
  position: Vec3;
  forward: Vec3;
  up: Vec3;
  heightAngle: number;
  near: number;
  far: number;
  aspect: number;
}

export class Camera {
  private position: Vec3;
  private forward: Vec3;
  private up: Vec3;
  private right: Vec3;
  private heightAngle: number;
  private near: number;
  private far: number;
  private aspect: number;

  constructor(cameraData: CameraData) {
    this.position = cameraData.position;
    this.forward = cameraData.forward;
    this.up = cameraData.up;
    this.heightAngle = cameraData.heightAngle;
    this.near = cameraData.near;
    this.far = cameraData.far;
    this.aspect = cameraData.aspect;

    this.right = vec3.create();
    this.updateVectors();
  }

  private updateVectors(): void {
    vec3.normalize(this.forward, this.forward);

    vec3.cross(this.right, this.forward, this.up);
    vec3.normalize(this.right, this.right);

    vec3.cross(this.up, this.right, this.forward);
    vec3.normalize(this.up, this.up);
  }

  getViewMatrix(): Mat4 {
    const translateMatrix = Mat4.fromValues(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      -this.position[0], -this.position[1], -this.position[2], 1
    );

    const rotateMatrix = Mat4.fromValues(
      this.right[0], this.up[0], -this.forward[0], 0,
      this.right[1], this.up[1], -this.forward[1], 0,
      this.right[2], this.up[2], -this.forward[2], 0,
      0, 0, 0, 1
    );

    const viewMatrix = Mat4.create();
    Mat4.multiply(viewMatrix, rotateMatrix, translateMatrix);
    return viewMatrix;
  }

  getProjectionMatrix(): Mat4 {
    const projection = Mat4.create();
    
    const scaleY = 1.0 / Math.tan(this.heightAngle / 2.0);
    const scaleX = scaleY / this.aspect;
    
    
    const rangeInv = 1.0 / (this.far - this.near);
    
    const matrix = Mat4.fromValues(
      scaleX, 0.0,    0.0,    0.0,
      0.0,    scaleY, 0.0,    0.0,
      0.0,    0.0,    -(this.far * rangeInv), -1.0,
      0.0,    0.0,    -(this.near * this.far * rangeInv), 0.0
    );
    
    Mat4.copy(projection, matrix);
    return projection;
  }

  getWidthAngle(): number {
    return 2.0 * Math.atan(this.aspect * Math.tan(this.heightAngle / 2.0));
  }

  rotateCamera(angleX: number, angleY: number): void {
    const rotationX = Mat4.create();
    Mat4.fromRotation(rotationX, angleX, this.up);

    const rotationY = Mat4.create();
    Mat4.fromRotation(rotationY, angleY, this.right);

    const combinedRotation = Mat4.create();
    Mat4.multiply(combinedRotation, rotationY, rotationX);

    vec3.transformMat4(this.forward, this.forward, combinedRotation);
    
    this.updateVectors();
  }

  updateAspect(width: number, height: number): void {
    this.aspect = width / height;
  }

  getPosition(): Vec3 {
    return this.position;
  }

  getForward(): Vec3 {
    return this.forward;
  }

  getUp(): Vec3 {
    return this.up;
  }

  getRight(): Vec3 {
    return this.right;
  }
}
