import { Vec3, Mat4 } from 'gl-matrix';

export interface CameraData {
  position: Vec3;
  lookAt: Vec3;
  up: Vec3;
  heightAngle: number;
  near: number;
  far: number;
  aspect: number;
}

export class Camera {
  private position: Vec3;
  private lookAt: Vec3;
  private up: Vec3;
  private heightAngle: number;
  private near: number;
  private far: number;
  private aspect: number;

  constructor(cameraData: CameraData) {
    this.position = cameraData.position;
    this.lookAt = cameraData.lookAt;
    this.up = cameraData.up;
    this.heightAngle = cameraData.heightAngle;
    this.near = cameraData.near;
    this.far = cameraData.far;
    this.aspect = cameraData.aspect;
  }

  getViewMatrix(): Mat4 {
    const view = Mat4.create();
    Mat4.lookAt(view, this.position, this.lookAt, this.up);
    return view;
  }

  getProjectionMatrix(): Mat4 {
    const projection = Mat4.create();
    
    // Calculate basic scale factors
    const scaleY = 1.0 / Math.tan(this.heightAngle / 2.0);
    const scaleX = scaleY / this.aspect;
    
    // Calculate depth range factors for WebGPU's [0,1] range
    // but maintaining right-handed convention (-Z forward)
    const rangeInv = 1.0 / (this.far - this.near);
    
    // Build right-handed perspective matrix for WebGPU depth range
    const matrix = Mat4.fromValues(
        scaleX, 0.0,    0.0,    0.0,
        0.0,    scaleY, 0.0,    0.0,
        0.0,    0.0,    -(this.far * rangeInv), -1.0,  // Note the negatives here
        0.0,    0.0,    -(this.near * this.far * rangeInv), 0.0
    );
    
    Mat4.copy(projection, matrix);
    return projection;
}
  
  

  getWidthAngle(): number {
    return 2.0 * Math.atan(this.aspect * Math.tan(this.heightAngle / 2.0));
  }

  updateCamera(newLook: Vec3, newUp: Vec3, newPos: Vec3): void {
    this.lookAt = newLook;
    this.up = newUp;
    this.position = newPos;
  }

  updateAspect(width: number, height: number): void {
    this.aspect = width / height;
  }
}
