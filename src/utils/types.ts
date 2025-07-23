export interface TexturePair {
  current: GPUTextureView;
  next: GPUTextureView;
}

export interface SimulationState {
  velocity: TexturePair;
  density: TexturePair;
  temperature: TexturePair;
  pressure: TexturePair;
  divergence: TexturePair;
  vorticity: TexturePair;
}
