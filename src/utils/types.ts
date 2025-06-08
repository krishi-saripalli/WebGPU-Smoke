export interface TexturePair {
  current: GPUTexture;
  next: GPUTexture;
}

export interface SimulationState {
  velocity: TexturePair;
  density: TexturePair;
  temperature: TexturePair;
  pressure: TexturePair;
  divergence: TexturePair;
  vorticity: TexturePair;
}
