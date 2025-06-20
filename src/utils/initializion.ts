const _initializeDensity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): number => {
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  if (
    internalX >= 0 &&
    internalX < gridSize &&
    internalY >= 0 &&
    internalY < gridSize &&
    internalZ >= 0 &&
    internalZ < gridSize
  ) {
    const centerX = gridSize / 2;
    const centerZ = gridSize / 2;
    const radius = 6;
    const height = 3;

    if (
      internalY <= height &&
      internalX >= centerX - radius &&
      internalX <= centerX + radius &&
      internalZ >= centerZ - radius &&
      internalZ <= centerZ + radius
    ) {
      return 1.0;
    }
    return 0.0;
  }

  return 0.0;
};

const _initializeTemperature = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): number => {
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  if (
    internalX >= 0 &&
    internalX < gridSize &&
    internalY >= 0 &&
    internalY < gridSize &&
    internalZ >= 0 &&
    internalZ < gridSize
  ) {
    const centerX = gridSize / 2;
    const centerZ = gridSize / 2;
    const radius = 10;
    const height = gridSize / 2;

    if (
      internalY <= height &&
      internalX >= centerX - radius &&
      internalX <= centerX + radius &&
      internalZ >= centerZ - radius &&
      internalZ <= centerZ + radius
    ) {
      return 1000.0; // hot!!;
    }
    return 0.0; //ambient temp
  }
  return 0.0;
};

/**
 * Interface for velocity components
 */
export interface VelocityComponents {
  x: number;
  y: number;
  z: number;
}

const _initializeVelocity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): VelocityComponents => {
  return { x: 0, y: 0, z: 0 };
};

const _initializePressure = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): number => {
  return 0.0;
};

/**
 * Interface for the simulation data arrays
 */
export interface SimulationData {
  densityData: Float32Array;
  temperatureData: Float32Array;
  velocityData: Float32Array;
  pressureData: Float32Array;
}

/**
 * Initialize all data for the simulation
 */
export const initializeSimulationData = (
  totalGridSize: number,
  halosSize: number,
  gridSize: number
): SimulationData => {
  // Create arrays for all simulation fields
  const initDensityData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);
  const initTemperatureData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);
  const initVelocityData = new Float32Array(totalGridSize * totalGridSize * totalGridSize * 4); //vec4
  const initPressureData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);

  for (let z = 0; z < totalGridSize; z++) {
    for (let y = 0; y < totalGridSize; y++) {
      for (let x = 0; x < totalGridSize; x++) {
        const i = x + y * totalGridSize + z * totalGridSize * totalGridSize;
        const velocityIndex = i * 4;

        initDensityData[i] = _initializeDensity(x, y, z, halosSize, gridSize);
        initTemperatureData[i] = _initializeTemperature(x, y, z, halosSize, gridSize);

        const velocity = _initializeVelocity(x, y, z, halosSize, gridSize);
        initVelocityData[velocityIndex] = velocity.x;
        initVelocityData[velocityIndex + 1] = velocity.y;
        initVelocityData[velocityIndex + 2] = velocity.z;
        initVelocityData[velocityIndex + 3] = 0.0;

        initPressureData[i] = _initializePressure(x, y, z, halosSize, gridSize);
      }
    }
  }

  return {
    densityData: initDensityData,
    temperatureData: initTemperatureData,
    velocityData: initVelocityData,
    pressureData: initPressureData,
  };
};
