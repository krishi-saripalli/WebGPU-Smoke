// Create array buffers for initial data
const _initializeDensity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): number => {
  // Convert to internal grid coordinates (excluding halos)
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Check if we're in the internal grid area
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
    const radius = gridSize;
    const bottomHeight = gridSize;

    if (
      internalY < bottomHeight &&
      Math.sqrt(
        (internalX - centerX) * (internalX - centerX) +
          (internalZ - centerZ) * (internalZ - centerZ)
      ) < radius
    ) {
      return Math.exp(-(Math.pow(internalX - centerX, 2) + Math.pow(internalZ - centerZ, 2)) / 5);
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
  // Convert to internal grid coordinates (excluding halos)
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Check if we're in the internal grid area
  if (
    internalX >= 0 &&
    internalX < gridSize &&
    internalY >= 0 &&
    internalY < gridSize &&
    internalZ >= 0 &&
    internalZ < gridSize
  ) {
    // Initialize temperature: Heat source coinciding with density source
    const centerX = gridSize / 2;
    const centerZ = gridSize / 2;
    const radius = gridSize / 10;
    const bottomHeight = gridSize / 10;

    if (
      internalY < bottomHeight &&
      Math.sqrt(
        (internalX - centerX) * (internalX - centerX) +
          (internalZ - centerZ) * (internalZ - centerZ)
      ) < radius
    ) {
      return 10.0; // Heat source (this value affects buoyancy)
    }
    return 0.0; // Ambient temperature
  }

  // For halo cells, return 0 as default
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

/**
 * Initialize velocity components for a given cell in the grid
 */
const _initializeVelocity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): VelocityComponents => {
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Default values
  const velocity: VelocityComponents = { x: 0.0, y: 0.0, z: 0.0 };

  // Check if we're in the internal grid area
  if (
    internalX >= 0 &&
    internalX < gridSize &&
    internalY >= 0 &&
    internalY < gridSize &&
    internalZ >= 0 &&
    internalZ < gridSize
  ) {
    // Initialize upward velocity at source points
    const centerX = gridSize / 2;
    const centerZ = gridSize / 2;
    const radius = gridSize / 10;
    const bottomHeight = gridSize / 10;

    if (
      internalY < bottomHeight &&
      Math.sqrt(
        (internalX - centerX) * (internalX - centerX) +
          (internalZ - centerZ) * (internalZ - centerZ)
      ) < radius
    ) {
      // velocity.y = 0.5; // Initial upward velocity at source
      velocity.y = 1.5; // Increased upward velocity for testing

      // Add a slight swirl with X and Z components for more interesting motion
      // This creates a gentle vortex-like initial condition
      const dx = internalX - centerX;
      const dz = internalZ - centerZ;
      const distance = Math.sqrt(dx * dx + dz * dz);
      if (distance > 0) {
        // Tangential velocity components to create swirl - DISABLED FOR TESTING
        // velocity.x = 0.005 * (-dz / distance);
        // velocity.z = 0.005 * (dx / distance);
      }
    }
  }

  return velocity;
};

/**
 * Initialize pressure for a given cell in the grid
 */
const _initializePressure = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  gridSize: number
): number => {
  // Pressure is initialized to zero everywhere
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
