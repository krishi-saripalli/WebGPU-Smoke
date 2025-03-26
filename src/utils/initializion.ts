// Create array buffers for initial data
export const initializeDensity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  internalGridSize: number
): number => {
  // Convert to internal grid coordinates (excluding halos)
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Check if we're in the internal grid area
  if (
    internalX >= 0 &&
    internalX < internalGridSize &&
    internalY >= 0 &&
    internalY < internalGridSize &&
    internalZ >= 0 &&
    internalZ < internalGridSize
  ) {
    // Initial density: Smoke source at the bottom center area
    const centerX = internalGridSize / 2;
    const centerZ = internalGridSize / 2;
    const radius = internalGridSize / 10;
    const bottomHeight = internalGridSize / 10;

    if (
      internalY < bottomHeight &&
      Math.sqrt(
        (internalX - centerX) * (internalX - centerX) +
          (internalZ - centerZ) * (internalZ - centerZ)
      ) < radius
    ) {
      return 1.0; // Density source
    }
    return 0.0;
  }

  // For halo cells, return 0 as default
  return 0.0;
};

export const initializeTemperature = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  internalGridSize: number
): number => {
  // Convert to internal grid coordinates (excluding halos)
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Check if we're in the internal grid area
  if (
    internalX >= 0 &&
    internalX < internalGridSize &&
    internalY >= 0 &&
    internalY < internalGridSize &&
    internalZ >= 0 &&
    internalZ < internalGridSize
  ) {
    // Initialize temperature: Heat source coinciding with density source
    const centerX = internalGridSize / 2;
    const centerZ = internalGridSize / 2;
    const radius = internalGridSize / 10;
    const bottomHeight = internalGridSize / 10;

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
export const initializeVelocity = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  internalGridSize: number
): VelocityComponents => {
  // Convert to internal grid coordinates (excluding halos)
  const internalX = x - halosSize;
  const internalY = y - halosSize;
  const internalZ = z - halosSize;

  // Default values
  const velocity: VelocityComponents = { x: 0.0, y: 0.0, z: 0.0 };

  // Check if we're in the internal grid area
  if (
    internalX >= 0 &&
    internalX < internalGridSize &&
    internalY >= 0 &&
    internalY < internalGridSize &&
    internalZ >= 0 &&
    internalZ < internalGridSize
  ) {
    // Initialize upward velocity at source points
    const centerX = internalGridSize / 2;
    const centerZ = internalGridSize / 2;
    const radius = internalGridSize / 10;
    const bottomHeight = internalGridSize / 10;

    if (
      internalY < bottomHeight &&
      Math.sqrt(
        (internalX - centerX) * (internalX - centerX) +
          (internalZ - centerZ) * (internalZ - centerZ)
      ) < radius
    ) {
      velocity.y = 0.5; // Initial upward velocity at source

      // Add a slight swirl with X and Z components for more interesting motion
      // This creates a gentle vortex-like initial condition
      const dx = internalX - centerX;
      const dz = internalZ - centerZ;
      const distance = Math.sqrt(dx * dx + dz * dz);
      if (distance > 0) {
        // Tangential velocity components to create swirl
        velocity.x = 0.05 * (-dz / distance);
        velocity.z = 0.05 * (dx / distance);
      }
    }
  }

  return velocity;
};

/**
 * Initialize pressure for a given cell in the grid
 */
export const initializePressure = (
  x: number,
  y: number,
  z: number,
  halosSize: number,
  internalGridSize: number
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
  velocityData: Float32Array; // Consolidated RGBA format (xyz in rgb, a is unused)
  pressureData: Float32Array;
}

/**
 * Initialize all simulation data for the fluid simulation
 */
export const initializeSimulationData = (
  totalGridSize: number,
  halosSize: number,
  internalGridSize: number
): SimulationData => {
  // Create arrays for all simulation fields
  const initDensityData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);
  const initTemperatureData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);
  // For velocity, we need 4 components per cell (RGBA) where we'll use RGB for the XYZ velocity components
  const initVelocityData = new Float32Array(totalGridSize * totalGridSize * totalGridSize * 4);
  const initPressureData = new Float32Array(totalGridSize * totalGridSize * totalGridSize);

  // Fill all data arrays
  for (let z = 0; z < totalGridSize; z++) {
    for (let y = 0; y < totalGridSize; y++) {
      for (let x = 0; x < totalGridSize; x++) {
        const i = x + y * totalGridSize + z * totalGridSize * totalGridSize;
        const velocityIndex = i * 4; // Each cell has 4 components (RGBA)

        // Initialize density and temperature
        initDensityData[i] = initializeDensity(x, y, z, halosSize, internalGridSize);
        initTemperatureData[i] = initializeTemperature(x, y, z, halosSize, internalGridSize);

        // Initialize velocity components
        const velocity = initializeVelocity(x, y, z, halosSize, internalGridSize);
        initVelocityData[velocityIndex] = velocity.x; // R channel = X component
        initVelocityData[velocityIndex + 1] = velocity.y; // G channel = Y component
        initVelocityData[velocityIndex + 2] = velocity.z; // B channel = Z component
        initVelocityData[velocityIndex + 3] = 0.0; // A channel = unused

        // Initialize pressure
        initPressureData[i] = initializePressure(x, y, z, halosSize, internalGridSize);
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
