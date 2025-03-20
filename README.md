# Primary Quantities

## Velocity (u) - Vector field
- 3D vector at each voxel face (staggered grid)
- Components: u, v, w for x, y, z directions
- Updated each frame through force addition, advection, and projection

## Pressure (p) - Scalar field
- Single scalar value at each voxel center
- Solved for at each time step (not stored between frames)
- Used only to make the velocity field incompressible

## Density (ρ) - Scalar field
- Single scalar value at each voxel center
- Represents the amount of smoke at each point
- Advected by the velocity field

## Temperature (T) - Scalar field
- Single scalar value at each voxel center
- Affects buoyancy (hot smoke rises)
- Advected by the velocity field

# Derived Quantities

## Vorticity (ω) - Vector field
- 3D vector at each voxel center
- Computed from the velocity field (curl of velocity)
- Used for vorticity confinement force

## Vorticity Confinement Force - Vector field
- 3D vector at each voxel center
- Computed from vorticity
- Added to velocity to counter numerical dissipation

## Buoyancy Force - Vector field
- 3D vector at each voxel center
- Computed from temperature and density
- Added to velocity to make hot smoke rise

# For Rendering (Hardware Renderer)

## Light Intensity - Scalar field
- Single scalar value at each voxel center
- Computed using line drawing algorithm through the density field
- Represents direct lighting at each voxel

## Voxel Transparency - Scalar field
- Computed from density
- Used during rendering phase
