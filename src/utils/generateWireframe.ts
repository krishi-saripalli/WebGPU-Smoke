export function generateWireframe(gridSize: number) {
  const lineWidth = 0.009;

  /////////////////////////////////////////////////////////////////////////
  // Evenly space the points from -1 to +1 in (gridSize) steps
  /////////////////////////////////////////////////////////////////////////
  const positions: number[] = [];
  for (let idx = 0; idx < gridSize; idx++) {
    positions.push(-1.0 + (2.0 * idx) / (gridSize - 1));
  }

  const vertexPositions: number[] = [];
  const indicesList: number[] = [];

  // helper so we can add four vertices (quad) + six indices at once.
  function addQuad(
    p0: [number, number, number],
    p1: [number, number, number],
    p2: [number, number, number],
    p3: [number, number, number]
  ) {
    // "startIndex" = how many vertices are in vertexPositions so far
    const startIndex = vertexPositions.length / 3;

    // Push the 4 corner vertices of this quad
    vertexPositions.push(...p0, ...p1, ...p2, ...p3);

    // Add two triangles (6 indices) in CCW order:
    //   0--1
    //   | /|
    //   2--3
    indicesList.push(
      startIndex + 0,
      startIndex + 2,
      startIndex + 1,
      startIndex + 2,
      startIndex + 3,
      startIndex + 1
    );
  }

  ///////////////////////////////////////////////////////////////////////////////
  // z-aligned lines: x,y vary, z goes from -1 to +1.
  // Only draw the quad if we are on the boundary plane (|x|=1 or |y|=1).
  ///////////////////////////////////////////////////////////////////////////////
  for (let x of positions) {
    for (let y of positions) {
      if (Math.abs(x) === 1 || Math.abs(y) === 1) {
        const zStart = -1.0;
        const zEnd = 1.0;

        // top-left, top-right, bottom-left, bottom-right
        const p0: [number, number, number] = [x + lineWidth, y + lineWidth, zEnd];
        const p1: [number, number, number] = [x + lineWidth, y + lineWidth, zStart];
        const p2: [number, number, number] = [x - lineWidth, y - lineWidth, zEnd];
        const p3: [number, number, number] = [x - lineWidth, y - lineWidth, zStart];

        addQuad(p0, p1, p2, p3);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  // x-aligned lines: y,z vary, x goes from -1 to +1.
  // Only draw if we're on the boundary plane (|y|=1 or |z|=1).
  ///////////////////////////////////////////////////////////////////////////////
  for (let y of positions) {
    for (let z of positions) {
      if (Math.abs(y) === 1 || Math.abs(z) === 1) {
        const xStart = -1.0;
        const xEnd = 1.0;

        const p0: [number, number, number] = [xStart, y + lineWidth, z];
        const p1: [number, number, number] = [xEnd, y + lineWidth, z];
        const p2: [number, number, number] = [xStart, y - lineWidth, z];
        const p3: [number, number, number] = [xEnd, y - lineWidth, z];

        addQuad(p0, p1, p2, p3);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  // y-aligned lines: x,z vary, y goes from -1 to +1.
  // Only draw if we're on the boundary plane (|x|=1 or |z|=1).
  ///////////////////////////////////////////////////////////////////////////////
  for (let z of positions) {
    for (let x of positions) {
      if (Math.abs(z) === 1 || Math.abs(x) === 1) {
        const yStart = -1.0;
        const yEnd = 1.0;

        const p0: [number, number, number] = [x + lineWidth, yStart, z];
        const p1: [number, number, number] = [x + lineWidth, yEnd, z];
        const p2: [number, number, number] = [x - lineWidth, yStart, z];
        const p3: [number, number, number] = [x - lineWidth, yEnd, z];

        addQuad(p0, p1, p2, p3);
      }
    }
  }

  return { vertexPositions, indicesList };
}
