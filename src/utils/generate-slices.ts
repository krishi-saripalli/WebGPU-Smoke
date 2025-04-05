export function generateSlices(gridSize: number) {
  /////////////////////////////////////////////////////////////////////////
  // Evenly space the points from -1 to +1 in (gridSize) steps
  /////////////////////////////////////////////////////////////////////////
  const positions: number[] = [];
  for (let idx = 0; idx < gridSize + 1; idx++) {
    positions.push(-1.0 + idx * (2.0 / gridSize));
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

    // push the 4 corner vertices of this quad
    vertexPositions.push(...p0, ...p1, ...p2, ...p3);

    // add two triangles (6 indices) in CCW order:
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

  for (let idx = 0; idx <= gridSize + 1; idx++) {
    const z = -1.0 + idx * (2.0 / gridSize);
    addQuad([-1, -1, z], [1, -1, z], [-1, 1, z], [1, 1, z]);
  }
  //TODO: Add the other axis-aligned slices to support camera movement

  // add a left and right boundary quad
  const left_x = -1;
  const right_x = 1;
  addQuad([left_x, -1, -1], [left_x, 1, -1], [left_x, -1, 1], [left_x, 1, 1]);
  addQuad([right_x, -1, -1], [right_x, 1, -1], [right_x, -1, 1], [right_x, 1, 1]);

  return { vertexPositions, indicesList };
}
