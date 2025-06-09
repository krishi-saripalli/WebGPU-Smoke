export function generateSlices(gridSize: number) {
  /////////////////////////////////////////////////////////////////////////
  // Evenly space the points from -1 to +1 in (gridSize) steps
  /////////////////////////////////////////////////////////////////////////

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
  //TODO: Add the dynamic axis-aligned slices to support camera movement

  return { vertexPositions, indicesList };
}

export function generateWireframe() {
  const vertexPositions: number[] = [];
  const indicesList: number[] = [];

  // 8 vertices of a cube
  const vertices = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1], // back face
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1], // front face
  ];

  for (const vertex of vertices) {
    vertexPositions.push(...vertex);
  }

  // 12 edges of the cube
  const edges = [
    // Back face edges
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    // Front face edges
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    // Connecting edges
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];

  for (const [start, end] of edges) {
    indicesList.push(start, end);
  }

  return { vertexPositions, indicesList };
}
