// helper so we can add four vertices (quad) + six indices at once.
function addQuad(
  vertexPositions: number[],
  indicesList: number[],
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

export function generateBox() {
  const vertexPositions: number[] = [];
  const indicesList: number[] = [];

  // Define the 8 corners of a cube from -1 to +1
  const corners = [
    [-1, -1, -1], // 0: back-bottom-left
    [+1, -1, -1], // 1: back-bottom-right
    [+1, +1, -1], // 2: back-top-right
    [-1, +1, -1], // 3: back-top-left
    [-1, -1, +1], // 4: front-bottom-left
    [+1, -1, +1], // 5: front-bottom-right
    [+1, +1, +1], // 6: front-top-right
    [-1, +1, +1], // 7: front-top-left
  ];

  // Add all 8 vertices to the vertex buffer
  for (const corner of corners) {
    vertexPositions.push(...corner);
  }

  // Define the 6 faces of the cube
  // Each face is defined by 4 corner indices in counter-clockwise order when viewed from outside
  const faces = [
    [0, 1, 2, 3], // Back face   (z = -1)
    [4, 7, 6, 5], // Front face  (z = +1)
    [0, 4, 5, 1], // Bottom face (y = -1)
    [2, 6, 7, 3], // Top face    (y = +1)
    [0, 3, 7, 4], // Left face   (x = -1)
    [1, 5, 6, 2], // Right face  (x = +1)
  ];

  // Convert each face (quad) into two triangles
  for (const face of faces) {
    const [v0, v1, v2, v3] = face;

    // First triangle: v0 -> v1 -> v2
    indicesList.push(v0, v1, v2);

    // Second triangle: v0 -> v2 -> v3
    indicesList.push(v0, v2, v3);
  }

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
