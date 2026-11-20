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

  const edgeIndices = [
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
  ].flat();
  return { vertexPositions, indicesList, edgeIndices };
}
