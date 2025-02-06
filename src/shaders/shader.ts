export const shader = `

struct Point {
    @location(0) position: vec2f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
}
@vertex
fn vertexMain(point: Point, @builtin(vertex_index) quadIndex: u32) -> VertexOut {
 // create a quad for each point
  let corners = array(
        vec2f(-1.0, -1.0),  // bottom left 
        vec2f( 1.0, -1.0),  // bottom right
        vec2f(-1.0,  1.0),  // top left
        vec2f(-1.0,  1.0),  // top left again
        vec2f( 1.0, -1.0),  // bottom right again
        vec2f( 1.0,  1.0)   // top right
    );

    let pointSize = 0.01;  // size of quad in world space
    let quadOffset = corners[quadIndex] * pointSize;  // get corner position based on vertex index
    
    var output: VertexOut;
    output.position = vec4f(point.position + quadOffset, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(vertexOut: VertexOut) -> @location(0) vec4f {
  return vec4f(1.0, 0.0, 0.0, 1.0); //red
}
`;
