export const shader = `

struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOut {
    var output: VertexOut;
    output.position = vec4f(input.position, 1.0);
    return output;
}

@fragment
fn fragmentMain(vertexOut: VertexOut) -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0); //red
}
`;
