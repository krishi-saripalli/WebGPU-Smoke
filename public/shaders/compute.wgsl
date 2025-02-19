@binding(0) @group(0) var<storage, read_write> density: array<f32>;
    
@compute @workgroup_size(4,4,4)
fn computeMain() {
    density[0] = 1.0;
}