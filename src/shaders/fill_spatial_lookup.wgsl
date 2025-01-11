@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(2) var<storage, read> spatial_lookup_vals: array<u32>;

const PARTICLE_CNT: u32 = 1000;
const SMOOTHING_RADIUS: f32 = 0.04;
const CELL_CNT: vec3<u32> = vec3<u32>(10, 10, 10);

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.x + cell.y * CELL_CNT.x + cell.z * CELL_CNT.x * CELL_CNT.y; 
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id >= PARTICLE_CNT) {
        return;
    }

    let cell = vec3<u32>(particle_positions[global_id] / SMOOTHING_RADIUS);
    spatial_lookup_keys[global_id] = cell_key(cell);
    spatial_lookup_vals[global_id] = global_id;
}