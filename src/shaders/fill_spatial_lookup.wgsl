@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read_write> spatial_lookup_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> spatial_lookup_vals: array<u32>;

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.z + cell.y * CELL_CNT.z + cell.x * CELL_CNT.y * CELL_CNT.z;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= PARTICLE_CNT) {
        return;
    }

    let cell = vec3<u32>(particle_positions[gid] / SMOOTHING_RADIUS);
    spatial_lookup_keys[gid] = cell_key(cell);
    spatial_lookup_vals[gid] = gid;
}