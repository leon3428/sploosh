@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(2) var<storage, read> spatial_lookup_vals: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_lookup_index: array<u32>;
@group(0) @binding(4) var<storage, read_write> density: array<f32>;

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.z + cell.y * CELL_CNT.z + cell.x * CELL_CNT.y * CELL_CNT.z;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= arrayLength(&particle_positions)) {
        return;
    }

    let particle_cell = vec3<i32>(particle_positions[gid] / SMOOTHING_RADIUS);

    for (var i = -1; i <= 1; i += 1) {
        for (var j = -1; j <= 1; j += 1) {
            for (var k = -1; k <= 1; k += 1) {
                let neighbor_cell = vec3<i32>(
                    particle_cell.x + i, 
                    particle_cell.y + j, 
                    particle_cell.z + k
                );

                if (neighbor_cell.x < 0 || u32(neighbor_cell.x) >= CELL_CNT.x) {
                    continue;
                }

                if (neighbor_cell.y < 0 || u32(neighbor_cell.y) >= CELL_CNT.y) {
                    continue;
                }

                if (neighbor_cell.z < 0 || u32(neighbor_cell.z) >= CELL_CNT.z) {
                    continue;
                }

                let neighbor_cell_key = cell_key(vec3<u32>(neighbor_cell));
                for (var l = spatial_lookup_index[neighbor_cell_key]; spatial_lookup_keys[l] == neighbor_cell_key; l += 1u) {
                    let ind = spatial_lookup_vals[l];
                    if (distance(particle_positions[gid], particle_positions[ind]) < SMOOTHING_RADIUS) {
                        // hit
                        if (gid == 225) {
                            density[ind] = 0.95;
                        } 
                    }
                }
            }
        }
    }
}