@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(2) var<storage, read> spatial_lookup_vals: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_lookup_index: array<u32>;
@group(0) @binding(4) var<storage, read> particle_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> particle_force: array<vec3<f32>>;

const PI = 3.14159;
const SPIKY_GRAD = 15.0 / (PI * pow(SMOOTHING_RADIUS, 6.0));

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.z + cell.y * CELL_CNT.z + cell.x * CELL_CNT.y * CELL_CNT.z;
}

fn calculate_pressure(density: f32) -> f32 {
    return GAS_CONST * (density - REST_DENSITY);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= arrayLength(&particle_positions)) {
        return;
    }

    let particle_pos = particle_positions[gid];
    let particle_den = particle_density[gid];
    let particle_pressure = calculate_pressure(particle_den);
    let particle_cell = vec3<i32>(particle_pos / SMOOTHING_RADIUS);
    var force: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

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
                    if (ind == gid) {
                        continue;
                    }

                    let neighbor_pos = particle_positions[ind];

                    let dir = particle_pos - neighbor_pos;
                    let dist = length(dir);

                    if (dist < SMOOTHING_RADIUS) {
                        // hit
                        let neighbor_density = particle_density[ind];
                        let neighbor_pressure = calculate_pressure(neighbor_density);

                        let diff = (SMOOTHING_RADIUS - dist);
                        force += -normalize(dir) * MASS * (particle_pressure + neighbor_pressure) / (2.0 * neighbor_density + 1.0e-6) * SPIKY_GRAD * diff * diff * diff;
                    }
                }
            }
        }
    }

    particle_force[gid] = force;
}