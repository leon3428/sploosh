@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> particle_velocities: array<vec3<f32>>; 
@group(0) @binding(2) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_lookup_vals: array<u32>;
@group(0) @binding(4) var<storage, read> spatial_lookup_index: array<u32>;
@group(0) @binding(5) var<storage, read> particle_density: array<f32>;
@group(0) @binding(6) var<storage, read_write> particle_force: array<vec3<f32>>;

const PI = 3.14159;
const SPIKY_GRAD = 15.0 / (PI * pow(SMOOTHING_RADIUS, 6.0));
const VISC_LAP = 45.0 / (PI * pow(SMOOTHING_RADIUS, 6.0));

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.z + cell.y * CELL_CNT.z + cell.x * CELL_CNT.y * CELL_CNT.z;
}

fn calculate_pressure(density: f32) -> f32 {
    return GAS_CONST * (density - REST_DENSITY);
}

const dx = array(-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1);
const dy = array(-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1);
const dz = array(1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x + GHOST_PARTICLE_CNT;

    if (gid >= arrayLength(&particle_positions)) {
        return;
    }

    let particle_velocity = particle_velocities[gid];
    let particle_pos = particle_positions[gid];
    let particle_den = particle_density[gid];
    let particle_pressure = calculate_pressure(particle_den);
    let particle_cell = vec3<i32>(particle_pos / SMOOTHING_RADIUS);
    var force: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0; i < 27; i += 1) {
        let neighbor_cell = vec3<i32>(
            particle_cell.x + dx[i], 
            particle_cell.y + dy[i], 
            particle_cell.z + dz[i]
        );

        let is_valid_cell = all(neighbor_cell >= vec3<i32>(0)) && 
                            all(vec3<u32>(neighbor_cell) < CELL_CNT);

        if (!is_valid_cell) {
            continue;
        }

        let neighbor_cell_key = cell_key(vec3<u32>(neighbor_cell));
        for (var l = spatial_lookup_index[neighbor_cell_key]; spatial_lookup_keys[l] == neighbor_cell_key; l += 1u) {
            let ind = spatial_lookup_vals[l];
            if (ind == gid) {
                continue;
            }

            let neighbor_pos = particle_positions[ind];

            var dir: vec3<f32> = particle_pos - neighbor_pos;
            let dist = length(dir);

            if (dist < SMOOTHING_RADIUS) {
                // hit
                if (dist == 0) {
                    dir = vec3<f32>(1.0, 0.0, 0.0);
                }

                let neighbor_density = particle_density[ind];
                let neighbor_pressure = calculate_pressure(neighbor_density);
                let neighbor_velocity = particle_velocities[ind];

                let diff = (SMOOTHING_RADIUS - dist);
                let norm_dir = normalize(dir);
                force += norm_dir * MASS * (particle_pressure + neighbor_pressure)  * SPIKY_GRAD * diff * diff * diff / (2.0 * neighbor_density);
                force += VISCOSITY * MASS * (neighbor_velocity - particle_velocity) * VISC_LAP * diff / neighbor_density;
            }
        }

    }

    particle_force[gid] = force;
}