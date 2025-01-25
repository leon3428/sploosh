@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(2) var<storage, read> spatial_lookup_vals: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_lookup_index: array<u32>;
@group(0) @binding(4) var<storage, read_write> density: array<f32>;


const PI = 3.14159;
const HSQ = SMOOTHING_RADIUS * SMOOTHING_RADIUS;
const POLY6 = 315.0 / (64.0 * PI * pow(SMOOTHING_RADIUS, 9.0));

fn cell_key(cell: vec3<u32>) -> u32 {
    return cell.z + cell.y * CELL_CNT.z + cell.x * CELL_CNT.y * CELL_CNT.z;
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

    let particle_pos = particle_positions[gid];
    let particle_cell = vec3<i32>(particle_pos / SMOOTHING_RADIUS);
    var d: f32 = 0.0;

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

            let dist = distance(particle_pos, particle_positions[ind]);
            let dist_sq = dist * dist;
            let is_within_radius = dist < SMOOTHING_RADIUS;

            let diff = select(0.0, HSQ - dist_sq, is_within_radius);
            d += MASS * POLY6 * diff * diff * diff;
        }    
    }

    density[gid] = d;
}