@group(0) @binding(0) var<storage, read_write> particle_positions: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read_write> particle_velocity: array<vec3<f32>>; 

var<push_constant> dt: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= arrayLength(&particle_positions)) {
        return;
    }

    var position: vec3<f32> = particle_positions[gid] + particle_velocity[gid] * dt;
    var velocity: vec3<f32> = particle_velocity[gid] + G * dt ;

    if position.x - SMOOTHING_RADIUS < 0.0 {
        velocity.x *= DAMPING;
        position.x = 0.0 + SMOOTHING_RADIUS;
    }

    if position.x + SMOOTHING_RADIUS > BBOX.x {
        velocity.x *= DAMPING;
        position.x = BBOX.x - SMOOTHING_RADIUS;
    }

    if position.y - SMOOTHING_RADIUS < 0.0 {
        velocity.y *= DAMPING;
        position.y = 0.0 + SMOOTHING_RADIUS;
    }

    if position.y + SMOOTHING_RADIUS > BBOX.y {
        velocity.y *= DAMPING;
        position.y = BBOX.y - SMOOTHING_RADIUS;
    }

    if position.z - SMOOTHING_RADIUS < 0.0 {
        velocity.z *= DAMPING;
        position.z = 0.0 + SMOOTHING_RADIUS;
    }

    if position.z + SMOOTHING_RADIUS > BBOX.z {
        velocity.z *= DAMPING;
        position.z = BBOX.z - SMOOTHING_RADIUS;
    }

    particle_positions[gid] = position;
    particle_velocity[gid] = velocity;
}