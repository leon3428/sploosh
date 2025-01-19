struct ColoredParticle {
    position: vec3<f32>,
    color: vec4<f32>
}

@group(0) @binding(0) var<storage, read> position: array<vec3<f32>>; 
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read_write> display: array<ColoredParticle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= arrayLength(&position)) {
        return;
    }

    let cmin = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let cmax = vec4<f32>(1.0, 0.0, 0.0, 1.0);

    var particle: ColoredParticle;

    let alpha = clamp(density[gid] / 100.0, 0.0, 1.0);

    particle.position = position[gid] + OFFSET;
    particle.color = mix(cmin, cmax, alpha);

    display[gid] = particle;
}