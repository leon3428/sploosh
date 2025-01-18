@group(0) @binding(0) var<storage, read> spatial_lookup_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> spatial_lookup_index: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= PARTICLE_CNT) {
        return;
    }

    let key = spatial_lookup_keys[gid];
    if (gid == 0 || key != spatial_lookup_keys[gid - 1]) {
        spatial_lookup_index[key] = gid;
    }
}