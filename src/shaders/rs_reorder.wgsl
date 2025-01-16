@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> keys_output: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

const BLOCK_SIZE = 8u;
const MASK = (1u << BLOCK_SIZE) - 1u;

var<push_constant> pass_ind: u32;

fn get_bin(x: u32) -> u32 {
    let ret = x & (MASK << (pass_ind * BLOCK_SIZE));
    return ret >> (pass_ind * BLOCK_SIZE);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    let lid = local_id.x;
    let gid = global_id.x;

    if (gid >= arrayLength(&keys)) {
        return;
    }

    let key = keys[gid];
    let bin = get_bin(key);
    let i = atomicAdd(&counters[bin], 1u);

    keys_output[i] = key;
}