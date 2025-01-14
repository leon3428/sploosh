@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;

const MASK = (1u << BLOCK_SIZE) - 1;
const BIN_CNT = (1u << BLOCK_SIZE);

var<push_constant> pass_ind: u32;
var<workgroup> local_counters: array<atomic<u32>, BIN_CNT>;

fn get_bin(x: u32) -> u32 {
    let ret = x & (MASK << (pass_ind * BLOCK_SIZE));
    return ret >> (pass_ind * BLOCK_SIZE);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    let lid = local_id.x;
    let gid = global_id.x;

    if ( lid < BIN_CNT ) {
        local_counters[lid] = 0u;
    } 

    workgroupBarrier();

    if ( gid < arrayLength(&keys) ) {
        let bin = get_bin(keys[gid]);
        atomicAdd(&local_counters[bin], 1u);
    }

    workgroupBarrier();

    if ( lid < BIN_CNT ) {
        atomicAdd(&counters[lid], atomicLoad(&local_counters[lid]));
    }
}