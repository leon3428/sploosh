@group(0) @binding(0) var<storage, read_write> counters: array<u32>;

const HALF_ARRAY_LENGTH = 128u;
const ARRAY_LENGTH = 256u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lid = local_id.x;
    var offset = 1u;

    for (var d = HALF_ARRAY_LENGTH; d > 0; d >>= 1u) {
        
        if (lid < d) {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;

            counters[bi] += counters[ai];
        }

        offset *= 2u;
        workgroupBarrier();
    }

    if (lid == 0) {
        counters[ARRAY_LENGTH - 1u] = 0u;
    } 

    for (var d = 1u; d < ARRAY_LENGTH; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (lid < d) {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;

            let t = counters[ai];
            counters[ai] = counters[bi];
            counters[bi] += t;
        }
    }
    workgroupBarrier();
}