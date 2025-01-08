struct CameraUniform {
    view_projection: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) particle_pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normalized_coords: vec2<f32>,
};

const SIZE: f32 = 0.01;
const SIZE_SQ: f32 = SIZE * SIZE;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    vertex_input: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    var quad_vertices: array<vec3<f32>, 4> = array(
        vec3f(-1.0, -1.0, 0.0),
        vec3f( 1.0, -1.0, 0.0),
        vec3f(-1.0,  1.0, 0.0),
        vec3f( 1.0,  1.0, 0.0),
    );

    let camera_forward = normalize(camera.position - vertex_input.particle_pos);
    let up = vec3(0.0, 1.0, 0.0);
    let right = normalize(cross(up, camera_forward));
    let billboard_up = cross(camera_forward, right);

    let world_position = vertex_input.particle_pos +
        quad_vertices[in_vertex_index].x * right * SIZE +
        quad_vertices[in_vertex_index].y * billboard_up * SIZE;

    out.clip_position = camera.view_projection * vec4<f32>(world_position, 1.0);
    out.normalized_coords = quad_vertices[in_vertex_index].xy;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let dist_sq = dot(in.normalized_coords, in.normalized_coords);
    if (dist_sq > 1.0) {
        discard;
    }

    let normal = normalize(vec3f(in.normalized_coords, sqrt(max(0.0, 1.0 - dist_sq))));
    var light_direction: vec4f = transpose(camera.view_inv) * vec4f(0.0, 1.0, 0.0, 0.0);
    let brightness = max(dot(normal, light_direction.xyz), 0.0) + 0.05;

    var ret: FragmentOutput;
    ret.color = vec4<f32>(brightness, brightness, brightness, 1.0);

    return ret;
}