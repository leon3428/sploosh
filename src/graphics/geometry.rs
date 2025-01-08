use std::rc::Rc;

#[derive(Clone)]
pub enum Geometry {
    Array {
        vertex_buffer: Rc<wgpu::Buffer>,
        vertex_cnt: usize,
    },
    Instanced {
        vertex_cnt: usize,
        instance_buffer: Rc<wgpu::Buffer>,
        instance_cnt: usize
    }
}
