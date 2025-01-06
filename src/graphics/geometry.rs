use std::num::NonZero;

pub struct Geometry {
    vertex_buffer: wgpu::Buffer,
    vertex_cnt: usize,
}

impl Geometry {
    pub fn new<T>(device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[T]) -> Self {
        let len = vertices.len() * std::mem::size_of::<T>();
        let ptr = vertices.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry buffer"),
            size: len as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view = queue.write_buffer_with(&vertex_buffer, 0, NonZero::new(len as u64).unwrap());
        view.unwrap().copy_from_slice(data);

        Self {
            vertex_buffer,
            vertex_cnt: vertices.len(),
        }
    }

    pub fn get_vertices(&self) -> wgpu::BufferSlice {
        self.vertex_buffer.slice(..)
    }

    pub fn vertex_cnt(&self) -> usize {
        self.vertex_cnt
    }
}
