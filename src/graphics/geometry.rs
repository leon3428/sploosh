use nalgebra::Vector3;
use wgpu::util::DeviceExt as _;

pub trait Geometry {
    fn get_vertices(&self) -> wgpu::BufferSlice;
    fn buffer_desc() -> wgpu::VertexBufferLayout<'static>;
    fn vertex_cnt(&self) -> u32;
}

pub struct Line {
    vertices: Vec<Vector3<f32>>,
    buffer: wgpu::Buffer
}

impl Line {
    pub fn new(device: &wgpu::Device, vertices: Vec<Vector3<f32>>) -> Self {
        let len = vertices.len() * std::mem::size_of::<Vector3<f32>>();
        let ptr = vertices.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line buffer"),
            contents: data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self { vertices, buffer }
    }
}

impl Geometry for Line {
    fn get_vertices(&self) -> wgpu::BufferSlice {
        self.buffer.slice(..)
    }
    
    fn buffer_desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<nalgebra::Vector3<f32>>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
    
    fn vertex_cnt(&self) -> u32 {
        self.vertices.len() as u32
    }
}