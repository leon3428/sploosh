use super::geometry::Geometry;

pub trait Material {
    fn material_type(&self) -> MaterialType;
    fn bind_pipeline(&self, render_pass: &mut wgpu::RenderPass);
    fn draw(&self, geometry: &Geometry, render_pass: &mut wgpu::RenderPass);
}

#[derive(PartialEq, Eq, Hash)]
pub enum MaterialType {
    Line,
}

pub struct LineMaterial {
    pipeline: wgpu::RenderPipeline,
}

impl LineMaterial {
    pub fn new(
        device: &wgpu::Device,
        model_view_bind_group_layout: &wgpu::BindGroupLayout,
        render_surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/line_shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Line render pipeline layout"),
                bind_group_layouts: &[&model_view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<nalgebra::Vector3<f32>>()
                        as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self { pipeline }
    }
}

impl Material for LineMaterial {
    fn material_type(&self) -> MaterialType {
        MaterialType::Line
    }

    fn bind_pipeline(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
    }

    fn draw(&self, geometry: &Geometry, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_vertex_buffer(0, geometry.get_vertices());
        render_pass.draw(0..geometry.vertex_cnt() as u32, 0..1);
    }
}
