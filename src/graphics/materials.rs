use crate::RenderDevice;

pub trait Material {
    fn material_type(&self) -> MaterialType;
    fn bind_pipeline(&self, render_pass: &mut wgpu::RenderPass);
    fn draw_geometry_array(
        &self,
        vertex_buffer: &wgpu::Buffer,
        vertex_cnt: usize,
        render_pass: &mut wgpu::RenderPass,
    );
    fn draw_instanced(
        &self,
        vertex_cnt: usize,
        instance_buffer: &wgpu::Buffer,
        instance_cnt: usize,
        render_pass: &mut wgpu::RenderPass,
    );
}

#[derive(PartialEq, Eq, Hash)]
pub enum MaterialType {
    Line,
    Particle,
}

pub struct LineMaterial {
    pipeline: wgpu::RenderPipeline,
}

impl LineMaterial {
    pub fn new(
        render_device: &RenderDevice,
        model_view_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = render_device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Line Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/line_shader.wgsl").into(),
                ),
            });

        let render_pipeline_layout =
            render_device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Line render pipeline layout"),
                    bind_group_layouts: &[&model_view_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            render_device
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                            format: render_device.config.format,
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
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: render_device.depth_texture.format(),
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
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

    fn draw_geometry_array(
        &self,
        vertex_buffer: &wgpu::Buffer,
        vertex_cnt: usize,
        render_pass: &mut wgpu::RenderPass,
    ) {
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..vertex_cnt as u32, 0..1);
    }

    fn draw_instanced(
        &self,
        _vertex_cnt: usize,
        _instance_buffer: &wgpu::Buffer,
        _instance_cnt: usize,
        _render_pass: &mut wgpu::RenderPass,
    ) {
        panic!("Instanced rendering is not currently supported for the line pipeline");
    }
}

pub struct ParticleMaterial {
    pipeline: wgpu::RenderPipeline,
}

impl ParticleMaterial {
    pub fn new(
        render_device: &RenderDevice,
        model_view_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = render_device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/particle_shader.wgsl").into(),
                ),
            });

        let render_pipeline_layout =
            render_device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Particle render pipeline layout"),
                    bind_group_layouts: &[&model_view_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            render_device
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Particle render pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<nalgebra::Point3<f32>>()
                                as wgpu::BufferAddress,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                        }],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: render_device.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: render_device.depth_texture.format(),
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
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

impl Material for ParticleMaterial {
    fn material_type(&self) -> MaterialType {
        MaterialType::Particle
    }

    fn bind_pipeline(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
    }

    fn draw_geometry_array(
        &self,
        _vertex_buffer: &wgpu::Buffer,
        _vertex_cnt: usize,
        _render_pass: &mut wgpu::RenderPass,
    ) {
        panic!("Individual particle rendering is not supported for the particle pipeline");
    }

    fn draw_instanced(
        &self,
        vertex_cnt: usize,
        instance_buffer: &wgpu::Buffer,
        instance_cnt: usize,
        render_pass: &mut wgpu::RenderPass,
    ) {
        render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
        render_pass.draw(0..vertex_cnt as u32, 0..instance_cnt as u32);
    }
}
