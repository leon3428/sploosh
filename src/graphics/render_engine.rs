use std::{error::Error, sync::Arc};

use nalgebra::{Matrix4, Perspective3, Vector3};
use winit::window::Window;

use super::{
    camera::Camera, geometry::{self, Geometry}, line_pipeline
};

pub struct RenderEngine {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    camera: Camera,

    mvp_buffer: wgpu::Buffer,
    mvp_bind_group: wgpu::BindGroup,

    line_pipeline: wgpu::RenderPipeline,
    line_geometry: geometry::Line,
}

impl RenderEngine {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to crate an adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let mvp_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("MVP buffer"),
            size: std::mem::size_of::<Matrix4<f32>>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (line_pipeline, camera_bind_group_layout) = line_pipeline::create_pipeline(&device, &config);

        let mvp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mvp_buffer.as_entire_binding(),
                }
            ],
        });

        let lines = vec![
            Vector3::new(0.1, 0.1, 0.0),
            Vector3::new(-0.1, 0.1, 0.0),
            Vector3::new(-0.1, 0.1, 0.0),
            Vector3::new(-0.1, -0.1, 0.0),
            Vector3::new(-0.1, -0.1, 0.0),
            Vector3::new(0.1, -0.1, 0.0),
            Vector3::new(0.1, -0.1, 0.0),
            Vector3::new(0.1, 0.1, 0.0),
        ];
        let line_geometry = geometry::Line::new(&device, lines);

        let camera = Camera::new();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            line_pipeline,
            line_geometry,
            camera,
            mvp_buffer,
            mvp_bind_group
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn update(&self) {}

    pub fn update_camera(&mut self, right_button_pressed: bool, mouse_move_delta: (f32, f32), mouse_wheel_delta: f32) {
        self.camera.update(right_button_pressed, mouse_move_delta, mouse_wheel_delta);
    }

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mvp = self.get_projection_matrix() * self.camera.get_view_matrix();
        let len = std::mem::size_of::<Matrix4<f32>>();
        let ptr = mvp.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        self.queue.write_buffer(&self.mvp_buffer, 0, data);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.mvp_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_geometry.get_vertices());
            render_pass.draw(0..self.line_geometry.vertex_cnt(), 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn get_projection_matrix(&self) -> Matrix4<f32> {
        let aspect = self.config.width as f32 / self.config.height as f32;
        Perspective3::new(aspect, self.camera.fov, self.camera.z_near, self.camera.z_far).to_homogeneous()
    }
}
