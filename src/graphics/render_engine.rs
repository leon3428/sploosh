use std::{collections::HashMap, error::Error, num::NonZero, rc::Rc, sync::Arc};

use nalgebra::{Matrix4, Perspective3, Point3};
use winit::window::Window;

use super::{
    camera::Camera,
    geometry::Geometry,
    materials::{LineMaterial, Material, MaterialType, ParticleMaterial},
    texture::Texture,
};

pub struct RenderRequest {
    pub material_type: MaterialType,
    pub geometry: Geometry,
}

#[repr(C)]
struct CameraUniform {
    pub view_proj: Matrix4<f32>,
    pub view_inv: Matrix4<f32>,
    pub position: Point3<f32>,
    pub _padding: f32,
}

pub struct RenderEngine {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    depth_texture: Texture,
    camera: Camera,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    materials: HashMap<MaterialType, Box<dyn Material>>,
    render_queue: Vec<RenderRequest>,
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
        let depth_texture = Texture::depth_texture(&device, &config);

        // Model view buffer initialization

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Material initialization

        let mut materials: HashMap<MaterialType, Box<dyn Material>> = HashMap::new();
        materials.insert(
            MaterialType::Line,
            Box::new(LineMaterial::new(
                &device,
                &camera_bind_group_layout,
                config.format,
            )),
        );
        materials.insert(
            MaterialType::Particle,
            Box::new(ParticleMaterial::new(
                &device,
                &camera_bind_group_layout,
                config.format,
            )),
        );

        // Camera initialization

        let camera = Camera::new();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            camera,
            camera_buffer,
            camera_bind_group,
            materials,
            render_queue: Vec::new(),
        })
    }

    pub fn create_geometry_array<T>(&self, vertices: &[T]) -> Geometry {
        Geometry::Array {
            vertex_buffer: self.create_buffer(vertices),
            vertex_cnt: vertices.len(),
        }
    }

    pub fn create_buffer<T>(&self, data: &[T]) -> Rc<wgpu::Buffer> {
        let len = data.len() * std::mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry buffer"),
            size: len as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view = self
            .queue
            .write_buffer_with(&buffer, 0, NonZero::new(len as u64).unwrap());
        view.unwrap().copy_from_slice(data);

        Rc::new(buffer)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Texture::depth_texture(&self.device, &self.config);
        }
    }

    pub fn update(&self) {}

    pub fn update_camera(
        &mut self,
        left_button_pressed: bool,
        mouse_move_delta: (f32, f32),
        mouse_wheel_delta: f32,
    ) {
        self.camera
            .update(left_button_pressed, mouse_move_delta, mouse_wheel_delta);
    }

    pub fn submit_render_request(&mut self, render_request: RenderRequest) {
        self.render_queue.push(render_request);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let view_mat = self.camera.get_view_matrix();

        let camera_data = CameraUniform {
            view_proj: self.get_projection_matrix() * view_mat,
            view_inv: view_mat.try_inverse().unwrap(),
            position: self.camera.position,
            _padding: 0.0,
        };

        let len = std::mem::size_of::<CameraUniform>();
        let ptr = camera_data.view_proj.as_ptr() as *const u8;
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        self.queue.write_buffer(&self.camera_buffer, 0, data);

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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_texture.view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for request in &self.render_queue {
                let material = self.materials.get(&request.material_type).unwrap();
                material.bind_pipeline(&mut render_pass);

                match &request.geometry {
                    Geometry::Array {
                        vertex_buffer,
                        vertex_cnt,
                    } => {
                        material.draw_geometry_array(&vertex_buffer, *vertex_cnt, &mut render_pass)
                    }
                    Geometry::Instanced {
                        vertex_cnt,
                        instance_buffer,
                        instance_cnt,
                    } => {
                        material.draw_instanced(
                            *vertex_cnt,
                            &instance_buffer,
                            *instance_cnt,
                            &mut render_pass,
                        );
                    }
                }
            }

            self.render_queue.clear();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn get_projection_matrix(&self) -> Matrix4<f32> {
        let aspect = self.config.width as f32 / self.config.height as f32;
        Perspective3::new(
            aspect,
            self.camera.fov,
            self.camera.z_near,
            self.camera.z_far,
        )
        .to_homogeneous()
    }
}
