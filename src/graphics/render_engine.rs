use std::{cell::RefCell, collections::HashMap, num::NonZero, rc::Rc};

use nalgebra::{Matrix4, Point3};

use crate::RenderDevice;

use super::{
    camera::Camera,
    geometry::Geometry,
    materials::{LineMaterial, Material, MaterialType, ParticleMaterial},
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
    render_device: Rc<RefCell<RenderDevice>>,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    materials: HashMap<MaterialType, Box<dyn Material>>,
    render_queue: Vec<RenderRequest>,
}

impl<'a> RenderEngine {
    pub fn new(render_device: Rc<RefCell<RenderDevice>>) -> Self {
        let rd = render_device.borrow();

        // Model view buffer initialization

        let camera_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            rd.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let camera_bind_group = rd.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            Box::new(LineMaterial::new(&rd, &camera_bind_group_layout)),
        );
        materials.insert(
            MaterialType::Particle,
            Box::new(ParticleMaterial::new(&rd, &camera_bind_group_layout)),
        );

        drop(rd);

        Self {
            render_device,
            camera_buffer,
            camera_bind_group,
            materials,
            render_queue: Vec::new(),
        }
    }

    pub fn create_geometry_array<T>(&self, vertices: &[T]) -> Geometry {
        Geometry::Array {
            vertex_buffer: self.create_buffer(vertices),
            vertex_cnt: vertices.len(),
        }
    }

    pub fn create_buffer<T>(&self, data: &[T]) -> Rc<wgpu::Buffer> {
        let rd = self.render_device.borrow();

        let len = data.len() * std::mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        let buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry buffer"),
            size: len as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view = rd
            .queue
            .write_buffer_with(&buffer, 0, NonZero::new(len as u64).unwrap());
        view.unwrap().copy_from_slice(data);

        Rc::new(buffer)
    }

    pub fn update(&self) {}

    pub fn submit_render_request(&mut self, render_request: RenderRequest) {
        self.render_queue.push(render_request);
    }

    pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        let rd = self.render_device.borrow();
        let output = rd.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let view_mat = camera.get_view_matrix();
        let projection_mat =
            camera.get_projection_matrix(rd.config.width as f32 / rd.config.height as f32);

        let camera_data = CameraUniform {
            view_proj: projection_mat * view_mat,
            view_inv: view_mat.try_inverse().unwrap(),
            position: camera.position,
            _padding: 0.0,
        };

        let len = std::mem::size_of::<CameraUniform>();
        let ptr = camera_data.view_proj.as_ptr() as *const u8;
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        rd.queue.write_buffer(&self.camera_buffer, 0, data);

        let mut encoder = rd
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
                    view: rd.depth_texture.view(),
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

        rd.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
