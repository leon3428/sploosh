use std::{cell::RefCell, collections::HashMap, rc::Rc, time::Instant};

use egui::{ClippedPrimitive, TexturesDelta};
use egui_wgpu::Renderer;
use nalgebra::{Matrix4, Point3};

use crate::{ComputeTask, WgpuRenderDevice};

use super::{
    camera::Camera,
    geometry::Geometry,
    materials::{LineMaterial, Material, MaterialType, ParticleMaterial},
};

pub struct RenderRequest {
    pub material_type: MaterialType,
    pub geometry: Geometry,
}

pub struct GuiRenderRequest {
    pub textures_delta: TexturesDelta,
    pub tris: Vec<ClippedPrimitive>,
    pub scale_factor: f32,
}

pub struct ComputeRequest {
    pub compute_task: Rc<ComputeTask>,
}

#[repr(C)]
struct CameraUniform {
    pub view_proj: Matrix4<f32>,
    pub view_inv: Matrix4<f32>,
    pub position: Point3<f32>,
    pub _padding: f32,
}

pub struct RenderEngine {
    render_device: Rc<RefCell<WgpuRenderDevice>>,
    gui_renderer: Renderer,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    materials: HashMap<MaterialType, Box<dyn Material>>,
    render_queue: Vec<RenderRequest>,
    gui_request: Option<GuiRenderRequest>,
    compute_queue: Vec<ComputeRequest>,

    last_frame_time: f32,
}

impl<'a> RenderEngine {
    pub fn new(render_device: Rc<RefCell<WgpuRenderDevice>>) -> Self {
        let rd = render_device.borrow();

        // Model view buffer initialization

        let camera_buffer = rd.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            rd.device()
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

        let camera_bind_group = rd.device().create_bind_group(&wgpu::BindGroupDescriptor {
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

        // gui
        let gui_renderer = Renderer::new(
            &rd.device(),
            rd.config.format,
            Some(rd.depth_texture.format()),
            1,
            true,
        );

        drop(rd);

        Self {
            render_device,
            gui_renderer,
            camera_buffer,
            camera_bind_group,
            materials,
            render_queue: Vec::new(),
            compute_queue: Vec::new(),
            gui_request: None,
            last_frame_time: 0.0,
        }
    }

    pub fn create_geometry_array<T>(&self, vertices: &[T]) -> Geometry {
        Geometry::Array {
            vertex_buffer: self.render_device.borrow().create_buffer_init(
                vertices,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            ),
            vertex_cnt: vertices.len(),
        }
    }

    pub fn update(&self) {}

    pub fn submit_render_request(&mut self, render_request: RenderRequest) {
        self.render_queue.push(render_request);
    }

    pub fn submit_gui_render_request(&mut self, request: GuiRenderRequest) {
        self.gui_request = Some(request);
    }

    pub fn submit_compute_request(&mut self, request: ComputeRequest) {
        self.compute_queue.push(request);
    }

    pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        let start_time = Instant::now();

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

        rd.queue().write_buffer(&self.camera_buffer, 0, data);

        let mut encoder = rd
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute pass"),
                timestamp_writes: None,
            });

            for request in &self.compute_queue {
                request.compute_task.execute(&mut compute_pass);
            }

            self.compute_queue.clear();
        }

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

        if let Some(request) = self.gui_request.take() {
            for (id, image_delta) in &request.textures_delta.set {
                self.gui_renderer
                    .update_texture(&rd.device(), &rd.queue(), *id, image_delta);
            }

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [rd.config.width, rd.config.height],
                pixels_per_point: request.scale_factor,
            };

            self.gui_renderer.update_buffers(
                &rd.device(),
                &rd.queue(),
                &mut encoder,
                &request.tris,
                &screen_descriptor,
            );

            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Gui render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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

            self.gui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &request.tris,
                &screen_descriptor,
            );
            for x in &request.textures_delta.free {
                self.gui_renderer.free_texture(x);
            }
        }

        rd.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        let end_time = Instant::now();
        self.last_frame_time = (end_time - start_time).as_secs_f32() * 1000.0;

        Ok(())
    }

    pub fn last_frame_time(&self) -> f32 {
        self.last_frame_time
    }
}
