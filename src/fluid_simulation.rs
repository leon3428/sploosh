use std::rc::Rc;

use nalgebra::{Point3, Vector3};

use crate::{
    graphics::{
        geometry::Geometry,
        materials::MaterialType,
        render_engine::{ComputeRequest, RenderEngine, RenderRequest},
    },
    ComputeTask, RenderDevice,
};

pub struct FluidSimulation {
    particle_cnt: usize,
    smoothing_radius: f32,
    bbox_dimensions: Vector3<f32>,

    bbox_geometry: Geometry,
    position_buffer: Rc<wgpu::Buffer>,
    velocity_buffer: Rc<wgpu::Buffer>,

    spatial_lookup_keys: wgpu::Buffer,
    spatial_lookup_vals: wgpu::Buffer,

    spatial_lookup_task: Rc<ComputeTask>,
}

impl FluidSimulation {
    pub fn new(
        particle_cnt: usize,
        smoothing_radius: f32,
        render_engine: &RenderEngine,
        render_device: &RenderDevice,
    ) -> Self {
        let bbox_dimensions = Vector3::new(1.0, 0.8, 0.8);
        let bbox_geometry = render_engine
            .create_geometry_array(&FluidSimulation::create_bbox_geometry(&bbox_dimensions));

        let (position_buffer, velocity_buffer, spatial_lookup_keys, spatial_lookup_vals) =
            FluidSimulation::create_buffers(particle_cnt, smoothing_radius, render_device);
        let spatial_lookup_task = FluidSimulation::create_spatial_lookup_task(
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,
            &position_buffer,
            &spatial_lookup_keys,
            &spatial_lookup_vals,
            render_device,
        );

        Self {
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,

            bbox_geometry,
            position_buffer,
            velocity_buffer,
            spatial_lookup_keys,
            spatial_lookup_vals,

            spatial_lookup_task,
        }
    }

    fn create_buffers(
        particle_cnt: usize,
        smoothing_radius: f32,
        render_device: &RenderDevice,
    ) -> (
        Rc<wgpu::Buffer>,
        Rc<wgpu::Buffer>,
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        let positions = FluidSimulation::particle_start_positions(particle_cnt, smoothing_radius);

        let position_buffer = render_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let velocity_buffer = render_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let spatial_lookup_keys = render_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial lookup keys"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let spatial_lookup_vals = render_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial lookup keys"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        (
            position_buffer,
            velocity_buffer,
            spatial_lookup_keys,
            spatial_lookup_vals,
        )
    }

    fn create_spatial_lookup_task(
        particle_cnt: usize,
        smoothing_radius: f32,
        bbox_dimensions: Vector3<f32>,
        position_buffer: &wgpu::Buffer,
        spatial_lookup_keys: &wgpu::Buffer,
        spatial_lookup_vals: &wgpu::Buffer,
        render_device: &RenderDevice,
    ) -> Rc<ComputeTask> {
        let cell_cnt_x = (bbox_dimensions.x / smoothing_radius).ceil();
        let cell_cnt_y = (bbox_dimensions.y / smoothing_radius).ceil();
        let cell_cnt_z = (bbox_dimensions.z / smoothing_radius).ceil();

        let shader_source = format!(
            "const PARTICLE_CNT: u32 = {particle_cnt};\n
             const SMOOTHING_RADIUS: f32 = {smoothing_radius};\n
             const CELL_CNT: vec3<u32> = vec3<u32>({cell_cnt_x}, {cell_cnt_y}, {cell_cnt_z});\n 
             {}",
            include_str!("shaders/fill_spatial_lookup.wgsl")
        );

        let spatial_lookup_task = Rc::new(ComputeTask::new(
            render_device,
            "Spatial lookup",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_lookup_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_lookup_vals.as_entire_binding(),
                },
            ],
            shader_source.into(),
            (64, 1, 1),
        ));

        spatial_lookup_task
    }

    fn create_bbox_geometry(dimensions: &Vector3<f32>) -> [Vector3<f32>; 24] {
        [
            Vector3::new(-dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(
                -dimensions.x / 2.0,
                -dimensions.y / 2.0,
                -dimensions.z / 2.0,
            ),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(
                -dimensions.x / 2.0,
                -dimensions.y / 2.0,
                -dimensions.z / 2.0,
            ),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(
                -dimensions.x / 2.0,
                -dimensions.y / 2.0,
                -dimensions.z / 2.0,
            ),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y / 2.0, -dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, -dimensions.y / 2.0, -dimensions.z / 2.0),
        ]
    }

    fn particle_start_positions(particle_cnt: usize, smoothing_radius: f32) -> Vec<Point3<f32>> {
        let mut positions = Vec::with_capacity(particle_cnt);
        let n = f32::ceil(f32::powf(particle_cnt as f32, 1.0 / 3.0)) as usize;

        let half = ((n - 1) as f32 * smoothing_radius * 0.95) / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let jitter_x = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_y = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_z = (rand::random::<f32>() - 0.5) / 200.0;

                    positions.push(Point3::new(
                        j as f32 * smoothing_radius * 0.95 - half + jitter_x,
                        i as f32 * smoothing_radius * 0.95 - half + jitter_y,
                        k as f32 * smoothing_radius * 0.95 - half + jitter_z,
                    ));
                    if positions.len() >= particle_cnt {
                        break 'outer;
                    }
                }
            }
        }

        positions
    }

    pub fn update(&self, render_engine: &mut RenderEngine) {
        render_engine.submit_compute_request(ComputeRequest {
            compute_task: self.spatial_lookup_task.clone(),
        });

        render_engine.submit_render_request(RenderRequest {
            material_type: MaterialType::Line,
            geometry: self.bbox_geometry.clone(),
        });

        render_engine.submit_render_request(RenderRequest {
            material_type: MaterialType::Particle,
            geometry: Geometry::Instanced {
                vertex_cnt: 4,
                instance_buffer: self.position_buffer.clone(),
                instance_cnt: self.particle_cnt,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use pollster::FutureExt;

    use super::*;

    #[test]
    fn populating_spatial_lookup() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .block_on()
            .unwrap();

        let particle_cnt = 10;
        let smoothing_radius = 0.04;
        let bbox_dimensions = Vector3::new(1.0, 0.8, 0.8);
        let render_device = RenderDevice {
            surface: todo!(),
            device,
            queue,
            config: todo!(),
            depth_texture: todo!(),
        };
        let (position_buffer, velocity_buffer, spatial_lookup_keys, spatial_lookup_vals) =
            FluidSimulation::create_buffers(particle_cnt, smoothing_radius, &render_device);
        let spatial_lookup_task = FluidSimulation::create_spatial_lookup_task(
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,
            &position_buffer,
            &spatial_lookup_keys,
            &spatial_lookup_vals,
            &render_device,
        );

        let test_buffer_size = (particle_cnt * std::mem::size_of::<u32>()) as u64;
        let test_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: test_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            spatial_lookup_task.execute(&mut compute_pass);
        }
        encoder.copy_buffer_to_buffer(&spatial_lookup_keys, 0, &test_buffer, 0, test_buffer_size);

        // Submit the work
        queue.submit(Some(encoder.finish()));

        // Read the output buffer
        let buffer_slice = test_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        receiver.receive().block_on().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        test_buffer.unmap();

        println!("{:?}", result);
    }
}
