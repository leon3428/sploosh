use std::rc::Rc;

use nalgebra::{Point3, Vector3};

use crate::{
    graphics::{
        geometry::Geometry,
        materials::MaterialType,
        render_engine::{RenderEngine, RenderRequest},
    },
    RenderDevice,
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
        let positions = FluidSimulation::particle_start_positions(particle_cnt, smoothing_radius);

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

        let position_buffer = render_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let velocity_buffer = render_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
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
        }
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

    fn compute_pipeline(render_device: &RenderDevice) -> wgpu::ComputePipeline {
        let bind_group_layout = render_device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial lookup compute bind group layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = render_device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spatial lookup compute pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let shader = render_device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial lookup compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fill_spatial_lookup.wgsl").into()),
        });
    
        let pipeline = render_device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spatial lookup compute pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        pipeline
    }

    fn update_spatial_lookup() {}

    pub fn update(&self, render_engine: &mut RenderEngine) {
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
