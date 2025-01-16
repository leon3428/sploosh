use std::rc::Rc;

use nalgebra::{Point4, Vector3};

use crate::{
    graphics::{
        geometry::Geometry,
        materials::MaterialType,
        render_engine::{RenderEngine, RenderRequest},
    },
    SpatialLookup, WgpuDevice,
};

pub struct FluidSimulation {
    particle_cnt: usize,
    smoothing_radius: f32,
    bbox_dimensions: Vector3<f32>,

    bbox_geometry: Geometry,
    position_buffer: Rc<wgpu::Buffer>,
    velocity_buffer: Rc<wgpu::Buffer>,

    spatial_lookup: SpatialLookup,
}

impl FluidSimulation {
    pub fn new(
        particle_cnt: usize,
        smoothing_radius: f32,
        render_engine: &RenderEngine,
        wgpu_device: &WgpuDevice,
    ) -> Self {
        let bbox_dimensions = Vector3::new(1.0, 0.8, 0.8);
        let bbox_geometry = render_engine
            .create_geometry_array(&FluidSimulation::create_bbox_geometry(&bbox_dimensions));

        let positions = FluidSimulation::particle_start_positions(particle_cnt, smoothing_radius);

        let position_buffer = wgpu_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let velocity_buffer = wgpu_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let cell_cnt = Vector3::new(
            (bbox_dimensions.x / smoothing_radius).ceil() as u32,
            (bbox_dimensions.y / smoothing_radius).ceil() as u32,
            (bbox_dimensions.z / smoothing_radius).ceil() as u32,
        );

        let spatial_lookup = SpatialLookup::new(
            wgpu_device,
            particle_cnt,
            smoothing_radius,
            cell_cnt,
            &position_buffer,
        );

        Self {
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,

            bbox_geometry,
            position_buffer,
            velocity_buffer,

            spatial_lookup
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

    fn particle_start_positions(particle_cnt: usize, smoothing_radius: f32) -> Vec<Point4<f32>> {
        let mut positions = Vec::with_capacity(particle_cnt);
        let n = f32::ceil(f32::powf(particle_cnt as f32, 1.0 / 3.0)) as usize;

        let half = ((n - 1) as f32 * smoothing_radius * 0.95) / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let jitter_x = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_y = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_z = (rand::random::<f32>() - 0.5) / 200.0;

                    positions.push(Point4::new(
                        j as f32 * smoothing_radius * 0.95 - half + jitter_x,
                        i as f32 * smoothing_radius * 0.95 - half + jitter_y,
                        k as f32 * smoothing_radius * 0.95 - half + jitter_z,
                        1.0,
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
        self.spatial_lookup.update(render_engine);

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
