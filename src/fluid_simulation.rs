use std::rc::Rc;

use nalgebra::{Point4, Vector3};

use crate::{
    graphics::{
        geometry::Geometry,
        materials::{ColoredVertex, MaterialType},
        render_engine::{RenderEngine, RenderRequest},
    },
    ComputeTask, SpatialLookup, WgpuDevice,
};

pub struct FluidSimulation {
    particle_cnt: usize,
    smoothing_radius: f32,
    bbox_dimensions: Vector3<f32>,

    bbox_geometry: Geometry,
    position_buffer: Rc<wgpu::Buffer>,
    velocity_buffer: Rc<wgpu::Buffer>,
    density_buffer: Rc<wgpu::Buffer>,
    force_buffer: Rc<wgpu::Buffer>,

    spatial_lookup: SpatialLookup,
    compute_density_task: Rc<ComputeTask>,

    particle_display_buffer: Rc<wgpu::Buffer>,
    display_density_task: Rc<ComputeTask>,
    update_particle_task: Rc<ComputeTask>,
    compute_pressure_task: Rc<ComputeTask>,
}

impl FluidSimulation {
    pub fn new(
        particle_cnt: usize,
        smoothing_radius: f32,
        mass: f32,
        damping: f32,
        gas_const: f32,
        rest_density: f32,
        gravity: Vector3<f32>,
        render_engine: &RenderEngine,
        wgpu_device: &WgpuDevice,
    ) -> Self {
        let bbox_dimensions = Vector3::new(1.0, 0.8, 0.8);
        let bbox_geometry = render_engine
            .create_geometry_array(&FluidSimulation::create_bbox_geometry(&bbox_dimensions));

        let positions = FluidSimulation::particle_start_positions(
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,
        );

        let position_buffer = wgpu_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let density_buffer = Rc::new(wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density buffer"),
            size: (particle_cnt * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let force_buffer = Rc::new(wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Force buffer"),
            size: (particle_cnt * std::mem::size_of::<nalgebra::Vector4<f32>>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let velocity = vec![nalgebra::Vector4::<f32>::new(0.0, 0.0, 0.0, 1.0); particle_cnt];
        let velocity_buffer = wgpu_device.create_buffer_init(
            &velocity,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let particle_display_buffer =
            Rc::new(wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Display buffer"),
                size: (particle_cnt * std::mem::size_of::<ColoredVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));

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

        let compute_density_task = FluidSimulation::create_compute_density_task(
            wgpu_device,
            particle_cnt,
            smoothing_radius,
            mass,
            cell_cnt,
            &position_buffer,
            spatial_lookup.keys(),
            spatial_lookup.vals(),
            spatial_lookup.index(),
            &density_buffer,
        );

        let display_density_task = FluidSimulation::create_display_density_task(
            wgpu_device,
            particle_cnt,
            bbox_dimensions,
            &position_buffer,
            &density_buffer,
            &particle_display_buffer,
        );

        let update_particle_task = FluidSimulation::create_update_particles_task(
            wgpu_device,
            particle_cnt,
            smoothing_radius,
            damping,
            mass,
            gravity,
            bbox_dimensions,
            &position_buffer,
            &velocity_buffer,
            &density_buffer,
            &force_buffer,
        );

        let compute_pressure_task = FluidSimulation::create_compute_pressure_task(
            wgpu_device,
            particle_cnt,
            smoothing_radius,
            mass,
            gas_const,
            rest_density,
            cell_cnt,
            &position_buffer,
            spatial_lookup.keys(),
            spatial_lookup.vals(),
            spatial_lookup.index(),
            &density_buffer,
            &force_buffer,
        );

        Self {
            particle_cnt,
            smoothing_radius,
            bbox_dimensions,

            bbox_geometry,
            position_buffer,
            velocity_buffer,
            density_buffer,
            force_buffer,

            spatial_lookup,
            compute_density_task,

            particle_display_buffer,
            display_density_task,
            update_particle_task,
            compute_pressure_task,
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

    fn particle_start_positions(
        particle_cnt: usize,
        smoothing_radius: f32,
        bbox_dimensions: Vector3<f32>,
    ) -> Vec<Point4<f32>> {
        let mut positions = Vec::with_capacity(particle_cnt);
        let n = f32::ceil(f32::powf(particle_cnt as f32, 1.0 / 3.0)) as usize;

        let squeeze_const = 0.95;

        let half = ((n - 1) as f32 * smoothing_radius * squeeze_const) / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let jitter_x = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_y = (rand::random::<f32>() - 0.5) / 200.0;
                    let jitter_z = (rand::random::<f32>() - 0.5) / 200.0;

                    positions.push(Point4::new(
                        j as f32 * smoothing_radius * squeeze_const + jitter_x - half
                            + bbox_dimensions.x / 2.0,
                        i as f32 * smoothing_radius * squeeze_const + jitter_y - half
                            + bbox_dimensions.y / 2.0,
                        k as f32 * smoothing_radius * squeeze_const + jitter_z - half
                            + bbox_dimensions.z / 2.0,
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

    fn create_compute_density_task(
        wgpu_device: &WgpuDevice,
        particle_cnt: usize,
        smoothing_radius: f32,
        mass: f32,
        cell_cnt: Vector3<u32>,
        positions: &wgpu::Buffer,
        spatial_lookup_keys: &wgpu::Buffer,
        spatial_lookup_vals: &wgpu::Buffer,
        spatial_lookup_index: &wgpu::Buffer,
        density: &wgpu::Buffer,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "
             const SMOOTHING_RADIUS: f32 = {smoothing_radius};\n
             const CELL_CNT: vec3<u32> = vec3<u32>({}, {}, {});\n 
             const MASS: f32 = {mass};\n 
             {}",
            cell_cnt.x,
            cell_cnt.y,
            cell_cnt.z,
            include_str!("shaders/compute_density.wgsl")
        );

        Rc::new(ComputeTask::new(
            wgpu_device,
            "Compute density",
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_lookup_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_lookup_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spatial_lookup_index.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: density.as_entire_binding(),
                },
            ],
            &[],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ))
    }

    fn create_compute_pressure_task(
        wgpu_device: &WgpuDevice,
        particle_cnt: usize,
        smoothing_radius: f32,
        mass: f32,
        gas_const: f32,
        rest_density: f32,
        cell_cnt: Vector3<u32>,
        positions: &wgpu::Buffer,
        spatial_lookup_keys: &wgpu::Buffer,
        spatial_lookup_vals: &wgpu::Buffer,
        spatial_lookup_index: &wgpu::Buffer,
        density: &wgpu::Buffer,
        force: &wgpu::Buffer,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "
             const REST_DENSITY: f32 = {rest_density};\n
             const GAS_CONST: f32 = {gas_const};\n
             const SMOOTHING_RADIUS: f32 = {smoothing_radius};\n
             const CELL_CNT: vec3<u32> = vec3<u32>({}, {}, {});\n 
             const MASS: f32 = {mass};\n 
             {}",
            cell_cnt.x,
            cell_cnt.y,
            cell_cnt.z,
            include_str!("shaders/compute_pressure.wgsl")
        );

        Rc::new(ComputeTask::new(
            wgpu_device,
            "Compute pressure",
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_lookup_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_lookup_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spatial_lookup_index.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: density.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: force.as_entire_binding(),
                },
            ],
            &[],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ))
    }

    fn create_update_particles_task(
        wgpu_device: &WgpuDevice,
        particle_cnt: usize,
        smoothing_radius: f32,
        damping: f32,
        mass: f32,
        gravity: Vector3<f32>,
        bbox_dimensions: Vector3<f32>,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        densities: &wgpu::Buffer,
        forces: &wgpu::Buffer,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "
             const SMOOTHING_RADIUS: f32 = {smoothing_radius};\n
             const MASS: f32 = {mass};\n
             const BBOX: vec3<f32> = vec3<f32>({}, {}, {});\n 
             const G: vec3<f32> = vec3<f32>({}, {}, {});\n 
             const DAMPING: f32 = {damping};\n 
             {}",
            bbox_dimensions.x,
            bbox_dimensions.y,
            bbox_dimensions.z,
            gravity.x,
            gravity.y,
            gravity.z,
            include_str!("shaders/update_particles.wgsl")
        );

        Rc::new(ComputeTask::new(
            wgpu_device,
            "Update particles",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: densities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: forces.as_entire_binding(),
                },
            ],
            &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ))
    }

    fn create_display_density_task(
        wgpu_device: &WgpuDevice,
        particle_cnt: usize,
        bbox_dimensions: Vector3<f32>,
        positions: &wgpu::Buffer,
        density: &wgpu::Buffer,
        display_buffer: &wgpu::Buffer,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "
             const OFFSET: vec3<f32> = vec3<f32>({}, {}, {});\n 
             {}",
            -bbox_dimensions.x / 2.0,
            -bbox_dimensions.y / 2.0,
            -bbox_dimensions.z / 2.0,
            include_str!("shaders/fill_display_buffer.wgsl")
        );

        Rc::new(ComputeTask::new(
            wgpu_device,
            "Display density",
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
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: display_buffer.as_entire_binding(),
                },
            ],
            &[],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ))
    }

    pub fn update(&self, render_engine: &mut RenderEngine, dt: f32, simulation_paused: bool) {
        if !simulation_paused {
            self.spatial_lookup.update(render_engine);

            let compute_density_task = self.compute_density_task.clone();
            render_engine.submit_generic_request(Box::new(move |encoder, _| {
                compute_density_task.execute(encoder, &[]);
            }));

            let compute_pressure_task = self.compute_pressure_task.clone();
            render_engine.submit_generic_request(Box::new(move |encoder, _| {
                compute_pressure_task.execute(encoder, &[]);
            }));

            let update_particles_task = self.update_particle_task.clone();
            render_engine.submit_generic_request(Box::new(move |encoder, _| {
                update_particles_task.execute(encoder, bytemuck::bytes_of(&dt));
            }));

            let display_density_task = self.display_density_task.clone();
            render_engine.submit_generic_request(Box::new(move |encoder, _| {
                display_density_task.execute(encoder, &[]);
            }));
        }

        render_engine.submit_render_request(RenderRequest {
            material_type: MaterialType::Line,
            geometry: self.bbox_geometry.clone(),
        });

        render_engine.submit_render_request(RenderRequest {
            material_type: MaterialType::Particle,
            geometry: Geometry::Instanced {
                vertex_cnt: 4,
                instance_buffer: self.particle_display_buffer.clone(),
                instance_cnt: self.particle_cnt,
            },
        });
    }
}
