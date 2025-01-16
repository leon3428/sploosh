use std::{num::NonZeroU32, rc::Rc};

use nalgebra::Vector3;
use pollster::FutureExt;
use wgpu_sort::{utils::guess_workgroup_size, GPUSorter, SortBuffers};

use crate::{graphics::RenderEngine, ComputeTask, WgpuDevice};

pub struct SpatialLookup {
    sort: Rc<GPUSorter>,
    sort_buffers: Rc<SortBuffers>,

    spatial_lookup_task: Rc<ComputeTask>,
    spatial_lookup_index: wgpu::Buffer,
    spatial_lookup_index_task: Rc<ComputeTask>,
}

impl SpatialLookup {
    pub fn new(
        wgpu_device: &WgpuDevice,
        particle_cnt: usize,
        smoothing_radius: f32,
        cell_cnt: Vector3<u32>,
        position_buffer: &wgpu::Buffer,
    ) -> Self {
        let spatial_lookup_index = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial index buffer"),
            size: (cell_cnt.x * cell_cnt.y * cell_cnt.z * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let subgroup_size = guess_workgroup_size(&wgpu_device.device, &wgpu_device.queue)
            .block_on()
            .unwrap();
        let sort = Rc::new(GPUSorter::new(&wgpu_device.device, subgroup_size));
        let sort_buffers = Rc::new(sort.create_sort_buffers(
            &wgpu_device.device,
            NonZeroU32::new(particle_cnt as u32).unwrap(),
        ));

        let spatial_lookup_task = SpatialLookup::create_spatial_lookup_fill_task(
            particle_cnt,
            smoothing_radius,
            cell_cnt,
            &position_buffer,
            &sort_buffers.keys(),
            &sort_buffers.values(),
            wgpu_device,
        );

        let spatial_lookup_index_task = SpatialLookup::create_spatial_lookup_index_task(
            wgpu_device,
            &sort_buffers.keys(),
            &spatial_lookup_index,
            particle_cnt,
        );

        Self {
            sort,
            sort_buffers,
            spatial_lookup_task,
            spatial_lookup_index,
            spatial_lookup_index_task,
        }
    }

    pub fn update(&self, render_engine: &mut RenderEngine) {
        render_engine.submit_generic_request(self.update_fn());
    }

    pub fn buffer_a(&self) -> &wgpu::Buffer {
        &self.sort_buffers.keys()
    }

    pub fn buffer_b(&self) -> &wgpu::Buffer {
        &self.sort_buffers.values()
    }

    pub fn buffer_c(&self) -> &wgpu::Buffer {
        &self.spatial_lookup_index
    }

    fn update_fn(&self) -> Box<dyn Fn(&mut wgpu::CommandEncoder, &wgpu::Queue) -> ()> {
        let spatial_lookup_task = self.spatial_lookup_task.clone();
        let sort = self.sort.clone();
        let sort_buffers = self.sort_buffers.clone();
        let spatial_lookup_index_task = self.spatial_lookup_index_task.clone();

        Box::new(move |encoder, queue| {
            spatial_lookup_task.execute(encoder, &[]);
            sort.sort(encoder, queue, &sort_buffers, None);
            spatial_lookup_index_task.execute(encoder, &[]);
        })
    }

    fn create_spatial_lookup_fill_task(
        particle_cnt: usize,
        smoothing_radius: f32,
        cell_cnt: Vector3<u32>,
        position_buffer: &wgpu::Buffer,
        spatial_lookup_keys: &wgpu::Buffer,
        spatial_lookup_vals: &wgpu::Buffer,
        wgpu_device: &WgpuDevice,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "const PARTICLE_CNT: u32 = {particle_cnt};\n
             const SMOOTHING_RADIUS: f32 = {smoothing_radius};\n
             const CELL_CNT: vec3<u32> = vec3<u32>({}, {}, {});\n 
             {}",
            cell_cnt.x,
            cell_cnt.y,
            cell_cnt.z,
            include_str!("shaders/fill_spatial_lookup.wgsl")
        );

        let spatial_lookup_task = Rc::new(ComputeTask::new(
            wgpu_device,
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
            &[],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ));

        spatial_lookup_task
    }

    fn create_spatial_lookup_index_task(
        wgpu_device: &WgpuDevice,
        spatial_lookup_keys: &wgpu::Buffer,
        spatial_lookup_index: &wgpu::Buffer,
        particle_cnt: usize,
    ) -> Rc<ComputeTask> {
        let mut workgroup_cnt = particle_cnt as u32 / 256;
        if particle_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        let shader_source = format!(
            "const PARTICLE_CNT: u32 = {particle_cnt};\n
             {}",
            include_str!("shaders/spatial_lookup_index.wgsl")
        );

        let spatial_lookup_index_task = Rc::new(ComputeTask::new(
            wgpu_device,
            "Spatial lookup index",
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
            ],
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spatial_lookup_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_lookup_index.as_entire_binding(),
                },
            ],
            &[],
            shader_source.into(),
            (workgroup_cnt, 1, 1),
        ));

        spatial_lookup_index_task
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{ComplexField, Point4, Vector3};
    use pollster::FutureExt as _;
    use rand::Rng;

    use crate::test_utils::read_buffer;

    use super::*;

    #[test]
    fn populating_spatial_lookup() {
        let wgpu_device = WgpuDevice::new_compute_device().block_on().unwrap();

        // consts
        let particle_cnt = 27;
        let smoothing_radius = 1.0;
        let bbox_dimensions = Vector3::new(3.0, 3.0, 3.0);

        // buffers
        let mut positions = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    positions.push(Point4::new(i as f32, j as f32, k as f32, 1.0));
                }
            }
        }

        let position_buffer = wgpu_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let spatial_lookup_keys = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial lookup keys"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spatial_lookup_vals = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial lookup keys"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_cnt = Vector3::new(
            (bbox_dimensions.x / smoothing_radius).ceil() as u32,
            (bbox_dimensions.y / smoothing_radius).ceil() as u32,
            (bbox_dimensions.z / smoothing_radius).ceil() as u32,
        );

        // create compute task
        let spatial_lookup_task = SpatialLookup::create_spatial_lookup_fill_task(
            particle_cnt,
            smoothing_radius,
            cell_cnt,
            &position_buffer,
            &spatial_lookup_keys,
            &spatial_lookup_vals,
            &wgpu_device,
        );

        // create the test buffer
        let staging_buffer_keys = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: spatial_lookup_keys.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_buffer_vals = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: spatial_lookup_vals.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // execute the compute task
        let mut encoder =
            wgpu_device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        spatial_lookup_task.execute(&mut encoder, &[]);

        encoder.copy_buffer_to_buffer(
            &spatial_lookup_vals,
            0,
            &staging_buffer_vals,
            0,
            spatial_lookup_vals.size(),
        );
        encoder.copy_buffer_to_buffer(
            &spatial_lookup_keys,
            0,
            &staging_buffer_keys,
            0,
            spatial_lookup_keys.size(),
        );
        wgpu_device.queue.submit(Some(encoder.finish()));

        let keys = read_buffer::<u32>(&wgpu_device, &staging_buffer_keys);
        let vals = read_buffer::<u32>(&wgpu_device, &staging_buffer_vals);

        for i in 0..27 {
            if vals[i] != i as u32 {
                panic!("Vals are not correct")
            }
            if keys[i] != i as u32 {
                panic!("Keys are not correct")
            }
        }
    }

    #[test]
    fn spatial_lookup() {
        let wgpu_device = WgpuDevice::new_compute_device().block_on().unwrap();

        let particle_cnt = 10;
        let smoothing_radius = 0.05;
        let bbox_dimensions = Vector3::new(1.0, 1.0, 1.0);
        let cell_cnt = Vector3::new(
            (bbox_dimensions.x / smoothing_radius).ceil() as u32,
            (bbox_dimensions.y / smoothing_radius).ceil() as u32,
            (bbox_dimensions.z / smoothing_radius).ceil() as u32,
        );

        let mut rng = rand::thread_rng();
        let mut positions = Vec::new();
        for _ in 0..particle_cnt {
            let x = rng.gen_range(0.0..bbox_dimensions.x);
            let y = rng.gen_range(0.0..bbox_dimensions.y);
            let z = rng.gen_range(0.0..bbox_dimensions.z);

            positions.push(Point4::new(x, y, z, 1.0));
        }

        let mut neighbors = vec![Vec::new(); particle_cnt];
        for i in 0..particle_cnt {
            for j in i..particle_cnt {
                let dist = (positions[i] - positions[j]).norm();
                if dist <= smoothing_radius {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        let position_buffer = wgpu_device.create_buffer_init(
            &positions,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let spatial_lookup = SpatialLookup::new(
            &wgpu_device,
            particle_cnt,
            smoothing_radius,
            cell_cnt,
            &position_buffer,
        );

        let staging_buffer_a = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer A"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_buffer_b = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer B"),
            size: (particle_cnt * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_buffer_c = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer C"),
            size: (cell_cnt.x * cell_cnt.y * cell_cnt.z * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let tmp: Vec<u32> = (0..particle_cnt as u32).collect();
        wgpu_device
            .queue
            .write_buffer(spatial_lookup.buffer_a(), 0, bytemuck::cast_slice(&tmp));
        wgpu_device
            .queue
            .write_buffer(spatial_lookup.buffer_b(), 0, bytemuck::cast_slice(&tmp));
        wgpu_device.device.poll(wgpu::Maintain::Wait);

        let mut encoder =
            wgpu_device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        spatial_lookup.update_fn()(&mut encoder, &wgpu_device.queue);

        encoder.copy_buffer_to_buffer(
            spatial_lookup.buffer_a(),
            0,
            &staging_buffer_a,
            0,
            staging_buffer_a.size(),
        );

        encoder.copy_buffer_to_buffer(
            spatial_lookup.buffer_b(),
            0,
            &staging_buffer_b,
            0,
            staging_buffer_b.size(),
        );

        encoder.copy_buffer_to_buffer(
            spatial_lookup.buffer_c(),
            0,
            &staging_buffer_c,
            0,
            staging_buffer_c.size(),
        );

        wgpu_device.queue.submit(Some(encoder.finish()));
        wgpu_device.device.poll(wgpu::Maintain::Wait);

        let a = read_buffer::<u32>(&wgpu_device, &staging_buffer_a);
        let b = read_buffer::<u32>(&wgpu_device, &staging_buffer_b);
        let c = read_buffer::<u32>(&wgpu_device, &staging_buffer_c);

        println!("{}", spatial_lookup.buffer_c().size() / 4);
        println!("{:?}", a);
        println!("{:?}", b);
        println!("{:?}", c);
    }
}
