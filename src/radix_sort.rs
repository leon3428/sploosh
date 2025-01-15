use crate::{ComputeTask, WgpuDevice};

pub struct RadixSort {
    counter_buffer: wgpu::Buffer,
    fill_counters_task: ComputeTask,
}

impl RadixSort {
    pub fn new(
        wgpu_device: &WgpuDevice,
        keys_buffer: &wgpu::Buffer,
        vals_buffer: &wgpu::Buffer,
    ) -> Self {
        let block_size = 8;
        let bin_cnt = 1u64 << block_size;

        let counter_buffer = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial lookup keys"),
            size: bin_cnt * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let fill_counters_task =
            RadixSort::create_fill_counters_task(wgpu_device, keys_buffer, &counter_buffer);

        Self {
            counter_buffer,
            fill_counters_task,
        }
    }

    fn create_fill_counters_task(
        wgpu_device: &WgpuDevice,
        keys_buffer: &wgpu::Buffer,
        counter_buffer: &wgpu::Buffer,
    ) -> ComputeTask {
        let key_cnt = keys_buffer.size() / std::mem::size_of::<u32>() as u64;
        let mut workgroup_cnt = key_cnt / 256;
        if key_cnt % 256 != 0 {
            workgroup_cnt += 1;
        }

        ComputeTask::new(
            wgpu_device,
            "Fill counters",
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
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
            &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
            include_str!("shaders/rs_fill_counters.wgsl").into(),
            (workgroup_cnt as u32, 1, 1),
        )
    }

    fn create_prescan_task(wgpu_device: &WgpuDevice, counter_buffer: &wgpu::Buffer) -> ComputeTask {
        ComputeTask::new(
            wgpu_device,
            "Prescan counters",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter_buffer.as_entire_binding(),
            }],
            &[],
            include_str!("shaders/rs_prescan.wgsl").into(),
            (1, 1, 1),
        )
    }
}

#[cfg(test)]
mod tests {
    use pollster::FutureExt as _;
    use rand::{thread_rng, Rng};

    use crate::test_utils::read_buffer;

    use super::*;

    #[test]
    fn fill_counters() {
        let wgpu_device = WgpuDevice::new_compute_device().block_on().unwrap();

        let block_size = 8;
        let bin_cnt = 1u64 << block_size;
        let mask = (1u32 << block_size) - 1;
        let n = 100_000;
        let pass_cnt = 32 / block_size;

        let mut rng = thread_rng();
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            let x: u32 = rng.gen();
            keys.push(x);
        }
        let keys_buffer = wgpu_device.create_buffer_init(
            &keys,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );

        let counter_buffer = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counters"),
            size: bin_cnt * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fill_counters_task =
            RadixSort::create_fill_counters_task(&wgpu_device, &keys_buffer, &counter_buffer);

        // create the test buffer
        let staging_buffer = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: counter_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // execute the compute task
        for pass_ind in 0..pass_cnt {
            let counters = vec![0u32; bin_cnt as usize];
            wgpu_device
                .queue
                .write_buffer(&counter_buffer, 0, bytemuck::cast_slice(&counters));

            let mut encoder =
                wgpu_device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Command Encoder"),
                    });

            let mut expected_bins = vec![0u32; bin_cnt as usize];

            for key in &keys {
                let tmp = key & (mask << (pass_ind * block_size));
                let bin = tmp >> (pass_ind * block_size);
                expected_bins[bin as usize] += 1;
            }

            fill_counters_task.execute(&mut encoder, bytemuck::cast_slice(&[pass_ind]));

            encoder.copy_buffer_to_buffer(
                &counter_buffer,
                0,
                &staging_buffer,
                0,
                counter_buffer.size(),
            );
            wgpu_device.queue.submit(Some(encoder.finish()));

            let bins = read_buffer::<u32>(&wgpu_device, &staging_buffer);

            assert_eq!(bins, expected_bins);
        }
    }

    #[test]
    fn prescan() {
        let wgpu_device = WgpuDevice::new_compute_device().block_on().unwrap();

        let data: Vec<u32> = (0..256).collect();
        let buffer = wgpu_device.create_buffer_init(
            &data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let prescan_task = RadixSort::create_prescan_task(&wgpu_device, &buffer);

        // create the staging buffer
        let staging_buffer = wgpu_device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer.size(),
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

        prescan_task.execute(&mut encoder, &[]);

        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, buffer.size());
        wgpu_device.queue.submit(Some(encoder.finish()));
        let prescan = read_buffer::<u32>(&wgpu_device, &staging_buffer);

        println!("{:?}", prescan);
        println!("{:?}", (0..256).reduce(|acc, e| acc + e));
    }
}
