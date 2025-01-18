use std::{error::Error, num::NonZero, rc::Rc};

pub struct WgpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuDevice {
    pub async fn new_compute_device() -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to crate an adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 4,
                        ..Default::default()
                    },
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        Ok(Self { device, queue })
    }

    pub fn create_buffer_init<T>(&self, data: &[T], usage: wgpu::BufferUsages) -> Rc<wgpu::Buffer> {
        let len = data.len() * std::mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;

        let data = unsafe { std::slice::from_raw_parts(ptr, len) };

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer"),
            size: len as u64,
            usage,
            mapped_at_creation: false,
        });

        let view = self
            .queue
            .write_buffer_with(&buffer, 0, NonZero::new(len as u64).unwrap());
        view.unwrap().copy_from_slice(data);

        Rc::new(buffer)
    }
}
