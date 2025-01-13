use std::{error::Error, rc::Rc, sync::Arc};

use winit::window::Window;

use crate::{graphics::texture::Texture, WgpuDevice};

pub struct WgpuRenderDevice {
    pub surface: wgpu::Surface<'static>,
    pub wgpu_device: WgpuDevice,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_texture: Texture,
}

impl WgpuRenderDevice {
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

        Ok(Self {
            surface,
            wgpu_device: WgpuDevice { device, queue },
            config,
            depth_texture,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.wgpu_device.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.wgpu_device.queue
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(self.device(), &self.config);
            self.depth_texture = Texture::depth_texture(self.device(), &self.config);
        }
    }

    pub fn create_buffer_init<T>(&self, data: &[T], usage: wgpu::BufferUsages) -> Rc<wgpu::Buffer> {
        self.wgpu_device.create_buffer_init(data, usage)
    }
}
