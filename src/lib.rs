use std::error::Error;
use application::Application;
use winit;

pub mod application;
pub mod graphics;
pub mod input_helper;
pub mod fluid_simulation;
pub mod gui;
pub mod wgpu_render_device;
pub mod application_state;
pub mod camera_controller;
pub mod compute_task;
pub mod wgpu_device;
pub mod test_utils;
pub mod spatial_lookup;

pub use wgpu_render_device::WgpuRenderDevice;
pub use wgpu_device::WgpuDevice;
pub use fluid_simulation::FluidSimulation;
pub use application_state::ApplicationState;
pub use camera_controller::CameraController;
pub use compute_task::ComputeTask;
pub use spatial_lookup::SpatialLookup;

pub fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = Application::new();
    event_loop.run_app(&mut app)?;    

    Ok(())
}