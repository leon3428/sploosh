use std::error::Error;
use application::Application;
use winit;

pub mod application;
pub mod graphics;
pub mod input_helper;
pub mod fluid_simulation;
pub mod gui;
pub mod render_device;
pub mod application_state;
pub mod camera_controller;

pub use render_device::RenderDevice;
pub use fluid_simulation::FluidSimulation;
pub use application_state::ApplicationState;
pub use camera_controller::CameraController;

pub fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = Application::new();
    event_loop.run_app(&mut app)?;    

    Ok(())
}