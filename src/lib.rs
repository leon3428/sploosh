use std::error::Error;
use application::Application;
use winit;

pub mod application;
pub mod graphics;

pub fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = Application::new();
    event_loop.run_app(&mut app)?;    

    Ok(())
}