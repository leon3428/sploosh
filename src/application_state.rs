use std::{cell::RefCell, error::Error, rc::Rc, sync::Arc};

use winit::{dpi::PhysicalSize, window::Window};

use crate::{graphics::{Camera, RenderEngine}, input_helper::InputHelper, CameraController, FluidSimulation, RenderDevice};

pub struct ApplicationState {
    render_device: Rc<RefCell<RenderDevice>>,
    render_engine: RenderEngine,
    camera: Camera,
    camera_controller: CameraController,

    fluid_sim: FluidSimulation,
}

impl ApplicationState {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let render_device = Rc::new(RefCell::new(RenderDevice::new(window.clone()).await?));
        let render_engine = RenderEngine::new(render_device.clone());
        let fluid_sim = FluidSimulation::new(900, &render_engine);

        Ok(Self {
            render_device,
            render_engine,
            camera: Camera::new(),
            camera_controller: CameraController::new(),
            fluid_sim,
        })
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.render_device.borrow_mut().resize(size);
    }

    pub fn update(&mut self, input_helper: &InputHelper) {
        self.camera_controller.update_camera(input_helper, &mut self.camera);
        self.fluid_sim.update(&mut self.render_engine);
    }

    pub fn redraw(&mut self) {
        self.render_engine.render(&self.camera).expect("Render engine failed");
    }
}
