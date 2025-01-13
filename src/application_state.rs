use std::{cell::RefCell, collections::VecDeque, error::Error, rc::Rc, sync::Arc};

use egui::Slider;
use egui_plot::{Line, Plot, PlotPoints};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

use crate::{
    graphics::{Camera, RenderEngine},
    gui::Egui,
    input_helper::InputHelper,
    CameraController, FluidSimulation, WgpuRenderDevice,
};

pub struct ApplicationState {
    window: Arc<Window>,
    render_device: Rc<RefCell<WgpuRenderDevice>>,
    render_engine: RenderEngine,
    gui: Egui,
    camera: Camera,
    camera_controller: CameraController,

    fluid_sim: FluidSimulation,
    frame_times: VecDeque<f32>,

    particle_display_size: f32,
}

impl ApplicationState {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let render_device = Rc::new(RefCell::new(WgpuRenderDevice::new(window.clone()).await?));
        let render_engine = RenderEngine::new(render_device.clone());
        let fluid_sim = FluidSimulation::new(900, 0.04, &render_engine, &render_device.borrow().wgpu_device);
        let gui = Egui::new(&window);

        Ok(Self {
            window,
            render_device,
            render_engine,
            gui,
            camera: Camera::new(),
            camera_controller: CameraController::new(),
            fluid_sim,
            frame_times: VecDeque::new(),
            particle_display_size: 0.01,
        })
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) {
        self.gui.handle_input(&self.window, &event);
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.render_device.borrow_mut().resize(size);
    }

    pub fn update(&mut self, input_helper: &InputHelper) {
        self.camera_controller
            .update_camera(input_helper, &mut self.camera);
        self.fluid_sim.update(&mut self.render_engine);
    }

    pub fn redraw(&mut self) {
        if self.frame_times.len() > 1000 {
            self.frame_times.pop_front();
        }

        self.frame_times
            .push_back(self.render_engine.last_frame_time());

        self.gui.render(
            &self.window,
            &mut self.render_engine,
            "Fluid simulation",
            |ui| {
                let points: PlotPoints = self
                    .frame_times
                    .iter()
                    .enumerate()
                    .map(|(i, &time)| [i as f64, time as f64])
                    .collect();

                let line = Line::new(points)
                    .color(egui::Color32::LIGHT_BLUE)
                    .name("Frame Time (ms)");

                Plot::new("frame_time_plot")
                    .view_aspect(2.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
                ui.label("Particle display size:");
                ui.add(Slider::new(&mut self.particle_display_size, 0.001..=0.5).text("Size"));
            },
        );

        self.render_engine
            .render(&self.camera)
            .expect("Render engine failed");
    }
}
