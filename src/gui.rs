use egui::Context;
use egui_winit::State;
use winit::{event::WindowEvent, window::Window};

use crate::graphics::{render_engine::GuiRenderRequest, RenderEngine};

pub struct Egui {
    state: State,
}

impl Egui {
    pub fn new(window: &Window) -> Self {
        let ctx = Context::default();
        let state = egui_winit::State::new(
            ctx,
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );

        Self { state }
    }

    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self.state.on_window_event(window, event);
    }

    pub fn context(&self) -> &Context {
        self.state.egui_ctx()
    }

    pub fn ppp(&self) -> f32 {
        self.context().pixels_per_point()
    }

    pub fn render(&mut self, window: &Window, render_engine: &mut RenderEngine, title: &str, add_contents: impl FnOnce(&mut egui::Ui) -> ()) {
        let raw_input = self.state.take_egui_input(window);
        self.state.egui_ctx().begin_pass(raw_input);

        let scale_factor = window.scale_factor() as f32;

        egui::Window::new(title)
            .resizable(true)
            .vscroll(true)
            .default_open(false)
            .show(self.context(), add_contents);

        self.state.egui_ctx().set_pixels_per_point(scale_factor);
        let full_output = self.state.egui_ctx().end_pass();
        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .state
            .egui_ctx()
            .tessellate(full_output.shapes, self.ppp());

        render_engine.submit_gui_render_request(GuiRenderRequest {
            textures_delta: full_output.textures_delta,
            tris,
            scale_factor,
        });
    }
}
