use std::sync::Arc;

use pollster::FutureExt;
use winit::{
    application::ApplicationHandler, event::WindowEvent, window::Window
};

use crate::graphics::render_engine::RenderEngine;

pub struct Application {
    window: Option<Arc<Window>>,
    render_engine: Option<RenderEngine>,
}

impl Application {
    pub fn new() -> Self {
        Self {
            window: None,
            render_engine: None,
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Ok(window) = event_loop.create_window(Window::default_attributes()) {
            let window_arc = Arc::new(window);

            if let Ok(render_engine) = RenderEngine::new(window_arc.clone()).block_on() {
                self.render_engine = Some(render_engine);
            } else {
                self.render_engine = None;
            }

            self.window = Some(window_arc);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(window) = self.window.as_ref() {
            if window.id() == window_id {
                let render_engine = self.render_engine.as_mut().unwrap();

                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        render_engine.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        render_engine.update();
                        render_engine.render().expect("Render engine failed");
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
