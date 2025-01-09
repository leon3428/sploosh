use std::sync::Arc;

use pollster::FutureExt;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, MouseScrollDelta, WindowEvent},
    window::Window,
};

use crate::{input_helper::InputHelper, ApplicationState};

pub struct Application {
    window: Option<Arc<Window>>,
    state: Option<ApplicationState>,
    input_helper: InputHelper,
}

impl Application {
    pub fn new() -> Self {
        Self {
            window: None,
            state: None,
            input_helper: InputHelper::new(),
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Ok(window) = event_loop.create_window(Window::default_attributes()) {
            let window_arc = Arc::new(window);

            self.state = ApplicationState::new(window_arc.clone()).block_on().ok();
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
            if let Some(state) = &mut self.state {
                state.on_window_event(&event);
            }

            if window.id() == window_id {
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        if let Some(state) = &mut self.state {
                            state.resize(physical_size);
                        }
                    }
                    WindowEvent::KeyboardInput {
                        device_id: _,
                        event,
                        is_synthetic: _,
                    } => {
                        self.input_helper.key_event(&event);
                    }
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                    } => {
                        self.input_helper.mouse_key_event(&state, button);
                    }
                    WindowEvent::RedrawRequested => {
                        if let Some(state) = &mut self.state {
                            state.update(&self.input_helper);
                            state.redraw();
                        }
                        self.input_helper.reset();
                    }
                    _ => {}
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _: &winit::event_loop::ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.input_helper
                    .mouse_moved((delta.0 as f32, delta.1 as f32));
            }
            DeviceEvent::MouseWheel { delta } => {
                if let MouseScrollDelta::LineDelta(_, dy) = delta {
                    self.input_helper.mouse_wheel_moved(dy);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
