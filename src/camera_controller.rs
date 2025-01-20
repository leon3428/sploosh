use core::f32;

use crate::{graphics::Camera, input_helper::InputHelper};

pub struct CameraController {
    radius: f32,
    phi: f32,
    theta: f32,
    zoom_sensitivity: f32,
    orbit_sensitivity: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            radius: 10.0,
            phi: 0.0,
            theta: f32::consts::FRAC_2_PI,
            zoom_sensitivity: 0.01,
            orbit_sensitivity: 0.003,
        }
    }

    pub fn update_camera(&mut self, input_helper: &InputHelper, camera: &mut Camera) {
        self.radius += input_helper.mouse_wheel_delta() * self.zoom_sensitivity;
        self.radius = f32::max(self.radius, camera.z_near);

        if input_helper.is_mouse_button_pressed(winit::event::MouseButton::Left) {
            let (dx, dy) = input_helper.mouse_delta();
            self.phi += dx * self.orbit_sensitivity;
            self.theta -= dy * self.orbit_sensitivity;

            self.theta = self.theta.clamp(0.01, f32::consts::PI - 0.01);
        }

        camera.position.x = self.radius * self.theta.sin() * self.phi.cos();
        camera.position.y = self.radius * self.theta.cos();
        camera.position.z = self.radius * self.theta.sin() * self.phi.sin();
    }
}
