use std::collections::HashMap;

use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    keyboard::PhysicalKey,
};

pub struct InputHelper {
    mouse_button_map: HashMap<MouseButton, bool>,
    keyboard_button_map: HashMap<PhysicalKey, bool>,

    mouse_dx: f32,
    mouse_dy: f32,
    mouse_dw: f32,
}

impl InputHelper {
    pub fn new() -> Self {
        Self {
            mouse_button_map: HashMap::new(),
            keyboard_button_map: HashMap::new(),
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            mouse_dw: 0.0,
        }
    }

    pub fn key_event(&mut self, event: &KeyEvent) {
        self.keyboard_button_map
            .insert(event.physical_key, event.state.is_pressed());
    }

    pub fn mouse_key_event(&mut self, state: &ElementState, button: MouseButton) {
        self.mouse_button_map.insert(button, state.is_pressed());
    }

    pub fn mouse_moved(&mut self, delta: (f32, f32)) {
        self.mouse_dx += delta.0;
        self.mouse_dy += delta.1;
    }

    pub fn mouse_wheel_moved(&mut self, delta: f32) {
        self.mouse_dw += delta;
    }

    pub fn reset(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.mouse_dw = 0.0;
    }

    pub fn is_key_pressed(&self, key: PhysicalKey) -> bool {
        *self.keyboard_button_map.get(&key).unwrap_or(&false)
    }

    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        *self.mouse_button_map.get(&button).unwrap_or(&false)
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        (self.mouse_dx, self.mouse_dy)
    }
    
    pub fn mouse_wheel_delta(&self) -> f32 {
        self.mouse_dw
    }
    
}
