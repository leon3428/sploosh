use core::f32;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,

    pub z_near: f32,
    pub z_far: f32,
    pub fov: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, -1.0),
            target: Point3::origin(),
            z_near: 0.1,
            z_far: 100.0,
            fov: std::f32::consts::FRAC_PI_4,
        }
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        nalgebra::geometry::Isometry3::look_at_rh(&self.position, &self.target, &Vector3::y())
            .to_homogeneous()
    }

    pub fn get_projection_matrix(&self, aspect: f32) -> Matrix4<f32> {
        Perspective3::new(
            aspect,
            self.fov,
            self.z_near,
            self.z_far,
        )
        .to_homogeneous()
    }
}
