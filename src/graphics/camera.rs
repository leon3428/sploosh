use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

pub struct Camera {
    position: Point3<f32>,
    target: Point3<f32>,
    radius: f32,
    rotation: UnitQuaternion<f32>,
    pub z_near: f32,
    pub z_far: f32,
    pub fov: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, -1.0),
            target: Point3::origin(),
            radius: 1.0,
            rotation: UnitQuaternion::identity(),
            z_near: 0.1,
            z_far: 100.0,
            fov: std::f32::consts::FRAC_PI_4,
        }
    }

    fn update_position(&mut self) {
        let direction = self.rotation * Vector3::new(0.0, 0.0, -1.0);
        self.position = self.target + direction * self.radius;
    }

    pub fn update(
        &mut self,
        left_button_pressed: bool,
        mouse_move_delta: (f32, f32),
        mouse_wheel_delta: f32,
    ) {
        let sensitivity = 0.003;
        let zoom_speed = 0.001;

        self.radius += mouse_wheel_delta * zoom_speed;
        self.radius = f32::max(self.radius, self.z_near);

        if left_button_pressed {
            let yaw_rotation = UnitQuaternion::from_axis_angle(
                &Vector3::y_axis(),
                -mouse_move_delta.0 * sensitivity,
            );
            let pitch_axis = self.rotation * Vector3::x_axis();
            let pitch_rotation =
                UnitQuaternion::from_axis_angle(&pitch_axis, mouse_move_delta.1 * sensitivity);

            self.rotation = yaw_rotation * pitch_rotation * self.rotation;
        }

        self.update_position();
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        nalgebra::geometry::Isometry3::look_at_rh(&self.position, &self.target, &Vector3::y())
            .to_homogeneous()
    }
}
