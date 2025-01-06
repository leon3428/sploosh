use nalgebra::Vector3;

use crate::graphics::{geometry::Geometry, render_engine::RenderEngine};

pub struct FluidSimulation<'a> {
    bbox_dimensions: Vector3<f32>,

    bbox_geometry: Geometry,
    render_engine: &'a RenderEngine,
}

impl<'a> FluidSimulation<'a> {
    pub fn new(render_engine: &'a RenderEngine) -> Self {
        let bbox_dimensions = Vector3::new(3.0, 2.0, 1.0);
        let bbox_geometry =
            render_engine.create_geometry(&FluidSimulation::create_bbox_geometry(&bbox_dimensions));
            
        Self {
            bbox_dimensions,
            bbox_geometry,
            render_engine,
        }
    }

    fn create_bbox_geometry(dimensions: &Vector3<f32>) -> [Vector3<f32>; 8] {
        [
            Vector3::new(-dimensions.x / 2.0, 0.0, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, 0.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, 0.0, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y, dimensions.z / 2.0),
            Vector3::new(-dimensions.x / 2.0, dimensions.y, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, dimensions.y, dimensions.z / 2.0),
            Vector3::new(dimensions.x / 2.0, 0.0, dimensions.z / 2.0),
        ]
    }
}
