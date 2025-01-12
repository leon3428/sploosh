use std::borrow::Cow;

use crate::RenderDevice;

pub struct ComputeTask {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    workgroups: (u32, u32, u32),
}

impl ComputeTask {
    pub fn new(
        render_device: &RenderDevice,
        name: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        resources: &[wgpu::BindGroupEntry],
        shader_source: Cow<'_, str>,
        workgroups: (u32, u32, u32),
    ) -> Self {
        let bind_group_layout =
            render_device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{name} bind group layout")),
                    entries,
                });

        let bind_group = render_device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{name} bind group")),
                layout: &bind_group_layout,
                entries: resources,
            });

        let layout = render_device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name} pipeline layout")),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = render_device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{name} shader")),
                source: wgpu::ShaderSource::Wgsl(shader_source),
            });

        let pipeline =
            render_device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{name} pipeline")),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Self {
            bind_group,
            pipeline,
            workgroups,
        }
    }

    pub fn execute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(self.workgroups.0, self.workgroups.1, self.workgroups.2);
    }
}
