//! Shared GPU context handle for engine2 systems.

use std::sync::Arc;

/// Thin adapter around externally-owned wgpu handles.
///
/// Engine2 intentionally does not own global device bootstrap in this phase.
#[derive(Debug, Clone)]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    pub fn device(&self) -> &wgpu::Device {
        self.device.as_ref()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref()
    }

    pub fn create_encoder(&self, label: &'static str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }
}
