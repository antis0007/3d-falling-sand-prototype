//! GPU-owned hot queues for active/dirty/remesh brick scheduling.

use bytemuck::bytes_of;

use crate::engine2::gpu::buffers::{BufferPool, QueueCounters};
use crate::engine2::gpu::context::GpuContext;
use crate::engine2::phases::SimOutput;

#[derive(Debug, Default)]
pub struct HotQueues {
    draw_indirect_count: u32,
}

impl HotQueues {
    pub fn clear_frame(&mut self) {
        self.draw_indirect_count = 0;
    }

    pub fn set_draw_indirect_count(&mut self, count: u32) {
        self.draw_indirect_count = count;
    }

    pub fn upload(&self, gpu: &GpuContext, buffers: &BufferPool, sim: &SimOutput) {
        gpu.queue().write_buffer(
            &buffers.active_brick_queue,
            0,
            bytemuck::cast_slice(&sim.active_pages),
        );
        gpu.queue().write_buffer(
            &buffers.dirty_brick_queue,
            0,
            bytemuck::cast_slice(&sim.dirty_pages),
        );
        gpu.queue().write_buffer(
            &buffers.remesh_queue,
            0,
            bytemuck::cast_slice(&sim.remesh_pages),
        );

        let counters = QueueCounters {
            active_count: sim.active_pages.len() as u32,
            dirty_count: sim.dirty_pages.len() as u32,
            remesh_count: sim.remesh_pages.len() as u32,
            draw_indirect_count: self.draw_indirect_count,
        };
        gpu.queue()
            .write_buffer(&buffers.queue_counters, 0, bytes_of(&counters));
    }
}
