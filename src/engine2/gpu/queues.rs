//! GPU-owned hot queues for active/dirty/remesh brick scheduling.

use bytemuck::bytes_of;

use crate::engine2::gpu::buffers::{BufferPool, QueueCounters};
use crate::engine2::gpu::context::GpuContext;

#[derive(Debug, Default)]
pub struct HotQueues {
    active_bricks: Vec<u32>,
    dirty_bricks: Vec<u32>,
    remesh_bricks: Vec<u32>,
    draw_indirect_count: u32,
}

impl HotQueues {
    pub fn clear_frame(&mut self) {
        self.active_bricks.clear();
        self.dirty_bricks.clear();
        self.remesh_bricks.clear();
        self.draw_indirect_count = 0;
    }

    pub fn clear_active(&mut self) {
        self.active_bricks.clear();
    }

    pub fn push_active(&mut self, page_slot: u32) {
        self.active_bricks.push(page_slot);
    }

    pub fn push_dirty(&mut self, page_slot: u32) {
        self.dirty_bricks.push(page_slot);
    }

    pub fn push_remesh(&mut self, page_slot: u32) {
        self.remesh_bricks.push(page_slot);
    }

    pub fn set_draw_indirect_count(&mut self, count: u32) {
        self.draw_indirect_count = count;
    }

    pub fn upload(&self, gpu: &GpuContext, buffers: &BufferPool) {
        gpu.queue().write_buffer(
            &buffers.active_brick_queue,
            0,
            bytemuck::cast_slice(&self.active_bricks),
        );
        gpu.queue().write_buffer(
            &buffers.dirty_brick_queue,
            0,
            bytemuck::cast_slice(&self.dirty_bricks),
        );
        gpu.queue().write_buffer(
            &buffers.remesh_queue,
            0,
            bytemuck::cast_slice(&self.remesh_bricks),
        );

        let counters = QueueCounters {
            active_count: self.active_bricks.len() as u32,
            dirty_count: self.dirty_bricks.len() as u32,
            remesh_count: self.remesh_bricks.len() as u32,
            draw_indirect_count: self.draw_indirect_count,
        };
        gpu.queue()
            .write_buffer(&buffers.queue_counters, 0, bytes_of(&counters));
    }
}
