//! GPU-owned buffer pool declarations.

use bytemuck::{Pod, Zeroable};

use crate::engine2::world::brick::BRICK_VOXEL_CAPACITY;

pub const DEFAULT_RESIDENT_BRICK_CAPACITY: u32 = 4096;
pub const DEFAULT_COMMAND_UPLOAD_BYTES: u64 = 1 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BrickHeader {
    pub key_xyz: [i32; 3],
    pub page_slot: u32,
    pub revision: u32,
    pub flags: u32,
    pub _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QueueCounters {
    pub active_count: u32,
    pub dirty_count: u32,
    pub remesh_count: u32,
    pub draw_indirect_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferPoolConfig {
    pub resident_brick_capacity: u32,
    pub command_upload_bytes: u64,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            resident_brick_capacity: DEFAULT_RESIDENT_BRICK_CAPACITY,
            command_upload_bytes: DEFAULT_COMMAND_UPLOAD_BYTES,
        }
    }
}

#[derive(Debug)]
pub struct BufferPool {
    pub config: BufferPoolConfig,
    pub brick_headers: wgpu::Buffer,
    pub brick_state_pages: wgpu::Buffer,
    pub command_upload: wgpu::Buffer,
    pub active_brick_queue: wgpu::Buffer,
    pub dirty_brick_queue: wgpu::Buffer,
    pub remesh_queue: wgpu::Buffer,
    pub queue_counters: wgpu::Buffer,
    pub indirect_draw: wgpu::Buffer,
}

impl BufferPool {
    pub fn new(device: &wgpu::Device, config: BufferPoolConfig) -> Self {
        let cap_u64 = u64::from(config.resident_brick_capacity);
        let header_size = std::mem::size_of::<BrickHeader>() as u64;
        let queue_item_size = std::mem::size_of::<u32>() as u64;
        let state_page_bytes = (BRICK_VOXEL_CAPACITY as u64) * std::mem::size_of::<u16>() as u64;

        let brick_headers = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.brick_headers"),
            size: cap_u64 * header_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let brick_state_pages = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.brick_state_pages"),
            size: cap_u64 * state_page_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let command_upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.command_upload"),
            size: config.command_upload_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let queue_buffer_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let active_brick_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.active_brick_queue"),
            size: cap_u64 * queue_item_size,
            usage: queue_buffer_usage,
            mapped_at_creation: false,
        });

        let dirty_brick_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.dirty_brick_queue"),
            size: cap_u64 * queue_item_size,
            usage: queue_buffer_usage,
            mapped_at_creation: false,
        });

        let remesh_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.remesh_queue"),
            size: cap_u64 * queue_item_size,
            usage: queue_buffer_usage,
            mapped_at_creation: false,
        });

        let queue_counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.queue_counters"),
            size: std::mem::size_of::<QueueCounters>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_draw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine2.indirect_draw"),
            size: std::mem::size_of::<DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            config,
            brick_headers,
            brick_state_pages,
            command_upload,
            active_brick_queue,
            dirty_brick_queue,
            remesh_queue,
            queue_counters,
            indirect_draw,
        }
    }

    pub fn brick_state_page_bytes(&self) -> u64 {
        (BRICK_VOXEL_CAPACITY as u64) * std::mem::size_of::<u16>() as u64
    }
}
