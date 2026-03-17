//! Upload staging for command-driven brick updates.

use bytemuck::bytes_of;

use crate::engine2::gpu::buffers::{BrickHeader, BufferPool, DrawIndirectArgs};
use crate::engine2::gpu::context::GpuContext;
use crate::engine2::gpu::queues::HotQueues;
use crate::engine2::types::{BrickKey, GpuPageHandle};
use crate::engine2::world::brick::{BrickPayload, BRICK_VOXEL_CAPACITY};

#[derive(Debug, Clone)]
pub struct PendingBrickUpload {
    pub key: BrickKey,
    pub page: GpuPageHandle,
    pub revision: u32,
    /// One-shot page initialization payload.
    ///
    /// This staging copy is consumed at flush and is not a CPU authoritative resident state.
    pub init_payload: Option<BrickPayload>,
    pub mark_dirty: bool,
}

#[derive(Debug, Default)]
pub struct UploadQueue {
    pending: Vec<PendingBrickUpload>,
    pub pending_bytes: u64,
}

impl UploadQueue {
    pub fn enqueue(&mut self, upload: PendingBrickUpload) {
        self.pending_bytes = self
            .pending_bytes
            .saturating_add((BRICK_VOXEL_CAPACITY * std::mem::size_of::<u16>()) as u64);
        self.pending.push(upload);
    }

    pub fn flush(&mut self, gpu: &GpuContext, buffers: &BufferPool, queues: &mut HotQueues) {
        if self.pending.is_empty() {
            return;
        }

        let header_size = std::mem::size_of::<BrickHeader>() as u64;
        let page_bytes = buffers.brick_state_page_bytes();
        let zero_payload = vec![0u16; BRICK_VOXEL_CAPACITY];

        for upload in self.pending.drain(..) {
            let page_slot = upload.page.0;
            let header = BrickHeader {
                key_xyz: [upload.key.x, upload.key.y, upload.key.z],
                page_slot,
                revision: upload.revision,
                flags: u32::from(upload.mark_dirty),
                _pad0: [0, 0],
            };
            let header_offset = u64::from(page_slot) * header_size;
            gpu.queue()
                .write_buffer(&buffers.brick_headers, header_offset, bytes_of(&header));

            let payload = upload
                .init_payload
                .as_ref()
                .map_or(&zero_payload, |p| &p.material_ids);
            let page_offset = u64::from(page_slot) * page_bytes;
            gpu.queue().write_buffer(
                &buffers.brick_state_pages,
                page_offset,
                bytemuck::cast_slice(payload),
            );

            queues.push_active(page_slot);
            if upload.mark_dirty {
                queues.push_dirty(page_slot);
                queues.push_remesh(page_slot);
            }
        }

        let indirect = DrawIndirectArgs {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        gpu.queue()
            .write_buffer(&buffers.indirect_draw, 0, bytes_of(&indirect));

        self.pending_bytes = 0;
    }
}
