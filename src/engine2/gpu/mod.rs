//! GPU hot-state ownership modules.

pub mod buffers;
pub mod context;
pub mod page_table;
pub mod queues;
pub mod upload;

use std::sync::Arc;

use crate::engine2::gpu::buffers::{BufferPool, BufferPoolConfig};
use crate::engine2::gpu::context::GpuContext;
use crate::engine2::gpu::page_table::BrickPageTable;
use crate::engine2::gpu::queues::HotQueues;
use crate::engine2::gpu::upload::{PendingBrickUpload, UploadQueue};
use crate::engine2::types::BrickKey;
use crate::engine2::world::brick::BrickPayload;

#[derive(Debug, Default)]
pub struct Engine2Gpu {
    context: Option<GpuContext>,
    buffers: Option<BufferPool>,
    pub page_table: BrickPageTable,
    pub queues: HotQueues,
    pub uploads: UploadQueue,
}

impl Engine2Gpu {
    pub fn initialize(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: BufferPoolConfig,
    ) {
        let context = GpuContext::new(device, queue);
        let buffers = BufferPool::new(context.device(), config);
        self.page_table = BrickPageTable::with_capacity(config.resident_brick_capacity);
        self.context = Some(context);
        self.buffers = Some(buffers);
        self.queues.clear_frame();
    }

    pub fn is_initialized(&self) -> bool {
        self.context.is_some() && self.buffers.is_some()
    }

    pub fn enqueue_resident_brick(
        &mut self,
        key: BrickKey,
        revision: u32,
        init_payload: Option<BrickPayload>,
        mark_dirty: bool,
    ) -> Option<crate::engine2::types::GpuPageHandle> {
        let page = self.page_table.allocate(key)?;
        self.uploads.enqueue(PendingBrickUpload {
            key,
            page,
            revision,
            init_payload,
            mark_dirty,
        });
        Some(page)
    }

    pub fn evict_brick(&mut self, key: BrickKey) -> Option<crate::engine2::types::GpuPageHandle> {
        self.page_table.evict(key)
    }

    pub fn context_and_buffers(&self) -> Option<(&GpuContext, &BufferPool)> {
        Some((self.context.as_ref()?, self.buffers.as_ref()?))
    }

    pub fn flush(&mut self) {
        let (Some(context), Some(buffers)) = (&self.context, &self.buffers) else {
            return;
        };
        self.uploads.flush(context, buffers, &mut self.queues);
        self.queues.upload(context, buffers);
    }
}
