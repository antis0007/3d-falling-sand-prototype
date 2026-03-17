//! GPU-owned buffer pool declarations.

#[derive(Debug, Default)]
pub struct BufferPool {
    pub resident_brick_capacity: u32,
}
