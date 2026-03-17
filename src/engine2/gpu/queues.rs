//! GPU-owned hot queues represented in phase 1 as schema-only CPU mirrors.

#[derive(Debug, Default)]
pub struct HotQueues {
    pub active_bricks: u32,
    pub dirty_bricks: u32,
    pub remesh_bricks: u32,
    pub draw_indirect_count: u32,
}
