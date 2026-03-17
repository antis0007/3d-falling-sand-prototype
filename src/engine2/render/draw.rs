//! Draw packet schema backed by GPU indirect buffers.

#[derive(Debug, Default)]
pub struct DrawPacket {
    pub indirect_count: u32,
}
