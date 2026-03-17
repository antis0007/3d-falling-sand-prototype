//! Upload staging for command-driven brick updates.

#[derive(Debug, Default)]
pub struct UploadQueue {
    pub pending_bytes: u64,
}
