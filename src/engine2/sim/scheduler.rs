//! High-level simulation scheduler (CPU cold control only).

#[derive(Debug, Default)]
pub struct SimScheduler {
    pub tick_index: u64,
}
