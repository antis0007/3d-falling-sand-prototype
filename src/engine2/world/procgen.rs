//! Procgen hooks for filling non-resident bricks on demand.

use crate::engine2::types::BrickKey;

#[derive(Debug, Default)]
pub struct ProcgenScheduler;

impl ProcgenScheduler {
    pub fn request(&mut self, _brick: BrickKey) {}
}
