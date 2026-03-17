//! Procgen integration surface for engine2 world materialization.
//!
//! Ownership boundary: this module describes requests/results exchanged with a future
//! procgen backend, while leaving generation algorithms in legacy code until later phases.

use std::collections::VecDeque;

use crate::engine2::types::BrickKey;
use crate::engine2::world::brick::BrickPayload;

#[derive(Debug, Clone)]
pub struct ProcgenRequest {
    pub brick: BrickKey,
}

#[derive(Debug, Clone)]
pub struct ProcgenResult {
    pub brick: BrickKey,
    pub payload: BrickPayload,
}

#[derive(Debug, Default)]
pub struct ProcgenInterface {
    pending: VecDeque<ProcgenRequest>,
    completed: VecDeque<ProcgenResult>,
}

impl ProcgenInterface {
    pub fn request(&mut self, brick: BrickKey) {
        self.pending.push_back(ProcgenRequest { brick });
    }

    pub fn pop_request(&mut self) -> Option<ProcgenRequest> {
        self.pending.pop_front()
    }

    pub fn submit_result(&mut self, result: ProcgenResult) {
        self.completed.push_back(result);
    }

    pub fn pop_completed(&mut self) -> Option<ProcgenResult> {
        self.completed.pop_front()
    }
}
