//! Procgen integration surface for engine2 world materialization.
//!
//! Ownership boundary: this module describes requests/results exchanged with a future
//! procgen backend, while leaving generation algorithms in legacy code until later phases.

use std::collections::VecDeque;

use crate::engine2::types::BrickKey;
use crate::engine2::world::brick::{BrickPayload, BRICK_EDGE, BRICK_VOXEL_CAPACITY};

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

    pub fn synthesize_placeholder(&mut self, request: ProcgenRequest) {
        let mut payload = BrickPayload::default();
        let base_material = if request.brick.y <= 0 { 1 } else { 0 };
        if base_material != 0 {
            payload.material_ids.fill(base_material);
        }

        // Stamp a tiny checker on the top face so edits and uploads are visually obvious.
        if request.brick.y == 0 {
            for z in 0..BRICK_EDGE as usize {
                for x in 0..BRICK_EDGE as usize {
                    if (x + z) % 2 == 0 {
                        let idx =
                            ((BRICK_EDGE as usize - 1) * BRICK_EDGE as usize * BRICK_EDGE as usize)
                                + z * BRICK_EDGE as usize
                                + x;
                        if idx < BRICK_VOXEL_CAPACITY {
                            payload.material_ids[idx] = 2;
                        }
                    }
                }
            }
        }

        self.submit_result(ProcgenResult {
            brick: request.brick,
            payload,
        });
    }
}
