//! Command stream from CPU cold-state systems into GPU-owned hot state.

use crate::engine2::types::BrickKey;

#[derive(Debug, Clone, Copy)]
pub enum MaterialCommand {
    SetMaterial {
        brick: BrickKey,
        voxel_index: u16,
        material: u16,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum ResidencyCommand {
    LoadBrick { brick: BrickKey },
    UnloadBrick { brick: BrickKey },
}

#[derive(Debug, Clone, Copy)]
pub enum ToolCommand {
    InjectImpulse { brick: BrickKey, strength: f32 },
}
