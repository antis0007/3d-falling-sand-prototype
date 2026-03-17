//! Explicit command stream for engine2 world/simulation orchestration.

use std::collections::VecDeque;

use crate::engine2::types::{BrickKey, VoxelCoord};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadBrickCommand {
    pub brick: BrickKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnloadBrickCommand {
    pub brick: BrickKey,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditSphereCommand {
    pub center: VoxelCoord,
    pub radius_voxels: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditBoxCommand {
    pub min: VoxelCoord,
    pub max: VoxelCoord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InjectMaterialCommand {
    pub brick: BrickKey,
    pub voxel_index: u16,
    pub material: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EngineCommand {
    LoadBrick(LoadBrickCommand),
    UnloadBrick(UnloadBrickCommand),
    EditSphere(EditSphereCommand),
    EditBox(EditBoxCommand),
    InjectMaterial(InjectMaterialCommand),
}

#[derive(Debug, Default)]
pub struct CommandQueue {
    items: VecDeque<EngineCommand>,
}

impl CommandQueue {
    pub fn push(&mut self, command: EngineCommand) {
        self.items.push_back(command);
    }

    pub fn pop(&mut self) -> Option<EngineCommand> {
        self.items.pop_front()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}
