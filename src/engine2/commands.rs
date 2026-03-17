//! Explicit command stream for engine2 world/simulation orchestration.

use std::collections::VecDeque;

use bytemuck::{bytes_of, Pod, Zeroable};

use crate::engine2::gpu::buffers::BufferPool;
use crate::engine2::gpu::context::GpuContext;
use crate::engine2::types::VoxelCoord;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditSphereCommand {
    pub center: VoxelCoord,
    pub radius_voxels: f32,
    pub material: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditBoxCommand {
    pub min: VoxelCoord,
    pub max: VoxelCoord,
    pub material: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditCommand {
    Sphere(EditSphereCommand),
    Box(EditBoxCommand),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EngineCommand {
    EditSphere(EditSphereCommand),
    EditBox(EditBoxCommand),
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOpcode {
    Sphere = 1,
    Box = 2,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct EditCommandStreamHeader {
    pub command_count: u32,
    pub command_stride_bytes: u32,
    pub version: u32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEditCommand {
    pub opcode: u32,
    pub material: u32,
    pub p0: [i32; 4],
    pub p1: [i32; 4],
}

impl GpuEditCommand {
    fn from_edit(edit: EditCommand) -> Self {
        match edit {
            EditCommand::Sphere(sphere) => Self {
                opcode: EditOpcode::Sphere as u32,
                material: u32::from(sphere.material),
                p0: [
                    sphere.center.x,
                    sphere.center.y,
                    sphere.center.z,
                    sphere.radius_voxels.to_bits() as i32,
                ],
                p1: [0; 4],
            },
            EditCommand::Box(edit_box) => Self {
                opcode: EditOpcode::Box as u32,
                material: u32::from(edit_box.material),
                p0: [edit_box.min.x, edit_box.min.y, edit_box.min.z, 0],
                p1: [edit_box.max.x, edit_box.max.y, edit_box.max.z, 0],
            },
        }
    }
}

#[derive(Debug, Default)]
pub struct PackedEditCommands {
    pub header: EditCommandStreamHeader,
    pub commands: Vec<GpuEditCommand>,
}

impl PackedEditCommands {
    pub fn from_edits(edits: &[EditCommand]) -> Self {
        let mut commands = Vec::with_capacity(edits.len());
        for edit in edits {
            commands.push(GpuEditCommand::from_edit(*edit));
        }
        Self {
            header: EditCommandStreamHeader {
                command_count: commands.len() as u32,
                command_stride_bytes: std::mem::size_of::<GpuEditCommand>() as u32,
                version: 1,
                _pad0: 0,
            },
            commands,
        }
    }

    pub fn upload(&self, gpu: &GpuContext, buffers: &BufferPool) {
        gpu.queue()
            .write_buffer(&buffers.command_upload, 0, bytes_of(&self.header));
        if !self.commands.is_empty() {
            let offset = std::mem::size_of::<EditCommandStreamHeader>() as u64;
            gpu.queue().write_buffer(
                &buffers.command_upload,
                offset,
                bytemuck::cast_slice(&self.commands),
            );
        }
    }
}

#[derive(Debug, Default)]
pub struct CommandQueue {
    items: VecDeque<EngineCommand>,
}

impl CommandQueue {
    pub fn push(&mut self, command: EngineCommand) {
        self.items.push_back(command);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn drain_edit_commands(&mut self, max_commands: usize) -> Vec<EditCommand> {
        let mut edits = Vec::new();
        while edits.len() < max_commands {
            let Some(command) = self.items.pop_front() else {
                break;
            };
            let edit = match command {
                EngineCommand::EditSphere(cmd) => EditCommand::Sphere(cmd),
                EngineCommand::EditBox(cmd) => EditCommand::Box(cmd),
            };
            edits.push(edit);
        }
        edits
    }
}
