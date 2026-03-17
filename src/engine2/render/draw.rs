//! Engine2 draw packet schema backed by GPU indirect buffers.

use bytemuck::bytes_of;

use crate::engine2::gpu::buffers::{BufferPool, DrawIndirectArgs};
use crate::engine2::gpu::context::GpuContext;
use crate::engine2::render::extract::{ExtractOutput, ExtractedPrimitive};

#[derive(Debug, Clone, Copy)]
pub struct DrawCommand {
    pub page_slot: u32,
    pub vertex_count: u32,
}

impl From<ExtractedPrimitive> for DrawCommand {
    fn from(value: ExtractedPrimitive) -> Self {
        Self {
            page_slot: value.page_slot,
            vertex_count: value.vertex_count,
        }
    }
}

#[derive(Debug, Default)]
pub struct DrawPacket {
    pub commands: Vec<DrawCommand>,
    pub indirect_count: u32,
}

#[derive(Debug, Default, Clone)]
pub struct DrawInput {
    pub extracted: Vec<ExtractedPrimitive>,
}

impl From<ExtractOutput> for DrawInput {
    fn from(value: ExtractOutput) -> Self {
        Self {
            extracted: value.extracted,
        }
    }
}

#[derive(Debug, Default)]
pub struct Engine2Drawer;

impl Engine2Drawer {
    pub fn prepare(&mut self, draw_input: DrawInput) -> DrawPacket {
        let commands: Vec<DrawCommand> = draw_input
            .extracted
            .into_iter()
            .map(DrawCommand::from)
            .collect();

        DrawPacket {
            indirect_count: commands.len() as u32,
            commands,
        }
    }

    pub fn upload_indirect(&self, gpu: (&GpuContext, &BufferPool), draw_packet: &DrawPacket) {
        let (context, buffers) = gpu;
        let indirect = DrawIndirectArgs {
            vertex_count: draw_packet
                .commands
                .iter()
                .map(|cmd| cmd.vertex_count)
                .sum(),
            instance_count: u32::from(!draw_packet.commands.is_empty()),
            first_vertex: 0,
            first_instance: 0,
        };
        context
            .queue()
            .write_buffer(&buffers.indirect_draw, 0, bytes_of(&indirect));
    }
}

#[cfg(test)]
mod tests {
    use super::Engine2Drawer;
    use crate::engine2::render::draw::DrawInput;
    use crate::engine2::render::extract::ExtractedPrimitive;

    #[test]
    fn stages_indirect_count_without_gpu_context() {
        let mut drawer = Engine2Drawer;
        let output = DrawInput {
            extracted: vec![ExtractedPrimitive {
                page_slot: 4,
                vertex_count: 36,
            }],
        };

        let packet = drawer.prepare(output);

        assert_eq!(packet.indirect_count, 1);
        assert_eq!(packet.commands[0].page_slot, 4);
    }
}
