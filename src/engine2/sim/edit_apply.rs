//! Edit command staging and application for GPU execution.

use std::collections::HashSet;

use crate::engine2::commands::{CommandQueue, EditCommand, PackedEditCommands};
use crate::engine2::gpu::Engine2Gpu;
use crate::engine2::phases::EditOutput;
use crate::engine2::sim::scheduler::SimScheduler;
use crate::engine2::types::{voxel_to_brick_key, BrickKey, VoxelCoord};
use crate::engine2::world::brick::BrickPayload;
use crate::engine2::world::residency::ResidencyStateMap;

const MAX_EDIT_COMMANDS_PER_FRAME: usize = 4096;

pub struct EditApplier {
    pub applied_commands: u64,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl std::fmt::Debug for EditApplier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditApplier")
            .field("applied_commands", &self.applied_commands)
            .finish()
    }
}

impl Default for EditApplier {
    fn default() -> Self {
        Self {
            applied_commands: 0,
            pipeline: None,
            bind_group_layout: None,
        }
    }
}

impl EditApplier {
    pub fn apply_pending_edits(
        &mut self,
        commands: &mut CommandQueue,
        residency: &mut ResidencyStateMap,
        gpu: &mut Engine2Gpu,
        scheduler: &mut SimScheduler,
    ) -> EditOutput {
        let edits = commands.drain_edit_commands(MAX_EDIT_COMMANDS_PER_FRAME);
        if edits.is_empty() {
            return EditOutput::default();
        }

        let packed = PackedEditCommands::from_edits(&edits);
        if let Some((context, buffers)) = gpu.context_and_buffers() {
            packed.upload(context, buffers);
            self.dispatch_placeholder(context, buffers, packed.header.command_count);
        }

        let mut touched_pages = HashSet::new();
        let mut output = EditOutput::default();
        for edit in &edits {
            let material = edit_material(*edit);
            let (min, max) = touched_brick_bounds(*edit);
            for z in min.z..=max.z {
                for y in min.y..=max.y {
                    for x in min.x..=max.x {
                        let key = BrickKey { x, y, z };
                        let Some(page) = gpu.page_table.page_for(key) else {
                            continue;
                        };
                        let mut payload = BrickPayload::default();
                        payload.material_ids.fill(material);
                        gpu.enqueue_page_update(key, page, 1, payload, true);
                        residency.mark_dirty(key);
                        if touched_pages.insert(page.0) {
                            scheduler.wake_brick(page.0);
                            output.dirty_pages.push(page.0);
                            output.wake_pages.push(page.0);
                        }
                    }
                }
            }
        }

        self.applied_commands = self
            .applied_commands
            .saturating_add(packed.header.command_count as u64);
        output
    }

    fn dispatch_placeholder(
        &mut self,
        context: &crate::engine2::gpu::context::GpuContext,
        buffers: &crate::engine2::gpu::buffers::BufferPool,
        command_count: u32,
    ) {
        if command_count == 0 {
            return;
        }

        self.ensure_pipeline(context.device());
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine2.edit_apply.bind_group"),
                layout: self.bind_group_layout.as_ref().expect("layout initialized"),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.command_upload.as_entire_binding(),
                }],
            });

        let mut encoder = context.create_encoder("engine2.edit_apply.dispatch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine2.edit_apply.compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline.as_ref().expect("pipeline initialized"));
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = command_count.max(1);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        context.queue().submit(std::iter::once(encoder.finish()));
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device) {
        if self.pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine2.edit_apply.placeholder_wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0)
var<storage, read> command_stream: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        let _touch = command_stream[0u];
    }
}
"#
                .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine2.edit_apply.bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine2.edit_apply.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine2.edit_apply.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline = Some(pipeline);
    }
}

fn edit_material(edit: EditCommand) -> u16 {
    match edit {
        EditCommand::Sphere(sphere) => sphere.material,
        EditCommand::Box(edit_box) => edit_box.material,
    }
}

fn touched_brick_bounds(edit: EditCommand) -> (BrickKey, BrickKey) {
    match edit {
        EditCommand::Sphere(sphere) => {
            let radius = sphere.radius_voxels.ceil() as i32;
            let min_voxel = VoxelCoord {
                x: sphere.center.x - radius,
                y: sphere.center.y - radius,
                z: sphere.center.z - radius,
            };
            let max_voxel = VoxelCoord {
                x: sphere.center.x + radius,
                y: sphere.center.y + radius,
                z: sphere.center.z + radius,
            };
            (voxel_to_brick_key(min_voxel), voxel_to_brick_key(max_voxel))
        }
        EditCommand::Box(edit_box) => {
            let min_voxel = VoxelCoord {
                x: edit_box.min.x.min(edit_box.max.x),
                y: edit_box.min.y.min(edit_box.max.y),
                z: edit_box.min.z.min(edit_box.max.z),
            };
            let max_voxel = VoxelCoord {
                x: edit_box.min.x.max(edit_box.max.x),
                y: edit_box.min.y.max(edit_box.max.y),
                z: edit_box.min.z.max(edit_box.max.z),
            };
            (voxel_to_brick_key(min_voxel), voxel_to_brick_key(max_voxel))
        }
    }
}
