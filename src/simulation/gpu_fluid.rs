use std::collections::{HashMap, HashSet};

use crate::chunk_store::ChunkStore;
use crate::sim_world::Rng;
use crate::simulation::{SimulationBackend, SimulationStepMetadata, SimulationStepStats};
use crate::types::{chunk_to_world_min, voxel_to_chunk, ChunkCoord, VoxelCoord, CHUNK_SIZE_VOXELS};

const CHUNK_VOLUME: usize =
    CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize * CHUNK_SIZE_VOXELS as usize;

#[derive(Clone, Copy, Debug)]
struct SimCommand {
    coord: VoxelCoord,
    material_id: u16,
}

#[derive(Default)]
pub struct GpuFluidBackend {
    command_buffer: Vec<SimCommand>,
    frame_index: u64,
    #[cfg(feature = "gpu-compute")]
    gpu: Option<GpuFluidPipeline>,
}

impl SimulationBackend for GpuFluidBackend {
    fn step(
        &mut self,
        store: &mut ChunkStore,
        active_chunks: &HashSet<ChunkCoord>,
        _center: ChunkCoord,
        _rng: &mut Rng,
        _metadata: SimulationStepMetadata,
    ) -> SimulationStepStats {
        let mut region: HashSet<ChunkCoord> = active_chunks.iter().copied().collect();
        if region.is_empty() && !self.command_buffer.is_empty() {
            region.extend(
                self.command_buffer
                    .iter()
                    .map(|cmd| voxel_to_chunk(cmd.coord).0),
            );
        }

        #[cfg(feature = "gpu-compute")]
        {
            let stepped_chunks = self.step_gpu_native(store, &region);
            self.frame_index = self.frame_index.wrapping_add(1);
            return SimulationStepStats {
                stepped_chunks,
                ..SimulationStepStats::default()
            };
        }

        #[cfg(not(feature = "gpu-compute"))]
        {
            for cmd in self.command_buffer.drain(..) {
                store.set_voxel(cmd.coord, cmd.material_id);
            }
            self.frame_index = self.frame_index.wrapping_add(1);
            SimulationStepStats {
                stepped_chunks: region.len(),
                ..SimulationStepStats::default()
            }
        }
    }

    fn queue_place_edit(&mut self, coord: VoxelCoord, material_id: u16) {
        self.command_buffer.push(SimCommand { coord, material_id });
    }
}

#[cfg(feature = "gpu-compute")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidSimParams {
    cell_count: u32,
    jacobi_iterations: u32,
    edit_count: u32,
    _pad: u32,
}

#[cfg(feature = "gpu-compute")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEdit {
    index: u32,
    material: u32,
}

#[cfg(feature = "gpu-compute")]
struct GpuFluidPipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    force_advect_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    jacobi_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    material_advect_pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    edit_buffer: wgpu::Buffer,
    velocity_u: wgpu::Buffer,
    velocity_v: wgpu::Buffer,
    velocity_w: wgpu::Buffer,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    material_state: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity_cells: usize,
    chunk_slots: HashMap<ChunkCoord, u32>,
}

#[cfg(feature = "gpu-compute")]
impl GpuFluidBackend {
    fn step_gpu_native(&mut self, store: &mut ChunkStore, region: &HashSet<ChunkCoord>) -> usize {
        if region.is_empty() {
            self.command_buffer.clear();
            return 0;
        }
        let commands = std::mem::take(&mut self.command_buffer);
        let Some(gpu) = self.ensure_gpu() else {
            Self::apply_queued_edits_to_store(store, commands);
            return region.len();
        };

        gpu.ensure_slots(store, region);
        let total_cells = region.len() * CHUNK_VOLUME;
        gpu.ensure_capacity(total_cells.max(1));

        let edits = gpu.translate_edits(&commands);
        gpu.dispatch_pipeline(total_cells as u32, &edits, 24);
        Self::apply_queued_edits_to_store(store, commands);
        region.len()
    }

    fn ensure_gpu(&mut self) -> Option<&mut GpuFluidPipeline> {
        if self.gpu.is_none() {
            self.gpu = GpuFluidPipeline::new();
        }
        self.gpu.as_mut()
    }

    fn apply_queued_edits_to_store(store: &mut ChunkStore, commands: Vec<SimCommand>) {
        for cmd in commands {
            let below = VoxelCoord {
                x: cmd.coord.x,
                y: cmd.coord.y.saturating_sub(1),
                z: cmd.coord.z,
            };
            if below != cmd.coord && store.get_voxel(below) == crate::world::EMPTY {
                store.set_voxel(below, cmd.material_id);
                store.set_voxel(cmd.coord, crate::world::EMPTY);
            } else {
                store.set_voxel(cmd.coord, cmd.material_id);
            }
        }
    }
}

#[cfg(feature = "gpu-compute")]
impl GpuFluidPipeline {
    fn new() -> Option<Self> {
        use wgpu::util::DeviceExt;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("gpu-fluid-backend-device"),
            },
            None,
        ))
        .ok()?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-fluid-backend-bgl"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let generated_material_ids_wgsl =
            include_str!(concat!(env!("OUT_DIR"), "/material_ids.wgsl"));
        let pipeline_shader_source = format!(
            "{}\n{}",
            generated_material_ids_wgsl,
            include_str!("../shaders/gpu_fluid_pipeline.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-fluid-pipeline-shader"),
            source: wgpu::ShaderSource::Wgsl(pipeline_shader_source.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-fluid-backend-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let force_advect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("force_advect"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "force_advect",
            });
        let divergence_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_divergence"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "compute_divergence",
            });
        let jacobi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jacobi_pressure"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "jacobi_pressure",
        });
        let project_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("project_velocity"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "project_velocity",
        });
        let material_advect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("advect_material"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "advect_material",
            });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-fluid-params"),
            contents: bytemuck::bytes_of(&FluidSimParams {
                cell_count: 1,
                jacobi_iterations: 24,
                edit_count: 0,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-edits"),
            size: 1024 * std::mem::size_of::<GpuEdit>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let new_storage_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let dummy_velocity_u = new_storage_buffer("gpu-fluid-dummy-velocity-u");
        let dummy_velocity_v = new_storage_buffer("gpu-fluid-dummy-velocity-v");
        let dummy_velocity_w = new_storage_buffer("gpu-fluid-dummy-velocity-w");
        let dummy_pressure = new_storage_buffer("gpu-fluid-dummy-pressure");
        let dummy_divergence = new_storage_buffer("gpu-fluid-dummy-divergence");
        let dummy_material = new_storage_buffer("gpu-fluid-dummy-material");

        let bind_group = create_bind_group(
            &device,
            &bind_group_layout,
            &dummy_velocity_u,
            &dummy_velocity_v,
            &dummy_velocity_w,
            &dummy_pressure,
            &dummy_divergence,
            &dummy_material,
            &params_buffer,
            &edit_buffer,
        );

        Some(Self {
            device,
            queue,
            bind_group_layout,
            force_advect_pipeline,
            divergence_pipeline,
            jacobi_pipeline,
            project_pipeline,
            material_advect_pipeline,
            params_buffer,
            edit_buffer,
            velocity_u: dummy_velocity_u,
            velocity_v: dummy_velocity_v,
            velocity_w: dummy_velocity_w,
            pressure: dummy_pressure,
            divergence: dummy_divergence,
            material_state: dummy_material,
            bind_group,
            capacity_cells: 1,
            chunk_slots: HashMap::new(),
        })
    }

    fn ensure_slots(&mut self, store: &ChunkStore, region: &HashSet<ChunkCoord>) {
        for &chunk in region {
            if self.chunk_slots.contains_key(&chunk) {
                continue;
            }
            let slot = self.chunk_slots.len() as u32;
            self.chunk_slots.insert(chunk, slot);
            self.upload_chunk_materials(store, chunk, slot);
        }
    }

    fn ensure_capacity(&mut self, requested_cells: usize) {
        if requested_cells <= self.capacity_cells {
            return;
        }
        let new_capacity = requested_cells.next_power_of_two();
        let size = (new_capacity * std::mem::size_of::<f32>()) as u64;
        let mat_size = (new_capacity * std::mem::size_of::<u32>()) as u64;
        self.velocity_u = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-velocity-u"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.velocity_v = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-velocity-v"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.velocity_w = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-velocity-w"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.pressure = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-pressure"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.divergence = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-divergence"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.material_state = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-fluid-material-state"),
            size: mat_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bind_group = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.velocity_u,
            &self.velocity_v,
            &self.velocity_w,
            &self.pressure,
            &self.divergence,
            &self.material_state,
            &self.params_buffer,
            &self.edit_buffer,
        );
        self.capacity_cells = new_capacity;
    }

    fn upload_chunk_materials(&self, store: &ChunkStore, chunk: ChunkCoord, slot: u32) {
        let origin = chunk_to_world_min(chunk);
        let mut dense = vec![0u32; CHUNK_VOLUME];
        for z in 0..CHUNK_SIZE_VOXELS as usize {
            for y in 0..CHUNK_SIZE_VOXELS as usize {
                for x in 0..CHUNK_SIZE_VOXELS as usize {
                    let idx = crate::chunk_store::Chunk::index(x, y, z);
                    dense[idx] = store.get_voxel(VoxelCoord {
                        x: origin.x + x as i32,
                        y: origin.y + y as i32,
                        z: origin.z + z as i32,
                    }) as u32;
                }
            }
        }
        let offset = slot as u64 * CHUNK_VOLUME as u64 * std::mem::size_of::<u32>() as u64;
        self.queue
            .write_buffer(&self.material_state, offset, bytemuck::cast_slice(&dense));
    }

    fn translate_edits(&self, command_buffer: &[SimCommand]) -> Vec<GpuEdit> {
        let mut edits = Vec::new();
        for &cmd in command_buffer {
            let (chunk, local) = voxel_to_chunk(cmd.coord);
            let Some(slot) = self.chunk_slots.get(&chunk).copied() else {
                continue;
            };
            let local_idx = crate::chunk_store::Chunk::index(
                local[0] as usize,
                local[1] as usize,
                local[2] as usize,
            ) as u32;
            edits.push(GpuEdit {
                index: slot * CHUNK_VOLUME as u32 + local_idx,
                material: cmd.material_id as u32,
            });
        }
        edits
    }

    fn dispatch_pipeline(&mut self, cell_count: u32, edits: &[GpuEdit], jacobi_iterations: u32) {
        self.queue
            .write_buffer(&self.edit_buffer, 0, bytemuck::cast_slice(edits));
        let params = FluidSimParams {
            cell_count,
            jacobi_iterations,
            edit_count: edits.len() as u32,
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let groups = cell_count.div_ceil(64);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-fluid-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-fluid-pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.force_advect_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.divergence_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.jacobi_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.project_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.material_advect_pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(feature = "gpu-compute")]
fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "gpu-compute")]
#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    velocity_u: &wgpu::Buffer,
    velocity_v: &wgpu::Buffer,
    velocity_w: &wgpu::Buffer,
    pressure: &wgpu::Buffer,
    divergence: &wgpu::Buffer,
    material_state: &wgpu::Buffer,
    params: &wgpu::Buffer,
    edits: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-fluid-backend-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: velocity_u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: velocity_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: velocity_w.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pressure.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: divergence.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: material_state.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: edits.as_entire_binding(),
            },
        ],
    })
}
