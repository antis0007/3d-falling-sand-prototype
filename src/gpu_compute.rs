use anyhow::Context;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::renderer::{ChunkMeshArtifact, MeshJob, Vertex};
use crate::types::GpuPageIndex;

pub const COMPUTE_STORAGE_BINDING_COUNT: u32 = 13;

// Keep these within common storage binding limits; slot sizing is derived in renderer.rs.
pub const GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES: u64 = 128 * 1024 * 1024; // 128 MiB
pub const GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
pub struct DrawIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
pub struct ChunkMeshMeta {
    pub dummy: u32,
}

// Renderer expects these.
pub fn gpu_page_capacity() -> u32 {
    256
}
pub fn mesh_pool_slot_capacity() -> u32 {
    128
}

pub fn required_storage_buffer_binding_size_bytes() -> u64 {
    GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES.max(GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshPipelineBackend {
    Disabled,
    Cpu,
    Gpu,
}

pub struct SharedMeshBuffers {
    pub chunk_vertex_buffer: Arc<wgpu::Buffer>,
    pub chunk_index_buffer: Arc<wgpu::Buffer>,
    pub draw_indirect_buffer: Arc<wgpu::Buffer>,
    pub page_indirect: Arc<wgpu::Buffer>,
    pub mesh_meta_buffer: Arc<wgpu::Buffer>,
    pub face_mask_buffer: Arc<wgpu::Buffer>,
    pub face_offset_buffer: Arc<wgpu::Buffer>,
    pub face_count_buffer: Arc<wgpu::Buffer>,
}

struct Runtime {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: SharedMeshBuffers,
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub struct GpuComputeRuntime;

impl GpuComputeRuntime {
    pub fn runtime_supported(adapter: &wgpu::Adapter, device_limits: &wgpu::Limits) -> bool {
        let _ = adapter;
        device_limits.max_storage_buffers_per_shader_stage >= COMPUTE_STORAGE_BINDING_COUNT
            && (device_limits.max_storage_buffer_binding_size as u64)
                >= required_storage_buffer_binding_size_bytes()
            && (device_limits.max_buffer_size as u64) >= required_storage_buffer_binding_size_bytes()
    }
}

pub fn initialize_gpu_compute_worker(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: SharedMeshBuffers,
) -> anyhow::Result<()> {
    RUNTIME
        .set(Runtime {
            device,
            queue,
            buffers,
        })
        .map_err(|_| anyhow::anyhow!("gpu_compute runtime already initialized"))?;
    Ok(())
}

pub fn update_gpu_page_fences_on_renderer() {
    // Stub hook for future real compute fencing.
}

pub fn dispatch_gpu_chunk_tasks_on_renderer(_max_tasks: u32) -> anyhow::Result<()> {
    // Stub hook for future real compute dispatch.
    Ok(())
}

pub struct ChunkJobOutput {
    pub mesh_artifact: ChunkMeshArtifact,
}

pub fn cpu_generate_material_field(job: &MeshJob) -> ChunkJobOutput {
    let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
        crate::renderer::mesh_chunk_snapshot(job.coord, &job.snapshot, job.lod, job.greedy);

    ChunkJobOutput {
        mesh_artifact: ChunkMeshArtifact::Cpu {
            indirect: DrawIndirectArgs {
                index_count: inds.len() as u32,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            verts,
            inds,
            aabb_min,
            aabb_max,
            chunk_origin_world,
        },
    }
}

/// GPU backend worker entrypoint.
/// Contract:
/// - uses `job.mesh_slot` (renderer-owned) for where to write geometry
/// - writes vertex/index bytes into the shared global buffers
/// - returns metadata with correct `index_count`
/// - renderer will author the indirect command (so no 0-index bug)
pub fn run_chunk_job_on_worker(job: &MeshJob) -> anyhow::Result<ChunkJobOutput> {
    let rt = RUNTIME.get().context("gpu_compute runtime not initialized")?;

    // Mesh on CPU for now; still runs on background worker threads and
    // writes directly into the global GPU buffers at the renderer-owned slot.
    let t0 = Instant::now();
    let (verts, inds, aabb_min, aabb_max, chunk_origin_world) =
        crate::renderer::mesh_chunk_snapshot(job.coord, &job.snapshot, job.lod, job.greedy);
    let dispatch_ms = t0.elapsed().as_secs_f32() * 1000.0;

    let slot = job.mesh_slot;
    if slot as usize >= mesh_pool_slot_capacity() as usize {
        return Ok(ChunkJobOutput {
            mesh_artifact: ChunkMeshArtifact::Skipped {
                reason: crate::renderer::MeshSkipReason::MeshSlotCapacitySaturated {
                    slot_capacity: mesh_pool_slot_capacity(),
                    in_flight_fences: 0,
                },
            },
        });
    }

    // Layout matches renderer.rs helpers (bytes-per-slot derived from globals).
    let v_bytes_per_slot = (GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES / mesh_pool_slot_capacity() as u64).max(1);
    let i_bytes_per_slot = (GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES / mesh_pool_slot_capacity() as u64).max(1);

    let v_off = slot as u64 * v_bytes_per_slot;
    let i_off = slot as u64 * i_bytes_per_slot;

    let v_bytes = (verts.len() * std::mem::size_of::<Vertex>()) as u64;
    let i_bytes = (inds.len() * std::mem::size_of::<u32>()) as u64;

    if v_bytes > v_bytes_per_slot || i_bytes > i_bytes_per_slot {
        return Ok(ChunkJobOutput {
            mesh_artifact: ChunkMeshArtifact::Failed {
                reason: format!(
                    "mesh overflow slot={} v_bytes={} (cap={}) i_bytes={} (cap={})",
                    slot, v_bytes, v_bytes_per_slot, i_bytes, i_bytes_per_slot
                ),
            },
        });
    }

    if !verts.is_empty() {
        rt.queue
            .write_buffer(&rt.buffers.chunk_vertex_buffer, v_off, bytemuck::cast_slice(&verts));
    }
    if !inds.is_empty() {
        rt.queue
            .write_buffer(&rt.buffers.chunk_index_buffer, i_off, bytemuck::cast_slice(&inds));
    }

    Ok(ChunkJobOutput {
        mesh_artifact: ChunkMeshArtifact::Gpu {
            page_index: GpuPageIndex(slot.min(gpu_page_capacity() - 1)),
            index_count: inds.len() as u32,
            lod: job.lod as u8,
            aabb_min,
            aabb_max,
            chunk_origin_world,
            dispatch_ms,
        },
    })
}