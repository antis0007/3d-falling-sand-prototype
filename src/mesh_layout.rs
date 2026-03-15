pub const GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES: u64 = 512 * 1024 * 1024;
pub const GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES: u64 = 256 * 1024 * 1024;
// Surgical contract repair: keep current global buffers and per-chunk maxima,
// and reduce slot cardinality so one full chunk fits into one deterministic slot.
pub const MESH_SLOT_COUNT: u32 = 64;
pub const CHUNK_VOLUME_VOXELS: u32 = 32 * 32 * 32;
pub const MESH_VERTEX_ELEMENTS_PER_CHUNK: u32 = CHUNK_VOLUME_VOXELS * 12;
pub const MESH_INDEX_ELEMENTS_PER_CHUNK: u32 = CHUNK_VOLUME_VOXELS * 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshBufferKind {
    Vertex,
    Index,
}

impl MeshBufferKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Index => "index",
        }
    }

    pub const fn global_size_bytes(self) -> u64 {
        match self {
            Self::Vertex => GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES,
            Self::Index => GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES,
        }
    }

    pub const fn element_size_bytes(self) -> u64 {
        match self {
            Self::Vertex => std::mem::size_of::<crate::renderer::Vertex>() as u64,
            Self::Index => std::mem::size_of::<u32>() as u64,
        }
    }

    pub const fn global_capacity_elements(self) -> u32 {
        (self.global_size_bytes() / self.element_size_bytes()) as u32
    }

    pub const fn slot_capacity_elements(self) -> u32 {
        self.global_capacity_elements() / MESH_SLOT_COUNT
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshWriteRange {
    pub offset_elements: u32,
    pub count_elements: u32,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSlotWriteRanges {
    pub vertex: MeshWriteRange,
    pub index: MeshWriteRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSliceContract {
    pub slot_count: u32,
    pub vertex_elements_per_chunk: u32,
    pub index_elements_per_chunk: u32,
    pub vertex_global_capacity_elements: u32,
    pub index_global_capacity_elements: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSliceLayout {
    pub contract: MeshSliceContract,
    pub vertex_slot_capacity_elements: u32,
    pub index_slot_capacity_elements: u32,
}

pub const fn gpu_mesh_slice_contract() -> MeshSliceContract {
    MeshSliceContract {
        slot_count: MESH_SLOT_COUNT,
        vertex_elements_per_chunk: MESH_VERTEX_ELEMENTS_PER_CHUNK,
        index_elements_per_chunk: MESH_INDEX_ELEMENTS_PER_CHUNK,
        vertex_global_capacity_elements: MeshBufferKind::Vertex.global_capacity_elements(),
        index_global_capacity_elements: MeshBufferKind::Index.global_capacity_elements(),
    }
}

pub fn validate_gpu_mesh_slice_contract() -> Result<MeshSliceLayout, String> {
    let contract = gpu_mesh_slice_contract();
    if contract.slot_count == 0 {
        return Err("gpu mesh slice contract invalid: slot_count must be > 0".to_string());
    }

    let vertex_slot_capacity = MeshBufferKind::Vertex.slot_capacity_elements();
    let index_slot_capacity = MeshBufferKind::Index.slot_capacity_elements();

    if contract.vertex_elements_per_chunk > vertex_slot_capacity {
        return Err(format!(
            "gpu mesh slice contract invalid: vertex per chunk {} exceeds per-slot vertex capacity {} (slots={} vertex_global_capacity={})",
            contract.vertex_elements_per_chunk,
            vertex_slot_capacity,
            contract.slot_count,
            contract.vertex_global_capacity_elements,
        ));
    }

    if contract.index_elements_per_chunk > index_slot_capacity {
        return Err(format!(
            "gpu mesh slice contract invalid: index per chunk {} exceeds per-slot index capacity {} (slots={} index_global_capacity={})",
            contract.index_elements_per_chunk,
            index_slot_capacity,
            contract.slot_count,
            contract.index_global_capacity_elements,
        ));
    }

    let last_slot = contract.slot_count.saturating_sub(1);
    for (kind, per_chunk, global_capacity) in [
        (
            MeshBufferKind::Vertex,
            contract.vertex_elements_per_chunk,
            contract.vertex_global_capacity_elements,
        ),
        (
            MeshBufferKind::Index,
            contract.index_elements_per_chunk,
            contract.index_global_capacity_elements,
        ),
    ] {
        let Some(base) = slot_base_offset_elements(kind, last_slot) else {
            return Err(format!(
                "gpu mesh slice contract invalid: {label} last-slot base offset unavailable (slot={} slots={})",
                last_slot,
                contract.slot_count,
                label = kind.label(),
            ));
        };
        let Some(limit) = base.checked_add(per_chunk) else {
            return Err(format!(
                "gpu mesh slice contract invalid: {label} offset overflow in last slot (slot={} base={} count={})",
                last_slot,
                base,
                per_chunk,
                label = kind.label(),
            ));
        };
        if limit > global_capacity {
            return Err(format!(
                "gpu mesh slice contract invalid: {label} last-slot write overruns global capacity (slot={} base={} count={} limit={} global={})",
                last_slot,
                base,
                per_chunk,
                limit,
                global_capacity,
                label = kind.label(),
            ));
        }
    }

    Ok(MeshSliceLayout {
        contract,
        vertex_slot_capacity_elements: vertex_slot_capacity,
        index_slot_capacity_elements: index_slot_capacity,
    })
}

fn checked_range(
    kind: MeshBufferKind,
    offset_elements: u32,
    count_elements: u32,
) -> Option<MeshWriteRange> {
    let element_size = kind.element_size_bytes();
    let offset_bytes = (offset_elements as u64).checked_mul(element_size)?;
    let size_bytes = (count_elements as u64).checked_mul(element_size)?;
    let end_bytes = offset_bytes.checked_add(size_bytes)?;
    if end_bytes > kind.global_size_bytes() {
        return None;
    }
    Some(MeshWriteRange {
        offset_elements,
        count_elements,
        offset_bytes,
        size_bytes,
    })
}

pub fn validate_element_range(
    kind: MeshBufferKind,
    offset_elements: u32,
    count_elements: u32,
) -> Option<MeshWriteRange> {
    checked_range(kind, offset_elements, count_elements)
}

pub fn slot_base_offset_elements(kind: MeshBufferKind, slot_index: u32) -> Option<u32> {
    if slot_index >= MESH_SLOT_COUNT {
        return None;
    }
    slot_index.checked_mul(kind.slot_capacity_elements())
}

pub fn validate_slot_write_ranges(
    slot_index: u32,
    vertex_count: u32,
    index_count: u32,
) -> Option<MeshSlotWriteRanges> {
    let vertex_slot_capacity = MeshBufferKind::Vertex.slot_capacity_elements();
    let index_slot_capacity = MeshBufferKind::Index.slot_capacity_elements();
    if vertex_count > vertex_slot_capacity || index_count > index_slot_capacity {
        return None;
    }

    let vertex_offset = slot_base_offset_elements(MeshBufferKind::Vertex, slot_index)?;
    let index_offset = slot_base_offset_elements(MeshBufferKind::Index, slot_index)?;

    let vertex = checked_range(MeshBufferKind::Vertex, vertex_offset, vertex_count)?;
    let index = checked_range(MeshBufferKind::Index, index_offset, index_count)?;

    Some(MeshSlotWriteRanges { vertex, index })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_base_offsets_stay_in_bounds_for_all_slots() {
        for slot in 0..MESH_SLOT_COUNT {
            let vertex_base = slot_base_offset_elements(MeshBufferKind::Vertex, slot).unwrap();
            let index_base = slot_base_offset_elements(MeshBufferKind::Index, slot).unwrap();

            let vertex_range = validate_element_range(
                MeshBufferKind::Vertex,
                vertex_base,
                MeshBufferKind::Vertex.slot_capacity_elements(),
            )
            .unwrap();
            let index_range = validate_element_range(
                MeshBufferKind::Index,
                index_base,
                MeshBufferKind::Index.slot_capacity_elements(),
            )
            .unwrap();

            assert!(
                vertex_range.offset_bytes + vertex_range.size_bytes
                    <= MeshBufferKind::Vertex.global_size_bytes()
            );
            assert!(
                index_range.offset_bytes + index_range.size_bytes
                    <= MeshBufferKind::Index.global_size_bytes()
            );
        }
    }

    #[test]
    fn invalid_slot_or_capacity_is_rejected() {
        assert!(validate_slot_write_ranges(MESH_SLOT_COUNT, 1, 1).is_none());
        assert!(validate_slot_write_ranges(
            0,
            MeshBufferKind::Vertex
                .slot_capacity_elements()
                .saturating_add(1),
            1
        )
        .is_none());
        assert!(validate_slot_write_ranges(
            0,
            1,
            MeshBufferKind::Index
                .slot_capacity_elements()
                .saturating_add(1)
        )
        .is_none());
    }

    #[test]
    fn per_slot_capacity_matches_global_capacity_math() {
        assert_eq!(
            MeshBufferKind::Vertex.slot_capacity_elements() * MESH_SLOT_COUNT,
            MeshBufferKind::Vertex.global_capacity_elements()
        );
        assert_eq!(
            MeshBufferKind::Index.slot_capacity_elements() * MESH_SLOT_COUNT,
            MeshBufferKind::Index.global_capacity_elements()
        );
    }

    #[test]
    fn gpu_contract_chunk_capacity_fits_each_slot() {
        let contract = gpu_mesh_slice_contract();
        assert_eq!(contract.slot_count, MESH_SLOT_COUNT);
        assert!(
            contract.vertex_elements_per_chunk <= MeshBufferKind::Vertex.slot_capacity_elements()
        );
        assert!(
            contract.index_elements_per_chunk <= MeshBufferKind::Index.slot_capacity_elements()
        );
    }

    #[test]
    fn slot_offsets_match_authoritative_contract_layout() {
        let contract = gpu_mesh_slice_contract();
        for slot in 0..contract.slot_count {
            let vertex = validate_slot_write_ranges(
                slot,
                contract.vertex_elements_per_chunk,
                contract.index_elements_per_chunk,
            )
            .expect("contract-sized slot range must validate");
            assert_eq!(
                vertex.vertex.offset_elements,
                slot * MeshBufferKind::Vertex.slot_capacity_elements()
            );
            assert_eq!(
                vertex.index.offset_elements,
                slot * MeshBufferKind::Index.slot_capacity_elements()
            );
        }
    }

    #[test]
    fn gpu_contract_validation_succeeds_and_matches_capacities() {
        let layout = validate_gpu_mesh_slice_contract().expect("gpu contract must be satisfiable");
        assert_eq!(layout.contract.slot_count, MESH_SLOT_COUNT);
        assert_eq!(
            layout.vertex_slot_capacity_elements,
            MeshBufferKind::Vertex.slot_capacity_elements()
        );
        assert_eq!(
            layout.index_slot_capacity_elements,
            MeshBufferKind::Index.slot_capacity_elements()
        );
        assert!(layout.contract.vertex_elements_per_chunk <= layout.vertex_slot_capacity_elements);
        assert!(layout.contract.index_elements_per_chunk <= layout.index_slot_capacity_elements);
    }

    #[test]
    fn no_valid_slot_path_can_overrun_global_buffers() {
        for slot in 0..MESH_SLOT_COUNT {
            let ranges = validate_slot_write_ranges(
                slot,
                MeshBufferKind::Vertex.slot_capacity_elements(),
                MeshBufferKind::Index.slot_capacity_elements(),
            )
            .unwrap();
            assert!(
                ranges.vertex.offset_bytes + ranges.vertex.size_bytes
                    <= MeshBufferKind::Vertex.global_size_bytes()
            );
            assert!(
                ranges.index.offset_bytes + ranges.index.size_bytes
                    <= MeshBufferKind::Index.global_size_bytes()
            );
        }
    }
}
