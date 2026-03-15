pub const GLOBAL_MESH_VERTEX_BUFFER_SIZE_BYTES: u64 = 512 * 1024 * 1024;
pub const GLOBAL_MESH_INDEX_BUFFER_SIZE_BYTES: u64 = 256 * 1024 * 1024;
pub const MESH_SLOT_COUNT: u32 = 256;
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

pub const fn gpu_mesh_slice_contract() -> MeshSliceContract {
    MeshSliceContract {
        slot_count: MESH_SLOT_COUNT,
        vertex_elements_per_chunk: MESH_VERTEX_ELEMENTS_PER_CHUNK,
        index_elements_per_chunk: MESH_INDEX_ELEMENTS_PER_CHUNK,
        vertex_global_capacity_elements: MeshBufferKind::Vertex.global_capacity_elements(),
        index_global_capacity_elements: MeshBufferKind::Index.global_capacity_elements(),
    }
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
