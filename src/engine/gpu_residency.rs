//! GPU residency ownership domain.
//! The renderer consumes residency state but does not own allocation identity.

use crate::engine::artifacts::{MeshArtifactHandle, MeshArtifactKey};
use crate::engine::render::FrameEpoch;
use crate::types::GpuPageIndex;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GpuResourceHandle(pub u64);

impl GpuResourceHandle {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GpuResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResidencyState {
    #[default]
    Unrequested,
    UploadQueued,
    Resident,
    EvictPending,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ResidencyUploadMetadata {
    pub page_index: Option<GpuPageIndex>,
    pub draw_indirect_index: Option<u32>,
    pub lod: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct ResidencyRecord {
    pub artifact_key: MeshArtifactKey,
    pub artifact_handle: Option<MeshArtifactHandle>,
    pub state: ResidencyState,
    pub gpu_resource: Option<GpuResourceHandle>,
    pub upload: ResidencyUploadMetadata,
    pub last_referenced_frame: Option<FrameEpoch>,
    pub evict_requested_frame: Option<FrameEpoch>,
}

impl ResidencyRecord {
    #[inline]
    pub fn is_resident(&self) -> bool {
        self.state == ResidencyState::Resident && self.gpu_resource.is_some()
    }
}

#[derive(Default)]
pub struct GpuResidencyManager {
    records: HashMap<MeshArtifactKey, ResidencyRecord>,
    resource_to_artifact: HashMap<GpuResourceHandle, MeshArtifactKey>,
    next_resource_handle: u64,
    completed_frame_epoch: FrameEpoch,
    submitted_frame_references: BTreeMap<FrameEpoch, BTreeSet<GpuResourceHandle>>,
}

impl GpuResidencyManager {
    pub fn enqueue_artifact_upload(
        &mut self,
        artifact_key: MeshArtifactKey,
        artifact_handle: Option<MeshArtifactHandle>,
        upload: ResidencyUploadMetadata,
    ) -> ResidencyRecord {
        let record = self.records.entry(artifact_key).or_insert(ResidencyRecord {
            artifact_key,
            artifact_handle,
            state: ResidencyState::Unrequested,
            gpu_resource: None,
            upload,
            last_referenced_frame: None,
            evict_requested_frame: None,
        });

        record.artifact_handle = artifact_handle.or(record.artifact_handle);
        record.upload = upload;
        if matches!(
            record.state,
            ResidencyState::Unrequested | ResidencyState::Released
        ) {
            record.state = ResidencyState::UploadQueued;
        }
        *record
    }

    pub fn mark_upload_complete_resident(
        &mut self,
        artifact_key: MeshArtifactKey,
        artifact_handle: Option<MeshArtifactHandle>,
        upload: ResidencyUploadMetadata,
    ) -> ResidencyRecord {
        let mut record = self.enqueue_artifact_upload(artifact_key, artifact_handle, upload);
        let resource = self.allocate_resource_handle();
        record.state = ResidencyState::Resident;
        record.gpu_resource = Some(resource);
        record.evict_requested_frame = None;
        self.resource_to_artifact.insert(resource, artifact_key);
        self.records.insert(artifact_key, record);
        debug_assert!(
            record.gpu_resource.is_some(),
            "resident record must have a resource handle"
        );
        record
    }

    pub fn residency(&self, artifact_key: MeshArtifactKey) -> Option<ResidencyRecord> {
        self.records.get(&artifact_key).copied()
    }

    pub fn resolve_resident_handle(
        &self,
        artifact_key: MeshArtifactKey,
    ) -> Option<GpuResourceHandle> {
        let record = self.records.get(&artifact_key)?;
        if record.state == ResidencyState::Resident {
            return record.gpu_resource;
        }
        None
    }

    pub fn mark_resource_referenced_by_frame(
        &mut self,
        resource: GpuResourceHandle,
        frame_epoch: FrameEpoch,
    ) {
        if let Some(artifact_key) = self.resource_to_artifact.get(&resource).copied() {
            if let Some(record) = self.records.get_mut(&artifact_key) {
                debug_assert_ne!(
                    record.state,
                    ResidencyState::Released,
                    "released resource cannot be referenced"
                );
                record.last_referenced_frame = Some(frame_epoch);
            }
        }
    }

    pub fn frame_submitted(
        &mut self,
        frame_epoch: FrameEpoch,
        referenced_resources: impl IntoIterator<Item = GpuResourceHandle>,
    ) {
        let resources: Vec<_> = referenced_resources.into_iter().collect();
        {
            let refs = self
                .submitted_frame_references
                .entry(frame_epoch)
                .or_default();
            for resource in &resources {
                refs.insert(*resource);
            }
        }
        for resource in resources {
            self.mark_resource_referenced_by_frame(resource, frame_epoch);
        }
    }

    pub fn frame_completed(&mut self, frame_epoch: FrameEpoch) {
        if frame_epoch > self.completed_frame_epoch {
            self.completed_frame_epoch = frame_epoch;
        }
        self.submitted_frame_references
            .retain(|epoch, _| *epoch > self.completed_frame_epoch);
    }

    pub fn request_evict_release(
        &mut self,
        artifact_key: MeshArtifactKey,
        frame_epoch: FrameEpoch,
    ) -> Option<ResidencyRecord> {
        let record = self.records.get_mut(&artifact_key)?;
        if matches!(
            record.state,
            ResidencyState::Resident | ResidencyState::UploadQueued
        ) {
            record.state = ResidencyState::EvictPending;
            record.evict_requested_frame = Some(frame_epoch);
        }
        Some(*record)
    }

    pub fn invalidate_artifact(&mut self, artifact_key: MeshArtifactKey, frame_epoch: FrameEpoch) {
        let _ = self.request_evict_release(artifact_key, frame_epoch);
    }

    pub fn complete_releases_after_frame_safety(&mut self) -> Vec<GpuResourceHandle> {
        let mut released = Vec::new();
        for record in self.records.values_mut() {
            if record.state != ResidencyState::EvictPending {
                continue;
            }
            let safe_epoch = record.last_referenced_frame.unwrap_or_default();
            if safe_epoch > self.completed_frame_epoch {
                continue;
            }
            if let Some(resource) = record.gpu_resource.take() {
                debug_assert!(
                    !self
                        .submitted_frame_references
                        .values()
                        .any(|handles| handles.contains(&resource)),
                    "resource released while still referenced by incomplete frame"
                );
                self.resource_to_artifact.remove(&resource);
                released.push(resource);
            }
            record.state = ResidencyState::Released;
        }
        released
    }

    pub fn clear(&mut self) {
        self.records.clear();
        self.resource_to_artifact.clear();
        self.submitted_frame_references.clear();
        self.completed_frame_epoch = FrameEpoch::default();
    }

    fn allocate_resource_handle(&mut self) -> GpuResourceHandle {
        self.next_resource_handle = self.next_resource_handle.saturating_add(1);
        GpuResourceHandle(self.next_resource_handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mesh::LodLevel;
    use crate::engine::world::ChunkVersion;
    use crate::types::ChunkCoord;

    fn key() -> MeshArtifactKey {
        MeshArtifactKey {
            chunk_id: ChunkCoord { x: 1, y: 2, z: 3 },
            chunk_version: ChunkVersion(7),
            lod: LodLevel(0),
        }
    }

    #[test]
    fn resident_requires_handle() {
        let mut mgr = GpuResidencyManager::default();
        let record = mgr.mark_upload_complete_resident(
            key(),
            Some(MeshArtifactHandle(11)),
            ResidencyUploadMetadata::default(),
        );
        assert_eq!(record.state, ResidencyState::Resident);
        assert!(record.gpu_resource.is_some());
    }

    #[test]
    fn release_waits_for_frame_completion() {
        let mut mgr = GpuResidencyManager::default();
        let key = key();
        let record = mgr.mark_upload_complete_resident(
            key,
            Some(MeshArtifactHandle(3)),
            ResidencyUploadMetadata::default(),
        );
        let handle = record.gpu_resource.expect("resident handle");
        mgr.frame_submitted(FrameEpoch(3), [handle]);
        mgr.request_evict_release(key, FrameEpoch(3));
        assert!(mgr.complete_releases_after_frame_safety().is_empty());
        mgr.frame_completed(FrameEpoch(3));
        let released = mgr.complete_releases_after_frame_safety();
        assert_eq!(released, vec![handle]);
    }
}
