# Macrochunk Migration Foundation

This document captures the target interfaces for a chunk-first world pipeline. This PR intentionally adds only type/module scaffolding so follow-up PRs can implement behavior incrementally while keeping the current macrochunk path intact.

## Canonical Types

- `types::MaterialId`: canonical voxel material identifier (currently re-exported from `world`).
- `types::VoxelCoord { x, y, z }`: world-space voxel coordinates.
- `types::ChunkCoord { x, y, z }`: chunk-grid coordinates (not macrochunk coordinates).
- `types::CHUNK_SIZE_VOXELS`: chunk edge length in voxels (currently mirrors `world::CHUNK_SIZE`).

Helpers:

- `voxel_to_chunk(voxel) -> (ChunkCoord, [u32; 3])`
  - floor-divides world voxel coordinates into chunk coordinates
  - returns non-negative local coordinates (`0..CHUNK_SIZE_VOXELS`)
- `chunk_to_world_min(chunk) -> VoxelCoord`
  - returns the minimum world voxel coordinate covered by a chunk

## ChunkStore API Draft

`chunk_store::ChunkStore` is intended to become the authoritative sparse chunk map.

Planned primary API:

- `get_voxel(coord) -> Option<MaterialId>`
- `set_voxel(coord, material)`
- `get_chunk(coord) -> Option<&Chunk>`
- `mark_dirty(coord)`

## Streaming API Draft

`streaming::StreamingState` will own chunk residency decisions.

Planned primary API:

- `ensure_resident(region)`
- `get_resident_set() -> &HashSet<ChunkCoord>`

## Floating Origin Draft

`floating_origin` will manage large-world recentering.

Planned responsibilities:

- recenter threshold configuration
- origin translation state and updates
- coordination hooks for renderer/simulation transforms
