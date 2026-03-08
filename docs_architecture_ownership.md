# Chunk Ownership Architecture

## Single source of truth

`ChunkStore` is the only long-lived owner of voxel chunk data. Generation, simulation, and rendering all read from or write to `ChunkStore::Chunk` instances.

## Ownership boundaries

- **Store (`chunk_store`)**
  - Owns chunk voxel memory and chunk lifecycle (insert/remove/dirty tracking).
  - Provides meshing border extraction and chunk dirty state transitions.
- **Simulation (`sim_world`, `sim`)**
  - Mutates voxels through `ChunkStore` APIs.
  - Does not own persistent chunk containers.
- **Procedural generation (`procgen`)**
  - Builds generated data directly into `chunk_store::Chunk` output for insertion.
  - Uses an internal transient generation volume only while running passes.
- **Rendering (`renderer`)**
  - Is the sole authority for *visible residency* decisions (what is drawable this frame).
  - Maintains continuity policy for renderer-visible state: the current drawable must survive candidate failure; lower detail is acceptable, void is not.
  - Uses `ChunkStore`/meshing outputs as candidate inputs and never treats candidate pipeline success as visibility authority.
  - Does not maintain authoritative voxel simulation state.

## Legacy world ownership removal

The former `world::World` / `world::Chunk` ownership path is removed from chunk generation and chunk insertion APIs. `world.rs` now only carries shared schema/constants used across UI/tooling.

## Coordinate and GPU slot contract

- `ChunkCoord` is the only logical chunk identity used by streaming/simulation/render scheduling.
- `ChunkOriginWorld`/`chunk_to_world_min` derives world-space placement from `ChunkCoord` exactly once.
- Mesh vertices are generated in **chunk-local** space.
- Render-time instance data carries chunk world origin; shader applies `chunk_origin_world - origin_offset` once.
- `GpuPageIndex` is storage residency only and must never be interpreted as a spatial transform.

## Renderer-visible authority contract

- GPU meshing/finalize/atlas upload stages are candidate producers/accelerators, not visibility authority.
- Renderer continuity policy is the production path for visible residency decisions.
- Any feature flag preserving migration-era behavior is legacy/debug-only and must not redefine architecture ownership.
