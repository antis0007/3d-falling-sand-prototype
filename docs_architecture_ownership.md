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
  - Treats `ChunkStore` as read-only input for meshing and draw submission.
  - Does not maintain authoritative voxel state.

## Legacy world ownership removal

The former `world::World` / `world::Chunk` ownership path is removed from chunk generation and chunk insertion APIs. `world.rs` now only carries shared schema/constants used across UI/tooling.
