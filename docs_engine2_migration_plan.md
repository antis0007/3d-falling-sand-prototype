# Engine2 migration plan (phase-oriented)

## Current architecture failures
- `app.rs` is a large monolithic orchestrator that mixes streaming, procgen dispatch, simulation stepping, edit tools, throttling, and render lifecycle in one loop.
- `renderer.rs` owns too much policy and lifecycle coordination (meshing, draw continuity, residency interactions, startup budget logic), coupling visibility, upload, and GPU mesh lifecycle.
- `gpu_compute.rs` is tightly bound to renderer mesh buffers and startup decisions, preventing clean separation between simulation hot state and render extraction.
- Hot-path ownership is split across CPU structures and GPU state with multiple control points, increasing synchronization complexity.

## Preserve in phase 1
- Coordinate/value definitions in `types.rs` and material/world definitions in `world.rs`.
- Procgen entry points (`procgen.rs`) as cold-state source material for future `engine2/world/procgen.rs` integration.
- Utility budgets and low-level reusable helpers where trivially decoupled in later phases.

## Freeze as legacy in phase 1
- App orchestration monolith (`app.rs` legacy path).
- Renderer orchestration and mesh lifecycle (`renderer.rs`).
- GPU compute lifecycle currently coupled to renderer internals (`gpu_compute.rs`).

## Compile-safe phases
1. **Phase 1: extraction skeleton + legacy quarantine** ✅
   - Introduced `src/engine2/` module tree with explicit storage/sim/render/gpu boundaries.
   - Added thin app adapter path: app entry calls `engine2::app_bridge::loop::run()` and then legacy fallback.
   - Added `legacy` namespace module documenting frozen modules.
2. **Phase 2: engine2 world + residency source of truth** ✅
   - Added explicit engine2 coordinate types and conversion helpers for voxel→brick→local resolution with fixed 16^3 bricks.
   - Added `world/brick.rs` metadata and payload placeholders, `world/storage.rs` cold-state catalogs, and `world/residency.rs` as sparse residency authority.
   - Added desired-vs-current residency tracking with `request`, `release`, `mark_resident`, `mark_dirty`, and load/evict decision collection.
   - Added compact explicit command queue types (`load`, `unload`, `edit sphere`, `edit box`, `inject material`).
   - Extended engine2 app bridge to construct `Engine2State` (residency + commands + storage + procgen) while still returning legacy fallback.
3. **Phase 3: GPU hot-state bring-up** ✅
   - Added an engine2-owned GPU subsystem (`engine2/gpu`) with a thin context adapter over externally-owned `wgpu::Device`/`wgpu::Queue`.
   - Added explicit storage buffers for brick headers, brick state pages, command upload staging, active/dirty/remesh queues, queue counters, and indirect draw placeholder args.
   - Added engine2 page table ownership for brick-key -> GPU page-slot mapping with explicit allocation/eviction semantics.
   - Added residency -> GPU upload plumbing: load decisions allocate page slots, enqueue placeholder/real payload upload, update residency page handles, and flush queue metadata.
   - Kept rendering switchover out of scope and did not reuse legacy renderer/gpu_compute architecture.
4. **Phase 4: render extraction switchover**
   - Bind engine2 extraction/draw to GPU indirect buffers and retire legacy mesh finalize path.
5. **Phase 5: app loop simplification**
   - Remove legacy orchestration branches and move scheduling concerns into engine2 bridge/schedulers.

## Phase 2 design decisions
- Brick edge size is fixed at 16 to preserve a consistent sparse page unit for later GPU residency.
- Residency map is the source of truth for current state transitions, while storage tracks payload/meta and procgen only handles request/result plumbing.
- Residency entries include optional GPU page handles now to avoid API churn in Phase 3, but no allocation is performed in this phase.
