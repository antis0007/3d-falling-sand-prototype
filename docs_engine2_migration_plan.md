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
1. **Phase 1 (this change): extraction skeleton + legacy quarantine**
   - Introduce `src/engine2/` module tree with explicit storage/sim/render/gpu boundaries.
   - Add thin app adapter path: app entry calls `engine2::app_bridge::loop::run()` and then legacy fallback.
   - Add `legacy` namespace module documenting frozen modules.
2. **Phase 2: engine2 world + residency source of truth**
   - Implement brick-key residency map (16^3 bricks), command queues, and CPU cold storage/procgen integration.
   - Keep renderer output minimal and legacy draw path still active.
3. **Phase 3: GPU hot-state bring-up**
   - Allocate resident brick pools/page tables/queues/indirect buffers in engine2 gpu modules.
   - Route edit/load/unload commands through command stream; no CPU meshing.
4. **Phase 4: render extraction switchover**
   - Bind engine2 extraction/draw to GPU indirect buffers and retire legacy mesh finalize path.
5. **Phase 5: app loop simplification**
   - Remove legacy orchestration branches and move scheduling concerns into engine2 bridge/schedulers.
