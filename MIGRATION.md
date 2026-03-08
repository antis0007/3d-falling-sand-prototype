# Simulation + Procgen Migration Notes

## Removed / reduced redundant systems

- Retained `SimulationRuntime` as the only simulation entry point and removed the old pseudo-fluid velocity/pressure path inside `GpuFluidBackend`.
- `GpuFluidBackend` now uses a deterministic state-class transport model (solid/powder/liquid/gas) that shares behavior classification with `sim::material`.
- Added explicit runtime warning when GPU simulation mode is selected but only deterministic CPU emulation is active, preventing silent feature failure.

## Unified data flow

1. **Generation** (`procgen.rs`): noise/biome/hydrology -> chunk voxels + vegetation intents.
2. **Simulation** (`simulation/mod.rs`, `simulation/gpu_fluid.rs`, `sim_world.rs`): runtime routes to a selected backend; both backends now use common material phase semantics.
3. **Meshing** (`meshing.rs`, `gpu_compute.rs`): consumes post-simulation chunk state.
4. **Rendering** (`renderer.rs`): consumes meshed geometry.

## Tunables

### Terrain

- `ProcGenConfig::terrain_scale`
- `ProcGenConfig::cave_density`
- Added stratified inland variation band in `terrain_height` for more ecosystem relief transitions.

### Vegetation

- `ProcGenConfig::tree_density` default increased from `0.028` to `0.042`.
- Wet-surface tree anchors are damped instead of hard-rejected (`tree_p *= 0.3`) to preserve riparian vegetation while avoiding full waterline overgrowth.

### Simulation

- `SUBSTEPS` in GPU-emulation backend increased to 3 for better settling/advection stability.
- State movement priority table in `movement_candidates` controls powder, liquid, and gas transport characteristics.

## Continuity and renderer authority (current state)

- Renderer is the sole authority for visible residency and draw continuity.
- GPU pipeline (meshing/finalize/atlas path) is a candidate producer/accelerator and is not the authority boundary.
- Current drawable must survive candidate failure; fallback to lower detail is acceptable, renderer-visible voids are not.
- Migration-era dual-ownership framing is obsolete: chunk ownership and renderer-visible authority are single-path responsibilities.

## Known limitations

- Full world-simulation WGSL execution is not yet wired into `SimulationRuntime`; current `GpuFluid` mode is deterministic CPU emulation with diagnostics.
- Cross-chunk fluid pressure continuity is still approximated by chunk-local movement and dirty-region readback (simulation limitation, not renderer-visible residency authority).
