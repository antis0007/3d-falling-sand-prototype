# Simulation + Procgen Audit

## Current code paths and conflicts

- CPU cellular simulation lives in `src/sim_world.rs` and is driven through `SimulationRuntime::CpuCellular`.
- The `GpuFluidBackend` in `src/simulation/gpu_fluid.rs` previously performed a second, conflicting simulation model (velocity/pressure arrays) but did not run WGSL or share material rules with CPU.
- `src/gpu_compute.rs` owns WGSL compute pipelines, but those pipelines are currently integrated with meshing jobs and not with the authoritative world simulation update loop.
- Legacy compatibility exports in `src/physics_gpu/mod.rs` still expose old names and can mask where the real backend lives.

## Why trees regressed

- Vegetation spawning had become over-constrained by hard wet-surface rejection, causing many viable anchors to be discarded near rivers/lakes and reducing visible tree coverage.
- Tree density baseline (`ProcGenConfig::tree_density`) was tuned too low for current biome weighting and hydrology suppression.

## Why terrain looked repetitive

- Terrain already had multi-noise composition, but lacked additional stratified variation bands in inland elevations, creating visible repetition in wide plains/highland transitions.

## Unification direction

- Keep one authoritative simulation routing point (`SimulationRuntime`).
- Keep one material behavior source (`sim::material` / `Phase`) for state logic decisions.
- Use GPU compute where available, but never silently fail: emit diagnostics and deterministic fallback behavior.
