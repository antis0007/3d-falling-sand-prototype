# 3D Falling Sand Prototype (Rust + wgpu + winit + egui)

A minimal, buildable voxel sandbox prototype inspired by Minecraft-style controls and NOITA-style material simulation, expanded into 3D.

## Features

- 64x64x64 default voxel world (configurable for new worlds, multiples of 16).
- Voxel size = `0.5m` render scale.
- Chunked world (`16x16x16` chunks), exposed-face chunk meshing, per-chunk dirty rebuild.
- Chunk frustum culling with chunk AABB clip test.
- Materials + phase rules:
  - Solids: Stone, Wood
  - Powders: Sand, Snow
  - Liquids: Water, Lava, Acid
  - Gases: Smoke, Steam
- Active-set simulation (per chunk) with settled cooldown for stable voxels.
- Pause/Run simulation + Step Once.
- Save/Load with versioned header + RLE chunk data + LZ4 compression.
- FPS controls with mouse look, jump + double-jump flight toggle.
- egui top bar, toolbar (0-9), pause menu, brush panel.

## Build / Run

```bash
cargo run
```

This repo sets a shared Cargo target directory at `/workspace/.cargo-target/3d-falling-sand-prototype` via `.cargo/config.toml` so dependency artifacts are reused across tasks instead of rebuilt each time.

Requires stable Rust on Linux or Windows.

## Controls

- `WASD`: Move
- `Mouse`: Look (when cursor is locked)
- `Space`: Jump
- `Space` double tap: Toggle flight mode
- `Left Shift`: Descend while flying
- `Esc`: Pause menu + cursor unlock/lock toggle
- `P`: Pause/Run simulation
- `B`: Toggle brush window
- `0-9`: Select toolbar slot
- `Mouse Wheel`: Cycle toolbar slot
- `LMB`: Place brush action
- `RMB`: Erase brush action (opposite convenience action)

## UI

Top bar includes:
- New World
- Save
- Load
- Pause/Run
- Step Once
- Day/Night toggle
- New world size slider
- Selected material + brush radius

Brush panel (`B`):
- Radius
- Ray distance
- Shape: Sphere/Cube
- Mode: Place/Erase

## Save/Load

Default save path:
- `./saves/world.vxl`

Format:
- Magic bytes: `VXL3`
- Version: `1`
- Dims: `[x, y, z]`
- Per-chunk RLE over `u16` material IDs
- Entire payload compressed with `lz4_flex`

Loading always pauses simulation and marks chunk meshes dirty for rebuild.

## Notes

This code intentionally keeps simulation and rendering separate (`sim`, `world`, `renderer`) so simulation can later move to worker-thread execution and more advanced fluid/pressure/temperature models can be swapped in.
