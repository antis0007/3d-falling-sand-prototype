# Material ID code generation

This repo now has a single source of truth for simulation pipeline material IDs:

- `material_ids.txt`

`build.rs` reads that file and generates two artifacts under Cargo's `OUT_DIR`:

1. `material_ids.rs` for Rust (`src/material_ids.rs` includes it)
2. `material_ids.wgsl` for WGSL compute shaders

## How to add or change IDs

1. Edit `material_ids.txt`.
2. Use the generated Rust constants from `crate::material_ids::*` in simulation code.
3. WGSL shader sources that need IDs should consume the generated snippet by prepending
   `material_ids.wgsl` (see `src/gpu_compute.rs`).
4. Run tests: `cargo test generated_material_ids_match_material_table`.

## Regression guard

`src/sim.rs` includes `generated_material_ids_match_material_table`, which asserts that every generated ID still maps to the matching `MATERIALS[id].id` entry.

If that test fails, update either `material_ids.txt` or the `MATERIALS` table so they match.
