//! Compatibility glue for callers that still import `physics_gpu`.
//! Simulation backends now live in `crate::simulation`.

#[allow(unused_imports)]
pub use crate::simulation::{GpuFluidBackend as PhysicsGpuSimulator, SimulationMode};
