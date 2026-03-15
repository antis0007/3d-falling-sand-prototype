//! GPU residency ownership domain.
//! The renderer consumes residency state but does not own allocation identity.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GpuResourceHandle(pub u64);

impl GpuResourceHandle {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GpuResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
