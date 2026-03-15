//! Render authority domain.
//! Render-visible truth advances through explicit frame epochs.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FrameEpoch(pub u64);

impl FrameEpoch {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderLifecycleState {
    Idle,
    CandidatePending,
    CandidateReady,
    CandidateRejected,
    CurrentDrawable,
}

impl fmt::Display for FrameEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
