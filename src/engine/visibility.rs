//! Visibility authority declarations.
//! The visibility domain decides drawable membership; it must not infer
//! ownership of artifacts or GPU allocations.

/// Forbidden state helper: visibility cannot publish a drawable when the
/// source candidate has been explicitly rejected.
#[inline]
pub fn debug_assert_not_rejected_before_drawable(rejected: bool, drawable: bool) {
    debug_assert!(
        !(rejected && drawable),
        "forbidden state: rejected candidate cannot become drawable"
    );
}
