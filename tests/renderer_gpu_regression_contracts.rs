use std::collections::{HashMap, HashSet};

use falling_sand_3d::streaming::{ChunkStreaming, WorkItem};
use falling_sand_3d::ChunkCoord;
use glam::Vec3;

fn cc(x: i32, y: i32, z: i32) -> ChunkCoord {
    ChunkCoord { x, y, z }
}

#[test]
fn continuity_replacement_failure_keeps_visible_chunk() {
    // Contract: if replacement generation fails, the currently visible resident chunk
    // must remain resident instead of disappearing ("hole" regression).
    let chunk = cc(0, 0, 0);
    let mut streaming = ChunkStreaming::new(7);
    streaming.resident.insert(chunk);
    streaming.mark_visible(chunk, 10);

    // Simulate an in-flight replacement attempt that gets dropped.
    streaming.scheduled_generate.insert(chunk);
    streaming.dispatched_generate.insert(chunk);
    streaming.mark_generation_dropped(chunk);

    assert!(streaming.resident.contains(&chunk));
    assert_eq!(streaming.residency_of(chunk), falling_sand_3d::streaming::Residency::Resident);
}

#[test]
fn pending_or_invalid_candidate_does_not_blank_current_drawable() {
    // Contract: a deferred/pending replacement candidate must not blank an already
    // resident chunk.
    let chunk = cc(3, 0, -2);
    let mut streaming = ChunkStreaming::new(99);
    streaming.resident.insert(chunk);

    streaming.mark_dispatch_failed_or_deferred(chunk);

    assert!(streaming.resident.contains(&chunk));
    assert!(streaming.scheduled_generate.contains(&chunk));
    assert!(streaming.pending_generate_count() >= 1);
    assert_eq!(streaming.residency_of(chunk), falling_sand_3d::streaming::Residency::Resident);
}

#[test]
fn backlog_processing_is_bounded_and_carries_over() {
    // Contract: backlog scheduling is capped per update, and remaining work carries over.
    let player = cc(0, 0, 0);
    let mut streaming = ChunkStreaming::new(1);
    streaming.max_generate_schedule_per_update = 2;

    let desired: Vec<_> = (0..6).map(|x| cc(10 + x, 0, 10)).collect();
    let keep = HashSet::new();

    let stats0 = streaming.update(&desired, &keep, player, 1);
    assert_eq!(stats0.queued_generate, 2);
    assert_eq!(streaming.pending_generate_count(), 2);

    // Drain one item, then update again: work should continue, but still stay bounded.
    let drained = streaming.drain_generate_requests(1);
    assert_eq!(drained.len(), 1);

    let stats1 = streaming.update(&desired, &keep, player, 2);
    assert_eq!(stats1.queued_generate, 2);
    assert_eq!(streaming.pending_generate_count(), 3);
}

#[test]
fn low_detail_or_fallback_is_preferred_over_void() {
    // Contract approximation (black-box): even at minimal radii, desired_set must
    // still include the player's chunk so the system has a non-void fallback target.
    let player = cc(0, 0, 0);
    let desired = ChunkStreaming::desired_set(
        player,
        Vec3::ZERO,
        Vec3::Z,
        None,
        0,
        0,
        Some(0),
        Some(0),
        0,
        16,
        16,
        &HashMap::new(),
        1,
    );

    assert!(desired.generation_order.contains(&player));
    assert!(desired.resident_keep.contains(&player));
}

#[test]
fn desired_backlog_does_not_imply_near_range_starvation() {
    // Contract: near-range (immediate horizontal neighbor) work should still be
    // scheduled even when backlog is dominated by far chunks.
    let player = cc(0, 0, 0);
    let near_neighbor = cc(1, 0, 0);
    let mut streaming = ChunkStreaming::new(123);
    streaming.max_generate_schedule_per_update = 4;

    let mut desired = Vec::new();
    desired.extend((0..40).map(|i| cc(50 + i, 0, 50)));
    desired.push(near_neighbor);

    let keep = HashSet::new();
    let stats = streaming.update(&desired, &keep, player, 1);
    assert_eq!(stats.queued_generate, 4);

    let work = streaming.drain_work_items();
    assert!(
        work.iter()
            .any(|item| matches!(item, WorkItem::Generate(coord) if *coord == near_neighbor)),
        "expected near neighbor to be scheduled despite far backlog"
    );
}

#[test]
fn reported_success_requires_drawable_candidate() {
    // We cannot directly assert renderer "drawable candidate" success semantics from
    // integration tests because candidate/drawable internals are not exposed publicly.
    // Proxy contract we *can* enforce via public API: dispatch success alone does not
    // imply residency success; only mark_generated transitions to Resident.
    let chunk = cc(2, 0, 2);
    let mut streaming = ChunkStreaming::new(5);
    streaming.mark_dispatch_failed_or_deferred(chunk);

    let requests = streaming.drain_generate_requests(1);
    assert_eq!(requests, vec![chunk]);
    assert_eq!(
        streaming.residency_of(chunk),
        falling_sand_3d::streaming::Residency::Generating
    );

    streaming.mark_generated(chunk, 42);
    assert_eq!(streaming.residency_of(chunk), falling_sand_3d::streaming::Residency::Resident);
}
