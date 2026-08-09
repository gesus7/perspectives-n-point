"""Unit tests for TrackSelector state machine (no I/O, no OpenCV)."""
import pytest
from infer.tracker_state import TrackSelector, State


def _track(track_id, cls=0, conf=0.9):
    return {"class_id": cls, "track_id": track_id, "det_conf": conf}


# ── auto-lock ──────────────────────────────────────────────────────────────────

def test_auto_lock_single():
    s = TrackSelector()
    s.update([_track(1)])
    assert s.state == State.LOCKED
    assert s.locked_id() == 1
    assert not s.needs_selection()


def test_stay_searching_when_empty():
    s = TrackSelector()
    s.update([])
    assert s.state == State.SEARCHING
    assert s.locked_id() is None


# ── popup (one-shot) ───────────────────────────────────────────────────────────

def test_popup_fires_on_multiple_bases():
    s = TrackSelector()
    s.update([_track(1), _track(2)])
    assert s.state == State.SELECTING
    assert s.needs_selection()  # consumes the flag
    assert not s.needs_selection()  # only once


def test_popup_only_fires_once():
    s = TrackSelector()
    # First encounter: two bases → popup
    s.update([_track(1), _track(2)])
    assert s.state == State.SELECTING
    s.needs_selection()  # consume
    # pilot selects track 1
    s.select(0, 1)
    assert s.state == State.LOCKED
    # Now target disappears, and two bases reappear → must NOT re-show popup
    for _ in range(s.lost_patience * 3):
        s.update([])
    assert s.state == State.SEARCHING
    s.update([_track(3), _track(4)])
    assert s.state == State.LOCKED  # auto-locked highest conf, no popup
    assert not s.needs_selection()


# ── selection ─────────────────────────────────────────────────────────────────

def test_select_locks_correct_id():
    s = TrackSelector()
    s.update([_track(1), _track(2)])
    s.needs_selection()
    s.select(0, 2)
    assert s.state == State.LOCKED
    assert s.locked_id() == 2
    assert s.locked_class() == 0


# ── lost / re-acquire ──────────────────────────────────────────────────────────

def test_lost_counter_triggers_searching():
    s = TrackSelector(lost_patience=5)
    s.update([_track(1)])
    assert s.state == State.LOCKED
    # Target disappears for exactly patience * 2 frames
    for _ in range(5 * 2):
        s.update([])
    assert s.state == State.SEARCHING
    assert s.locked_id() is None


def test_reappear_before_timeout_re_locks():
    s = TrackSelector(lost_patience=10)
    s.update([_track(1)])
    # Disappear for a few frames (< patience)
    for _ in range(5):
        s.update([])
    assert s.state in (State.LOCKED, State.LOST)
    # Re-appears before timeout
    s.update([_track(1)])
    assert s.state == State.LOCKED
    assert s.locked_id() == 1


def test_locked_ignores_new_targets():
    s = TrackSelector()
    s.update([_track(1)])
    assert s.state == State.LOCKED
    # New target 2 appears — must stay locked on 1, no popup
    s.update([_track(1), _track(2)])
    assert s.state == State.LOCKED
    assert s.locked_id() == 1
    assert not s.needs_selection()


def test_status_label_reflects_state():
    s = TrackSelector()
    assert "SEARCHING" in s.status_label()
    s.update([_track(1)])
    assert "LOCKED" in s.status_label()
