"""Track-selection state machine for multi-target scenarios.

States
------
SEARCHING   No target locked. Auto-lock if exactly 1 TowerBase visible; trigger a
            one-time manual selection popup if > 1 TowerBase appears.
SELECTING   Popup is on screen waiting for the pilot to click. Tracking is still
            running in the background; the frozen frame stays on display.
LOCKED      A specific (class_id, track_id) pair is locked. All other targets are
            ignored for pose/annotation. Remains locked even if new targets appear
            (one-shot selection policy).
LOST        Locked target's track_id vanished for > `lost_patience` frames. Returns
            to SEARCHING after the patience window.

Transitions
-----------
SEARCHING + 0 base tracks              -> SEARCHING (wait)
SEARCHING + 1 base track               -> LOCKED (auto)
SEARCHING + > 1 base tracks            -> SELECTING (show popup; first time only
                                          per run) or auto-lock highest-conf track
                                          if popup was already shown and dismissed
LOCKED + target still visible          -> LOCKED
LOCKED + target missing                -> LOST counter increments
LOST + patience exceeded               -> SEARCHING
LOST + target reappears before timeout -> LOCKED (same id) or auto-LOCKED (new id)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class State(Enum):
    SEARCHING = auto()
    SELECTING = auto()
    LOCKED    = auto()
    LOST      = auto()


@dataclass
class TrackSelector:
    """Pure state-machine with no I/O or OpenCV dependency.

    The video loop asks update() every frame and reads state / locked_target.
    When needs_selection() returns True the video loop should show the popup.
    """
    # Class id to track (0 = TowerBase).  Top (1) is auto-locked to the single
    # visible TowerTop once the base is locked; it has no multi-target ambiguity
    # in this application.
    tracked_class: int = 0

    # How many frames to wait after the target disappears before giving up.
    lost_patience: int = 30

    # Internal state
    state:         State       = field(default=State.SEARCHING, init=False)
    locked_target: Optional[tuple[int, int]] = field(default=None, init=False)  # (class_id, track_id)
    _lost_count:   int         = field(default=0, init=False)
    _popup_shown:  bool        = field(default=False, init=False)
    _needs_popup:  bool        = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, current_tracks: list[dict]) -> None:
        """Advance the state machine with this frame's track list.

        current_tracks: list of dicts, each with at least
            'class_id' (int), 'track_id' (int | None), 'det_conf' (float).
        Only tracks for self.tracked_class are considered for base-locking;
        the caller handles TowerTop separately.
        """
        base_tracks = [
            t for t in current_tracks
            if t["class_id"] == self.tracked_class and t.get("track_id") is not None
        ]

        if self.state == State.SEARCHING:
            self._step_searching(base_tracks)

        elif self.state == State.SELECTING:
            # Stay in SELECTING; the video loop will call select() when the pilot clicks.
            pass

        elif self.state == State.LOCKED:
            self._step_locked(base_tracks)

        elif self.state == State.LOST:
            self._step_lost(base_tracks)

    def select(self, class_id: int, track_id: int) -> None:
        """Called by the UI when the pilot clicks/chooses a target."""
        self.locked_target = (class_id, track_id)
        self.state = State.LOCKED
        self._needs_popup = False
        self._lost_count = 0

    def needs_selection(self) -> bool:
        """True for exactly one frame — the frame the popup should be shown."""
        if self._needs_popup:
            self._needs_popup = False
            return True
        return False

    def is_locked(self) -> bool:
        return self.state == State.LOCKED

    def locked_id(self) -> Optional[int]:
        """Return the locked track_id, or None."""
        return self.locked_target[1] if self.locked_target else None

    def locked_class(self) -> Optional[int]:
        return self.locked_target[0] if self.locked_target else None

    def status_label(self) -> str:
        if self.state == State.LOCKED:
            cls, tid = self.locked_target
            return f"LOCKED cls={cls} id={tid}"
        if self.state == State.LOST:
            return f"LOST ({self._lost_count}/{self.lost_patience})"
        if self.state == State.SELECTING:
            return "SELECTING — click target"
        return "SEARCHING"

    # ------------------------------------------------------------------ #
    #  Internal transitions                                                #
    # ------------------------------------------------------------------ #

    def _step_searching(self, base_tracks: list[dict]) -> None:
        if len(base_tracks) == 0:
            return  # wait
        if len(base_tracks) == 1:
            self._auto_lock(base_tracks[0])
        else:
            # More than one base track visible.
            if not self._popup_shown:
                # First time: ask pilot to pick.
                self._popup_shown = True
                self._needs_popup = True
                self.state = State.SELECTING
            else:
                # Popup was already shown once; auto-lock highest-confidence.
                best = max(base_tracks, key=lambda t: t.get("det_conf", 0.0))
                self._auto_lock(best)

    def _step_locked(self, base_tracks: list[dict]) -> None:
        _, locked_id = self.locked_target
        still_there = any(t["track_id"] == locked_id for t in base_tracks)
        if still_there:
            self._lost_count = 0
        else:
            # Target not in this frame — start patience countdown.
            self._lost_count += 1
            if self._lost_count >= self.lost_patience:
                self.state = State.LOST

    def _step_lost(self, base_tracks: list[dict]) -> None:
        _, locked_id = self.locked_target
        # Did the exact same ID come back?
        reappeared = any(t["track_id"] == locked_id for t in base_tracks)
        if reappeared:
            self.state = State.LOCKED
            self._lost_count = 0
            return
        self._lost_count += 1
        if self._lost_count >= self.lost_patience * 2:
            # Give up entirely and restart searching with a clean slate.
            # NOTE: _popup_shown intentionally NOT reset — the selection popup
            # fires at most once per session regardless of how many resets occur.
            self.state = State.SEARCHING
            self.locked_target = None
            self._lost_count = 0

    def _auto_lock(self, track: dict) -> None:
        self.locked_target = (track["class_id"], track["track_id"])
        self.state = State.LOCKED
        self._lost_count = 0
