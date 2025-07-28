"""
track_holder.py
----------------
Manages a set of TrackObject instances:
- ID creation
- batch predict/update
- simple lifecycle logic (pruning)
- (optional) association hooks

API Sketch:
    tm = TrackManager(max_age=10, min_hits=3)
    tm.predict_all(frame_idx=k)
    assignments, unassigned_meas = associate(tm.active_tracks(), measurements)
    tm.update_tracks(assignments, measurements, frame_idx=k)
    tm.handle_misses()  # remove dead tracks, etc.

Feel free to adapt names to your style.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional
import itertools
import numpy as np

from .track_object import TrackObject
from .kalman import KalmanFilter

Array = np.ndarray


@dataclass
class TrackManager:
    max_age: int = 10        # prune if track.misses > max_age
    min_hits: int = 3        # optional: require N hits before reporting track
    _id_gen: itertools.count = field(default_factory=lambda: itertools.count(1), init=False)
    tracks: Dict[int, TrackObject] = field(default_factory=dict, init=False)

    def new_track(self, kf: KalmanFilter) -> TrackObject:
        tid = next(self._id_gen)
        trk = TrackObject(kf=kf, track_id=tid)
        self.tracks[tid] = trk
        return trk

    def predict_all(self, frame_idx: Optional[int] = None):
        for trk in self.tracks.values():
            trk.predict(frame_idx)

    def update_tracks(self,
                      assignments: Iterable[Tuple[int, int]],
                      measurements: Array,
                      frame_idx: Optional[int] = None):
        """
        assignments: iterable of (track_id, meas_idx).
        measurements: shape (M, ny)
        """
        assigned_ids = set()
        for tid, midx in assignments:
            z = measurements[midx]
            self.tracks[tid].update(z, frame_idx)
            assigned_ids.add(tid)

        # mark misses for the others
        for tid, trk in self.tracks.items():
            if tid not in assigned_ids:
                trk.miss()

    def prune_dead(self):
        """Remove tracks with too many consecutive misses."""
        dead = [tid for tid, trk in self.tracks.items() if trk.misses > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    def active_tracks(self) -> List[TrackObject]:
        """Return tracks that are 'confirmed' (hits >= min_hits)."""
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    def all_tracks(self) -> List[TrackObject]:
        return list(self.tracks.values())
