"""
track_object.py
---------------
Wraps a KalmanFilter with bookkeeping useful for MOT:
- unique id
- hit/miss counters
- time since last update
- history storage (optional; can be disabled for speed)

You can extend this to carry appearance features, classification, etc.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Dict

from .kalman import KalmanFilter

Array = np.ndarray


@dataclass
class TrackObject:
    kf: KalmanFilter
    track_id: int

    # --- Bookkeeping ---
    age: int = 0                   # total frames since birth
    time_since_update: int = 0     # frames since last meas. update
    hits: int = 0                  # number of measurement updates
    misses: int = 0                # number of consecutive misses

    # Optional history (store for debugging / plots)
    store_history: bool = True
    history: Dict[str, List[Array]] = field(default_factory=lambda: {
        "t": [], "x": [], "P": [], "S": [], "y_pred": []
    })

    def predict(self, frame_idx: Optional[int] = None):
        """Run KF predict. Also increment counters and store history."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.store_history:
            y_pred, S = self.kf.project()
            self.history["t"].append(frame_idx)
            self.history["x"].append(self.kf.state.copy())
            self.history["P"].append(self.kf.covariance.copy())
            self.history["y_pred"].append(y_pred.copy())
            self.history["S"].append(S.copy())

    def update(self, z: Array, frame_idx: Optional[int] = None):
        """Measurement update + bookkeeping."""
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        if self.store_history:
            y_pred, S = self.kf.project()
            self.history["t"].append(frame_idx)
            self.history["x"].append(self.kf.state.copy())
            self.history["P"].append(self.kf.covariance.copy())
            self.history["y_pred"].append(y_pred.copy())
            self.history["S"].append(S.copy())

    def miss(self):
        """No measurement this frame."""
        self.misses += 1
        # Nothing else: predict() already advanced time_since_update

    # Convenience
    @property
    def state(self) -> Array:
        return self.kf.state

    @property
    def covariance(self) -> Array:
        return self.kf.covariance
