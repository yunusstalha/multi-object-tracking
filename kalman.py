"""
kalman.py
---------
Plain linear Kalman filter for a single target.

Design goals:
- No globals.
- Accepts the same A, Q, C, R you already have in models.py.
- Exposes small helper methods (project()) so you can reuse S, y_pred for gating/JPDAs.

Typical usage:
    kf = KalmanFilter(x0, P0, A, Q, C, R)
    x_pred, P_pred = kf.predict()
    y_pred, S = kf.project()
    kf.update(z)  # if a measurement arrived
    # or kf.miss()  # if no measurement (JPDA / PDA updates will call update() with weighted sums)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class KalmanFilter:
    x: Array          # state vector (nx,)
    P: Array          # state covariance (nx, nx)
    A: Array          # transition (nx, nx)
    Q: Array          # process noise (nx, nx)
    C: Array          # observation (ny, nx)
    R: Array          # measurement noise (ny, ny)

    # --- Internal caches (not strictly required but handy) ---
    _y_pred: Array | None = None     # C @ x
    _S: Array | None = None          # C P C^T + R

    def predict(self) -> tuple[Array, Array]:
        """
        Time update (a.k.a. 'predict').  x_k|k-1, P_k|k-1
        Returns (x_pred, P_pred) for convenience.
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        # Invalidate caches
        self._y_pred = None
        self._S = None
        return self.x, self.P

    def project(self) -> tuple[Array, Array]:
        """
        Compute predicted measurement y_pred and innovation covariance S, given current x,P.
        This is separate so external code can compute Mahalanobis distances etc. without
        doing a full update.
        """
        if self._y_pred is None:
            self._y_pred = self.C @ self.x
        if self._S is None:
            self._S = self.C @ self.P @ self.C.T + self.R
        return self._y_pred, self._S

    def update(self, z: Array) -> tuple[Array, Array]:
        """
        Measurement update.  Assumes z shape is (ny,) or (ny,1).
        Returns (x_upd, P_upd).
        """
        y_pred, S = self.project()
        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)
        # Residual
        r = z - y_pred
        # State and covariance update
        self.x = self.x + K @ r
        self.P = self.P - K @ S @ K.T
        # Invalidate caches
        self._y_pred = None
        self._S = None
        return self.x, self.P

    def miss(self):
        """
        Call when no measurement is associated (e.g., GNN miss or JPDA weight=0).
        Some filters (e.g., alpha-beta) would do nothing; standard KF doesn't need
        anything special beyond the predict step. Still provided for symmetry.
        """
        # Nothing beyond predict(); left here for readability / future logic.
        pass

    # Convenience getters (optional)
    @property
    def state(self) -> Array:
        return self.x

    @property
    def covariance(self) -> Array:
        return self.P
