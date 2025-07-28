
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------- Tiny type alias ----------
Array = np.ndarray


# ---------- Model definitions ----------

@dataclass(frozen=True)
class ProcessModel:
    """Linear-Gaussian motion model."""
    A: Array          # (nx, nx) state transition
    G: Array          # (nx, q)  noise shaping matrix
    Q_tilde: Array    # (q, q)   base process noise covariance (usually diagonal)

    @property
    def Q(self) -> Array:
        """Full process noise covariance in state space."""
        return self.G @ self.Q_tilde @ self.G.T


@dataclass(frozen=True)
class SensorModel:
    """Linear-Gaussian measurement model."""
    C: Array          # (ny, nx) observation matrix
    R: Array          # (ny, ny) measurement noise covariance


@dataclass(frozen=True)
class ObjectSpec:
    """Defines one object's initial state and lifetime within the simulation."""
    obj_id: int
    x0: Array          # (nx,)
    start: int         # inclusive frame index
    end: int           # exclusive frame index


@dataclass
class UniformClutter:
    """
    Simple uniform clutter generator.
    'rate' is the expected number of clutter points per frame (Poisson distributed).
    """
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    l_range: Tuple[float, float]
    w_range: Tuple[float, float]
    rate: float
    rng: np.random.Generator

    def sample(self) -> Array:
        n = self.rng.poisson(self.rate)
        if n == 0:
            return np.empty((0, 4))
        x = self.rng.uniform(*self.x_range, size=n)
        y = self.rng.uniform(*self.y_range, size=n)
        l = self.rng.uniform(*self.l_range, size=n)
        w = self.rng.uniform(*self.w_range, size=n)
        return np.stack([x, y, l, w], axis=1)
