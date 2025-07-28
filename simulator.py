"""
simulator.py
------------
Multi-object trajectory & measurement simulator for MOT experiments.

What you get:
- Deterministic/reproducible runs via numpy Generator
- 6D constant-velocity + box-size state: [x, vx, y, vy, length, width]
- 4D measurement: [x, y, length, width]  (no velocity observed)
- Object birth/death via simple start/end indices
- Uniform Poisson clutter per frame
- Results stored as:
    * X_truth[obj_id] : (T, nx) ground truth states (NaN when object not alive)
    * Y_truth[obj_id] : (T, ny) noisified true measurements (no clutter)
    * Y_all[k]        : (mk, ny) all measurements at frame k (incl. clutter)
- Quick XY scatter plot
- np.savez output for easy reload

Keep/extend:
- Swap in other motion/measurement models
- Change clutter distributions
- Add missed detections / detection probability
- YAML/JSON config if you want later
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------- Tiny type alias ----------
Array = np.ndarray

# ---------- Data structures ----------

@dataclass
class SimulationResult:
    X_truth: Dict[int, Array]           # per object: (T, nx)
    Y_truth: Dict[int, Array]           # per object: (T, ny)
    Y_all: List[Array]                  # per frame: (mk, ny)
    clutter_counts: Array               # (T,)
    meta: Dict                          # anything else you want to stash


class Simulator:
    """Core simulation loop."""
    def __init__(self,
                 process: ProcessModel,
                 sensor: SensorModel,
                 objects: List[ObjectSpec],
                 steps: int,
                 clutter: Optional[UniformClutter] = None,
                 seed: Optional[int] = None):
        self.process = process
        self.sensor = sensor
        self.objects = objects
        self.steps = steps
        self.clutter = clutter
        self.rng = np.random.default_rng(seed)

    def run(self) -> SimulationResult:
        nx = self.process.A.shape[0]
        ny = self.sensor.C.shape[0]

        # Allocate result buffers
        X_truth = {o.obj_id: np.full((self.steps, nx), np.nan) for o in self.objects}
        Y_truth = {o.obj_id: np.full((self.steps, ny), np.nan) for o in self.objects}
        Y_all: List[Array] = []
        clutter_counts = np.zeros(self.steps, dtype=int)

        # Copy initial states
        current_state = {o.obj_id: o.x0.copy() for o in self.objects}

        for k in range(self.steps):
            frame_meas = []

            # Propagate & measure each alive object
            for o in self.objects:
                if o.start <= k < o.end:
                    # Process noise in w-space, then mapped by G
                    w = self.rng.multivariate_normal(
                        mean=np.zeros(self.process.Q_tilde.shape[0]),
                        cov=self.process.Q_tilde
                    )
                    x_next = self.process.A @ current_state[o.obj_id] + self.process.G @ w
                    current_state[o.obj_id] = x_next
                    X_truth[o.obj_id][k] = x_next

                    # Measurement noise
                    v = self.rng.multivariate_normal(
                        mean=np.zeros(self.sensor.R.shape[0]),
                        cov=self.sensor.R
                    )
                    y = self.sensor.C @ x_next + v
                    Y_truth[o.obj_id][k] = y
                    frame_meas.append(y)

            # Clutter
            if self.clutter is not None:
                clutter = self.clutter.sample()
                clutter_counts[k] = clutter.shape[0]
                if clutter.shape[0] > 0:
                    frame_meas.append(clutter)

            # Combine this frame's measurements
            if len(frame_meas) == 0:
                Y_all.append(np.empty((0, ny)))
            else:
                # Each element is either (ny,) or (n_clutter, ny) -> stack
                Y_all.append(np.vstack(frame_meas))

        meta = {
            "seed_state": self.rng.bit_generator.state,
            "steps": self.steps,
            "nx": nx,
            "ny": ny,
        }

        return SimulationResult(X_truth=X_truth,
                                Y_truth=Y_truth,
                                Y_all=Y_all,
                                clutter_counts=clutter_counts,
                                meta=meta)


# ---------- Convenience plotting ----------

def plot_xy(Y_list: List[Array],
            xlim: Tuple[float, float] = (-3000, 3000),
            ylim: Tuple[float, float] = (-3000, 3000),
            title: str = "Cluttered Data") -> None:
    """Quick scatter of all measurements in XY plane (no time dimension)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for Y in Y_list:
        if Y.size:
            ax.plot(Y[:, 0], Y[:, 1], 'x', markersize=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(title)
    ax.grid(True)
    plt.show()


# ---------- Example usage ----------

if __name__ == "__main__":
    # ---- Parameters (tweak here or swap in a YAML later) ----
    dt = 1.0
    steps = 350
    var_p = 0.3

    # State transition (constant velocity for x,y; sizes static)
    A = np.array([[1, dt, 0,  0, 0, 0],
                  [0,  1, 0,  0, 0, 0],
                  [0,  0, 1, dt, 0, 0],
                  [0,  0, 0,  1, 0, 0],
                  [0,  0, 0,  0, 1, 0],
                  [0,  0, 0,  0, 0, 1]])

    G = np.array([[dt**2/2, 0,        0,        0],
                  [dt,       0,        0,        0],
                  [0,        dt**2/2,  0,        0],
                  [0,        dt,       0,        0],
                  [0,        0,        1,        0],
                  [0,        0,        0,        1]])

    # Process noise on [x, y, length, width] driving terms
    Q_tilde = np.diag([var_p**2, var_p**2, var_p**2, var_p**2])

    # Measurement: observe x,y,l,w
    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    sensor_sigma_pos = 1 / 3  # 3 sigma ~ 1 m
    sensor_sigma_box = 1.0
    R = np.diag([sensor_sigma_pos**2, sensor_sigma_pos**2,
                 sensor_sigma_box**2, sensor_sigma_box**2])

    process = ProcessModel(A=A, G=G, Q_tilde=Q_tilde)
    sensor = SensorModel(C=C, R=R)

    # Objects: (id, initial state, birth, death)
    objs = [
        ObjectSpec(1, np.array([0,    1,  0,  1, 20, 25]),   0,   250),
        ObjectSpec(2, np.array([-250, 1,  0,  1, 30, 36]),  50,   250),
        ObjectSpec(3, np.array([-500, 1, -70, 1, 25, 28]), 100,   350),
    ]

    # Optional clutter
    clutter = UniformClutter(
        x_range=(-1000, 1000),
        y_range=(-1000, 1000),
        l_range=(4, 20),
        w_range=(4, 30),
        rate=5.0,
        rng=np.random.default_rng(123)
    )

    # Run
    sim = Simulator(process, sensor, objs, steps=steps, clutter=clutter, seed=7)
    result = sim.run()

    # Save measurements (object arrays require dtype=object)
    np.savez(
        "demo_data.npz",
        Y=np.array(result.Y_all, dtype=object),
        clutter_counts=result.clutter_counts,
        meta=result.meta,
    )

    # Quick plot
    plot_xy(result.Y_all)