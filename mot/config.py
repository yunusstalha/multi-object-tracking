import numpy as np
from scipy.stats.distributions import chi2


def default_config():
    return {
        "seed": 7,
        "steps": 180,
        "dt": 1.0,
        "area": [-600.0, 600.0, -350.0, 350.0],
        "object_size": [20.0, 12.0],
        "process_std": [0.18, 0.18, 0.04, 0.04],
        "position_meas_std": 5.0,
        "size_meas_std": 1.0,
        "p_det": 0.95,
        "p_gate": 0.997,
        "clutter_per_step": 6,
        "clutter_size_range": [6.0, 28.0],
        "confirm_m": 2,
        "confirm_n": 3,
        "delete_m": 5,
        "delete_n": 7,
        "initial_pos_std": 10.0,
        "initial_vel_std": 3.0,
        "initial_size_std": 3.0,
        "max_jpda_hypotheses": 20000,
    }


def model_matrices(config):
    dt = config["dt"]
    qx, qy, qw, qh = np.square(config["process_std"])

    F = np.array(
        [
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )

    G = np.array(
        [
            [0.5 * dt * dt, 0, 0, 0],
            [dt, 0, 0, 0],
            [0, 0.5 * dt * dt, 0, 0],
            [0, dt, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    Q = G @ np.diag([qx, qy, qw, qh]) @ G.T

    H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    R = np.diag(
        [
            config["position_meas_std"] ** 2,
            config["position_meas_std"] ** 2,
            config["size_meas_std"] ** 2,
            config["size_meas_std"] ** 2,
        ]
    )
    return F, H, Q, R


def gate_threshold(config):
    if "gate_threshold" in config:
        return config["gate_threshold"]
    return chi2.ppf(config["p_gate"], df=4)


def initial_covariance(config):
    return np.diag(
        [
            config["initial_pos_std"] ** 2,
            config["initial_vel_std"] ** 2,
            config["initial_pos_std"] ** 2,
            config["initial_vel_std"] ** 2,
            config["initial_size_std"] ** 2,
            config["initial_size_std"] ** 2,
        ]
    )
