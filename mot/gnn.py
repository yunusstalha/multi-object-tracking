import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

from mot.config import gate_threshold
from mot.track import TrackManager


def assignment_cost(track, measurement, config):
    d2 = track.kf.mahalanobis2(measurement)
    if d2 > gate_threshold(config):
        return np.inf

    innovation = measurement - track.kf.predicted_measurement
    likelihood = multivariate_normal.pdf(
        innovation,
        mean=np.zeros(4),
        cov=track.kf.innovation_covariance,
        allow_singular=False,
    )
    return -np.log(max(config["p_det"] * likelihood, 1e-300))


def associate(tracks, measurements, config):
    if len(tracks) == 0 or len(measurements) == 0:
        return [], set()

    cost = np.full((len(tracks), len(measurements)), np.inf)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            cost[i, j] = assignment_cost(track, measurement, config)

    finite = np.isfinite(cost)
    if not finite.any():
        return [], set()

    big = 1e9
    rows, cols = linear_sum_assignment(np.where(finite, cost, big))
    matches = []
    used = set()
    for row, col in zip(rows, cols):
        if np.isfinite(cost[row, col]):
            matches.append((row, col))
            used.add(col)
    return matches, used


def run_gnn(measurements, config):
    manager = TrackManager(config)

    for time, frame in enumerate(measurements):
        frame = np.asarray(frame, dtype=float)
        tracks = manager.all_active()
        for track in tracks:
            track.predict()

        matches, used_measurements = associate(tracks, frame, config)
        matched_tracks = set()

        for track_index, measurement_index in matches:
            tracks[track_index].update(frame[measurement_index], time)
            matched_tracks.add(track_index)

        for track_index, track in enumerate(tracks):
            if track_index not in matched_tracks:
                track.miss(time)

        unused = [m for i, m in enumerate(frame) if i not in used_measurements]
        manager.create_tracks(unused)
        manager.step_lifecycle()

    return manager.result("gnn", config)

