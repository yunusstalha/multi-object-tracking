import itertools

import numpy as np
from scipy.stats import multivariate_normal

from mot.config import gate_threshold
from mot.track import TrackManager


def validation_matrix(tracks, measurements, config):
    matrix = np.zeros((len(tracks), len(measurements)), dtype=bool)
    threshold = gate_threshold(config)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            matrix[i, j] = track.kf.mahalanobis2(measurement) <= threshold
    return matrix


def generate_hypotheses(valid):
    choices = []
    for row in valid:
        choices.append([-1] + np.where(row)[0].tolist())

    hypotheses = []
    for hypothesis in itertools.product(*choices):
        assigned = [m for m in hypothesis if m >= 0]
        if len(assigned) == len(set(assigned)):
            hypotheses.append(hypothesis)
    return hypotheses


def hypothesis_probability(hypothesis, tracks, measurements, config):
    probability = 1.0
    used = set()

    for track_index, measurement_index in enumerate(hypothesis):
        track = tracks[track_index]
        if measurement_index < 0:
            probability *= 1.0 - config["p_det"] * config["p_gate"]
            continue

        used.add(measurement_index)
        innovation = measurements[measurement_index] - track.kf.predicted_measurement
        likelihood = multivariate_normal.pdf(
            innovation,
            mean=np.zeros(4),
            cov=track.kf.innovation_covariance,
            allow_singular=False,
        )
        probability *= config["p_det"] * likelihood

    false_alarms = len(measurements) - len(used)
    clutter_density = config["clutter_per_step"] / max(_area(config), 1.0)
    probability *= max(clutter_density, 1e-12) ** false_alarms
    return probability


def marginal_probabilities(tracks, measurements, config):
    valid = validation_matrix(tracks, measurements, config)
    hypotheses = generate_hypotheses(valid)
    max_hypotheses = config.get("max_jpda_hypotheses", 20000)
    if len(hypotheses) > max_hypotheses:
        marginals = approximate_marginals(tracks, measurements, valid, config)
        miss = 1.0 - np.minimum(1.0, marginals.sum(axis=1))
        return marginals, miss, [], np.array([])

    weights = np.array(
        [hypothesis_probability(h, tracks, measurements, config) for h in hypotheses],
        dtype=float,
    )
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(hypotheses)) / len(hypotheses)
    else:
        weights = weights / total

    marginals = np.zeros((len(tracks), len(measurements)), dtype=float)
    miss = np.zeros(len(tracks), dtype=float)
    for hypothesis, weight in zip(hypotheses, weights):
        for track_index, measurement_index in enumerate(hypothesis):
            if measurement_index < 0:
                miss[track_index] += weight
            else:
                marginals[track_index, measurement_index] += weight

    return marginals, miss, hypotheses, weights


def approximate_marginals(tracks, measurements, valid, config):
    likelihoods = np.zeros((len(tracks), len(measurements)), dtype=float)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            if not valid[i, j]:
                continue
            innovation = measurement - track.kf.predicted_measurement
            likelihoods[i, j] = multivariate_normal.pdf(
                innovation,
                mean=np.zeros(4),
                cov=track.kf.innovation_covariance,
                allow_singular=False,
            )

    marginals = np.zeros_like(likelihoods)
    for j in range(len(measurements)):
        total = likelihoods[:, j].sum()
        if total > 0:
            marginals[:, j] = config["p_det"] * likelihoods[:, j] / total

    for i in range(len(tracks)):
        total = marginals[i].sum()
        if total > config["p_det"]:
            marginals[i] *= config["p_det"] / total
    return marginals


def run_jpda(measurements, config):
    manager = TrackManager(config)

    for time, frame in enumerate(measurements):
        frame = np.asarray(frame, dtype=float)
        tracks = manager.all_active()
        for track in tracks:
            track.predict()

        if len(tracks) > 0 and len(frame) > 0:
            marginals = component_marginals(tracks, frame, config)
            used = set(np.where(marginals.sum(axis=0) > 0.05)[0])
            for i, track in enumerate(tracks):
                probs = marginals[i]
                positive = probs > 1e-9
                if positive.any():
                    track.update_jpda(frame[positive], probs[positive], time)
                else:
                    track.miss(time)
        else:
            used = set()
            for track in tracks:
                track.miss(time)

        unused = [m for i, m in enumerate(frame) if i not in used]
        manager.create_tracks(unused)
        manager.step_lifecycle()

    return manager.result("jpda", config)


def component_marginals(tracks, measurements, config):
    valid = validation_matrix(tracks, measurements, config)
    marginals = np.zeros((len(tracks), len(measurements)), dtype=float)
    visited_tracks = set()
    visited_measurements = set()

    for start_track in range(len(tracks)):
        if start_track in visited_tracks:
            continue

        track_group, measurement_group = _connected_component(valid, start_track)
        visited_tracks.update(track_group)
        visited_measurements.update(measurement_group)

        if len(measurement_group) == 0:
            continue

        local_tracks = [tracks[i] for i in track_group]
        local_measurements = measurements[measurement_group]
        local_marginals, _, _, _ = marginal_probabilities(local_tracks, local_measurements, config)
        for local_i, global_i in enumerate(track_group):
            for local_j, global_j in enumerate(measurement_group):
                marginals[global_i, global_j] = local_marginals[local_i, local_j]

    return marginals


def _connected_component(valid, start_track):
    tracks = set()
    measurements = set()
    queue = [("track", start_track)]

    while queue:
        kind, index = queue.pop()
        if kind == "track":
            if index in tracks:
                continue
            tracks.add(index)
            for measurement_index in np.where(valid[index])[0]:
                if measurement_index not in measurements:
                    queue.append(("measurement", int(measurement_index)))
        else:
            if index in measurements:
                continue
            measurements.add(index)
            for track_index in np.where(valid[:, index])[0]:
                if track_index not in tracks:
                    queue.append(("track", int(track_index)))

    return sorted(tracks), sorted(measurements)


def _area(config):
    xmin, xmax, ymin, ymax = config["area"]
    return (xmax - xmin) * (ymax - ymin)
