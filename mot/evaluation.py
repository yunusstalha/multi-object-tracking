import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_result(scenario, result, max_distance=60.0):
    matches = []
    misses = 0
    false_tracks = 0
    id_switches = 0
    previous_match = {}
    truth_track_ids = {}

    for time in range(len(scenario["truth"])):
        truth = np.asarray(scenario["truth"][time], dtype=float)
        truth_ids = np.asarray(scenario["truth_ids"][time], dtype=int)
        estimates = _estimates_at_time(result, time)

        if len(truth) == 0:
            false_tracks += len(estimates)
            continue
        if len(estimates) == 0:
            misses += len(truth)
            continue

        distance = np.zeros((len(truth), len(estimates)))
        for i, state in enumerate(truth):
            for j, estimate in enumerate(estimates):
                distance[i, j] = np.linalg.norm(state[[0, 2]] - estimate[[2, 4]])

        rows, cols = linear_sum_assignment(distance)
        matched_truth = set()
        matched_estimates = set()

        for row, col in zip(rows, cols):
            if distance[row, col] > max_distance:
                continue

            truth_id = int(truth_ids[row])
            track_id = int(estimates[col][1])
            if truth_id in previous_match and previous_match[truth_id] != track_id:
                id_switches += 1
            previous_match[truth_id] = track_id
            truth_track_ids.setdefault(truth_id, []).append(track_id)

            matches.append(distance[row, col])
            matched_truth.add(row)
            matched_estimates.add(col)

        misses += len(truth) - len(matched_truth)
        false_tracks += len(estimates) - len(matched_estimates)

    match_distances = np.asarray(matches, dtype=float)
    rmse = float(np.sqrt(np.mean(match_distances**2))) if len(match_distances) else float("nan")
    mean_error = float(np.mean(match_distances)) if len(match_distances) else float("nan")
    truth_frames = sum(len(frame) for frame in scenario["truth"])
    mostly_tracked_ids = _mostly_tracked_ids(truth_track_ids)

    return {
        "tracker": result["tracker"],
        "matched_truth_frames": int(len(matches)),
        "truth_frames": int(truth_frames),
        "missed_truth_frames": int(misses),
        "false_track_frames": int(false_tracks),
        "id_switches": int(id_switches),
        "mostly_tracked_ids": mostly_tracked_ids,
        "position_rmse": rmse,
        "mean_position_error": mean_error,
    }


def evaluate_results(scenario, results, max_distance=60.0):
    return [evaluate_result(scenario, result, max_distance) for result in results]


def evaluate_detection_result(scenario, result, max_distance=45.0):
    matched = 0
    track_frames = 0
    unmatched_track_frames = 0
    distances = []
    smoothness = []

    for time, frame in enumerate(scenario["measurements"]):
        detections = np.asarray(frame, dtype=float)
        estimates = _estimates_at_time(result, time)
        track_frames += len(estimates)

        if len(detections) == 0:
            unmatched_track_frames += len(estimates)
            continue
        if len(estimates) == 0:
            continue

        distance = np.zeros((len(detections), len(estimates)))
        for i, detection in enumerate(detections):
            for j, estimate in enumerate(estimates):
                distance[i, j] = np.linalg.norm(detection[:2] - estimate[[2, 4]])

        rows, cols = linear_sum_assignment(distance)
        matched_estimates = set()
        for row, col in zip(rows, cols):
            if distance[row, col] <= max_distance:
                matched += 1
                distances.append(distance[row, col])
                matched_estimates.add(col)
        unmatched_track_frames += len(estimates) - len(matched_estimates)

    for track in result["tracks"]:
        if len(track) < 3:
            continue
        xy = track[:, [2, 4]]
        accel = np.diff(xy, n=2, axis=0)
        if len(accel):
            smoothness.extend(np.linalg.norm(accel, axis=1))

    lengths = [len(track) for track in result["tracks"] if len(track)]
    detection_count = sum(len(frame) for frame in scenario["measurements"])
    distances = np.asarray(distances, dtype=float)
    smoothness = np.asarray(smoothness, dtype=float)

    return {
        "tracker": result["tracker"],
        "detections": int(detection_count),
        "matched_detections": int(matched),
        "coverage": float(matched / detection_count) if detection_count else 0.0,
        "tracks": int(len(lengths)),
        "track_frames": int(track_frames),
        "unmatched_track_frames": int(unmatched_track_frames),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "mean_detection_error": float(np.mean(distances)) if len(distances) else float("nan"),
        "mean_acceleration": float(np.mean(smoothness)) if len(smoothness) else float("nan"),
    }


def evaluate_detection_results(scenario, results, max_distance=45.0):
    return [evaluate_detection_result(scenario, result, max_distance) for result in results]


def save_metrics(path, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def print_metrics(metrics):
    if metrics and "coverage" in metrics[0]:
        print_detection_metrics(metrics)
        return

    header = (
        "tracker  matched/truth  missed  false_tracks  id_switches  "
        "rmse   mean_error  dominant_ids"
    )
    print(header)
    print("-" * len(header))
    for item in metrics:
        print(
            f"{item['tracker']:<8} "
            f"{item['matched_truth_frames']:>4}/{item['truth_frames']:<5} "
            f"{item['missed_truth_frames']:>6} "
            f"{item['false_track_frames']:>13} "
            f"{item['id_switches']:>11} "
            f"{item['position_rmse']:>6.2f} "
            f"{item['mean_position_error']:>11.2f} "
            f"{item['mostly_tracked_ids']}"
        )


def print_detection_metrics(metrics):
    header = (
        "tracker  coverage  matched/dets  tracks  track_frames  unmatched  "
        "mean_len  med_len  det_err  accel"
    )
    print(header)
    print("-" * len(header))
    for item in metrics:
        print(
            f"{item['tracker']:<8} "
            f"{item['coverage']:>7.3f} "
            f"{item['matched_detections']:>5}/{item['detections']:<5} "
            f"{item['tracks']:>6} "
            f"{item['track_frames']:>12} "
            f"{item['unmatched_track_frames']:>9} "
            f"{item['mean_track_length']:>8.1f} "
            f"{item['median_track_length']:>7.1f} "
            f"{item['mean_detection_error']:>7.2f} "
            f"{item['mean_acceleration']:>6.2f}"
        )


def _estimates_at_time(result, time):
    rows = []
    for track in result["tracks"]:
        if len(track) == 0:
            continue
        at_time = track[track[:, 0] == time]
        rows.extend(at_time)
    return np.asarray(rows, dtype=float)


def _mostly_tracked_ids(truth_track_ids):
    summary = {}
    for truth_id, track_ids in truth_track_ids.items():
        values, counts = np.unique(track_ids, return_counts=True)
        summary[int(truth_id)] = int(values[np.argmax(counts)])
    return summary
