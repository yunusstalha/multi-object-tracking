import json
import os

import numpy as np

from mot.config import default_config, model_matrices


def crossing_targets_config():
    config = default_config()
    config["objects"] = [
        {"id": 1, "start": 0, "end": 160, "state": [-420, 5.4, -120, 1.5, 22, 12]},
        {"id": 2, "start": 20, "end": 180, "state": [420, -5.2, 120, -1.4, 20, 14]},
        {"id": 3, "start": 55, "end": 150, "state": [-80, 1.8, 260, -4.0, 18, 10]},
    ]
    return config


def four_tracks_config():
    config = default_config()
    config["seed"] = 11
    config["steps"] = 170
    config["clutter_per_step"] = 5
    config["objects"] = [
        {"id": 1, "start": 0, "end": 125, "state": [-470, 5.8, -150, 1.8, 22, 12]},
        {"id": 2, "start": 18, "end": 170, "state": [430, -5.1, 140, -1.5, 20, 14]},
        {"id": 3, "start": 45, "end": 135, "state": [-120, 2.0, 280, -4.0, 18, 10]},
        {"id": 4, "start": 82, "end": 165, "state": [60, 0.9, -280, 4.3, 24, 13]},
    ]
    return config


def jpda_advantage_config():
    config = default_config()
    config["seed"] = 23
    config["steps"] = 150
    config["p_det"] = 0.98
    config["p_gate"] = 0.999
    config["clutter_per_step"] = 3
    config["position_meas_std"] = 13.0
    config["size_meas_std"] = 1.2
    config["process_std"] = [0.08, 0.08, 0.03, 0.03]
    config["confirm_m"] = 1
    config["confirm_n"] = 1
    config["delete_m"] = 6
    config["delete_n"] = 8
    config["objects"] = [
        {"id": 1, "start": 0, "end": 130, "state": [-300, 4.8, -20, 0.18, 22, 12]},
        {"id": 2, "start": 0, "end": 130, "state": [300, -4.8, 20, -0.18, 22, 12]},
        {"id": 3, "start": 35, "end": 110, "state": [-170, 2.4, 220, -2.2, 18, 10]},
        {"id": 4, "start": 70, "end": 150, "state": [90, 0.7, -250, 3.3, 24, 13]},
    ]
    return config


def scenario_config(name):
    if name == "pets09":
        return pets09_config()
    if name == "jpda_advantage":
        return jpda_advantage_config()
    if name == "four":
        return four_tracks_config()
    return crossing_targets_config()


def pets09_config():
    config = default_config()
    config.update(
        {
            "seed": 0,
            "steps": 794,
            "dt": 0.4,
            "area": [0.0, 768.0, 0.0, 576.0],
            "process_std": [10.0, 10.0, 3.0, 3.0],
            "position_meas_std": 10.0,
            "size_meas_std": 8.0,
            "p_det": 1.0,
            "p_gate": 0.997,
            "gate_threshold": 15.0,
            "clutter_per_step": 2,
            "confirm_m": 2,
            "confirm_n": 3,
            "delete_m": 9,
            "delete_n": 10,
            "initial_pos_std": 30.0,
            "initial_vel_std": 12.0,
            "initial_size_std": 12.0,
            "max_jpda_hypotheses": 50000,
        }
    )
    return config


def generate_scenario(config=None):
    if config is None:
        config = crossing_targets_config()

    rng = np.random.default_rng(config["seed"])
    F, H, Q, R = model_matrices(config)
    steps = config["steps"]
    xmin, xmax, ymin, ymax = config["area"]

    measurements = []
    measurement_truth_ids = []
    truth = []
    truth_ids = []
    live_states = {}

    objects = config.get("objects", crossing_targets_config()["objects"])
    process_std = np.sqrt(np.diag(Q))

    for time in range(steps):
        frame_measurements = []
        frame_truth_ids = []
        frame_truth = []
        frame_ids = []

        for obj in objects:
            if time == obj["start"]:
                live_states[obj["id"]] = np.asarray(obj["state"], dtype=float)

            if obj["start"] <= time < obj["end"]:
                if time > obj["start"]:
                    noise = rng.normal(0.0, process_std)
                    live_states[obj["id"]] = F @ live_states[obj["id"]] + noise

                state = live_states[obj["id"]].copy()
                frame_truth.append(state)
                frame_ids.append(obj["id"])

                if rng.random() <= config["p_det"]:
                    measurement = H @ state + rng.multivariate_normal(np.zeros(4), R)
                    frame_measurements.append(measurement)
                    frame_truth_ids.append(obj["id"])

        for _ in range(config["clutter_per_step"]):
            w = rng.uniform(*config["clutter_size_range"])
            h = rng.uniform(*config["clutter_size_range"])
            frame_measurements.append([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax), w, h])
            frame_truth_ids.append(-1)

        order = rng.permutation(len(frame_measurements))
        frame_measurements = np.asarray(frame_measurements, dtype=float)
        frame_truth_ids = np.asarray(frame_truth_ids, dtype=int)
        if len(order) > 0:
            frame_measurements = frame_measurements[order]
            frame_truth_ids = frame_truth_ids[order]

        measurements.append(frame_measurements)
        measurement_truth_ids.append(frame_truth_ids)
        truth.append(np.asarray(frame_truth, dtype=float))
        truth_ids.append(np.asarray(frame_ids, dtype=int))

    return {
        "measurements": _object_array(measurements),
        "measurement_truth_ids": _object_array(measurement_truth_ids),
        "truth": _object_array(truth),
        "truth_ids": _object_array(truth_ids),
        "config": dict(config),
    }


def save_scenario(path, scenario):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        measurements=scenario["measurements"],
        measurement_truth_ids=scenario["measurement_truth_ids"],
        truth=scenario["truth"],
        truth_ids=scenario["truth_ids"],
        config=json.dumps(scenario["config"]),
    )


def load_scenario(path):
    if path.endswith(".npy"):
        measurements = np.load(path, allow_pickle=True)
        return {
            "measurements": measurements,
            "measurement_truth_ids": _object_array([np.full(len(m), -1) for m in measurements]),
            "truth": _object_array([np.empty((0, 6)) for _ in measurements]),
            "truth_ids": _object_array([np.empty((0,), dtype=int) for _ in measurements]),
            "config": default_config(),
        }

    data = np.load(path, allow_pickle=True)
    return {
        "measurements": data["measurements"],
        "measurement_truth_ids": data["measurement_truth_ids"],
        "truth": data["truth"],
        "truth_ids": data["truth_ids"],
        "config": json.loads(str(data["config"])),
    }


def load_pets09_detection_file(path="data/pets09_s2l1_detection.txt", max_frames=None):
    frames = []
    for line in open(path, "r", encoding="utf-8"):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue

        frame_id = int(parts[0])
        count = int(parts[1])
        detections = []
        for i in range(count):
            offset = 2 + 4 * i
            x, y, w, h = [float(value) for value in parts[offset : offset + 4]]
            detections.append([x + w / 2.0, y + h / 2.0, w, h])

        while len(frames) < frame_id:
            frames.append(np.empty((0, 4), dtype=float))
        frames[frame_id - 1] = np.asarray(detections, dtype=float)

        if max_frames is not None and len(frames) >= max_frames:
            frames = frames[:max_frames]
            break

    config = pets09_config()
    config["steps"] = len(frames)
    config["image_dir"] = "data/pets09_s2l1/img1"
    return {
        "measurements": _object_array(frames),
        "measurement_truth_ids": _object_array([np.full(len(frame), -1) for frame in frames]),
        "truth": _object_array([np.empty((0, 6), dtype=float) for _ in frames]),
        "truth_ids": _object_array([np.empty((0,), dtype=int) for _ in frames]),
        "config": config,
    }


def save_result(path, result):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        tracker=result["tracker"],
        tracks=result["tracks"],
        track_ids=result["track_ids"],
        config=json.dumps(result["config"]),
    )


def load_result(path):
    data = np.load(path, allow_pickle=True)
    return {
        "tracker": str(data["tracker"]),
        "tracks": data["tracks"],
        "track_ids": data["track_ids"],
        "config": json.loads(str(data["config"])),
    }


def _object_array(items):
    array = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        array[i] = np.asarray(item)
    return array
