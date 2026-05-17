import os
import tempfile
from functools import lru_cache

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mot_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "mot_cache"))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_scenario(scenario, path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_measurements(ax, scenario["measurements"], alpha=0.25)
    _plot_truth(ax, scenario)
    _finish_axes(ax, "Synthetic measurements and ground truth")
    return _save_or_show(fig, path)


def plot_tracks(scenario, results, path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_measurements(ax, scenario["measurements"], alpha=0.12)
    _plot_truth(ax, scenario)

    colors = {"gnn": "tab:blue", "jpda": "tab:green"}
    for result in results:
        color = colors.get(result["tracker"], None)
        for track in result["tracks"]:
            if len(track) == 0:
                continue
            ax.plot(track[:, 2], track[:, 4], linewidth=1.7, color=color, alpha=0.9)

    _finish_axes(ax, "Tracker comparison")
    return _save_or_show(fig, path)


def plot_track_count(scenario, results, path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    truth_count = [len(ids) for ids in scenario["truth_ids"]]
    ax.plot(truth_count, label="true objects", color="black", linewidth=2)

    for result in results:
        counts = np.zeros(len(scenario["measurements"]))
        for track in result["tracks"]:
            for time in track[:, 0].astype(int):
                if 0 <= time < len(counts):
                    counts[time] += 1
        ax.plot(counts, label=result["tracker"])

    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, path)


def plot_frame(scenario, result, time, path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    frame = np.asarray(scenario["measurements"][time], dtype=float)
    for measurement in frame:
        _add_box(ax, measurement, "tab:red", alpha=0.45)

    for track in result["tracks"]:
        rows = track[track[:, 0] == time]
        for row in rows:
            state = np.array([row[2], row[4], row[6], row[7]])
            _add_box(ax, state, "tab:blue", alpha=0.75)
            ax.text(row[2], row[4], str(int(row[1])), fontsize=8)

    _plot_truth_frame(ax, scenario, time)
    _finish_axes(ax, f"{result['tracker']} frame {time}")
    return _save_or_show(fig, path)


def plot_image_frame(scenario, results, time, image_dir=None, path=None, tail_length=30):
    image_dir = image_dir or scenario["config"].get("image_dir")
    if image_dir is None:
        return plot_frame(scenario, results[0], time, path)

    image = plt.imread(_image_path(image_dir, time))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)

    frame = np.asarray(scenario["measurements"][time], dtype=float)
    for measurement in frame:
        _add_box(ax, measurement, "yellow", alpha=0.65)

    _draw_image_tracks(ax, results, time, tail_length)

    ax.set_title(f"PETS09 frame {time + 1}")
    ax.set_axis_off()
    _image_tracking_legend(ax)
    return _save_or_show(fig, path)


def render_image_frames(scenario, results, image_dir, out_dir, start=0, end=None):
    end = len(scenario["measurements"]) if end is None else min(end, len(scenario["measurements"]))
    os.makedirs(out_dir, exist_ok=True)
    for time in range(start, end):
        path = os.path.join(out_dir, f"frame_{time + 1:06d}.png")
        plot_image_frame(scenario, results, time, image_dir=image_dir, path=path)


def animate_image_frames(scenario, results, image_dir, path, start=0, end=None, tail_length=30):
    end = len(scenario["measurements"]) if end is None else min(end, len(scenario["measurements"]))
    fig, ax = plt.subplots(figsize=(11, 8))

    def update(time):
        ax.clear()
        image = plt.imread(_image_path(image_dir, time))
        ax.imshow(image)
        frame = np.asarray(scenario["measurements"][time], dtype=float)
        for measurement in frame:
            _add_box(ax, measurement, "yellow", alpha=0.65)
        _draw_image_tracks(ax, results, time, tail_length)
        ax.set_title(f"PETS09 frame {time + 1}")
        ax.set_axis_off()
        _image_tracking_legend(ax)

    animation = FuncAnimation(fig, update, frames=range(start, end), blit=False)
    try:
        animation.save(path, fps=7)
    except Exception:
        frame_dir = os.path.splitext(path)[0] + "_frames"
        render_image_frames(scenario, results, image_dir, frame_dir, start=start, end=end)
    plt.close(fig)


def animate(scenario, result, path, tail_length=30):
    fig, ax = plt.subplots(figsize=(8, 6))
    limits = _scenario_limits(scenario, [result])

    def update(time):
        ax.clear()
        plot_frame_on_axes(ax, scenario, result, time, limits=limits, tail_length=tail_length)

    animation = FuncAnimation(fig, update, frames=len(scenario["measurements"]), blit=False)
    try:
        animation.save(path, fps=12)
    except Exception:
        frame_dir = os.path.splitext(path)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        for time in range(len(scenario["measurements"])):
            ax.clear()
            plot_frame_on_axes(ax, scenario, result, time, limits=limits, tail_length=tail_length)
            fig.savefig(os.path.join(frame_dir, f"frame_{time:04d}.png"), dpi=140)
    plt.close(fig)


def plot_frame_on_axes(ax, scenario, result, time, limits=None, tail_length=30):
    frame = np.asarray(scenario["measurements"][time], dtype=float)
    for measurement in frame:
        _add_box(ax, measurement, "tab:red", alpha=0.45)
    for track in result["tracks"]:
        if len(track) == 0:
            continue
        tail = track[(track[:, 0] >= time - tail_length) & (track[:, 0] <= time)]
        if len(tail) > 1:
            ax.plot(tail[:, 2], tail[:, 4], color="tab:blue", linewidth=1.8, alpha=0.85)
        rows = track[track[:, 0] == time]
        for row in rows:
            _add_box(ax, [row[2], row[4], row[6], row[7]], "tab:blue", alpha=0.75)
            ax.text(row[2], row[4], str(int(row[1])), fontsize=8)
    _plot_truth_frame(ax, scenario, time)
    _finish_axes(ax, f"{result['tracker'].upper()} frame {time}", limits=limits)
    _tracking_legend(ax)


def _plot_measurements(ax, measurements, alpha):
    xs = []
    ys = []
    for frame in measurements:
        frame = np.asarray(frame, dtype=float)
        if len(frame):
            xs.extend(frame[:, 0])
            ys.extend(frame[:, 1])
    ax.scatter(xs, ys, s=8, c="tab:red", alpha=alpha, label="measurements")


def _plot_truth(ax, scenario):
    object_ids = sorted({int(i) for ids in scenario["truth_ids"] for i in ids})
    for object_id in object_ids:
        points = []
        for frame_truth, frame_ids in zip(scenario["truth"], scenario["truth_ids"]):
            for state, truth_id in zip(frame_truth, frame_ids):
                if int(truth_id) == object_id:
                    points.append([state[0], state[2]])
        if points:
            points = np.asarray(points)
            ax.plot(points[:, 0], points[:, 1], "--", linewidth=2, label=f"truth {object_id}")


def _plot_truth_frame(ax, scenario, time):
    for state, truth_id in zip(scenario["truth"][time], scenario["truth_ids"][time]):
        _add_box(ax, [state[0], state[2], state[4], state[5]], "black", alpha=0.9)
        ax.text(state[0], state[2], f"T{int(truth_id)}", fontsize=8)


def _add_box(ax, measurement, color, alpha):
    x, y, w, h = np.asarray(measurement, dtype=float)
    rect = patches.Rectangle(
        (x - w / 2, y - h / 2),
        w,
        h,
        linewidth=1.4,
        edgecolor=color,
        facecolor="none",
        alpha=alpha,
    )
    ax.add_patch(rect)


def _finish_axes(ax, title, limits=None):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")


def _save_or_show(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.show()
    return None


def _image_path(image_dir, time):
    numbered_path = os.path.join(image_dir, f"{time + 1:06d}.jpg")
    if os.path.exists(numbered_path):
        return numbered_path

    paths = _sorted_image_paths(image_dir)
    if time < len(paths):
        return paths[time]
    return numbered_path


@lru_cache(maxsize=8)
def _sorted_image_paths(image_dir):
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def _scenario_limits(scenario, results=None, pad_ratio=0.08):
    xs = []
    ys = []
    for frame in scenario["measurements"]:
        frame = np.asarray(frame, dtype=float)
        if len(frame):
            xs.extend(frame[:, 0])
            ys.extend(frame[:, 1])
    for frame in scenario["truth"]:
        frame = np.asarray(frame, dtype=float)
        if len(frame):
            xs.extend(frame[:, 0])
            ys.extend(frame[:, 2])
    for result in results or []:
        for track in result["tracks"]:
            if len(track):
                xs.extend(track[:, 2])
                ys.extend(track[:, 4])

    if not xs or not ys:
        return (-1, 1, -1, 1)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    xpad = width * pad_ratio
    ypad = height * pad_ratio
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _tracking_legend(ax):
    handles = [
        Patch(edgecolor="tab:red", facecolor="none", label="measurement"),
        Patch(edgecolor="tab:blue", facecolor="none", label="track box"),
        Line2D([0], [0], color="tab:blue", linewidth=1.8, label="track tail"),
        Patch(edgecolor="black", facecolor="none", label="truth"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)


def _draw_image_tracks(ax, results, time, tail_length):
    colors = {"gnn": "deepskyblue", "jpda": "lime"}
    for result in results:
        color = colors.get(result["tracker"], "white")
        for track in result["tracks"]:
            if len(track) == 0:
                continue

            tail = track[(track[:, 0] >= time - tail_length) & (track[:, 0] <= time)]
            if len(tail) > 1:
                ax.plot(tail[:, 2], tail[:, 4], color=color, linewidth=1.8, alpha=0.9)

            rows = track[track[:, 0] == time]
            for row in rows:
                _add_box(ax, [row[2], row[4], row[6], row[7]], color, alpha=0.95)
                ax.text(
                    row[2],
                    row[4],
                    f"{result['tracker']}:{int(row[1])}",
                    fontsize=7,
                    color=color,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
                )


def _image_tracking_legend(ax):
    handles = [
        Patch(edgecolor="yellow", facecolor="none", label="detection"),
        Patch(edgecolor="deepskyblue", facecolor="none", label="GNN box"),
        Line2D([0], [0], color="deepskyblue", linewidth=1.8, label="GNN tail"),
        Patch(edgecolor="lime", facecolor="none", label="JPDAF box"),
        Line2D([0], [0], color="lime", linewidth=1.8, label="JPDAF tail"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.75)
