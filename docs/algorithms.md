# Algorithms

## Kalman Model

The state is:

```text
[x, vx, y, vy, width, height]
```

Measurements are:

```text
[x, y, width, height]
```

Both trackers use the same constant-velocity Kalman filter and the same chi-square validation gate.

## GNN

Global Nearest Neighbor computes a gated cost for every track-measurement pair and solves the resulting assignment with the Hungarian algorithm. Each track receives at most one measurement, and each measurement can update at most one track.

This is simple and strong when targets are well separated. Around crossings, it can switch identities because it commits to one hard assignment per frame.

## JPDAF

JPDAF builds the validation matrix, enumerates all valid joint association hypotheses, normalizes their probabilities, and computes marginal association probabilities for each track-measurement pair.

Instead of choosing one measurement, each track receives a weighted Kalman update from all measurements with non-zero marginal probability. This is why JPDAF is usually smoother than GNN around ambiguous crossings.

The implementation is intentionally small-scale. If hypothesis count grows beyond `max_jpda_hypotheses`, it raises an error instead of silently doing something expensive.

