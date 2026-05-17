# Data Generation

`mot/data.py` creates deterministic synthetic scenarios from a plain config dictionary.

The default scenario has three objects with different birth/death times and crossing trajectories. Each frame contains:

- true object states,
- noisy detections with probability `p_det`,
- uniformly distributed clutter measurements,
- truth IDs for analysis and plotting.

The most useful config fields are:

- `seed`: random seed.
- `steps`: number of frames.
- `area`: `[xmin, xmax, ymin, ymax]` clutter area.
- `p_det`: detection probability.
- `clutter_per_step`: number of false alarms per frame.
- `position_meas_std` and `size_meas_std`: measurement noise.
- `process_std`: motion and size process noise.
- `confirm_m`, `confirm_n`, `delete_m`, `delete_n`: track lifecycle rules.

Use `python main.py generate --steps 60 --seed 2` for a smaller reproducible scenario.

