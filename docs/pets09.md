# PETS09 Detection Example

The PETS09-S2L1 report outputs in this repo were generated from the public `apennisi/jpdaf_tracking` visual tracking example. The source PETS detection file and image frames are not redistributed here; users should get them from the original repository.

Expected local paths for regeneration:

```text
data/pets09_s2l1_detection.txt
data/pets09_s2l1/img1/
```

The original file stores each frame as:

```text
frame_id, detection_count, x, y, width, height, ...
```

where `x, y` are top-left bounding-box coordinates. The loader converts each detection to this repo's measurement format:

```text
[center_x, center_y, width, height]
```

After users copy the PETS image sequence to `data/pets09_s2l1/img1`, results can be rendered on the original video frames.

The PETS config uses the original repository's main values where they map cleanly:

- `PD = 1`
- `DT = 0.4`
- `R_MATRIX = [[100, 0], [0, 100]]`, mapped to `position_meas_std = 10`
- `LOCAL_GSIGMA = 15`, mapped to `gate_threshold = 15`
- `MAX_MISSED_RATE = 9`, mapped to deletion over a 10-frame window

The original `PG = 0.4` is not used as our chi-square gate probability because that made the bbox tracker reject most PETS associations. Instead, this repo uses the explicit `LOCAL_GSIGMA` threshold for gating and keeps `p_gate` for JPDA probability weighting.

Useful commands:

```bash
git clone https://github.com/apennisi/jpdaf_tracking.git /tmp/jpdaf_tracking
mkdir -p data/pets09_s2l1
cp /tmp/jpdaf_tracking/PETS09-S2L1/detection.txt data/pets09_s2l1_detection.txt
cp -R /tmp/jpdaf_tracking/PETS09-S2L1/img1 data/pets09_s2l1/img1
python main.py generate --scenario pets09 --data outputs/pets09_scenario.npz
python main.py run --tracker gnn --data outputs/pets09_scenario.npz --result outputs/pets09_gnn.npz
python main.py run --tracker jpda --data outputs/pets09_scenario.npz --result outputs/pets09_jpda.npz
python main.py evaluate --data outputs/pets09_scenario.npz --gnn outputs/pets09_gnn.npz --jpda outputs/pets09_jpda.npz --out outputs/pets09_metrics.json --max-distance 45
python main.py compare --data outputs/pets09_scenario.npz --gnn outputs/pets09_gnn.npz --jpda outputs/pets09_jpda.npz --out outputs/pets09_comparison.png
python main.py plot --data outputs/pets09_scenario.npz --gnn outputs/pets09_gnn.npz --jpda outputs/pets09_jpda.npz --frame 120 --out outputs/pets09_frame_0121.png
python main.py animate --data outputs/pets09_scenario.npz --gnn outputs/pets09_gnn.npz --jpda outputs/pets09_jpda.npz --start 100 --end 130 --out outputs/pets09_frames_100_130
```

The bundled PETS detections do not include ground-truth identities, so evaluation reports detection coverage, track count, track length, unmatched track frames, detection error, and motion smoothness.
