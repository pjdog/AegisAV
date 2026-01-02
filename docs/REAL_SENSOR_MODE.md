# Real Sensor Mode (Calibration + Capture Walkthrough)

This walkthrough covers the real-camera capture flow and how to feed it into the SLAM/splat pipeline.

## 1) Enable the Real Sensor Profile

Copy the profile into the active config:

```bash
cp configs/aegis_config.real_sensor.yaml configs/aegis_config.yaml
```

Key defaults:
- Simulation off
- Mapping on
- Preflight mapping off

## 2) Calibrate the Camera

Capture chessboard images and run the calibration tool:

```bash
python mapping/calibrate_camera.py path/to/chessboard_images \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size-m 0.025 \
  --output data/calibration/camera_calibration.json
```

## 3) Capture a Real Sequence

Run the capture tool (adjust frames/interval as needed):

```bash
python mapping/real_capture.py \
  --output-dir data/maps/real_capture \
  --frames 300 \
  --interval-s 0.3 \
  --calibration data/calibration/camera_calibration.json
```

This creates:
- `data/maps/real_capture/frames/*.png`
- `data/maps/real_capture/frames/*.json` (metadata per frame)

## 4) Run SLAM on the Capture

```bash
python mapping/slam_runner.py \
  --input-dir data/maps/real_capture \
  --output-dir data/slam_runs/run_real_001
```

## 5) Train a Splat Scene (Optional)

```bash
python mapping/splat_trainer.py data/slam_runs/run_real_001/pose_graph.json
```

## 6) View the Map

- Open the dashboard map page: `http://<host>:<port>/dashboard/maps`
- The fused map list will populate as SLAM runs are processed.

## Notes

- If you have an external SLAM backend, set `ORB_SLAM3_CMD` or `VINS_FUSION_CMD`
  to point at your runner. The command receives `{input_dir}` and `{output_dir}`.
- For IMU/GPS integration, ensure the capture metadata JSON includes `imu` or
  update your capture pipeline to add those fields.
