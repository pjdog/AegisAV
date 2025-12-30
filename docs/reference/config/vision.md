# Vision Configuration

Location: `configs/aegis_config.yaml` under `vision`.

Fields:

- `enabled` (bool) - Enable vision pipeline.
- `use_real_detector` (bool) - Use real YOLO (GPU recommended).
- `model_path` (string) - Model name or path.
- `device` (string) - `auto`, `cpu`, `cuda`, or `cuda:0`.
- `confidence_threshold` (float) - Detector confidence threshold.
- `iou_threshold` (float) - Detector IOU threshold.
- `image_size` (int) - Input image size.
- `camera_resolution` (tuple) - `[width, height]`.
- `save_images` (bool) - Save captures.
- `image_output_dir` (string) - Output directory for captures.

Environment overrides:

- `AEGIS_VISION_ENABLED`
- `AEGIS_VISION_MODEL`
- `AEGIS_VISION_DEVICE`
- `AEGIS_VISION_REAL_DETECTOR`
