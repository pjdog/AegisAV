# Selected SLAM + Splat Stack

This file captures the chosen stacks for Phase 2 and Phase 3 and how to run the
current scaffolding in this repo.

## SLAM (Phase 2)

Primary: ORB-SLAM3 (RGB-D) for AirSim depth captures.  
Fallback: VINS-Fusion (RGB + IMU) when depth is not available.

Scaffold runner (telemetry-backed in simulation):
```
python mapping/slam_runner.py data/maps/sequence_YYYYMMDD_HHMMSS
```

Outputs (under `data/slam_runs/run_*`):
- `pose_graph.json`
- `slam_status.json`
- `map_points.ply` (sparse preview from depth)

## Gaussian Splatting (Phase 3)

Primary: Nerfstudio 3D Gaussian Splatting pipeline (offline).  
Alternate: Inria 3DGS / gsplat if Nerfstudio is not available.

Scaffold trainer:
```
python mapping/splat_trainer.py data/slam_runs/run_*/pose_graph.json
```

Outputs (under `data/splats/scene_*`):
- `scene.json`
- `preview.ply`
