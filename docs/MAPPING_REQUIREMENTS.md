# Mapping Requirements (SLAM + Gaussian Splatting)

This document captures the baseline requirements for mapping outputs that feed AegisAV agents
and the performance targets used to validate the pipeline.

## Map Outputs for Agents
- 2D occupancy grid (meters per cell, bounds, unknown vs free vs occupied)
- 3D voxel map (optional, downsampled for planning)
- Obstacle list (bounding cylinders with center, radius, height)
- Costmap (inflated occupancy with distance-based penalties)
- Metadata (map_id, timestamp, frame_id, confidence, source, resolution, bounds)

## Update Cadence and Latency Targets
- Map update cadence: 1 Hz minimum for incremental updates.
- End-to-end latency: less than 500 ms from capture to map update.
- Staleness budget: map_age_s less than 2.0 before planner fallback.

## Acceptance Metrics
- Pose drift: less than 1.0 percent of distance traveled or 0.5 m after loop closure.
- Obstacle recall: at least 0.90 on scenario fixtures.
- Obstacle precision: at least 0.80 on scenario fixtures.
- Collision rate reduction: at least 50 percent vs baseline in simulation benchmarks.

## Coordinate Frames and Transforms
- Primary frame: AirSim NED world frame.
- Map origin: record the world pose used to anchor map tiles and obstacles.
- Export rules: include transforms for ENU or GPS when available; otherwise keep NED.
