"""Map update loop for SLAM + splat fusion outputs."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from mapping.decision_context import MapContext, map_decision_logger
from mapping.map_fusion import MapFusion, MapFusionConfig
from mapping.map_storage import MapArtifactStore, MapArtifactStoreConfig
from mapping.safety_gates import MapUpdateGate, SafetyGateConfig, SafetyGateResult
from mapping.splat_proxy import build_planning_proxy, write_planning_proxy
from mapping.splat_change_detection import ChangeDetectionConfig, detect_splat_changes

logger = structlog.get_logger(__name__)


@dataclass
class MapUpdateConfig:
    """Configuration for the map update service."""

    enabled: bool = False
    update_interval_s: float = 2.0
    prefer_splat: bool = True
    max_map_age_s: float = 60.0
    min_quality_score: float = 0.3
    slam_dir: Path = Path("data/slam_runs")
    splat_dir: Path = Path("data/splats")
    map_resolution_m: float = 2.0
    tile_size_cells: int = 120
    voxel_size_m: float | None = None
    max_points: int = 200000
    min_points: int = 50
    fused_map_dir: Path = Path("data/maps/fused")
    fused_map_max_versions: int | None = None
    fused_map_max_age_days: int | None = None
    fused_map_keep_last: int = 3
    proxy_regen_interval_s: float = 60.0
    proxy_max_points: int = 120000

    # Proxy regeneration settings (Agent B Phase 3)
    proxy_regeneration_enabled: bool = True
    proxy_regeneration_cadence_s: float = 30.0
    proxy_max_points: int = 100000

    # Change detection (splat vs live depth)
    change_detection_enabled: bool = True
    change_detection_cooldown_s: float = 30.0
    change_detection_match_distance_m: float = 12.0
    change_detection_min_new_obstacles: int = 1
    proxy_force_regenerate: bool = False


class MapUpdateService:
    """Background service that updates server navigation maps."""

    def __init__(self, config: MapUpdateConfig, server_state: Any) -> None:
        self.config = config
        self.server_state = server_state
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_valid_map: dict[str, Any] | None = None
        self._decision_logger = map_decision_logger
        self._update_gate = MapUpdateGate(
            SafetyGateConfig(min_map_confidence=config.min_quality_score)
        )
        self._fusion = MapFusion(
            MapFusionConfig(
                resolution_m=config.map_resolution_m,
                tile_size_cells=getattr(config, "tile_size_cells", 120),
                voxel_size_m=config.voxel_size_m,
                min_points=config.min_points,
                max_points=config.max_points,
            )
        )
        self._store = MapArtifactStore(
            MapArtifactStoreConfig(
                base_dir=config.fused_map_dir,
                max_versions=config.fused_map_max_versions,
                max_age_days=config.fused_map_max_age_days,
                keep_last=config.fused_map_keep_last,
            )
        )
        # Proxy regeneration tracking (Agent B Phase 3)
        self._last_proxy_regen_time: float = 0.0
        self._proxy_regen_count: int = 0
        self._last_change_emit_time: float = 0.0

    async def start(self) -> None:
        if self._task:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("map_update_started", interval_s=self.config.update_interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("map_update_stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.update_once()
            except Exception as exc:
                self._record_update_error(f"map_update_exception: {exc}")
                logger.warning("map_update_failed", error=str(exc))
            await asyncio.sleep(self.config.update_interval_s)

    async def update_once(self) -> None:
        slam_pose_graph = self._find_latest(self.config.slam_dir, "pose_graph.json")
        slam_status = self._find_latest(self.config.slam_dir, "slam_status.json")
        splat_scene = self._find_latest(self.config.splat_dir, "scene.json")

        slam_confidence = 1.0
        slam_status_data: dict[str, Any] = {}
        pose_graph_summary: dict[str, Any] | None = None
        splat_quality = 1.0
        splat_proxy_path: Path | None = None
        splat_proxy_map: dict[str, Any] | None = None
        if slam_status and slam_status.exists():
            slam_status_data = self._load_slam_status(slam_status)
            try:
                slam_confidence = float(slam_status_data.get("pose_confidence", 1.0))
            except (TypeError, ValueError):
                slam_confidence = 1.0
        if slam_pose_graph and slam_pose_graph.exists():
            pose_graph_summary = self._load_pose_graph_summary(slam_pose_graph)
            if pose_graph_summary and slam_status_data:
                if "drift_estimate_m" not in slam_status_data:
                    slam_status_data["drift_estimate_m"] = pose_graph_summary.get(
                        "start_end_distance_m", 0.0
                    )

        splat_preview = None
        if splat_scene and splat_scene.exists():
            splat_preview, splat_quality = self._load_splat_scene(splat_scene)
            if splat_preview and splat_preview.exists():
                splat_proxy_path = splat_scene.parent / "planning_proxy.json"

        point_cloud_path = None
        map_id = None
        source = "slam"
        point_cloud_path, map_id, source, selection_info = self._select_map_source(
            splat_preview=splat_preview,
            splat_scene=splat_scene,
            splat_quality=splat_quality,
            slam_pose_graph=slam_pose_graph,
            slam_confidence=slam_confidence,
        )

        if not point_cloud_path:
            depth_only_map = self._build_depth_only_map()
            if depth_only_map:
                nav_map = depth_only_map
                map_context = MapContext.from_navigation_map(
                    nav_map,
                    stale_threshold_s=self.config.max_map_age_s,
                    min_quality_score=self.config.min_quality_score,
                )
                nav_map["last_updated"] = datetime.now().isoformat()
                self.server_state.navigation_map = nav_map
                if self.server_state.airsim_action_executor:
                    executor = self.server_state.airsim_action_executor
                    if hasattr(executor, "set_navigation_map"):
                        executor.set_navigation_map(nav_map)
                    else:
                        executor.set_avoid_zones(nav_map.get("obstacles", []))
                self._decision_logger.log_map_update(map_context, "airsim_depth")
            else:
                logger.debug("map_update_no_sources")
            return

        result = self._fusion.build_navigation_map(
            point_cloud_path=point_cloud_path,
            map_id=map_id,
            source=source,
            slam_confidence=slam_confidence,
            splat_quality=splat_quality,
            geo_ref=getattr(self.server_state, "airsim_geo_ref", None),
        )

        base_map = result.navigation_map
        if selection_info:
            metadata = base_map.get("metadata", {})
            metadata["source_selection"] = selection_info
            base_map["metadata"] = metadata

        if source == "splat" and self.config.change_detection_enabled:
            change_config = ChangeDetectionConfig(
                match_distance_m=self.config.change_detection_match_distance_m,
                min_new_obstacles=self.config.change_detection_min_new_obstacles,
            )
            changes = detect_splat_changes(
                base_map,
                getattr(self.server_state, "last_depth_capture", None),
                change_config,
            )
            if changes:
                await self._emit_map_change_anomaly(changes, map_id)
        nav_map = self._merge_depth_obstacles(dict(base_map))
        map_context = MapContext.from_navigation_map(
            nav_map,
            stale_threshold_s=self.config.max_map_age_s,
            min_quality_score=self.config.min_quality_score,
        )
        previous_map = self._last_valid_map
        gate_result = self._update_gate.check_update(nav_map, previous_map)

        # Record gate result in history (Agent B Phase 6)
        self._record_gate_result(gate_result, source, map_context.map_quality_score)

        apply_update = (
            gate_result.result in (SafetyGateResult.PASS, SafetyGateResult.WARN)
            and map_context.map_valid
        )
        if apply_update:
            nav_map["last_updated"] = datetime.now().isoformat()
            self._last_valid_map = base_map
            self.server_state.navigation_map = nav_map
            if hasattr(self.server_state, "last_valid_navigation_map"):
                self.server_state.last_valid_navigation_map = nav_map
            if self.server_state.airsim_action_executor:
                executor = self.server_state.airsim_action_executor
                if hasattr(executor, "set_navigation_map"):
                    executor.set_navigation_map(nav_map)
                else:
                    executor.set_avoid_zones(nav_map.get("obstacles", []))
            if getattr(self.server_state, "flight_controller", None):
                controller = self.server_state.flight_controller
                if hasattr(controller, "set_navigation_map"):
                    controller.set_navigation_map(nav_map)
            if getattr(self.server_state, "autonomy_pipeline", None):
                pipeline = self.server_state.autonomy_pipeline
                if hasattr(pipeline, "set_navigation_map"):
                    pipeline.set_navigation_map(nav_map)
            try:
                self.server_state.fused_map_artifact = self._store.store(nav_map)
            except Exception as exc:
                logger.warning("map_artifact_store_failed", error=str(exc))
            self._clear_update_error()
        elif self._last_valid_map:
            fallback_map = self._merge_depth_obstacles(dict(self._last_valid_map))
            self.server_state.navigation_map = fallback_map
            self._record_update_error(
                gate_result.reason
                if gate_result.result == SafetyGateResult.REJECT
                else "map_invalid"
            )
        else:
            if nav_map.get("obstacles"):
                self.server_state.navigation_map = nav_map
            self._record_update_error(
                gate_result.reason
                if gate_result.result == SafetyGateResult.REJECT
                else "map_invalid"
            )

        if splat_proxy_path and splat_preview and splat_preview.exists():
            if self._should_regenerate_proxy(
                splat_scene,
                splat_preview,
                splat_proxy_path,
                min_interval_s=self.config.proxy_regen_interval_s,
            ):
                if source == "splat" and point_cloud_path == splat_preview:
                    splat_proxy_map = dict(nav_map)
                    splat_proxy_map["source"] = "splat_proxy"
                    write_planning_proxy(splat_proxy_map, splat_scene.parent)
                else:
                    proxy_config = MapFusionConfig(
                        resolution_m=self._fusion.config.resolution_m,
                        tile_size_cells=self._fusion.config.tile_size_cells,
                        voxel_size_m=self._fusion.config.voxel_size_m,
                        min_points=self._fusion.config.min_points,
                        max_points=self.config.proxy_max_points,
                    )
                    splat_proxy_map, splat_proxy_path = build_planning_proxy(
                        splat_preview,
                        splat_scene.parent,
                        map_id=map_id or splat_scene.parent.name,
                        scenario_id=nav_map.get("scenario_id"),
                        source="splat_proxy",
                        splat_quality=splat_quality,
                        config=proxy_config,
                    )
            else:
                splat_proxy_map = self._read_json(splat_proxy_path)

        if slam_status_data:
            self.server_state.slam_status = slam_status_data
        if pose_graph_summary:
            self.server_state.slam_pose_graph_summary = pose_graph_summary
        if splat_scene:
            splat_data = self._read_json(splat_scene)
            splat_data["scene_path"] = str(splat_scene)
            if splat_proxy_path and splat_proxy_path.exists():
                if splat_proxy_map is None:
                    splat_proxy_map = self._read_json(splat_proxy_path)
                splat_data["planning_proxy"] = str(splat_proxy_path)
                splat_data["planning_proxy_updated_at"] = (
                    splat_proxy_map.get("last_updated")
                    or splat_proxy_map.get("generated_at")
                )
                splat_data["planning_proxy_obstacle_count"] = len(
                    splat_proxy_map.get("obstacles", []) if splat_proxy_map else []
                )
            self.server_state.splat_artifacts = splat_data

        obstacle_count_delta = 0
        if previous_map:
            previous_count = len(previous_map.get("obstacles", []))
            obstacle_count_delta = len(nav_map.get("obstacles", [])) - previous_count
        self._decision_logger.log_map_update(
            map_context,
            source,
            obstacle_count_delta=obstacle_count_delta,
        )

    @staticmethod
    def _find_latest(base_dir: Path, pattern: str) -> Path | None:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return None
        candidates = list(base_dir.rglob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def _resolve_point_cloud_from_pose_graph(path: Path) -> Path | None:
        data = MapUpdateService._read_json(path)
        if not data:
            return None
        point_path = data.get("point_cloud")
        if point_path:
            candidate = Path(point_path)
            if not candidate.is_absolute():
                candidate = path.parent / point_path
            if candidate.exists():
                return candidate
        map_points = path.parent / "map_points.ply"
        return map_points if map_points.exists() else None

    def _select_map_source(
        self,
        *,
        splat_preview: Path | None,
        splat_scene: Path | None,
        splat_quality: float,
        slam_pose_graph: Path | None,
        slam_confidence: float,
    ) -> tuple[Path | None, str | None, str, dict[str, Any] | None]:
        splat_candidate = None
        if splat_preview and splat_scene:
            splat_score = self._score_source(splat_quality, splat_scene)
            if self.config.prefer_splat:
                splat_score = min(1.0, splat_score + 0.05)
            splat_candidate = {
                "path": splat_preview,
                "map_id": splat_scene.parent.name,
                "source": "splat",
                "score": splat_score,
            }

        slam_candidate = None
        if slam_pose_graph:
            slam_preview = self._resolve_point_cloud_from_pose_graph(slam_pose_graph)
            if slam_preview:
                slam_score = self._score_source(slam_confidence, slam_pose_graph)
                slam_candidate = {
                    "path": slam_preview,
                    "map_id": slam_pose_graph.parent.name,
                    "source": "slam",
                    "score": slam_score,
                }

        if splat_candidate and slam_candidate:
            selected = (
                splat_candidate
                if splat_candidate["score"] >= slam_candidate["score"]
                else slam_candidate
            )
            selection_info = {
                "selected_source": selected["source"],
                "splat_score": splat_candidate["score"],
                "slam_score": slam_candidate["score"],
                "prefer_splat": self.config.prefer_splat,
            }
            logger.info(
                "map_source_selected",
                source=selected["source"],
                splat_score=splat_candidate["score"],
                slam_score=slam_candidate["score"],
                prefer_splat=self.config.prefer_splat,
            )
            return selected["path"], selected["map_id"], selected["source"], selection_info

        if splat_candidate:
            return (
                splat_candidate["path"],
                splat_candidate["map_id"],
                splat_candidate["source"],
                {
                    "selected_source": "splat",
                    "splat_score": splat_candidate["score"],
                    "slam_score": None,
                    "prefer_splat": self.config.prefer_splat,
                },
            )
        if slam_candidate:
            return (
                slam_candidate["path"],
                slam_candidate["map_id"],
                slam_candidate["source"],
                {
                    "selected_source": "slam",
                    "splat_score": None,
                    "slam_score": slam_candidate["score"],
                    "prefer_splat": self.config.prefer_splat,
                },
            )

        return None, None, "slam", None

    def _score_source(self, quality: float, timestamp_path: Path | None) -> float:
        score = max(0.0, min(1.0, float(quality)))
        age_s = self._file_age_seconds(timestamp_path) if timestamp_path else None
        if age_s is None:
            return score
        age_factor = 1.0 - (age_s / max(1.0, self.config.max_map_age_s))
        age_factor = max(0.0, min(1.0, age_factor))
        return (score * 0.7) + (age_factor * 0.3)

    @staticmethod
    def _file_age_seconds(path: Path | None) -> float | None:
        if not path or not path.exists():
            return None
        return max(0.0, time.time() - path.stat().st_mtime)

    @staticmethod
    def _load_slam_status(path: Path) -> dict[str, Any]:
        data = MapUpdateService._read_json(path)
        if not data:
            return {}
        keyframes = int(data.get("keyframe_count", 0))
        loop_closures = int(data.get("loop_closure_count", 0))
        data.setdefault("loop_closure_rate", loop_closures / max(1, keyframes))
        data.setdefault("reprojection_error", float(data.get("reprojection_error", 0.0)))
        return data

    @staticmethod
    def _load_pose_graph_summary(path: Path) -> dict[str, Any]:
        data = MapUpdateService._read_json(path)
        if not data:
            return {}
        frames = data.get("frames", [])
        summary: dict[str, Any] = {
            "pose_graph_path": str(path),
            "generated_at": data.get("generated_at"),
            "backend": data.get("backend"),
            "sequence_id": data.get("sequence_id"),
            "frame_count": data.get("frame_count", len(frames)),
            "keyframe_count": data.get("keyframe_count", 0),
            "point_cloud": data.get("point_cloud"),
        }
        if not frames:
            return summary

        positions = []
        path_length = 0.0
        prev_pos = None
        for frame in frames:
            pose = frame.get("pose") or {}
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            z = float(pose.get("z", 0.0))
            positions.append((x, y, z))
            if prev_pos:
                dx = x - prev_pos[0]
                dy = y - prev_pos[1]
                dz = z - prev_pos[2]
                path_length += (dx * dx + dy * dy + dz * dz) ** 0.5
            prev_pos = (x, y, z)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        summary["bounds"] = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
        }
        summary["path_length_m"] = path_length
        start = positions[0]
        end = positions[-1]
        summary["start_end_distance_m"] = (
            (end[0] - start[0]) ** 2
            + (end[1] - start[1]) ** 2
            + (end[2] - start[2]) ** 2
        ) ** 0.5
        return summary

    @staticmethod
    def _load_splat_scene(path: Path) -> tuple[Path | None, float]:
        data = MapUpdateService._read_json(path)
        if not data:
            return None, 1.0
        preview = data.get("preview_point_cloud")
        if not preview:
            preview = data.get("preview")
        preview_path = None
        if preview:
            preview_path = Path(preview)
            if not preview_path.is_absolute():
                preview_path = path.parent / preview
        quality = 1.0
        metrics = data.get("metrics") or {}
        if "psnr" in metrics:
            quality = min(1.0, float(metrics.get("psnr", 1.0)) / 30.0)
        return preview_path, quality

    async def _emit_map_change_anomaly(
        self,
        changes: list[dict[str, Any]],
        map_id: str | None,
    ) -> None:
        now = time.time()
        if now - self._last_change_emit_time < self.config.change_detection_cooldown_s:
            return
        self._last_change_emit_time = now

        severity = min(0.9, 0.3 + (len(changes) * 0.15))
        anomaly_id = f"map_change_{int(now * 1000)}"
        description = f"Splat change detected ({len(changes)} unmatched obstacle(s))"

        try:
            from agent.server.world_model import Anomaly

            anomaly = Anomaly(
                anomaly_id=anomaly_id,
                asset_id=f"map_change_{map_id or 'unknown'}",
                detected_at=datetime.now(),
                severity=severity,
                description=description,
            )
            self.server_state.world_model.add_anomaly(anomaly)
        except Exception as exc:
            logger.warning("map_change_anomaly_failed", error=str(exc))

        try:
            from agent.server.events import Event, EventSeverity, EventType
            from agent.server.state import connection_manager

            await connection_manager.broadcast(
                Event(
                    event_type=EventType.ANOMALY_CREATED,
                    timestamp=datetime.now(),
                    data={
                        "anomaly_id": anomaly_id,
                        "source": "splat_change_detection",
                        "map_id": map_id,
                        "severity": severity,
                        "description": description,
                        "change_count": len(changes),
                        "changes": changes,
                    },
                    severity=EventSeverity.WARNING,
                )
            )
        except Exception as exc:
            logger.warning("map_change_broadcast_failed", error=str(exc))

    def _merge_depth_obstacles(self, nav_map: dict[str, Any]) -> dict[str, Any]:
        depth_capture = getattr(self.server_state, "last_depth_capture", None) or {}
        obstacles = depth_capture.get("obstacles") or []
        if not obstacles:
            return nav_map

        timestamp = depth_capture.get("timestamp")
        if timestamp:
            try:
                observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_s = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
                if age_s > 10.0:
                    return nav_map
            except Exception:
                pass

        existing = [obs for obs in nav_map.get("obstacles", []) if obs.get("source") != "airsim_depth"]
        for obs in obstacles:
            merged = dict(obs)
            merged.setdefault("source", "airsim_depth")
            if timestamp:
                merged.setdefault("detected_at", timestamp)
            existing.append(merged)

        nav_map["obstacles"] = existing
        metadata = nav_map.get("metadata", {})
        metadata["obstacle_count"] = len(existing)
        nav_map["metadata"] = metadata
        return nav_map

    def _build_depth_only_map(self) -> dict[str, Any] | None:
        depth_capture = getattr(self.server_state, "last_depth_capture", None) or {}
        obstacles = depth_capture.get("obstacles") or []
        if not obstacles:
            return None

        timestamp = depth_capture.get("timestamp") or datetime.now().isoformat()
        try:
            observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
            if age_s > 10.0:
                return None
        except Exception:
            pass
        nav_map = {
            "scenario_id": None,
            "generated_at": timestamp,
            "last_updated": timestamp,
            "source": "airsim_depth",
            "obstacles": [dict(obs) for obs in obstacles],
            "metadata": {
                "map_id": "airsim_depth_only",
                "version": 1,
                "bounds_min_x": 0.0,
                "bounds_max_x": 0.0,
                "bounds_min_y": 0.0,
                "bounds_max_y": 0.0,
                "bounds_min_z": 0.0,
                "bounds_max_z": 0.0,
                "resolution_m": self.config.map_resolution_m,
                "voxel_size_m": self.config.voxel_size_m,
                "obstacle_count": len(obstacles),
                "map_quality_score": max(self.config.min_quality_score, 0.3),
                "slam_confidence": 0.0,
                "splat_quality": 0.0,
            },
            "tiles": [],
        }
        return nav_map

    @staticmethod
    def _should_regenerate_proxy(
        scene_path: Path | None,
        preview_path: Path,
        proxy_path: Path,
        *,
        min_interval_s: float = 0.0,
    ) -> bool:
        if not proxy_path.exists():
            return True
        proxy_mtime = proxy_path.stat().st_mtime
        if scene_path and scene_path.exists() and scene_path.stat().st_mtime > proxy_mtime:
            updated = True
        else:
            updated = preview_path.stat().st_mtime > proxy_mtime
        if not updated:
            return False
        if min_interval_s <= 0:
            return True
        return (datetime.now().timestamp() - proxy_mtime) >= min_interval_s

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _record_update_error(self, message: str) -> None:
        self.server_state.map_update_last_error = message
        self.server_state.map_update_last_error_at = datetime.now().isoformat()

    def _clear_update_error(self) -> None:
        self.server_state.map_update_last_error = None
        self.server_state.map_update_last_error_at = None

    def _record_gate_result(
        self,
        gate_result: Any,
        source: str,
        quality_score: float,
    ) -> None:
        """Record gate result to history (Agent B Phase 6)."""
        if not hasattr(self.server_state, "map_gate_history"):
            self.server_state.map_gate_history = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "result": gate_result.result.value,
            "reason": gate_result.reason,
            "source": source,
            "quality_score": quality_score,
            "details": gate_result.details,
        }

        self.server_state.map_gate_history.append(entry)

        # Keep only last 50 entries
        if len(self.server_state.map_gate_history) > 50:
            self.server_state.map_gate_history = self.server_state.map_gate_history[-50:]
