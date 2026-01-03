"""Vision API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent.server.state import server_state


def register_vision_routes(app: FastAPI) -> None:
    """Register vision-related routes."""

    @app.get("/api/vision/last")
    async def get_last_vision_observation() -> dict:
        """Get the latest vision observation summary."""
        if not server_state.vision_enabled:
            return {"enabled": False, "observation": None}

        return {
            "enabled": True,
            "observation": server_state.last_vision_observation,
        }

    @app.get("/api/vision/statistics")
    async def get_vision_statistics() -> dict:
        """Get vision system statistics."""
        if not server_state.vision_enabled or not server_state.vision_service:
            raise HTTPException(status_code=503, detail="Vision service not available")

        stats = server_state.vision_service.get_statistics()
        return {
            "enabled": True,
            "statistics": stats,
        }

    @app.get("/api/vision/observations")
    async def get_vision_observations(limit: int = 100) -> dict:
        """Get recent vision observations."""
        if not server_state.vision_enabled or not server_state.vision_service:
            raise HTTPException(status_code=503, detail="Vision service not available")

        observations = server_state.vision_service.get_recent_observations(limit=limit)
        return {
            "observations": [
                {
                    "observation_id": obs.observation_id,
                    "asset_id": obs.asset_id,
                    "timestamp": obs.timestamp.isoformat(),
                    "defect_detected": obs.defect_detected,
                    "max_confidence": obs.max_confidence,
                    "max_severity": obs.max_severity,
                    "anomaly_created": obs.anomaly_created,
                    "anomaly_id": obs.anomaly_id,
                    "detections": obs.detections,
                    "detection_count": len(obs.detections),
                }
                for obs in observations
            ],
            "total": len(observations),
        }

    @app.get("/api/vision/observations/{asset_id}")
    async def get_asset_observations(asset_id: str) -> dict:
        """Get vision observations for a specific asset."""
        if not server_state.vision_enabled or not server_state.vision_service:
            raise HTTPException(status_code=503, detail="Vision service not available")

        observations = server_state.vision_service.get_observations_for_asset(asset_id)
        return {
            "asset_id": asset_id,
            "observations": [
                {
                    "observation_id": obs.observation_id,
                    "timestamp": obs.timestamp.isoformat(),
                    "defect_detected": obs.defect_detected,
                    "max_confidence": obs.max_confidence,
                    "max_severity": obs.max_severity,
                    "anomaly_created": obs.anomaly_created,
                    "detections": obs.detections,
                    "detection_count": len(obs.detections),
                }
                for obs in observations
            ],
            "total": len(observations),
        }

    @app.get("/api/vision/captures")
    async def get_recent_captures(limit: int = 20) -> dict:
        """Get recent vision captures with image URLs for dashboard display.

        Returns captures with image paths and detection summaries suitable
        for the Recent Captures section of the dashboard.

        Args:
            limit: Maximum number of captures to return (default 20).

        Returns:
            Dictionary with captures list and metadata.
        """
        if not server_state.vision_enabled or not server_state.vision_service:
            return {"captures": [], "total": 0, "enabled": False}

        observations = server_state.vision_service.get_recent_observations(limit=limit)

        # Get asset names from world model if available
        asset_names: dict[str, str] = {}
        if server_state.world_model:
            world_snap = server_state.world_model.get_snapshot()
            if world_snap:
                for asset in world_snap.assets:
                    asset_names[asset.asset_id] = asset.name

        captures = []
        for obs in observations:
            # Build image URL if image path exists
            image_url = None
            thumbnail_url = None
            if obs.image_path:
                image_url = f"/api/vision/image/{obs.observation_id}"
                thumbnail_url = f"/api/vision/image/{obs.observation_id}?thumbnail=true"

            # Format detections for display
            formatted_detections = []
            for det in obs.detections:
                formatted_detections.append({
                    "class": det.get("class", det.get("detection_class", "unknown")),
                    "confidence": det.get("confidence", 0.0),
                    "severity": det.get("severity", 0.0),
                })

            captures.append({
                "observation_id": obs.observation_id,
                "asset_id": obs.asset_id,
                "asset_name": asset_names.get(obs.asset_id, obs.asset_id),
                "timestamp": obs.timestamp.isoformat(),
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "defect_detected": obs.defect_detected,
                "detections": formatted_detections,
                "detection_count": len(obs.detections),
                "max_confidence": obs.max_confidence,
                "max_severity": obs.max_severity,
                "anomaly_created": obs.anomaly_created,
                "anomaly_id": obs.anomaly_id,
            })

        return {
            "captures": captures,
            "total": len(captures),
            "enabled": True,
        }

    @app.get("/api/vision/image/{observation_id}")
    async def get_capture_image(observation_id: str, thumbnail: bool = False) -> FileResponse:
        """Serve a captured image by observation ID.

        Args:
            observation_id: The observation ID to get the image for.
            thumbnail: If True, return a smaller thumbnail version (not implemented yet).

        Returns:
            FileResponse with the image file.

        Raises:
            HTTPException: If observation not found or image not available.
        """
        if not server_state.vision_enabled or not server_state.vision_service:
            raise HTTPException(status_code=503, detail="Vision service not available")

        # Get the observation
        observations = server_state.vision_service.observations
        if observation_id not in observations:
            raise HTTPException(status_code=404, detail="Observation not found")

        obs = observations[observation_id]
        if not obs.image_path:
            raise HTTPException(status_code=404, detail="No image available for this observation")

        image_path = Path(obs.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        # Determine media type from extension
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        return FileResponse(
            path=image_path,
            media_type=media_type,
            filename=f"capture_{observation_id}{suffix}",
        )
