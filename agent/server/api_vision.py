"""Vision API routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agent.server.state import server_state


def register_vision_routes(app: FastAPI) -> None:
    """Register vision-related routes."""

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
                }
                for obs in observations
            ],
            "total": len(observations),
        }
