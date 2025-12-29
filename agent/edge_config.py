"""Edge Compute Configuration.

Defines configurable "edge compute" profiles that simulate different on-drone compute tiers
(FC-only, MCU, SBC, Jetson) and how they impact capture cadence, inference latency,
anomaly gating, and uplink payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EdgeComputeProfile(str, Enum):
    """Compute tier profiles for on-drone edge processing."""

    FC_ONLY = "fc_only"
    MCU_HEURISTIC = "mcu_heuristic"
    MCU_TINY_CNN = "mcu_tiny_cnn"
    SBC_CPU = "sbc_cpu"
    SBC_ACCEL = "sbc_accel"
    JETSON_FULL = "jetson_full"


class AnomalyGateMode(str, Enum):
    """How the edge agent turns detections into anomaly flags."""

    ANY = "any"
    N_OF_M = "n_of_m"
    SEVERITY_OVERRIDE = "severity_override"


class AnomalyGateConfig(BaseModel):
    """Configuration for anomaly gating."""

    mode: AnomalyGateMode = AnomalyGateMode.ANY

    # Thresholds applied to client detections.
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_severity: float = Field(default=0.3, ge=0.0, le=1.0)

    # Used when mode == N_OF_M, or as fallback gate for SEVERITY_OVERRIDE.
    n: int | None = Field(default=None, ge=1, le=100)
    m: int | None = Field(default=None, ge=1, le=100)

    # Used when mode == SEVERITY_OVERRIDE.
    min_severity_override: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_gate(self) -> AnomalyGateConfig:
        if self.mode == AnomalyGateMode.N_OF_M:
            if self.n is None or self.m is None:
                raise ValueError("n and m are required when mode == 'n_of_m'")
            if self.n > self.m:
                raise ValueError("n must be <= m")

        if self.mode == AnomalyGateMode.SEVERITY_OVERRIDE:
            if self.min_severity_override is None:
                raise ValueError("min_severity_override is required when mode == 'severity_override'")
            if (self.n is None) != (self.m is None):
                raise ValueError("n and m must be provided together (or omitted) for severity_override")
            if self.n is not None and self.m is not None and self.n > self.m:
                raise ValueError("n must be <= m")

        return self


class EdgeUplinkConfig(BaseModel):
    """Uplink shaping for client→server feedback payloads."""

    summary_only: bool = False
    send_images: bool = True
    max_images: int = Field(default=1, ge=0, le=10)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_uplink(self) -> EdgeUplinkConfig:
        if not self.send_images and self.max_images != 0:
            raise ValueError("max_images must be 0 when send_images is false")
        return self


class EdgeComputeConfig(BaseModel):
    """Full edge compute configuration (profile + overrides)."""

    profile: EdgeComputeProfile = EdgeComputeProfile.SBC_CPU

    # Vision capture + screening.
    vision_enabled: bool = True
    capture_interval_s: float = Field(default=2.0, ge=0.1, le=60.0)
    max_captures_per_inspection: int = Field(default=10, ge=0, le=50)
    simulated_inference_delay_ms: int = Field(default=0, ge=0, le=10_000)
    client_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # Decision policy.
    anomaly_gate: AnomalyGateConfig = Field(default_factory=AnomalyGateConfig)

    # Payload shaping.
    uplink: EdgeUplinkConfig = Field(default_factory=EdgeUplinkConfig)

    model_config = {"extra": "forbid"}


def available_edge_profiles() -> list[str]:
    """Return profile values for UI selection."""
    return [p.value for p in EdgeComputeProfile]


def default_edge_compute_config(profile: EdgeComputeProfile) -> EdgeComputeConfig:
    """Return opinionated defaults for a given profile."""
    match profile:
        case EdgeComputeProfile.FC_ONLY:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=False,
                capture_interval_s=60.0,
                max_captures_per_inspection=0,
                simulated_inference_delay_ms=0,
                client_confidence_threshold=1.0,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.ANY, min_confidence=1.0, min_severity=1.0
                ),
                uplink=EdgeUplinkConfig(summary_only=True, send_images=False, max_images=0),
            )
        case EdgeComputeProfile.MCU_HEURISTIC:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=True,
                capture_interval_s=5.0,
                max_captures_per_inspection=1,
                simulated_inference_delay_ms=20,
                client_confidence_threshold=0.85,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.ANY, min_confidence=0.85, min_severity=0.6
                ),
                uplink=EdgeUplinkConfig(summary_only=True, send_images=False, max_images=0),
            )
        case EdgeComputeProfile.MCU_TINY_CNN:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=True,
                capture_interval_s=3.0,
                max_captures_per_inspection=2,
                simulated_inference_delay_ms=180,
                client_confidence_threshold=0.7,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.ANY, min_confidence=0.75, min_severity=0.5
                ),
                uplink=EdgeUplinkConfig(summary_only=False, send_images=False, max_images=0),
            )
        case EdgeComputeProfile.SBC_CPU:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=True,
                capture_interval_s=1.5,
                max_captures_per_inspection=5,
                simulated_inference_delay_ms=450,
                client_confidence_threshold=0.5,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.N_OF_M,
                    n=2,
                    m=5,
                    min_confidence=0.6,
                    min_severity=0.4,
                ),
                uplink=EdgeUplinkConfig(summary_only=False, send_images=True, max_images=1),
            )
        case EdgeComputeProfile.SBC_ACCEL:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=True,
                capture_interval_s=1.0,
                max_captures_per_inspection=10,
                simulated_inference_delay_ms=60,
                client_confidence_threshold=0.45,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.SEVERITY_OVERRIDE,
                    min_confidence=0.55,
                    min_severity=0.35,
                    n=1,
                    m=5,
                    min_severity_override=0.75,
                ),
                uplink=EdgeUplinkConfig(summary_only=False, send_images=True, max_images=2),
            )
        case EdgeComputeProfile.JETSON_FULL:
            return EdgeComputeConfig(
                profile=profile,
                vision_enabled=True,
                capture_interval_s=0.6,
                max_captures_per_inspection=15,
                simulated_inference_delay_ms=20,
                client_confidence_threshold=0.35,
                anomaly_gate=AnomalyGateConfig(
                    mode=AnomalyGateMode.ANY, min_confidence=0.45, min_severity=0.3
                ),
                uplink=EdgeUplinkConfig(summary_only=False, send_images=True, max_images=3),
            )


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_edge_compute_update(current: EdgeComputeConfig, update: dict[str, Any]) -> EdgeComputeConfig:
    """Apply a partial update dict to the current edge config.

    If ``profile`` is present, the base is reset to that profile's defaults, then
    the remaining fields are merged in.
    """
    if not isinstance(update, dict):
        raise TypeError("edge config update must be a dict")

    base = current
    if "profile" in update:
        base = default_edge_compute_config(EdgeComputeProfile(update["profile"]))

    merged = _deep_merge(base.model_dump(mode="json"), update)
    return EdgeComputeConfig.model_validate(merged)
