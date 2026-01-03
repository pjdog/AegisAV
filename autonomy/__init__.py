"""AegisAV Autonomy Package.

Platform-agnostic flight interface supporting MAVLink-based autopilots.
"""

# Core types that don't require external dependencies
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSInfo,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)

__all__ = [
    "Attitude",
    "BatteryState",
    "FlightMode",
    "GPSInfo",
    "GPSState",
    # Vehicle state
    "Position",
    "VehicleHealth",
    "VehicleState",
    "Velocity",
]

# Optional imports that require pymavlink
try:
    from autonomy.mavlink_interface import (  # noqa: F401
        ConnectionState,
        MAVLinkConfig,
        MAVLinkInterface,
    )
    from autonomy.mission_primitives import (  # noqa: F401
        MissionPrimitives,
        PrimitiveConfig,
        PrimitiveResult,
    )

    __all__.extend([
        "ConnectionState",
        "MAVLinkConfig",
        "MAVLinkInterface",
        "MissionPrimitives",
        "PrimitiveConfig",
        "PrimitiveResult",
    ])
except ImportError:
    # pymavlink not installed - MAVLink functionality not available
    pass

# New autonomous flight components
try:
    from autonomy.autonomous_pipeline import (  # noqa: F401
        AutonomousPipeline,
        AutonomousPipelineConfig,
        PipelineState,
        PipelineStatus,
        create_autonomous_pipeline,
    )
    from autonomy.flight_backend import (  # noqa: F401
        AirSimBackendConfig,
        ArduPilotBackendConfig,
        BackendType,
        ConnectionStatus,
        FlightBackend,
        FlightBackendConfig,
        WaypointGPS,
        WaypointNED,
        create_backend,
    )
    from autonomy.flight_controller import (  # noqa: F401
        AbortReason,
        FlightController,
        FlightControllerConfig,
        FlightControllerEvent,
        FlightPhase,
    )
    from autonomy.mission_planner import (  # noqa: F401
        InspectionTarget,
        MissionPlan,
        MissionPlanner,
        MissionPlannerConfig,
        MissionWaypoint,
    )
    from autonomy.path_planner import (  # noqa: F401
        FlightPath,
        Obstacle,
        PathPlanner,
        PathPlannerConfig,
        Waypoint,
    )
    from autonomy.state_estimator import (  # noqa: F401
        EstimatedState,
        LocalizationMode,
        StateEstimator,
        StateEstimatorConfig,
    )

    __all__.extend([
        # Flight Controller
        "AbortReason",
        # Flight Backend
        "AirSimBackendConfig",
        "ArduPilotBackendConfig",
        # Autonomous Pipeline
        "AutonomousPipeline",
        "AutonomousPipelineConfig",
        "BackendType",
        "ConnectionStatus",
        # State Estimator
        "EstimatedState",
        "FlightBackend",
        "FlightBackendConfig",
        "FlightController",
        "FlightControllerConfig",
        "FlightControllerEvent",
        # Path Planner
        "FlightPath",
        "FlightPhase",
        # Mission Planner
        "InspectionTarget",
        "LocalizationMode",
        "MissionPlan",
        "MissionPlanner",
        "MissionPlannerConfig",
        "MissionWaypoint",
        "Obstacle",
        "PathPlanner",
        "PathPlannerConfig",
        "PipelineState",
        "PipelineStatus",
        "StateEstimator",
        "StateEstimatorConfig",
        "Waypoint",
        "WaypointGPS",
        "WaypointNED",
        "create_autonomous_pipeline",
        "create_backend",
    ])
except ImportError:
    # Some dependencies missing for autonomous flight components
    pass
