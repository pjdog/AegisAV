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
    from autonomy.mission_planner import (  # noqa: F401
        InspectionTarget,
        MissionPlan,
        MissionPlanner,
        MissionPlannerConfig,
        MissionWaypoint,
    )
    from autonomy.flight_controller import (  # noqa: F401
        AbortReason,
        FlightController,
        FlightControllerConfig,
        FlightControllerEvent,
        FlightPhase,
    )
    from autonomy.autonomous_pipeline import (  # noqa: F401
        AutonomousPipeline,
        AutonomousPipelineConfig,
        PipelineState,
        PipelineStatus,
        create_autonomous_pipeline,
    )

    __all__.extend([
        # Flight Backend
        "AirSimBackendConfig",
        "ArduPilotBackendConfig",
        "BackendType",
        "ConnectionStatus",
        "FlightBackend",
        "FlightBackendConfig",
        "WaypointGPS",
        "WaypointNED",
        "create_backend",
        # Path Planner
        "FlightPath",
        "Obstacle",
        "PathPlanner",
        "PathPlannerConfig",
        "Waypoint",
        # State Estimator
        "EstimatedState",
        "LocalizationMode",
        "StateEstimator",
        "StateEstimatorConfig",
        # Mission Planner
        "InspectionTarget",
        "MissionPlan",
        "MissionPlanner",
        "MissionPlannerConfig",
        "MissionWaypoint",
        # Flight Controller
        "AbortReason",
        "FlightController",
        "FlightControllerConfig",
        "FlightControllerEvent",
        "FlightPhase",
        # Autonomous Pipeline
        "AutonomousPipeline",
        "AutonomousPipelineConfig",
        "PipelineState",
        "PipelineStatus",
        "create_autonomous_pipeline",
    ])
except ImportError as e:
    # Some dependencies missing for autonomous flight components
    pass
