"""Advanced Agentic Decision Making.

Implements sophisticated decision-making capabilities for truly autonomous aerial monitoring.
This module provides hierarchical planning, adaptive learning, and explainable AI reasoning.
"""

import asyncio
import heapq
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from agent.server.goals import Goal, GoalType
from agent.server.llm_config import get_default_llm_model, llm_credentials_present
from agent.server.world_model import Asset, WorldSnapshot
from autonomy.vehicle_state import Position

logger = logging.getLogger(__name__)


class CognitiveLevel(Enum):
    """Cognitive processing levels for hierarchical decision making."""

    REACTIVE = "reactive"  # Immediate response to stimuli
    DELIBERATIVE = "deliberative"  # Planning and reasoning
    REFLECTIVE = "reflective"  # Meta-cognition and learning
    PREDICTIVE = "predictive"  # Anticipatory planning


class UrgencyLevel(Enum):
    """Mission urgency levels."""

    CRITICAL = 5.0  # Immediate action required
    HIGH = 4.0  # High priority, act soon
    MEDIUM = 3.0  # Normal priority
    LOW = 2.0  # Can defer action
    ROUTINE = 1.0  # Low priority, regular operation


class ConfidenceType(Enum):
    """Types of decision confidence."""

    CERTAIN = 1.0  # Very high confidence
    HIGH = 0.8  # High confidence
    MEDIUM = 0.6  # Moderate confidence
    LOW = 0.4  # Low confidence
    VERY_LOW = 0.2  # Very low confidence


@dataclass
class DecisionContext:
    """Context information for decision making."""

    timestamp: datetime
    cognitive_level: CognitiveLevel
    urgency_level: UrgencyLevel
    confidence_type: ConfidenceType
    uncertainty_factors: list[str] = field(default_factory=list)
    learning_weight: float = 1.0
    metacognitive_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of prediction/forecasting."""

    predicted_value: Any
    confidence: float
    prediction_horizon: timedelta
    factors_considered: list[str]
    uncertainty_margin: float


@dataclass
class ExperienceOutcome:
    """Outcome details for a learning experience."""

    outcome: str
    success: bool
    reward_signal: float
    lesson_learned: str


@dataclass
class LearningExperience:
    """Stored learning experience for adaptive behavior."""

    timestamp: datetime
    situation_hash: str
    decision_made: str
    contextual_features: dict[str, Any]
    result: ExperienceOutcome


class MissionDecision(BaseModel):
    """Structured output from the Mission Planner agent."""

    goal_type: GoalType
    priority: float = Field(..., ge=0.0, le=100.0)
    reason: str = Field(..., min_length=1)
    target_asset_id: str | None = None


def _create_mission_planner() -> Agent[WorldSnapshot, MissionDecision]:
    """Factory function to create the mission planner agent.

    Uses the configured LLM provider when credentials are available,
    or TestModel for testing when no API key is available.
    """
    resolved_model = get_default_llm_model()
    if llm_credentials_present():
        model: str | TestModel = resolved_model
    else:
        logger.warning(
            "LLM credentials not set - using TestModel for mission_planner. "
            "Configure AEGIS_LLM_PROVIDER/AEGIS_LLM_MODEL and provider API key for production."
        )
        model = TestModel()

    agent = Agent(
        model,
        output_type=MissionDecision,
        system_prompt=(
            "You are the AegisAV Mission Planner, the high-level intelligence for an "
            "autonomous drone. Your task is to analyze the current world state and "
            "select the most appropriate mission goal. Prioritize safety (battery, "
            "health, weather) above all else. If multiple assets need inspection, "
            "prioritize based on priority level and scheduled time. Provide clear, "
            "concise reasoning for your decisions. Use the path-planning tools "
            "(Dijkstra planner, Markov policy, neural ranker) when route choice or "
            "asset ordering matters."
        ),
    )

    @agent.tool
    def get_asset_details(ctx: RunContext[WorldSnapshot], asset_id: str) -> Asset | None:
        """Get detailed information about a specific asset."""
        for asset in ctx.deps.assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    @agent.tool
    def plan_path_dijkstra(
        ctx: RunContext[WorldSnapshot],
        start_id: str,
        goal_id: str,
        neighbor_k: int = 4,
        avoid_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find a shortest path using Dijkstra over nearby neighbors.

        start_id/goal_id can be asset IDs, "vehicle", or "dock".
        """
        nodes = _collect_nodes(ctx.deps, include_vehicle=True, include_dock=True)
        if start_id not in nodes or goal_id not in nodes:
            return {
                "path": [],
                "distance_m": None,
                "error": "unknown_start_or_goal",
            }
        graph = _build_neighbor_graph(nodes, neighbor_k=max(1, neighbor_k))
        path, distance_m = _dijkstra_path(graph, start_id, goal_id, avoid_ids=avoid_ids)
        return {
            "path": path,
            "distance_m": None if not path else distance_m,
            "neighbor_k": neighbor_k,
        }

    @agent.tool
    def markov_next_asset(
        ctx: RunContext[WorldSnapshot],
        current_id: str,
        candidate_ids: list[str] | None = None,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Suggest the next asset using a Markov policy."""
        ranked = _markov_rank_assets(
            ctx.deps,
            current_id=current_id,
            candidate_ids=candidate_ids,
            temperature=temperature,
        )
        next_asset = ranked[0]["asset_id"] if ranked else None
        return {"current_id": current_id, "next_asset_id": next_asset, "distribution": ranked}

    @agent.tool
    def neural_rank_assets(
        ctx: RunContext[WorldSnapshot],
        current_id: str,
        candidate_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Rank assets using a lightweight neural heuristic."""
        ranked = _neural_rank_assets(ctx.deps, current_id=current_id, candidate_ids=candidate_ids)
        next_asset = ranked[0]["asset_id"] if ranked else None
        return {"current_id": current_id, "next_asset_id": next_asset, "ranked": ranked}

    @agent.tool
    def get_mission_metrics(ctx: RunContext[WorldSnapshot]) -> dict[str, Any]:
        """Get current mission success metrics for optimization decisions.

        Returns metrics about inspection coverage, anomaly detection rate,
        and success rate to help optimize goal selection and prioritization.

        Use this tool to:
        - Check current mission progress before deciding on next action
        - Identify coverage gaps (uninspected high-priority assets)
        - Balance between coverage completion and anomaly investigation
        """
        world = ctx.deps
        metrics: dict[str, Any] = {
            "inspection_coverage_percent": 0.0,
            "anomaly_detection_rate": 0.0,
            "success_rate": 0.0,
            "assets_remaining": [],
            "high_priority_anomalies": [],
            "optimization_hints": [],
        }

        # Calculate inspection coverage
        if world.mission:
            total = world.mission.assets_total
            inspected = world.mission.assets_inspected
            metrics["inspection_coverage_percent"] = (
                (inspected / total * 100) if total > 0 else 0.0
            )

            # Find remaining uninspected assets
            inspected_ids = {a.asset_id for a in world.assets if a.last_inspected}
            remaining = []
            for asset in world.assets:
                if asset.asset_id not in inspected_ids:
                    remaining.append({
                        "asset_id": asset.asset_id,
                        "name": asset.name,
                        "priority": asset.priority,
                    })
            # Sort by priority (highest first)
            remaining.sort(key=lambda x: x.get("priority", 0), reverse=True)
            metrics["assets_remaining"] = remaining[:10]  # Top 10

        # Calculate anomaly detection metrics
        total_anomalies = len(world.anomalies)
        if total_anomalies > 0:
            resolved = sum(1 for a in world.anomalies if a.resolved)
            metrics["anomaly_detection_rate"] = total_anomalies
            metrics["anomalies_resolved"] = resolved
            metrics["anomalies_pending"] = total_anomalies - resolved

            # Find high-priority unresolved anomalies
            high_priority = []
            for anomaly in world.anomalies:
                if not anomaly.resolved and anomaly.severity >= 0.7:
                    high_priority.append({
                        "anomaly_id": anomaly.anomaly_id,
                        "asset_id": anomaly.asset_id,
                        "severity": anomaly.severity,
                        "anomaly_type": anomaly.anomaly_type,
                    })
            high_priority.sort(key=lambda x: x.get("severity", 0), reverse=True)
            metrics["high_priority_anomalies"] = high_priority[:5]  # Top 5

        # Generate optimization hints
        hints = []
        coverage = metrics["inspection_coverage_percent"]
        if coverage < 50:
            hints.append("Low coverage - prioritize uninspected assets")
        elif coverage < 80:
            hints.append("Good progress - continue systematic inspection")
        else:
            hints.append("High coverage - focus on anomaly investigation")

        if metrics.get("high_priority_anomalies"):
            hints.append(f"{len(metrics['high_priority_anomalies'])} high-severity anomalies need attention")

        if world.vehicle and world.vehicle.battery:
            battery = world.vehicle.battery.remaining_percent
            if battery < 30:
                hints.append("Low battery - prioritize nearby assets or return")
            elif battery < 50:
                hints.append("Moderate battery - plan efficient route")

        metrics["optimization_hints"] = hints

        return metrics

    return agent


def _collect_nodes(
    world: WorldSnapshot, *, include_vehicle: bool = True, include_dock: bool = True
) -> dict[str, Position]:
    nodes: dict[str, Position] = {}
    if include_vehicle:
        nodes["vehicle"] = world.vehicle.position
    if include_dock:
        nodes["dock"] = world.dock.position
    for asset in world.assets:
        nodes[asset.asset_id] = asset.position
    return nodes


def _build_neighbor_graph(
    nodes: dict[str, Position], neighbor_k: int
) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {}
    for node_id, position in nodes.items():
        distances: list[tuple[str, float]] = []
        for other_id, other_pos in nodes.items():
            if other_id == node_id:
                continue
            distances.append((other_id, position.distance_to(other_pos)))
        distances.sort(key=lambda item: item[1])
        graph[node_id] = distances[: min(neighbor_k, len(distances))]
    return graph


def _dijkstra_path(
    graph: dict[str, list[tuple[str, float]]],
    start_id: str,
    goal_id: str,
    *,
    avoid_ids: list[str] | None = None,
) -> tuple[list[str], float]:
    if start_id not in graph or goal_id not in graph:
        return [], float("inf")

    avoid = set(avoid_ids or [])
    if start_id in avoid or goal_id in avoid:
        return [], float("inf")

    distances: dict[str, float] = {start_id: 0.0}
    previous: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, start_id)]

    while queue:
        current_distance, node = heapq.heappop(queue)
        if node == goal_id:
            break
        if current_distance > distances.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, []):
            if neighbor in avoid:
                continue
            next_distance = current_distance + weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                previous[neighbor] = node
                heapq.heappush(queue, (next_distance, neighbor))

    if goal_id not in distances:
        return [], float("inf")

    path = [goal_id]
    while path[-1] != start_id:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[goal_id]


def _candidate_assets(world: WorldSnapshot, candidate_ids: list[str] | None) -> list[Asset]:
    if candidate_ids:
        asset_map = {asset.asset_id: asset for asset in world.assets}
        return [asset_map[a_id] for a_id in candidate_ids if a_id in asset_map]

    assets: list[Asset] = []
    seen: set[str] = set()
    for asset in world.get_anomaly_assets() + world.get_pending_assets():
        if asset.asset_id not in seen:
            assets.append(asset)
            seen.add(asset.asset_id)
    if assets:
        return assets
    return list(world.assets)


def _markov_rank_assets(
    world: WorldSnapshot,
    *,
    current_id: str,
    candidate_ids: list[str] | None = None,
    temperature: float = 1.0,
) -> list[dict[str, float | str]]:
    nodes = _collect_nodes(world, include_vehicle=True, include_dock=True)
    if current_id not in nodes:
        return []

    current_pos = nodes[current_id]
    assets = _candidate_assets(world, candidate_ids)
    if not assets:
        return []

    anomalies = {asset.asset_id for asset in world.get_anomaly_assets()}
    distances = [current_pos.distance_to(asset.position) for asset in assets]
    max_distance = max(distances) if distances else 1.0
    battery_norm = world.vehicle.battery.remaining_percent / 100.0

    scores: list[float] = []
    for asset, distance in zip(assets, distances, strict=False):
        distance_score = 1.0 - min(distance / max_distance, 1.0)
        priority_score = 1.0 / (1.0 + asset.priority)
        anomaly_bonus = 0.4 if asset.asset_id in anomalies else 0.0
        score = 0.5 * distance_score + 0.4 * priority_score + anomaly_bonus + 0.1 * battery_norm
        scores.append(score)

    temp = max(0.1, temperature)
    logits = np.array(scores, dtype=float) / temp
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum() if exp.sum() else np.zeros_like(exp)

    ranked = [
        {
            "asset_id": asset.asset_id,
            "probability": float(prob),
            "score": float(score),
            "distance_m": float(distance),
        }
        for asset, prob, score, distance in zip(assets, probs, scores, distances, strict=False)
    ]
    ranked.sort(key=lambda item: item["probability"], reverse=True)
    return ranked


def _neural_rank_assets(
    world: WorldSnapshot,
    *,
    current_id: str,
    candidate_ids: list[str] | None = None,
) -> list[dict[str, float | str]]:
    nodes = _collect_nodes(world, include_vehicle=True, include_dock=True)
    if current_id not in nodes:
        return []

    current_pos = nodes[current_id]
    assets = _candidate_assets(world, candidate_ids)
    if not assets:
        return []

    distances = [current_pos.distance_to(asset.position) for asset in assets]
    max_distance = max(distances) if distances else 1.0
    anomalies = {asset.asset_id for asset in world.get_anomaly_assets()}
    battery_norm = world.vehicle.battery.remaining_percent / 100.0

    weights_1 = np.array([
        [-1.4, 1.2, 0.6, 0.3],
        [-1.0, 0.8, 1.1, 0.2],
        [-1.8, 1.0, 0.4, 0.4],
    ])
    bias_1 = np.array([0.1, 0.0, 0.05])
    weights_2 = np.array([1.1, 0.9, 0.7])
    bias_2 = 0.0

    ranked: list[dict[str, float | str]] = []
    for asset, distance in zip(assets, distances, strict=False):
        distance_norm = min(distance / max_distance, 1.0)
        priority_norm = 1.0 / (1.0 + asset.priority)
        anomaly_flag = 1.0 if asset.asset_id in anomalies else 0.0
        features = np.array([distance_norm, priority_norm, anomaly_flag, battery_norm], dtype=float)
        hidden = np.maximum(0.0, weights_1 @ features + bias_1)
        score = float(1.0 / (1.0 + np.exp(-(weights_2 @ hidden + bias_2))))
        ranked.append({
            "asset_id": asset.asset_id,
            "score": score,
            "distance_m": float(distance),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def collect_nodes(
    world: WorldSnapshot, *, include_vehicle: bool = True, include_dock: bool = True
) -> dict[str, Position]:
    """Collect node positions from the world snapshot."""
    return _collect_nodes(world, include_vehicle=include_vehicle, include_dock=include_dock)


def build_neighbor_graph(
    nodes: dict[str, Position], neighbor_k: int
) -> dict[str, list[tuple[str, float]]]:
    """Build a neighbor graph for pathfinding."""
    return _build_neighbor_graph(nodes, neighbor_k)


def dijkstra_path(
    graph: dict[str, list[tuple[str, float]]],
    start_id: str,
    goal_id: str,
    *,
    avoid_ids: list[str] | None = None,
) -> tuple[list[str], float]:
    """Compute a shortest path using Dijkstra's algorithm."""
    return _dijkstra_path(graph, start_id, goal_id, avoid_ids=avoid_ids)


def markov_rank_assets(
    world: WorldSnapshot,
    *,
    current_id: str,
    candidate_ids: list[str] | None = None,
    temperature: float = 1.0,
) -> list[dict[str, float | str]]:
    """Rank assets using a Markov-style heuristic."""
    return _markov_rank_assets(
        world, current_id=current_id, candidate_ids=candidate_ids, temperature=temperature
    )


def neural_rank_assets(
    world: WorldSnapshot,
    *,
    current_id: str,
    candidate_ids: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Rank assets using a lightweight neural heuristic."""
    return _neural_rank_assets(world, current_id=current_id, candidate_ids=candidate_ids)


class PredictiveModel(ABC):
    """Base class for predictive models."""

    @abstractmethod
    async def predict(self, context: DecisionContext, world: WorldSnapshot) -> PredictionResult:
        """Make prediction based on current context."""
        raise NotImplementedError

    @abstractmethod
    async def update_model(self, experience: LearningExperience) -> None:
        """Update model based on new experience."""
        raise NotImplementedError


class BatteryPredictor(PredictiveModel):
    """Predicts battery consumption and remaining flight time."""

    def __init__(self) -> None:
        """Initialize the BatteryPredictor."""
        self.consumption_history = deque(maxlen=100)
        self.weather_factors = {}

    async def predict(self, _context: DecisionContext, world: WorldSnapshot) -> PredictionResult:
        """Predict battery consumption for upcoming actions."""
        current_consumption = self._calculate_current_consumption(world)
        # Predict consumption for different action types
        predicted_consumption = current_consumption * 1.2  # Conservative estimate
        try:
            mission_context = world.mission_context
        except AttributeError:
            mission_context = None
        if mission_context:
            action_type = mission_context.get("next_action_type")
            predicted_consumption = self._predict_action_consumption(action_type, world)
        confidence = self._calculate_prediction_confidence()
        return PredictionResult(
            predicted_value=predicted_consumption,
            confidence=confidence,
            prediction_horizon=timedelta(minutes=30),
            factors_considered=["current_consumption", "flight_time", "battery_age"],
            uncertainty_margin=predicted_consumption * 0.2,
        )

    async def update_model(self, experience: LearningExperience) -> None:
        """Update battery model based on actual outcomes."""
        if "battery_consumption" in experience.contextual_features:
            actual_consumption = experience.contextual_features["battery_consumption"]
            predicted_consumption = experience.contextual_features.get(
                "predicted_consumption", actual_consumption
            )
            error = actual_consumption - predicted_consumption
            self.consumption_history.append(error)
            # Update learning parameters
            self.weather_factors[experience.situation_hash] = error

    def _calculate_current_consumption(self, world: WorldSnapshot) -> float:
        """Calculate current power consumption."""
        if world.vehicle.in_air:
            return world.vehicle.battery.current
        return world.vehicle.battery.current * 0.1  # Lower consumption when grounded

    def _predict_action_consumption(self, action_type: str, world: WorldSnapshot) -> float:
        """Predict consumption for specific action type."""
        base_consumption = self._calculate_current_consumption(world)
        multipliers = {
            "takeoff": 2.5,
            "orbit": 1.3,
            "goto": 1.1,
            "inspect": 1.2,
            "land": 0.8,
        }
        return base_consumption * multipliers.get(action_type, 1.0)

    def _estimate_max_flight_time(self) -> float:
        """Estimate maximum flight time with current battery."""
        return 1800.0  # 30 minutes in seconds

    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence in prediction."""
        if len(self.consumption_history) < 10:
            return 0.5  # Low confidence with little data
        recent_errors = list(self.consumption_history)[-20:]
        variance = np.variance(recent_errors) if recent_errors else 1.0
        # Higher confidence with more consistent predictions
        confidence = max(0.1, 1.0 - (variance / 10.0))
        return min(0.9, confidence)


class WeatherPredictor(PredictiveModel):
    """Predicts weather changes and their impact on operations."""

    def __init__(self) -> None:
        """Initialize the WeatherPredictor."""
        self.weather_history = deque(maxlen=50)
        self.wind_patterns = {}

    async def predict(self, _context: DecisionContext, world: WorldSnapshot) -> PredictionResult:
        """Predict weather changes."""
        current_weather = 0.0
        if world.environment is not None:
            try:
                current_weather = world.environment.wind_speed_ms
            except AttributeError:
                logger.warning("world_environment_missing_wind_speed")
        # Simple trend analysis
        if len(self.weather_history) >= 3:
            recent = list(self.weather_history)[-3:]
            trend = np.mean([recent[-1] - recent[-2], recent[-2] - recent[-3]])
        else:
            trend = 0
        # Predict conditions in 30 minutes
        predicted_wind = max(0, current_weather + trend * 6)
        confidence = 0.7 if abs(trend) < 2 else 0.4
        return PredictionResult(
            predicted_value=predicted_wind,
            confidence=confidence,
            prediction_horizon=timedelta(minutes=30),
            factors_considered=["current_wind", "trend", "seasonal_patterns"],
            uncertainty_margin=abs(trend) * 2,
        )

    async def update_model(self, experience: LearningExperience) -> None:
        """Update weather model with actual conditions."""
        if "actual_wind_speed" in experience.contextual_features:
            actual_wind = experience.contextual_features["actual_wind_speed"]
            predicted_wind = experience.contextual_features.get("predicted_wind_speed", actual_wind)
            error = actual_wind - predicted_wind
            self.weather_history.append(actual_wind)
            # Update pattern recognition
            hour = experience.timestamp.hour
            if hour not in self.wind_patterns:
                self.wind_patterns[hour] = []
            self.wind_patterns[hour].append(error)

    def _calculate_safety_factor(self, wind_speed: float) -> float:
        """Calculate safety factor for wind conditions."""
        if wind_speed < 5:
            return 1.0  # Safe
        if wind_speed < 10:
            return 0.7  # Moderate risk
        if wind_speed < 15:
            return 0.4  # High risk
        return 0.1  # Very high risk


class GoalHierarchy:
    """Hierarchical goal planning with multi-level objectives."""

    def __init__(self) -> None:
        """Initialize the GoalHierarchy."""
        self.goal_stack = []
        self.current_level = 0
        self.max_depth = 5

    async def generate_hierarchical_plan(
        self, world: WorldSnapshot, primary_goal: Goal
    ) -> list[Goal]:
        """Generate hierarchical plan with sub-goals."""
        plan = []
        if primary_goal.goal_type == GoalType.INSPECT_ASSET:
            # Break down inspection into sub-goals
            plan.extend(self._plan_inspection_mission(world, primary_goal))
        elif primary_goal.goal_type == GoalType.RETURN_LOW_BATTERY:
            # Plan efficient return route
            plan.extend(self._plan_battery_return(world, primary_goal))
        elif primary_goal.goal_type == GoalType.ABORT:
            # Plan safe abort sequence
            plan.extend(self._plan_abort_sequence(world, primary_goal))
        return plan[: self.max_depth]

    def _plan_inspection_mission(self, _world: WorldSnapshot, goal: Goal) -> list[Goal]:
        """Plan detailed inspection mission."""
        sub_goals = []
        # 1. Navigation to inspection area
        sub_goals.append(
            Goal(
                goal_type=GoalType.INSPECT_ASSET,
                target_asset=goal.target_asset,
                reasoning="Navigate to inspection area",
                priority=1.0,
            )
        )
        # 2. Establish observation orbit
        sub_goals.append(
            Goal(
                goal_type=GoalType.INSPECT_ASSET,
                target_asset=goal.target_asset,
                reasoning="Establish observation orbit",
                priority=0.9,
            )
        )
        # 3. Detailed inspection
        sub_goals.append(
            Goal(
                goal_type=GoalType.INSPECT_ASSET,
                target_asset=goal.target_asset,
                reasoning="Perform detailed inspection",
                priority=0.8,
            )
        )
        # 4. Documentation and analysis
        sub_goals.append(
            Goal(
                goal_type=GoalType.WAIT,
                target_asset=goal.target_asset,
                reasoning="Document and analyze inspection results",
                priority=0.6,
            )
        )
        return sub_goals

    def _plan_battery_return(self, _world: WorldSnapshot, _goal: Goal) -> list[Goal]:
        """Plan efficient battery return."""
        sub_goals = []
        # 1. Optimize return route
        sub_goals.append(
            Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reasoning="Calculate optimal return route",
                priority=1.0,
            )
        )
        # 2. Climbing for efficiency
        sub_goals.append(
            Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reasoning="Climb to efficient altitude",
                priority=0.9,
            )
        )
        # 3. Return to home
        sub_goals.append(
            Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reasoning="Execute return to launch",
                priority=0.8,
            )
        )
        return sub_goals

    def _plan_abort_sequence(self, _world: WorldSnapshot, _goal: Goal) -> list[Goal]:
        """Plan safe abort sequence."""
        sub_goals = []
        # 1. Immediate safety actions
        sub_goals.append(
            Goal(
                goal_type=GoalType.WAIT, reasoning="Assess current safety conditions", priority=1.0
            )
        )
        # 2. Safe landing approach
        sub_goals.append(
            Goal(
                goal_type=GoalType.WAIT,
                reasoning="Navigate to safe landing zone",
                priority=0.9,
            )
        )
        # 3. Emergency landing
        sub_goals.append(
            Goal(goal_type=GoalType.ABORT, reasoning="Execute emergency landing", priority=1.0)
        )
        return sub_goals


class MetaCognitiveMonitor:
    """Meta-cognitive monitoring for self-awareness."""

    def __init__(self) -> None:
        """Initialize the MetaCognitiveMonitor."""
        self.decision_history = deque(maxlen=100)
        self.performance_metrics = {}
        self.learning_rate = 0.1

    async def analyze_decision_quality(self, decision: Goal, outcome: dict[str, Any]) -> float:
        """Analyze quality of past decisions."""
        self.decision_history.append({
            "timestamp": datetime.now(),
            "decision": decision,
            "outcome": outcome,
            "success": outcome.get("success", False),
        })
        # Calculate success rate for recent decisions
        recent_decisions = list(self.decision_history)[-20:]
        if len(recent_decisions) > 0:
            success_rate = sum(1 for d in recent_decisions if d["success"]) / len(recent_decisions)
            return success_rate
        return 0.5  # Default neutral quality

    async def update_metacognition(
        self, context: DecisionContext, feedback: dict[str, Any]
    ) -> None:
        """Update meta-cognitive state based on feedback."""
        performance_score = await self.analyze_decision_quality(context, feedback)
        context.metacognitive_state.update({
            "recent_performance": performance_score,
            "adaptation_level": self._calculate_adaptation_level(performance_score),
            "confidence_adjustment": self._calculate_confidence_adjustment(performance_score),
        })

    def _calculate_adaptation_level(self, performance: float) -> float:
        """Calculate how much to adapt based on performance."""
        if performance > 0.8:
            return 1.2  # Increase adaptation
        if performance < 0.5:
            return 0.8  # Decrease adaptation
        return 1.0

    def _calculate_confidence_adjustment(self, performance: float) -> float:
        """Calculate confidence adjustment factor."""
        if performance > 0.8:
            return 1.1  # Be more confident
        if performance < 0.5:
            return 0.9  # Be less confident
        return 1.0


@dataclass
class DecisionModules:
    """Grouped decision engine components to keep initialization concise."""

    battery_predictor: BatteryPredictor
    weather_predictor: WeatherPredictor
    goal_hierarchy: GoalHierarchy
    metacognitive_monitor: MetaCognitiveMonitor
    agent: Agent[WorldSnapshot, MissionDecision]
    experience_buffer: deque[LearningExperience]


class AdvancedDecisionEngine:
    """Advanced agentic decision engine with learning and prediction."""

    def __init__(self) -> None:
        """Initialize the AdvancedDecisionEngine."""
        self.modules = DecisionModules(
            battery_predictor=BatteryPredictor(),
            weather_predictor=WeatherPredictor(),
            goal_hierarchy=GoalHierarchy(),
            metacognitive_monitor=MetaCognitiveMonitor(),
            agent=_create_mission_planner(),
            experience_buffer=deque(maxlen=500),
        )
        # Cognitive architecture parameters
        self.cognitive_parameters = {
            "reactive_threshold": 0.1,  # When to use reactive mode
            "deliberative_threshold": 0.3,  # When to use deliberative mode
            "learning_rate": 0.05,  # How fast to learn
            "prediction_weight": 0.7,  # Weight for predictions
            "confidence_decay": 0.95,  # How confidence decays
        }

    async def make_advanced_decision(
        self, world: WorldSnapshot, learning_experiences: list[LearningExperience] | None = None
    ) -> tuple[Goal, DecisionContext]:
        """Make advanced agentic decision with PydanticAI."""
        # Update learning models if experiences provided
        if learning_experiences:
            for experience in learning_experiences:
                await self.modules.battery_predictor.update_model(experience)
                await self.modules.weather_predictor.update_model(experience)
                self.modules.experience_buffer.append(experience)
        # Identify cognitive level
        cognitive_level = self._determine_cognitive_level(world)
        urgency = self._assess_urgency(world)
        confidence_type = self._determine_confidence_type(world, cognitive_level)
        context = DecisionContext(
            timestamp=datetime.now(),
            cognitive_level=cognitive_level,
            urgency_level=urgency,
            confidence_type=confidence_type,
            uncertainty_factors=self._identify_uncertainties(world),
            learning_weight=self._calculate_learning_weight(cognitive_level),
            metacognitive_state={},
        )
        try:
            # Use PydanticAI for decision making
            result = await self.modules.agent.run(
                f"Current State: {world.model_dump_json()}",
                deps=world,
            )
            mission_decision = result.output
            # Convert to Goal object
            target_asset = None
            if mission_decision.target_asset_id:
                for asset in world.assets:
                    if asset.asset_id == mission_decision.target_asset_id:
                        target_asset = asset
                        break
            goal = Goal(
                goal_type=mission_decision.goal_type,
                priority=int(mission_decision.priority),
                reason=mission_decision.reason,
                target_asset=target_asset,
            )
            # Update meta-cognition
            await self.modules.metacognitive_monitor.update_metacognition(
                context, {"prediction_used": True}
            )
            return goal, context
        except Exception as e:
            logger.error(f"PydanticAI decision failed: {e}", exc_info=True)
            # Fallback to reactive decision
            goal = await self._make_reactive_decision(world, context)
            return goal, context

    def _determine_cognitive_level(self, world: WorldSnapshot) -> CognitiveLevel:
        """Determine appropriate cognitive level."""
        emergency_level = self._calculate_emergency_level(world)
        if emergency_level > 0.8:
            return CognitiveLevel.REACTIVE
        if emergency_level > 0.4:
            return CognitiveLevel.DELIBERATIVE
        if emergency_level > 0.1:
            return CognitiveLevel.REFLECTIVE
        return CognitiveLevel.PREDICTIVE

    def _calculate_emergency_level(self, world: WorldSnapshot) -> float:
        """Calculate emergency/safety level."""
        factors = []
        # Battery emergency
        if world.vehicle.battery.remaining_percent < 20:
            factors.append(1.0 - (world.vehicle.battery.remaining_percent / 20.0))
        # GPS emergency
        if world.vehicle.gps and not world.vehicle.gps.has_fix:
            factors.append(0.9)
        # Health emergency
        if world.vehicle.health and not world.vehicle.health.is_healthy:
            factors.append(0.7)
        # Weather emergency
        if world.environment is not None:
            try:
                wind_speed = world.environment.wind_speed_ms
            except AttributeError:
                logger.warning("world_environment_missing_wind_speed")
                wind_speed = 0
            if wind_speed > 15:
                factors.append(0.8)
        return max(factors) if factors else 0.0

    def _assess_urgency(self, world: WorldSnapshot) -> UrgencyLevel:
        """Assess mission urgency."""
        emergency_score = self._calculate_emergency_level(world)
        if emergency_score > 0.7:
            return UrgencyLevel.CRITICAL
        if emergency_score > 0.5:
            return UrgencyLevel.HIGH
        if emergency_score > 0.3:
            return UrgencyLevel.MEDIUM
        if emergency_score > 0.1:
            return UrgencyLevel.LOW
        return UrgencyLevel.ROUTINE

    def _determine_confidence_type(
        self, world: WorldSnapshot, cognitive_level: CognitiveLevel
    ) -> ConfidenceType:
        """Determine confidence level based on situation and cognitive mode."""
        # High confidence in routine situations
        if cognitive_level == CognitiveLevel.PREDICTIVE:
            return ConfidenceType.HIGH
        # Lower confidence in emergency situations
        emergency = self._calculate_emergency_level(world)
        if emergency > 0.5:
            return ConfidenceType.LOW
        # Medium confidence for deliberative decisions
        if cognitive_level == CognitiveLevel.DELIBERATIVE:
            return ConfidenceType.MEDIUM
        return ConfidenceType.MEDIUM

    def _identify_uncertainties(self, world: WorldSnapshot) -> list[str]:
        """Identify sources of uncertainty."""
        uncertainties = []
        # Sensor uncertainties
        if world.vehicle.gps and world.vehicle.gps.hdop > 2.0:
            uncertainties.append("gps_accuracy_low")
        if world.vehicle.battery.remaining_percent < 30:
            uncertainties.append("battery_degradation_uncertain")
        # Environmental uncertainties
        if world.environment is not None:
            env = world.environment
            try:
                visibility_m = env.visibility_m
            except AttributeError:
                visibility_m = None
            if visibility_m is not None and visibility_m < 1000:
                uncertainties.append("reduced_visibility")
            try:
                wind_gusts_ms = env.wind_gusts_ms
            except AttributeError:
                wind_gusts_ms = None
            if wind_gusts_ms is not None and wind_gusts_ms > 5:
                uncertainties.append("wind_turbulence")
        return uncertainties

    def _calculate_learning_weight(self, cognitive_level: CognitiveLevel) -> float:
        """Calculate learning weight based on cognitive level."""
        weights = {
            CognitiveLevel.REACTIVE: 0.1,
            CognitiveLevel.DELIBERATIVE: 0.3,
            CognitiveLevel.REFLECTIVE: 0.5,
            CognitiveLevel.PREDICTIVE: 0.7,
        }
        return weights.get(cognitive_level, 0.3)

    async def _make_reactive_decision(
        self, world: WorldSnapshot, _context: DecisionContext
    ) -> Goal:
        """Make immediate reactive decision."""
        # Prioritize safety above all else
        emergency = self._calculate_emergency_level(world)
        if emergency > 0.7:
            return Goal(
                goal_type=GoalType.ABORT,
                reason=f"Emergency abort (severity: {emergency:.2f})",
                priority=100,
            )
        # Simple reactive rules
        if world.vehicle.battery.remaining_percent < 25:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reason="Low battery - immediate return required",
                priority=90,
            )
        # Default wait
        return Goal(goal_type=GoalType.WAIT, reason="No immediate action required", priority=10)

    async def _make_deliberative_decision(
        self,
        world: WorldSnapshot,
        _context: DecisionContext,
        battery_prediction: PredictionResult,
        weather_prediction: PredictionResult,
    ) -> Goal:
        """Make deliberative decision with planning."""
        # Consider predictions in decision making
        if battery_prediction.confidence > 0.6:
            battery_risk = "Battery prediction indicates adequate power"
        else:
            battery_risk = "Battery prediction indicates power concern"
        # Multi-criteria decision making
        criteria = {
            "safety": self._evaluate_safety(world),
            "efficiency": self._evaluate_efficiency(world, battery_prediction),
            "mission_progress": self._evaluate_mission_progress(world),
            "weather_suitability": 1.0 - (weather_prediction.predicted_value / 20.0)
            if weather_prediction
            else 1.0,
        }
        # Weighted decision
        weights = {
            "safety": 0.4,
            "efficiency": 0.3,
            "mission_progress": 0.2,
            "weather_suitability": 0.1,
        }
        score = sum(value * weights[key] for key, value in criteria.items())
        # Select action based on score
        if criteria["safety"] < 0.5:
            return Goal(
                goal_type=GoalType.ABORT,
                reason=f"Safety risk too high (score: {score:.2f}): {battery_risk}",
                priority=100,
            )
        if world.vehicle.battery.remaining_percent < 30:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reason=f"Efficient return needed (score: {score:.2f}): {battery_risk}",
                priority=80,
            )
        return Goal(
            goal_type=GoalType.WAIT,
            reason=f"Continue monitoring (score: {score:.2f}): {battery_risk}",
            priority=50,
        )

    async def _make_reflective_decision(
        self,
        world: WorldSnapshot,
        context: DecisionContext,
        _battery_prediction: PredictionResult,
        _weather_prediction: PredictionResult,
    ) -> Goal:
        """Make reflective decision with meta-cognition."""
        # Analyze recent performance
        recent_performance = self.modules.metacognitive_monitor.decision_history
        # Get insights from past decisions
        similar_situations = self._find_similar_situations(world)
        insights = self._extract_insights(similar_situations)
        # Reflective decision with learning integration
        confidence_adjustment = context.metacognitive_state.get("confidence_adjustment", 1.0)
        if len(recent_performance) > 10:
            success_rate = sum(1 for d in list(recent_performance)[-10:] if d["success"]) / 10
            if success_rate < 0.7:
                # Be more cautious if recent performance is poor
                confidence_adjustment *= 0.8
        reason = f"Reflective decision with insights: {insights} (adj: {confidence_adjustment:.2f})"
        # Make decision with adjusted confidence
        if world.vehicle.battery.remaining_percent < 25 * confidence_adjustment:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reason=reason,
                priority=int(80 * confidence_adjustment),
            )
        return Goal(
            goal_type=GoalType.WAIT,
            reason=reason,
            priority=int(50 * confidence_adjustment),
        )

    async def _make_predictive_decision(
        self,
        _world: WorldSnapshot,
        _context: DecisionContext,
        battery_prediction: PredictionResult,
        weather_prediction: PredictionResult,
    ) -> Goal:
        """Make predictive decision with anticipatory planning."""
        # Use predictions to optimize future decisions
        # Anticipate future conditions
        future_battery_risk = (
            battery_prediction.predicted_value < 25.0 if battery_prediction else False
        )
        future_weather_risk = (
            weather_prediction.predicted_value > 12.0 if weather_prediction else False
        )
        # Proactive decision making
        if future_battery_risk and future_weather_risk:
            reason = (
                "Predictive action: Battery "
                f"({battery_prediction.predicted_value:.1f}%) and weather "
                f"({weather_prediction.predicted_value:.1f} m/s) risks anticipated"
            )
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reason=reason,
                priority=90,
            )
        if future_battery_risk:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                reason=(
                    "Predictive battery management: "
                    f"{battery_prediction.predicted_value:.1f}% predicted"
                ),
                priority=70,
            )
        if future_weather_risk:
            return Goal(
                goal_type=GoalType.RETURN_WEATHER,
                reason=(
                    "Predictive weather avoidance: "
                    f"{weather_prediction.predicted_value:.1f} m/s predicted"
                ),
                priority=80,
            )
        return Goal(
            goal_type=GoalType.INSPECT_ASSET,
            reason="Predictive optimization: Favorable conditions predicted",
            priority=60,
        )

    def _evaluate_safety(self, world: WorldSnapshot) -> float:
        """Evaluate safety of current situation."""
        safety_score = 1.0
        # Battery safety
        if world.vehicle.battery.remaining_percent < 20:
            safety_score -= 0.4
        elif world.vehicle.battery.remaining_percent < 30:
            safety_score -= 0.2
        # GPS safety
        if world.vehicle.gps and not world.vehicle.gps.has_fix:
            safety_score -= 0.5
        # Vehicle health
        if world.vehicle.health and not world.vehicle.health.is_healthy:
            safety_score -= 0.3
        return max(0.0, safety_score)

    def _evaluate_efficiency(
        self, world: WorldSnapshot, battery_prediction: PredictionResult
    ) -> float:
        """Evaluate operational efficiency."""
        efficiency = 0.8  # Base efficiency
        # Adjust based on battery prediction
        if battery_prediction and battery_prediction.confidence > 0.6:
            efficiency += 0.1
        elif battery_prediction and battery_prediction.confidence < 0.4:
            efficiency -= 0.2
        # Altitude efficiency
        if world.vehicle.position.altitude_agl and world.vehicle.position.altitude_agl > 50:
            efficiency -= 0.1  # Less efficient at high altitude
        return max(0.0, efficiency)

    def _evaluate_mission_progress(self, world: WorldSnapshot) -> float:
        """Evaluate mission progress."""
        try:
            mission = world.mission
        except AttributeError:
            mission = None
        if not mission:
            return 0.5  # Neutral if no mission
        if mission.assets_total == 0:
            return 0.0
        progress = mission.assets_inspected / mission.assets_total
        return progress

    def _find_similar_situations(self, world: WorldSnapshot) -> list[dict[str, Any]]:
        """Find similar past situations."""
        similar_situations = []
        # Extract current world features for comparison
        current_features = self._extract_world_features(world)
        # Simple similarity matching based on key parameters
        for experience in self.modules.experience_buffer:
            if self._situation_similarity(experience.contextual_features, current_features) > 0.7:
                similar_situations.append(experience.contextual_features)
        return similar_situations[-5:]  # Return 5 most similar

    def _extract_world_features(self, world: WorldSnapshot) -> dict[str, Any]:
        """Extract comparable features from a WorldSnapshot as a dict."""
        features: dict[str, Any] = {
            "battery_percent": world.vehicle.battery.remaining_percent,
        }
        if world.vehicle.gps:
            features["gps_fix"] = world.vehicle.gps.has_fix
            features["gps_hdop"] = world.vehicle.gps.hdop
        if world.vehicle.health:
            features["is_healthy"] = world.vehicle.health.is_healthy
        return features

    def _situation_similarity(self, features1: dict[str, Any], features2: dict[str, Any]) -> float:
        """Calculate similarity between two situations."""
        similarities = []
        # Battery similarity
        if "battery_percent" in features1 and "battery_percent" in features2:
            batt_diff = abs(features1["battery_percent"] - features2["battery_percent"])
            similarities.append(1.0 - (batt_diff / 100.0))
        # GPS similarity
        if "gps_fix" in features1 and "gps_fix" in features2:
            if features1.get("gps_fix") == features2.get("gps_fix"):
                similarities.append(0.8)
            else:
                similarities.append(0.2)
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _extract_insights(self, similar_situations: list[dict[str, Any]]) -> str:
        """Extract insights from similar situations."""
        if not similar_situations:
            return "No similar past situations"
        # Analyze outcomes in similar situations
        outcomes = [s.get("success", False) for s in similar_situations]
        success_rate = sum(outcomes) / len(outcomes)
        if success_rate > 0.8:
            return f"Similar situations had {success_rate:.0%} success rate"
        if success_rate < 0.5:
            return f"Similar situations had {success_rate:.0%} success rate - be cautious"
        return f"Similar situations had mixed success rate ({success_rate:.0%})"


async def create_advanced_decision_engine() -> AdvancedDecisionEngine:
    """Factory function to create advanced decision engine."""
    await asyncio.sleep(0)
    return AdvancedDecisionEngine()


class EnhancedGoalSelector:
    """Enhanced goal selector using advanced decision engine."""

    def __init__(self) -> None:
        """Initialize the EnhancedGoalSelector."""
        self.advanced_engine = None  # Will be initialized asynchronously

    async def initialize(self) -> None:
        """Initialize advanced components."""
        self.advanced_engine = await create_advanced_decision_engine()

    async def select_goal(self, world: WorldSnapshot) -> Goal:
        """Select goal using advanced agentic decision making."""
        if not self.advanced_engine:
            await self.initialize()
        # Use advanced decision engine
        goal, context = await self.advanced_engine.make_advanced_decision(world)
        # Log the decision context for explainability
        logger.info(
            "advanced_decision_made",
            goal_type=goal.goal_type.value,
            cognitive_level=context.cognitive_level.value,
            confidence=context.confidence_type.value,
            urgency=context.urgency_level.value,
            reasoning=goal.reason,
        )
        return goal
