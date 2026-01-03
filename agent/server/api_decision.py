"""Decision-making, feedback, and execution event API routes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Response

from agent.api_models import ActionType, DecisionResponse, VehicleStateRequest
from agent.server.decision import Decision
from agent.server.events import Event, EventSeverity, EventType
from agent.server.feedback_store import persist_feedback
from agent.server.goals import Goal, GoalType
from agent.server.models import DecisionFeedback
from agent.server.state import connection_manager, server_state
from agent.server.unreal_stream import (
    CriticVerdict,
    CognitiveLevel as UnrealCognitiveLevel,
    thinking_tracker,
    unreal_manager,
)
from autonomy.vehicle_state import Position
from metrics.logger import DecisionLogContext
from mapping.decision_context import MapContext, map_decision_logger
from mapping.safety_gates import PlannerSafetyGate, SafetyGateResult
from vision.data_models import DetectionResult

logger = structlog.get_logger(__name__)


async def _record_telemetry(
    state: VehicleStateRequest,
    vehicle_id: str,
    source: str,
) -> None:
    payload = state.model_dump(mode="json")
    payload["vehicle_id"] = vehicle_id
    payload["source"] = source
    server_state.known_vehicles.add(vehicle_id)
    server_state.latest_telemetry[vehicle_id] = payload

    try:
        server_state.telemetry_logger.log_telemetry(vehicle_id, payload, source=source)
    except Exception as exc:
        logger.warning("telemetry_log_failed", error=str(exc), vehicle_id=vehicle_id)

    if server_state.store:
        stored = await server_state.store.add_telemetry(vehicle_id, payload)
        if not stored:
            logger.warning("telemetry_store_failed", vehicle_id=vehicle_id)


async def _make_decision(snapshot, risk) -> tuple[Decision, Goal | None]:
    # pylint: disable=too-many-return-statements
    """Core decision-making logic.

    Combines goal selection with risk assessment to produce a decision.
    """
    if server_state.risk_evaluator.should_abort(risk):
        return Decision.abort(
            reason=risk.abort_reason or "Risk threshold exceeded",
            risk_factors={name: f.value for name, f in risk.factors.items()},
        ), None

    goal: Goal = await server_state.goal_selector.select_goal(snapshot)

    if goal.goal_type == GoalType.ABORT:
        return Decision.abort(goal.reason), goal

    if goal.goal_type in (
        GoalType.RETURN_LOW_BATTERY,
        GoalType.RETURN_MISSION_COMPLETE,
        GoalType.RETURN_WEATHER,
    ):
        return Decision.return_to_dock(
            reason=goal.reason,
            confidence=goal.confidence,
        ), goal

    if goal.goal_type == GoalType.INSPECT_ASSET and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl
                + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            inspection={
                "orbit_radius_m": goal.target_asset.orbit_radius_m,
                "dwell_time_s": goal.target_asset.dwell_time_s,
            },
        ), goal

    if goal.goal_type == GoalType.INSPECT_ANOMALY and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl
                + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            inspection={
                "orbit_radius_m": goal.target_asset.orbit_radius_m * 0.75,
                "dwell_time_s": goal.target_asset.dwell_time_s * 2,
            },
        ), goal

    if goal.goal_type == GoalType.WAIT:
        return Decision.wait(goal.reason), goal

    return Decision(
        action=ActionType.NONE,
        reasoning="No actionable goal",
    ), None


async def _process_state(state: VehicleStateRequest, *, source: str) -> DecisionResponse:
    logger.debug("state_received", armed=state.armed, mode=state.mode)

    vehicle_id = state.vehicle_id or "drone_001"
    await _record_telemetry(state, vehicle_id, source)

    vehicle_state = state.to_dataclass()

    server_state.world_model.update_vehicle(vehicle_state)

    planner_gate = server_state.planner_safety_gate or PlannerSafetyGate()
    if server_state.planner_safety_gate is None:
        server_state.planner_safety_gate = planner_gate
    gate_config = planner_gate.config
    map_context = MapContext.from_navigation_map(
        server_state.navigation_map,
        stale_threshold_s=gate_config.max_map_age_s,
        min_quality_score=gate_config.min_map_confidence,
    )
    gate_result = planner_gate.check_planning_allowed(map_context)

    snapshot = server_state.world_model.get_snapshot(map_context=map_context)
    if snapshot is None:
        raise HTTPException(status_code=500, detail="World model not initialized")

    drone_id = vehicle_id
    if unreal_manager.active_connections > 0:
        await thinking_tracker.start_thinking(
            drone_id=drone_id,
            cognitive_level=UnrealCognitiveLevel.DELIBERATIVE,
        )

    risk = server_state.risk_evaluator.evaluate(snapshot)
    await connection_manager.broadcast(
        Event(
            event_type=EventType.RISK_UPDATE,
            timestamp=datetime.now(),
            data={
                "risk_level": risk.overall_level.value,
                "risk_score": risk.overall_score,
                "factors": {name: f.value for name, f in risk.factors.items()},
                "abort_recommended": risk.abort_recommended,
                "warnings": risk.warnings,
            },
            severity=EventSeverity.WARNING if risk.abort_recommended else EventSeverity.INFO,
        )
    )

    for warning in risk.warnings:
        logger.warning("risk_warning", warning=warning)

    if unreal_manager.active_connections > 0:
        await thinking_tracker.update_thinking(
            drone_id=drone_id,
            situation=(
                f"Evaluating situation. Battery: {snapshot.vehicle.battery.remaining_percent:.0f}%"
            ),
            considerations=[
                f"Risk level: {risk.overall_level.value}",
                f"Battery remaining: {snapshot.vehicle.battery.remaining_percent:.0f}%",
                *risk.warnings[:3],
            ],
            risk_score=risk.overall_score,
            risk_level=risk.overall_level.value,
            risk_factors={name: f.value for name, f in risk.factors.items()},
        )

    decision, goal = await _make_decision(snapshot, risk)
    if gate_result.result == SafetyGateResult.FREEZE and decision.is_movement:
        decision = Decision.wait(reason=f"Planner frozen: {gate_result.reason}")
        goal = None

    if goal:
        await connection_manager.broadcast(
            Event(
                event_type=EventType.GOAL_SELECTED,
                timestamp=datetime.now(),
                data={
                    "goal_type": goal.goal_type.value,
                    "priority": goal.priority,
                    "target": goal.target_asset.asset_id if goal.target_asset else None,
                    "confidence": goal.confidence,
                    "reasoning": goal.reason,
                },
                severity=EventSeverity.INFO,
            )
        )

    approved, escalation = await server_state.critic_orchestrator.validate_decision(
        decision, snapshot, risk
    )

    await connection_manager.broadcast(
        Event(
            event_type=EventType.CRITIC_VALIDATION,
            timestamp=datetime.now(),
            data={
                "agent_label": "Orchestration AG",
                "drone_id": drone_id,
                "drone_name": drone_id,
                "approved": approved,
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "risk_level": risk.overall_level.value,
                "risk_score": risk.overall_score,
                "escalation": (
                    {
                        "reason": escalation.reason,
                        "severity": escalation.severity.value,
                    }
                    if escalation
                    else None
                ),
            },
            severity=EventSeverity.WARNING if not approved else EventSeverity.INFO,
        )
    )

    if unreal_manager.active_connections > 0:
        verdict = CriticVerdict.APPROVE if approved else CriticVerdict.REJECT
        concerns = [escalation.reason] if escalation else []
        for critic_name in ["safety", "efficiency", "goal_alignment"]:
            await thinking_tracker.report_critic(
                drone_id=drone_id,
                critic_name=critic_name,
                verdict=verdict,
                confidence=0.9 if approved else 0.7,
                concerns=concerns if critic_name == "safety" else [],
            )

    if not approved:
        logger.warning(
            "decision_blocked_by_critics",
            decision_id=decision.decision_id,
            original_action=decision.action.value,
            reason=escalation.reason if escalation else "Unknown",
        )
        reason = (
            "Decision blocked by critics: "
            f"{escalation.reason if escalation else 'Safety concerns'}"
        )
        decision = Decision.abort(reason=reason)

    server_state.outcome_tracker.create_outcome(decision)

    server_state.last_decision = decision
    server_state.decisions_made += 1

    map_decision_logger.log_planner_gate(
        map_context,
        decision.decision_id,
        decision.action.value,
        gate_result,
    )
    if map_context.map_available:
        map_decision_logger.log_map_decision(
            map_context,
            decision.decision_id,
            decision.action.value,
        )

    server_state.decision_logger.log_decision(
        DecisionLogContext(
            decision=decision,
            risk=risk,
            world=snapshot,
            goal=goal,
            escalation=escalation,
            map_context=map_context,
        )
    )

    logger.info(
        "decision_made",
        decision_id=decision.decision_id,
        action=decision.action.value,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_level=risk.overall_level.value,
    )

    await connection_manager.broadcast(
        Event(
            event_type=EventType.SERVER_DECISION,
            timestamp=datetime.now(),
            data={
                "agent_label": "Drone AG",
                "drone_id": drone_id,
                "drone_name": drone_id,
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "parameters": decision.parameters,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "risk_level": risk.overall_level.value,
                "risk_score": risk.overall_score,
                "battery_percent": snapshot.vehicle.battery.remaining_percent,
                "reasoning_context": {
                    "available_assets": len(getattr(snapshot, "assets", []) or []),
                    "pending_inspections": len(
                        snapshot.get_pending_assets() if hasattr(snapshot, "get_pending_assets") else []
                    ),
                    "fleet_in_progress": 0,
                    "fleet_completed": getattr(snapshot.mission, "assets_inspected", 0)
                    if getattr(snapshot, "mission", None)
                    else 0,
                    "battery_ok": snapshot.vehicle.battery.remaining_percent > 25,
                    "battery_percent": snapshot.vehicle.battery.remaining_percent,
                    "weather_ok": (
                        getattr(getattr(snapshot, "environment", None), "wind_speed_ms", 0.0) < 12
                    ),
                    "wind_speed": getattr(getattr(snapshot, "environment", None), "wind_speed_ms", 0.0),
                },
                "target_asset": (
                    {
                        "asset_id": goal.target_asset.asset_id,
                        "name": goal.target_asset.name,
                    }
                    if goal and goal.target_asset
                    else None
                ),
            },
            severity=EventSeverity.INFO,
        )
    )

    if unreal_manager.active_connections > 0:
        await thinking_tracker.complete_thinking(
            drone_id=drone_id,
            action=decision.action.value,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            parameters=decision.parameters,
        )

    return DecisionResponse(
        decision_id=decision.decision_id,
        action=decision.action.value,
        parameters=decision.parameters,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_assessment={name: f.value for name, f in risk.factors.items()},
        timestamp=decision.timestamp,
        map_context=map_context.to_dict() if map_context else None,
    )


def register_decision_routes(app: FastAPI) -> None:
    """Register decision, feedback, and execution-event routes."""

    @app.post("/api/execution_event")
    async def receive_execution_event(event_data: dict) -> dict:
        """Receive execution event from client for WebSocket broadcast."""
        await connection_manager.broadcast(
            Event(
                event_type=EventType.CLIENT_EXECUTION,
                timestamp=datetime.now(),
                data=event_data,
                severity=EventSeverity.INFO,
            )
        )

        return {"status": "received"}

    @app.post("/feedback")
    async def receive_feedback(feedback: DecisionFeedback) -> dict:
        """Receive feedback from client about decision execution outcomes."""
        logger.info(
            "feedback_received",
            decision_id=feedback.decision_id,
            status=feedback.status.value,
            battery_consumed=feedback.battery_consumed,
        )

        outcome = await server_state.outcome_tracker.process_feedback(feedback)
        await persist_feedback(server_state.store, feedback, outcome)

        vision_observation = None
        # Record vision captures for all inspections, not just anomaly detections
        # This ensures coverage and captures stay in sync
        if feedback.asset_inspected and server_state.vision_enabled and server_state.vision_service:
            try:
                client_detection: DetectionResult | None = None
                image_path: Path | None = None
                vehicle_state: dict | None = None
                asset_id = feedback.asset_inspected or "unknown"

                inspection_data = feedback.inspection_data or {}
                if isinstance(inspection_data, dict):
                    asset_id_from_payload = inspection_data.get("asset_id")
                    if isinstance(asset_id_from_payload, str) and asset_id_from_payload:
                        asset_id = asset_id_from_payload

                    vehicle_state = inspection_data.get("vehicle_state")
                    if vehicle_state is not None and not isinstance(vehicle_state, dict):
                        vehicle_state = None

                    vision_payload = inspection_data.get("vision", {})
                    if isinstance(vision_payload, dict):
                        vehicle_state = vehicle_state or vision_payload.get("vehicle_state")
                        if vehicle_state is not None and not isinstance(vehicle_state, dict):
                            vehicle_state = None

                        image_path_str = (
                            vision_payload.get("best_image_path")
                            or vision_payload.get("best_detection_image")
                            or vision_payload.get("image_path")
                        )
                        if isinstance(image_path_str, str) and image_path_str:
                            image_path = Path(image_path_str)

                        detection_payload = (
                            vision_payload.get("best_detection")
                            or vision_payload.get("client_detection")
                            or vision_payload.get("detection_result")
                        )
                        if isinstance(detection_payload, dict):
                            try:
                                client_detection = DetectionResult.model_validate(detection_payload)
                            except Exception as exc:
                                logger.warning(
                                    "feedback_client_detection_parse_failed",
                                    error=str(exc),
                                )

                vision_observation = await server_state.vision_service.process_inspection_result(
                    asset_id=asset_id,
                    client_detection=client_detection,
                    image_path=image_path,
                    vehicle_state=vehicle_state,
                )

                await connection_manager.broadcast(
                    Event(
                        event_type=EventType.VISION_DETECTION,
                        timestamp=datetime.now(),
                        data={
                            "observation_id": vision_observation.observation_id,
                            "asset_id": vision_observation.asset_id,
                            "defect_detected": vision_observation.defect_detected,
                            "max_confidence": vision_observation.max_confidence,
                            "max_severity": vision_observation.max_severity,
                        },
                        severity=(
                            EventSeverity.WARNING
                            if vision_observation.defect_detected
                            else EventSeverity.INFO
                        ),
                    )
                )

                if vision_observation.anomaly_created:
                    logger.info(
                        "vision_anomaly_created",
                        asset_id=vision_observation.asset_id,
                        anomaly_id=vision_observation.anomaly_id,
                        severity=vision_observation.max_severity,
                        confidence=vision_observation.max_confidence,
                    )

                    await connection_manager.broadcast(
                        Event(
                            event_type=EventType.ANOMALY_CREATED,
                            timestamp=datetime.now(),
                            data={
                                "anomaly_id": vision_observation.anomaly_id or "",
                                "asset_id": vision_observation.asset_id,
                                "severity": vision_observation.max_severity,
                                "confidence": vision_observation.max_confidence,
                                "description": (
                                    f"Vision anomaly detected on {vision_observation.asset_id}"
                                ),
                            },
                            severity=EventSeverity.CRITICAL,
                        )
                    )

            except Exception as exc:
                logger.error("vision_processing_failed", error=str(exc))

        if outcome:
            response = {
                "status": "feedback_received",
                "decision_id": feedback.decision_id,
                "outcome_status": outcome.execution_status.value,
            }

            if vision_observation:
                response["vision"] = {
                    "observation_id": vision_observation.observation_id,
                    "defect_detected": vision_observation.defect_detected,
                    "anomaly_created": vision_observation.anomaly_created,
                    "max_confidence": vision_observation.max_confidence,
                    "max_severity": vision_observation.max_severity,
                }

            return response

        logger.warning("feedback_for_unknown_decision", decision_id=feedback.decision_id)
        return {
            "status": "feedback_received_but_unknown_decision",
            "decision_id": feedback.decision_id,
        }

    @app.post("/state", response_model=DecisionResponse)
    async def receive_state(state: VehicleStateRequest) -> DecisionResponse:
        """Receive vehicle state and return decision."""
        return await _process_state(state, source="http")

    @app.post("/state/async")
    async def receive_state_async(state: VehicleStateRequest) -> dict:
        """Receive vehicle state and enqueue decision for async retrieval."""
        decision = await _process_state(state, source="async")
        vehicle_id = state.vehicle_id or "drone_001"
        await server_state.decision_queue.put(vehicle_id, decision)
        return {
            "status": "queued",
            "vehicle_id": vehicle_id,
            "decision_id": decision.decision_id,
        }

    @app.get("/decisions/next", response_model=DecisionResponse)
    async def get_next_decision(
        vehicle_id: str, timeout_s: float = 10.0
    ) -> DecisionResponse | Response:
        """Long-poll for the next decision for a vehicle."""
        timeout_s = max(0.1, min(timeout_s, 30.0))
        decision = await server_state.decision_queue.get(vehicle_id, timeout_s)
        if decision is None:
            return Response(status_code=204)
        return decision
