"""Agent Client Package.

Lightweight components for vehicle-side execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.client.action_executor import ActionExecutor
    from agent.client.state_collector import StateCollector

__all__ = ["ActionExecutor", "StateCollector"]


def __getattr__(name: str):
    if name == "ActionExecutor":
        from agent.client.action_executor import ActionExecutor

        return ActionExecutor
    if name == "StateCollector":
        from agent.client.state_collector import StateCollector

        return StateCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
