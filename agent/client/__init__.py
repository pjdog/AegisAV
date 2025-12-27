"""
Agent Client Package

Lightweight components for vehicle-side execution.
"""

from agent.client.state_collector import StateCollector
from agent.client.action_executor import ActionExecutor

__all__ = [
    "StateCollector",
    "ActionExecutor",
]
