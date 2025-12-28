import sys
from unittest.mock import MagicMock
sys.path.append("/home/pjdog/code/AegisAV")

from agent.server.dashboard import _calculate_relative_pos, _recent
from agent.server.goals import Goal, GoalType
from agent.server.decision import Decision, ActionType
from agent.server.risk_evaluator import RiskAssessment, RiskLevel, RiskFactor
from metrics.logger import DecisionLogEntry
from autonomy.vehicle_state import Position, VehicleState, VehicleMode
from agent.server.world_model import Asset, AssetType, WorldSnapshot

def test_relative_pos():
    # Test 1 deg offset (approx 111km)
    res = _calculate_relative_pos(0, 0, 1, 0)
    print(f"1 deg Lat diff (expect ~111,111m Y): {res}")
    assert abs(res['y'] - 111111) < 100

    res = _calculate_relative_pos(0, 0, 0, 1)
    print(f"1 deg Lon diff at equator (expect ~111,111m X): {res}")
    assert abs(res['x'] - 111111) < 100

def test_log_entry_structure():
    # Simulate a log entry dictionary
    mock_entry = {
        "timestamp": "2023-01-01T00:00:00",
        "action": "wait",
        "vehicle_position": {"lat": 0, "lon": 0, "alt": 10},
        "assets": [
            {"id": "a1", "type": "solar_panel", "lat": 0.001, "lon": 0.001, "alt": 0}
        ]
    }
    
    recent_items = _recent([mock_entry], limit=1)
    item = recent_items[0]
    print(f"Recent item spatial context: {item.get('spatial_context')}")
    
    ctx = item.get('spatial_context')[0]
    assert ctx['id'] == 'a1'
    assert abs(ctx['y'] - 111) < 10 # 0.001 deg ~ 111m
    assert abs(ctx['x'] - 111) < 10

if __name__ == "__main__":
    test_relative_pos()
    test_log_entry_structure()
    print("Verification Passed")
