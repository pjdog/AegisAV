# Decision Logs

This directory contains decision trace logs from agent runs.

Each run creates a `.jsonl` file with one decision per line.

## Log Format

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "decision_id": "dec_20240101_120000_123456",
  "action": "inspect",
  "parameters": {...},
  "confidence": 0.95,
  "reasoning": "Scheduled inspection of Solar Panel A",
  "risk_level": "low",
  "risk_score": 0.15,
  "risk_factors": {...},
  "vehicle_position": {"lat": 47.398, "lon": 8.546, "alt": 500},
  "battery_percent": 75.0,
  "mode": "GUIDED",
  "armed": true,
  "goal_type": "inspect_asset",
  "target_asset": "asset-001"
}
```

## Analysis

Use the metrics package to analyze logs:

```python
from metrics.logger import DecisionLogger, create_summary_report

logger = DecisionLogger(Path("logs"))
entries = logger.load_run("20240101_120000")
report = create_summary_report(entries)
print(report)
```
