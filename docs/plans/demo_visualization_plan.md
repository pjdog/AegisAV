# Demo Visualization Plan

## Goals
- Show agent autonomy in action with clear decision context
- Make risk and safety reasoning visible
- Demonstrate mission progress and outcome at a glance

## Primary Views
1. Mission Overview
   - Total decisions, risk summary, mission time window
   - Latest decision with confidence and rationale
2. Risk and Battery Timeline
   - Line chart of risk score over time
   - Line chart of battery percentage over time
3. Decision Mix
   - Action distribution (inspect, return, wait, abort)
4. Recent Decisions Table
   - Timestamp, action, confidence, risk level, battery

## Demo Narrative (5-7 minutes)
1. Setup (30s)
   - Introduce mission goal and live dashboard
2. Agent in Motion (2m)
   - Highlight decision cadence and action mix
   - Call out a sample decision and reasoning
3. Risk Response (2m)
   - Show risk spike and corresponding behavior change
   - Tie to battery trend or weather thresholds
4. Outcome (1-2m)
   - Return to dock or mission completion
   - Summarize key metrics

## Data Requirements
- Decision logs written as JSONL per run
- Risk score and risk level
- Battery percentage
- Decision action and confidence
- World position (optional for future map view)

## Stretch Goals (Optional)
- Map panel with asset locations and vehicle path
- Timeline scrubber to replay decisions
- Compare runs side-by-side for benchmarking
