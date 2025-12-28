"""
Integration tests for AegisAV system.

These tests validate end-to-end workflows across multiple subsystems:
- Vision pipeline (camera → detection → anomaly → re-inspection)
- Decision pipeline (state → goal → decision → critics → execution)
- Feedback loops (outcomes → learning → adaptation)
"""
