# Simulated One-Night Sleep MVP Dataset

This dataset is synthetic test data for the individualized sleep regulation MVP.

## User
- user_id: u_001
- date: 2026-06-11
- timezone: Asia/Singapore / UTC+08:00

## Files
- user_profiles.csv: static user profile
- user_baselines.csv: resting baseline and 7-day summary fields
- physiology_stream.csv: 1-minute physiology stream from 2026-06-11T22:10:00+08:00 to 2026-06-12T06:40:00+08:00
- state_snapshots.csv: 3-minute rule-classifier snapshots
- action_logs.csv: 1-minute control-action logs for the first 90 minutes
- episode_outcomes.csv: one-night outcome labels
- reward_logs.csv: illustrative dense reward logs aligned with action_logs
- personalized_strategy.json: generated personalized strategy snapshot

## Simulation Notes
- Sleep onset is simulated at 2026-06-11T22:47:00+08:00, so sleep latency is 37.0 minutes.
- Micro-arousal events are simulated near 01:18 and 03:42.
- A short wake event is simulated near 04:55.
- This dataset is not real physiology and should only be used for adapter, pipeline, controller, and training-interface tests.
