# Evaluation Notes & Metrics

This section describes how results can be evaluated and validated.

## Outputs produced
- `duration_seconds` (float)
- `transcript` (str)
- `gestures` (markdown summary)
- `depth` (overall_depth_score: float 0.0-1.0, reasoning: str, per_segment: list)

## Suggested quantitative checks
- **Transcription sanity:** word count > 10 for sessions (>30s). If transcript contains errors, verify ffmpeg extraction and whisper availability.
- **Gesture heuristics:** use % frames with detected hands, % frames with face/nose in front region for eye contact. Validate thresholds empirically.
- **Depth score validation:** compare local heuristic vs. human labels on 50-100 sample videos; compute Pearson/Spearman correlation.

## Future evaluation pipeline
- Create labeled dataset: human raters score teaching quality (1-5) for sample videos.
- Train/fit regression model mapping transcript & gesture features to human scores.
- Report metrics: RMSE, MAE, correlation, classification accuracy (if binned).
