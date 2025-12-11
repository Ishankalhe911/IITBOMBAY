Evaluation Notes

This document summarizes how the system computes its scores, how each metric is interpreted, and how the evaluation pipeline aligns with the official judging criteria of the Hackathon.

1. Scoring Philosophy

The goal of Mentor-Scoring-AI is to provide:

Objective and consistent evaluation

Multimodal analysis (audio + visual + text)

Interpretable results (segment-wise, metric-wise)

The system is aligned with the Hackathon Problem Statement #2 evaluation structure.

2. Metric Breakdown
2.1 Engagement (20%)

Derived from gesture analysis:

Hand movement frequency

Face visibility across frames

Eye contact ratio

Detected pose confidence

YOLO pose estimations are aggregated into a normalized engagement score.

2.2 Communication (20%)

Based on transcript features:

Speech continuity

Pause frequency

Basic clarity indicators

Sentence segmentation

Whisper output is analyzed for quality and fluency.

2.3 Technical Depth (30%)

Computed using:

Ollama Llama depth analysis (if available)

Fallback heuristic engine, based on:

Vocabulary richness

Technical terminology density

Long-word ratio

Sentence complexity

This produces:

overall_depth_score

reasoning

per_segment[] breakdown

2.4 Clarity (20%)

Derived from:

Transcription confidence

Word uniqueness

Repetition patterns

Structural readability

The clarity score is independent of the depth score.

2.5 Interaction Quality (10%)

Defined by:

Eye-contact stability

Face orientation

Basic non-verbal communication cues

Pose sampling (~25 frames) captures variation across the video.

3. How Scores Are Aggregated

Each metric is normalized from 0.0 → 1.0 and then weighted:

final_score =
    0.20 * engagement +
    0.20 * communication +
    0.30 * technical_depth +
    0.20 * clarity +
    0.10 * interaction_quality


Outputs include:

Final mentor score

Component-wise breakdown

Depth reasoning text

Segment table

Transcript and gesture interpretation

4. Explanation Examples
Sample Engagement Interpretation

“Active hand gestures detected in 65%+ of sampled frames”

“Strong eye contact maintained consistently”

“Face visible in majority of tracked frames”

Sample Depth Interpretation

“Uses technical terminology with moderate density”

“Explanation structure suggests intermediate conceptual depth”

Sample Clarity Interpretation

“Good sentence segmentation, minimal filler words”

5. Evaluation Alignment with Problem Statement

The hackathon requires:

Requirement	Implemented In
Communication clarity analysis	Transcription module
Engagement analysis	Gesture module
Technical correctness / depth	Depth module
Interaction quality	Eye/contact tracking
Multimodal evaluation	Combined pipeline
Parameter-wise scoring report	Streamlit UI

All metrics map 1:1 with the expected outcome specification.

6. Limitations & Future Improvements

Depth scoring relies partly on heuristics without LLM

Gesture model may miss subtle micro-expressions

Does not currently detect tone or emotion in speech

Future roadmap includes:

Real-time mentor scoring

Comparative dashboards

Bias mitigation

Calibration against human evaluators
