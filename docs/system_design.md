##System Design

**This document outlines the end-to-end workflow, component responsibilities, and data flow of the Mentor-Scoring-AI system. The design focuses on modularity, reproducibility, and CPU-friendly multimodal processing
*1. High-Level Workflow
                ┌────────────────────────┐
                │     Video Input        │
                │ (Upload or URL Fetch)  │
                └─────────────┬──────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Audio Extraction (FFmpeg)│
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼─────────────┐
                │   Speech-to-Text (Whisper)│
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼────────────┐
                │ Transcript Processing    │
                │ (Clarity + Segmentation) │
                └─────────────┬────────────┘
                              │
     ┌────────────────────────▼────────────────────────┐
     │         Parallel Multimodal Processing          │
     │                                                 │
     │   ┌──────────────────────────┐   ┌───────────┐  │
     │   │ Visual/Gesture Analysis  │   │ Depth AI  │ │
     │   │ (YOLO Pose Sampling)     │   │ LLM/Heur. │ │
     │   └──────────────────────────┘   └───────────┘ │
     └────────────────────────┬───────────────────────┘
                              │
                ┌─────────────▼────────────┐
                │     Metric Aggregation   │
                │  (Weighted Scoring Model)│
                └─────────────┬────────────┘
                              │
                ┌─────────────▼────────────┐
                │       Streamlit UI       │
                │  Results + Visualization │
                └──────────────────────────┘

2. Components
2.1 Streamlit UI (src/app.py)

Manages input: upload or URL-based videos

Displays processing steps, results, and tables

Orchestrates calls to all AI modules

Suitable for demo and prototype evaluation

2.2 Audio Extraction (utils/ffmpeg_utils.py)

Converts video → WAV using FFmpeg

Normalizes audio for Whisper STT

Handles file safety and temporary paths

2.3 Transcription Module (ai/transcribe.py)

Uses faster-whisper for CPU-efficient transcription

Falls back to standard Whisper when needed

Produces:

transcript text

word count

clarity metrics

2.4 Gesture Analysis (ai/gesture_analysis.py)

YOLOv11n/YOLOv8n pose estimation

Samples ~25 frames for fast evaluation

Tracks:

Hand movement frequency

Eye contact

Face visibility

Engagement score

2.5 Depth Analysis (ai/depth_analysis.py)

Two-layer depth scoring:

Ollama Llama-based reasoning (if available)

Fallback heuristic engine (vocabulary richness, technical terms, segmentation)

Returns structured depth JSON:

{
  "overall_depth_score": 0.71,
  "reasoning": "...",
  "per_segment": [...]
}

2.6 Scoring & Metrics

Weighted according to problem statement:

Metric	Weight
Engagement	20%
Communication	20%
Technical Depth	30%
Clarity	20%
Interaction Quality	10%

Modules contribute to specific metrics.

3. Data Flow
Input

.mp4, .mov, .avi

YouTube/public URLs

Intermediate Outputs

WAV audio

Transcript

Pose keypoint metadata

Depth JSON

Final Output

Mentor Performance Report:

Transcript

Gesture analysis

Depth analysis

Consolidated scores

Segment-wise interpretation

4. Design Considerations

CPU-first optimization: No GPU required

Graceful fallbacks: Ollama optional

Isolated modules: Easy replacement/swapping

Caching: Model loading occurs once per session

Reproducibility: Each component is deterministic when LLM is not used

5. Deployment Notes

Suitable for:

Local evaluation

Institutional desktop use

Cloud deployment with Streamlit/Gradio

Can be expanded with:

REST API

Batch processing

Comparative dashboards
