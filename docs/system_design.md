# System Design

## Overview
The system ingests a mentor's recorded video and outputs a structured analysis including:
- transcript (text)
- engagement/gesture summary
- concept depth score and per-segment analysis

## Pipeline steps
1. **Upload video (Streamlit UI)** - user uploads mp4/mov/avi
2. **Audio extraction** - ffmpeg extracts a mono 16kHz WAV
3. **Transcription** - faster-whisper converts audio → transcript
4. **Gesture analysis** - YOLO pose model samples frames and computes gesture/eye-contact metrics
5. **Depth analysis** - local heuristic or Ollama model to estimate concept depth (0-1)
6. **Report generation** - Streamlit displays metrics, transcript, and suggestions

## Data flow
Video file -> ffmpeg -> WAV -> whisper -> transcript
Video file -> frame sampling -> YOLO pose -> gestures summary
Transcript -> depth analysis -> depth score
All outputs -> final response object
