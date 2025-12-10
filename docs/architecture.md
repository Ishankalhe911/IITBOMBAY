# Architecture (text diagram)

Streamlit UI (src/app.py)
    ├─> utils/ffmpeg_utils.py  -- extracts WAV
    ├─> ai/transcribe.py       -- faster-whisper wrapper
    ├─> ai/gesture_analysis.py -- YOLO pose sampling analysis
    └─> ai/depth_analysis.py   -- transcript depth scoring (local + Ollama)

Notes:
- Models (YOLO, Whisper) are loaded on demand and cached where possible.
- Ollama is used as an optional local LLM fallback if available.
- Design favors modular testing: each ai/ module exposes a single function.
