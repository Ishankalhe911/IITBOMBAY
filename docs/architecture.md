# Architecture

The system follows a modular, pipeline-based design where the Streamlit frontend orchestrates independent AI components for transcription, gesture analysis, and depth scoring.

```
Streamlit UI  (src/app.py)
│
├── utils/ffmpeg_utils.py
│     └─ Audio extraction (FFmpeg → WAV)
│
├── ai/transcribe.py
│     └─ Speech-to-text module (faster-whisper / Whisper fallback)
│
├── ai/gesture_analysis.py
│     └─ Visual engagement analysis (YOLO pose detection)
│
└── ai/depth_analysis.py
      └─ Concept depth scoring (heuristic + optional Ollama LLM)
```

## Architectural Notes

* **On-demand Model Loading**
  Both Whisper and YOLO models are loaded lazily and cached to reduce startup time and improve repeat analysis performance.

* **Local LLM Support (Optional)**
  If an Ollama instance is detected, the system automatically routes depth analysis to a local Llama model; otherwise, it falls back to deterministic heuristics.

* **Modular AI Components**
  Each module under `ai/` exposes a single responsibility function, enabling easier testing, replacement, and parallel development.

* **Separation of Concerns**

  * `app.py` handles orchestration and UI.
  * `utils/` provides system-level helpers (FFmpeg, file operations).
  * `ai/` contains self-contained processing units for multimodal evaluation.

* **Lightweight CPU-Friendly Design**
  All components run efficiently on CPU, allowing deployment on low-resource systems or offline environments.

---
