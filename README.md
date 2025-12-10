# Mentor-Scoring-AI

**Project goal:** Evaluate mentor/instructor teaching quality and effectiveness from recorded video sessions.

This repository contains a Streamlit-based demo and modular AI components:
- Audio extraction + transcription (faster-whisper)
- Gesture & engagement analysis (YOLO pose)
- Concept depth analysis (local heuristics or Ollama fallback)
- Simple Streamlit UI to upload and analyze videos

## Repo structure (precise files included)
```
Mentor-Scoring-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── system_design.md
│   ├── architecture.md
│   └── evaluation_notes.md
├── src/
│   ├── app.py
│   ├── ai/
│   │   ├── transcribe.py
│   │   ├── gesture_analysis.py
│   │   └── depth_analysis.py
│   └── utils/
│       └── ffmpeg_utils.py
└── models/
    └── model_download_links.txt
```

## Quick start (development)
1. Create virtual env:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Update `FFMPEG_PATH` in `src/utils/ffmpeg_utils.py` (or ensure ffmpeg is on PATH).
3. Run demo:
   ```bash
   streamlit run src/app.py
   ```
4. Upload a teacher video (mp4/mov/avi) and click **Analyze**.

## Notes
- Large model files are not included (YOLO weights, Whisper weights). See `models/model_download_links.txt`.
- The code is modular and annotated for easy extension and productionization.

## Contact
Repository created for submission; author: Abhishek (from conversation).

