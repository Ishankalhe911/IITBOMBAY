# System Design

This document outlines the end-to-end workflow, component responsibilities, and data flow of the Mentor-Scoring-AI system. The design emphasizes modularity, reproducibility, and CPU-friendly multimodal processing.

---

## 1. High-Level Workflow

```
                ┌────────────────────────┐
                │     Video Input        │
                │ (Upload or URL Fetch)  │
                └─────────────┬──────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Audio Extraction (FFmpeg) │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Speech-to-Text (Whisper) │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼────────────┐
                │ Transcript Processing     │
                │ (Clarity + Segmentation)  │
                └─────────────┬────────────┘
                              │
     ┌────────────────────────▼────────────────────────┐
     │              Parallel Multimodal Processing      │
     │                                                 │
     │   ┌──────────────────────────┐   ┌───────────┐ │
     │   │ Visual/Gesture Analysis  │   │ Depth AI   │ │
     │   │ (YOLO Pose Sampling)     │   │ LLM/Heur.  │ │
     │   └──────────────────────────┘   └───────────┘ │
     └────────────────────────┬───────────────────────┘
                              │
                ┌─────────────▼────────────┐
                │     Metric Aggregation    │
                │  (Weighted Scoring Model) │
                └─────────────┬────────────┘
                              │
                ┌─────────────▼────────────┐
                │        Streamlit UI       │
                │  Results + Visualization  │
                └──────────────────────────┘
```

---

## 2. Components

### **2.1 Streamlit UI (`src/app.py`)**

* Manages user input: video upload or URL
* Displays progress indicators and evaluation results
* Orchestrates calls to the audio, gesture, and depth modules
* Serves as the unified interface for the prototype

---

### **2.2 Audio Extraction (`utils/ffmpeg_utils.py`)**

* Converts MP4 → WAV using FFmpeg
* Ensures consistent sampling rate (16kHz)
* Handles temporary file creation and cleanup

---

### **2.3 Transcription (`ai/transcribe.py`)**

* Uses `faster-whisper` for CPU-optimized STT
* Falls back to standard Whisper if necessary
* Produces:

  * transcript text
  * word counts
  * clarity indicators

---

### **2.4 Gesture Analysis (`ai/gesture_analysis.py`)**

* Performs pose detection using YOLOv11n/YOLOv8n
* Samples ~25 frames for fast, representative analysis
* Extracts engagement-related signals:

  * hand movement
  * face visibility
  * gaze orientation
  * pose confidence

---

### **2.5 Depth Analysis (`ai/depth_analysis.py`)**

* Two-mode depth evaluation:

  * **LLM-based analysis** (Ollama Llama)
  * **Deterministic heuristics** when LLM is unavailable
* Outputs structured depth JSON:

  * `overall_depth_score`
  * `reasoning`
  * `per_segment[]`

---

### **2.6 Scoring Model**

Weighted scoring in alignment with the hackathon’s problem statement:

| Metric              | Weight |
| ------------------- | ------ |
| Engagement          | 20%    |
| Communication       | 20%    |
| Technical Depth     | 30%    |
| Clarity             | 20%    |
| Interaction Quality | 10%    |

Each module contributes to one or more metrics.

---

## 3. Data Flow

### **Input Formats**

* `.mp4`, `.mov`, `.avi`
* Public YouTube/video URLs

### **Intermediate Outputs**

* WAV audio
* Transcript text
* Pose metadata
* Depth analysis JSON

### **Final Output**

A consolidated evaluation report containing:

* Transcript + clarity analysis
* Engagement metrics
* Technical depth assessment
* Segment breakdown
* Composite mentor score

---

## 4. Design Considerations

* **CPU-first optimization**
  All models selected for lightweight, real-time CPU execution.

* **Graceful fallback mechanism**
  If Ollama is not available, deterministic heuristics ensure consistent results.

* **Modular separation**
  Each AI module is isolated for easy replacement and independent testing.

* **Caching & performance**
  YOLO and Whisper models are loaded once and reused across evaluations.

* **Reproducibility**
  Ensures deterministic scoring when no LLM is used.

---

## 5. Deployment Notes

The system is designed for:

* Local offline use
* Institutional computer labs
* Lightweight cloud deployments (Streamlit/Gradio)

Potential extensions:

* Batch processing service
* REST API for institutions
* Multi-mentor comparison dashboards

---

