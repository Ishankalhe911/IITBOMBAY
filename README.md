# 📘 Mentor-Scoring-AI

### *AI-Driven Evaluation of Teaching Quality from Recorded Video Sessions*

**Submission for:** *UpSkill India Challenge — Techfest IIT Bombay x HCL GUVI*

---

## 🚀 Overview

Evaluating teaching quality across large education ecosystems is slow, subjective, and inconsistent.
**Mentor-Scoring-AI** is an **AI-powered multimodal assessment system** that analyzes video lectures and automatically scores a mentor on:

* Communication clarity
* Engagement & gestures
* Technical depth
* Confidence & pacing
* Interaction quality

All using **audio, visual pose analysis, and transcript intelligence**.

Built as a **lightweight, on-device, offline-capable prototype**, our solution focuses on **scalability, objectivity, and automation**—precisely addressing the Problem Statement #2 of the hackathon.

---

## 🧠 Key Features

### 🔊 **Audio Intelligence (Whisper + faster-whisper)**

* Clean audio extraction (FFmpeg)
* Fast speech-to-text transcription
* Sentence & vocabulary-based clarity scoring

### 👁️ **Visual + Gesture Intelligence (YOLOv11 Pose)**

* Hand movement analysis → engagement score
* Face visibility & eye-contact tracking → confidence score
* Lightweight YOLOv11n → **5× faster** processing

### 📄 **Concept Depth & Explanation Analysis**

* Ollama Llama3 local inference (if available)
* Smart fallback heuristic scoring
* Segment-wise depth metrics

### 🎛️ **Unified Streamlit Dashboard**

* Upload or URL-based analysis
* Instant breakdown of all metrics
* Tab-wise transcript & gesture insights
* 1-click evaluation report

---

## 🎯 Why This Matters

Institutions often deal with:

* Highly variable mentor performance
* Lack of standardized evaluation
* Manual review overhead
* Difficulty scaling quality checks

**Our system solves this by offering:**
✔ Consistent, unbiased scoring
✔ Automated evaluation — scalable to 1,000+ videos
✔ Actionable insights for teacher improvement
✔ Offline/on-device capability → low-cost deployment
✔ Multimodal analysis like real human evaluators

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────┐
                    │       Video Input         │
                    │ (Upload or URL Download)  │
                    └───────────────┬──────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
        ┌─────────▼──────┐ ┌────────▼────────┐ ┌────────▼─────────┐
        │  Audio Extract  │ │   Video Frames   │ │ Transcript Engine │
        │   (FFmpeg)      │ │  Sampling (cv2)  │ │ (Whisper/Faster)  │
        └─────────┬──────┘ └────────┬────────┘ └────────┬─────────┘
                  │                 │                   │
        ┌─────────▼──────┐ ┌────────▼────────┐ ┌────────▼─────────┐
        │ Speech-to-Text  │ │   YOLO Pose      │ │ Concept Depth AI │
        │ Whisper Model   │ │ Hand/Eye/Face    │ │ (Ollama / local) │
        └─────────┬──────┘ │   Detection       │ └────────┬─────────┘
                  │         └────────┬──────────┘         │
                  │                  │                    │
                  └──────────┬───────┴─────────┬──────────┘
                             ▼                 ▼
                     ┌────────────────────────────────┐
                     │     Scoring & Aggregation      │
                     │ (Engagement, Clarity, Depth)   │
                     └────────────────┬───────────────┘
                                     ▼
                           ┌────────────────────┐
                           │  Streamlit Report   │
                           └────────────────────┘
```

---

## 📂 Repository Structure

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

---

## ⚙️ Tech Stack

### **AI Models**

* Whisper + faster-whisper (speech-to-text)
* YOLOv11n-Pose (gesture + visual cues)
* Llama3 (Ollama) or heuristic fallback (depth scoring)

### **Core Libraries**

* OpenCV
* MoviePy
* YOLO (Ultralytics)
* FFmpeg
* FastAPI (for model structure)
* Streamlit (demo UI)

---

## 🧪 Evaluation Metrics

The judge-provided metric distribution is *natively integrated* into our scoring:

| Skill Metric        | Weight | Data Source           |
| ------------------- | ------ | --------------------- |
| Engagement          | 20%    | YOLO Pose (gestures)  |
| Communication       | 20%    | Whisper transcript    |
| Technical Depth     | 30%    | Depth Analysis (LLM)  |
| Clarity             | 20%    | Transcript complexity |
| Interaction Quality | 10%    | Eye contact + pacing  |

---

## 🛠️ Quick Start (Development)

### 1️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 2️⃣ FFmpeg configuration

Update path in:
`src/utils/ffmpeg_utils.py`
OR ensure ffmpeg is in PATH.

### 3️⃣ Run Streamlit demo

```bash
streamlit run src/app.py
```

### 4️⃣ Upload video → Get full AI report

Accepted formats: `.mp4`, `.mov`, `.avi`.

---

## 🎥 Demo Output (What Judges Will See)

### ✔ Transcript Summary

* Word count
* Sentence clarity
* Key concepts detected
* Complexity measure

### ✔ Gesture & Engagement Analysis

* Hand movement intensity
* Eye contact %, face visibility
* Confidence cues
* 25-frame sampled evaluation

### ✔ Depth Score & Reasoning

* JSON-based segment evaluation
* LLM reasoning text
* Overall depth score (0–1)

### ✔ Final “Mentor Score”

Weighted composite score aligned with hackathon criteria.

---

## 📈 Innovation & Differentiators

🔥 **5× faster** multimodal processing (YOLO11n + optimized sampling)
🔥 Local + cloud-free analysis (Ollama fallback)
🔥 Multi-segment depth scoring
🔥 Built with low compute footprint (runs on CPU)
🔥 URL-based YouTube lecture evaluation
🔥 Production-ready modular architecture

---

## 🧭 Roadmap (Post-Hackathon)

* Add **bias-free scoring calibration**
* Introduce **Live Mentor Evaluation** (real-time camera)
* Mentor benchmarking dashboard
* Session comparison & trend analytics
* Institution-wide scoring API

---



## Things Left in Implementation (Current Status)

The following components are planned and partially implemented but not fully integrated:

1. **Database and Dashboard Integration**

   * Dashboard UI structure is built
   * Database linking/connection logic is pending

2. **Database Connection With Main Website**

   * Backend–DB binding still needs to be implemented
   * Intended for storing mentor scores, video metadata, and analytics

These features will complete the system’s ability to store evaluations, visualize historical insights, and integrate end-to-end with a centralized platform.




## 👥 Team

**Abhishek Boyane**
**Ishan Kalhe**
**Yash Bhosale**
**Chetan Patel**
Roles include:

* AI/ML(**Abhishek Boyane**)
* Backend
* Vision Processing(**Yash Bhosale**)
* Full-stack(**Ishan Kalhe**)
* UI/UX(**Chetan Patel**)

---

## 📩 Contact

**Email:ishankalhe1@gmail.com** 


