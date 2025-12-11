# 📘 Mentor-Scoring-AI

### *AI-Driven Evaluation of Teaching Quality from Recorded Video Sessions*

**Submission for:** *UpSkill India Challenge — Techfest IIT Bombay x HCL GUVI*

---

## 🚀 Overview

Evaluating teaching quality across large education ecosystems is slow, subjective, and inconsistent.
**Mentor-Scoring-AI** solves this using a **multimodal, fully automated evaluation system** that scores teaching quality directly from recorded lecture videos.

Our AI analyzes:

* Communication clarity
* Engagement & gestures
* Technical depth
* Confidence & pacing
* Interaction quality

Using **video pose detection, audio transcription, and transcript content intelligence** — all running **locally**, CPU-friendly, no external LLM or cloud dependency.

---

## 🧠 Key Features

### 🔊 **Audio Intelligence (Whisper Speech-to-Text)**

* Local Whisper base model
* Accurate transcription on CPU
* Clean audio extraction using FFmpeg
* Sentence clarity & vocabulary richness scoring

### 👁️ **Visual + Gesture Intelligence (YOLO Pose)**

* YOLOv11n-Pose → extremely fast pose tracking
* Hand movement → engagement score
* Face visibility & eye-contact cues → confidence score
* Frame sampling optimized for speed

### 🧮 **Local Technical Depth Estimation (No LLM Required)**

Your updated code uses **pure local heuristic-based depth scoring**, including:

* Vocabulary diversity
* Rare/long word usage
* Technical keyword detection
* Sentence structure
* Automatic segment-based depth scoring

### 🎛️ **Streamlit Evaluation Dashboard**

* Upload video or use URL (YouTube supported)
* Real-time transcript + gesture analysis
* Segment-wise depth scoring
* Final weighted “Mentor Score”
* Downloadable evaluation report

---

## 🎯 Why This Matters

Institutions face major challenges:

* Manual evaluation is slow
* Scoring varies between reviewers
* No standardized metrics
* No large-scale automation

**Mentor-Scoring-AI delivers:**

✔ Consistent, objective scoring
✔ Fully automated workflow
✔ Scalable to thousands of videos
✔ CPU-only & offline-friendly
✔ Multimodal evaluation similar to human observation

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
        │  Audio Extract  │ │   Frame Sampler  │ │ Transcript Engine │
        │   (FFmpeg)      │ │   (OpenCV)       │ │    (Whisper)      │
        └─────────┬──────┘ └────────┬────────┘ └────────┬─────────┘
                  │                 │                   │
        ┌─────────▼──────┐ ┌────────▼────────┐ ┌────────▼─────────┐
        │ Speech-to-Text  │ │ YOLO Pose Model  │ │ Local Depth Logic │
        │    Whisper      │ │ Gesture/Face     │ │   (Heuristic)     │
        └─────────┬──────┘ │   Analysis        │ └────────┬─────────┘
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

* Whisper (local CPU transcription)
* YOLOv11n-Pose (gesture and face tracking)
* Local heuristic depth-engine (no LLM)

### **Core Libraries**

* OpenCV
* MoviePy
* YOLO (Ultralytics)
* FFmpeg
* Streamlit (UI)
* FastAPI-style processing structure

---

## 🧪 Evaluation Metrics

Your scoring system exactly matches the hackathon guidelines:

| Skill Metric        | Weight | Source                    |
| ------------------- | ------ | ------------------------- |
| Engagement          | 20%    | YOLO Pose (hands, motion) |
| Communication       | 20%    | Whisper transcript        |
| Technical Depth     | 30%    | Local depth heuristics    |
| Clarity             | 20%    | Transcript complexity     |
| Interaction Quality | 10%    | Eye-contact + pacing      |

---

## 🛠️ Quick Start (Development)

### 1️⃣ Setup Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Ensure FFmpeg is Installed

Update the FFmpeg path inside:

```
src/utils/ffmpeg_utils.py
```

### 3️⃣ Run Streamlit App

```bash
streamlit run src/app.py
```

### 4️⃣ Upload or paste URL → Get instant evaluation

---

## 🎥 Demo Output (Judges Will See)

### ✔ Transcript Summary

* Total word count
* Key topics
* Sentence clarity
* Vocabulary richness

### ✔ Gesture & Engagement Analysis

* Hand movement score
* Eye contact and face visibility
* Confidence cues

### ✔ Technical Depth

* Automatic heuristic-based depth
* Segment-wise scoring
* Explanation + reasoning

### ✔ Final Mentor Score

Weighted blended score with interpretation.

---

## 📈 Innovation & Differentiators

🔥 5× faster processing using YOLOv11n-Pose
🔥 Fully local — **no API costs, no LLMs, no internet needed**
🔥 Local depth analysis engine (unique approach)
🔥 Optimized for low-end hardware
🔥 URL + file support
🔥 Modular AI pipeline

---

## 🧭 Roadmap

* Database storage for mentor scores
* Historical insights & comparison charts
* API endpoints for institution portals
* Real-time evaluation mode (webcam)
* Improvement recommendations using analytics

---

## 🔧 Current Implementation Status

### ✔ Streamlit dashboard connected

### ✔ Whisper transcription working

### ✔ YOLO gesture tracking functional

### ✔ Local depth analysis implemented

### ✔ URL → video → transcript pipeline complete

### ⏳ Remaining:

1. **Database integration**
2. **Linking dashboard to main website**
3. **Final UI polishing**

---

## 👥 Team

**Abhishek Boyane** – AI/ML
**Ishan Kalhe** – Full-Stack, Backend
**Yash Bhosale** – Vision Processing
**Chetan Patel** – UI/UX

---

## 📩 Contact

**Email:** *[ishankalhe1@gmail.com](mailto:ishankalhe1@gmail.com)*

