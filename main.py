import os
import tempfile
import json
import subprocess
import time
import requests
from typing import Optional, Any
from pydantic import BaseModel, Field
import cv2
from ultralytics import YOLO
from fastapi import UploadFile
import streamlit as st
import re

# --- Pydantic Models (TOP) ---
class SegmentDepth(BaseModel):
    start: float
    end: float
    depth_score: float = Field(..., ge=0.0, le=1.0)

class DepthAnalysis(BaseModel):
    overall_depth_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    per_segment: list[SegmentDepth]

class AnalysisResponse(BaseModel):
    duration_seconds: float
    transcript: str
    gestures: str
    depth: DepthAnalysis

# --- Global Clients ---
openai_client: Optional[Any] = None
gemini_client: Optional[Any] = None
ollama_available: bool = False

@st.cache_resource
def get_pose_model():
    """Cached YOLO model loader."""
    try:
        st.info("⬇️ Loading YOLOv8 pose model...")
        model = YOLO("yolov8n-pose.pt")
        st.success("✅ YOLOv8 Pose Model Loaded!")
        return model
    except Exception as e:
        import traceback
        st.error("❌ YOLO failed to load")
        st.code(traceback.format_exc())
        return None

def check_ollama():
    """Check if Ollama is running."""
    global ollama_available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_available = True
            return True
    except:
        pass
    ollama_available = False
    return False

def initialize_clients():
    """Initialize with Ollama priority."""
    global openai_client, gemini_client
    
    gemini_client = None
    openai_client = None
    
    # ✅ PRIORITY 1: Ollama (FREE unlimited)
    if check_ollama():
        st.success("✅ Ollama detected - FREE unlimited AI!")
    else:
        st.info("ℹ️ Ollama not running - using smart local analysis")
    
    # OpenAI fallback (optional)
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if openai_api_key:
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_api_key)
            st.success("✅ OpenAI ready")
    except:
        st.info("ℹ️ OpenAI skipped")

# --- FIXED DEPTH ANALYSIS ---
def analyze_concept_depth(transcript: str) -> DepthAnalysis:
    """Ollama → Smart Local fallback."""
    
    # PRIORITY 1: Ollama (FREE, UNLIMITED)
    if ollama_available:
        st.info("🧠 Analyzing with Ollama (FREE)...")
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                                   json={
                                       "model": "llama3.2:1b",
                                       "prompt": f'Analyze presentation transcript depth (0-1 scale). Transcript: "{transcript[:1000]}"\nReturn ONLY valid JSON:\n{{"overall_depth_score":0.75,"reasoning":"Brief analysis","per_segment":[{"start":0.0,"end":15.0,"depth_score":0.8},{"start":15.0,"end":30.0,"depth_score":0.7}]}}',
                                       "stream": False
                                   }, timeout=30)
            
            if response.status_code == 200:
                ollama_response = response.json()["response"].strip()
                # Clean JSON response
                json_start = ollama_response.find('{')
                json_end = ollama_response.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    clean_json = ollama_response[json_start:json_end]
                    data = json.loads(clean_json)
                    st.success("✅ Ollama depth analysis complete!")
                    return DepthAnalysis(**data)
        except Exception as e:
            st.warning(f"⚠️ Ollama failed: {str(e)[:50]} - using smart local")
    
    # PRIORITY 2: SMART LOCAL ANALYSIS (FIXED)
    st.info("🧠 Smart local depth analysis...")
    words = [w.strip() for w in transcript.lower().split() if w.strip()]
    word_count = len(words)
    
    if word_count == 0:
        return DepthAnalysis(
            overall_depth_score=0.0,
            reasoning="No transcript available",
            per_segment=[SegmentDepth(start=0, end=10, depth_score=0.0)]
        )
    
    unique_words = len(set(words))
    long_words = sum(1 for w in words if len(w) > 6)
    sentences = max(1, len(re.split(r'[.!?]+', transcript)))
    
    # ✅ FIXED TECH TERMS LIST (shortened + proper syntax)
    tech_terms_list = [
        'data', 'model', 'system', 'process', 'method', 'analysis', 'algorithm', 
        'solution', 'framework', 'architecture', 'neural', 'training', 'dataset',
        'accuracy', 'precision', 'recall', 'feature', 'vector', 'matrix', 
        'gradient', 'optimizer', 'loss', 'layer', 'neuron', 'network'
    ]
    tech_terms = sum(1 for w in words if any(term in w for term in tech_terms_list))
    
    complexity = unique_words / word_count
    depth_score = min(0.95, 0.2 + complexity * 0.3 + (long_words/word_count)*0.2 + (tech_terms/word_count)*0.3)
    
    reasoning = f"{word_count} words, {sentences} sentences, {unique_words} unique terms. Complexity: {complexity:.1%}, Technical: {tech_terms/word_count:.0%}."
    
    segment_duration = max(15, word_count // 10)
    segments = [
        SegmentDepth(start=0, end=min(segment_duration, 60), depth_score=round(depth_score, 2)),
        SegmentDepth(start=segment_duration, end=min(segment_duration*2, 120), depth_score=round(max(0, depth_score-0.1), 2)),
    ]
    
    return DepthAnalysis(
        overall_depth_score=round(depth_score, 2),
        reasoning=reasoning,
        per_segment=segments
    )

# --- FAST LOCAL WHISPER ---
def transcribe(video_path: str) -> str:
    audio_path = tempfile.mktemp(suffix=".wav")
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # UPDATE FOR YOUR PATH!
    
    st.info("🔊 Extracting audio...")
    try:
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-ar", "16000",
            "-ac", "1", "-acodec", "pcm_s16le", audio_path
        ], check=True, capture_output=True, timeout=120)
    except Exception as e:
        if os.path.exists(audio_path): os.remove(audio_path)
        return f"❌ FFmpeg failed: {e}"
    
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        if os.path.exists(audio_path): os.remove(audio_path)
        return "❌ No audio detected"
    
    st.info("🎤 Transcribing...")
    try:
        import faster_whisper
        model = faster_whisper.WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        os.remove(audio_path)
        st.success(f"✅ Transcribed {len(transcript.split())} words")
        return transcript or "[No speech detected]"
    except ImportError:
        st.error("❌ Install: `pip install faster-whisper`")
        return "❌ faster-whisper missing"
    except Exception as e:
        if os.path.exists(audio_path): os.remove(audio_path)
        return f"❌ Transcription failed: {e}"

# --- Gesture Analysis ---
def analyze_gestures(video_path: str, model: Optional[YOLO]) -> str:
    if model is None:
        return "❌ YOLO model unavailable"
    
    st.info("🤲 Analyzing gestures...")
    try:
        results = model(video_path, stream=True, save=False, verbose=False, max_det=1)
        hand_frames = total_frames = 0
        
        for r in list(results)[:100]:
            total_frames += 1
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints.xy[0].cpu().numpy()
                if len(kpts) >= 15:
                    left_hand = kpts[9]
                    right_hand = kpts[10]
                    if (0 < left_hand[0] < 1920 and 0 < left_hand[1] < 1080) or \
                       (0 < right_hand[0] < 1920 and 0 < right_hand[1] < 1080):
                        hand_frames += 1
        
        hand_ratio = hand_frames / total_frames if total_frames else 0
        findings = []
        
        if hand_ratio > 0.5: findings.append("👍 Active hand gestures")
        elif hand_ratio > 0.2: findings.append("👌 Moderate hand usage")
        else: findings.append("🙌 Hands mostly static")
        findings.append(f"📊 {total_frames} frames analyzed")
        
        return "**Gestures:** " + " | ".join(findings)
    except Exception as e:
        return f"❌ Gesture error: {str(e)[:50]}"

# --- Main Pipeline ---
def run_full_analysis(video_file: UploadFile, temp_video_path: str) -> AnalysisResponse:
    pose_model = get_pose_model()
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0
    cap.release()
    
    st.metric("📹 Duration", f"{duration:.1f}s")
    
    transcript = transcribe(temp_video_path)
    gestures = analyze_gestures(temp_video_path, pose_model)
    depth = analyze_concept_depth(transcript)
    
    return AnalysisResponse(duration_seconds=duration, transcript=transcript, gestures=gestures, depth=depth)

# --- Display Results ---
def display_analysis_results(response: AnalysisResponse):
    st.success("✅ Analysis Complete!")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("⏱️ Duration", f"{response.duration_seconds:.1f}s")
        st.metric("🎯 Depth Score", f"{response.depth.overall_depth_score:.2f}")
    with col2:
        st.info(f"**Reasoning:** {response.depth.reasoning}")
    
    st.subheader("📊 Segment Analysis")
    segments = [{"Start": f"{s.start:.0f}s", "End": f"{s.end:.0f}s", "Depth": f"{s.depth_score:.1f}"} 
                for s in response.depth.per_segment[:5]]
    st.table(segments)
    
    tab1, tab2 = st.tabs(["📝 Transcript", "🤲 Gestures"])
    with tab1: st.text_area("Transcript", response.transcript, height=200)
    with tab2: st.markdown(response.gestures)

# --- Main App ---
def main():
    st.set_page_config(page_title="Mentor AI", layout="wide")
    st.title("🎓 Mentor AI: Presentation Analyzer")
    
    initialize_clients()
    pose_model = get_pose_model()
    
    st.info("🚀 **Ollama + Local** - FREE unlimited analysis!")
    
    uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        st.video(uploaded_file)
        st.info(f"**{uploaded_file.name}** ({uploaded_file.size/1024/1024:.1f} MB)")
        
        if st.button("🚀 Analyze Video", type="primary"):
            with st.spinner("🔬 Processing..."):
                temp_path = tempfile.mktemp(suffix=f".{uploaded_file.name.split('.')[-1]}")
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    result = run_full_analysis(uploaded_file, temp_path)
                    display_analysis_results(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)
    else:
        st.info("👆 Upload video to start!")
        st.markdown("""
        ### ✅ **Works with:**
        - 🎤 Local Whisper (faster-whisper)
        - 🤲 YOLOv8 gestures
        - 🧠 Ollama AI (FREE) **OR** Smart local analysis
        
        ### ⚙️ **Quick Setup:**
        ```
        pip install streamlit ultralytics faster-whisper opencv-python pydantic requests
        ```
        **FFmpeg:** C:\\ffmpeg\\bin\\ffmpeg.exe
        **Ollama:** Download OllamaSetup.exe from ollama.com
        """)

if __name__ == "__main__":
    main()

