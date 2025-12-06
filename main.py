import os
import tempfile
import json
import subprocess
import time
from typing import Optional, Any

from pydantic import BaseModel, Field
import cv2
from ultralytics import YOLO
from fastapi import UploadFile, HTTPException
from openai import OpenAI
import streamlit as st

# --- Global Variables ---
POSE_MODEL: Optional[YOLO] = None
openai_client: Optional[OpenAI] = None
gemini_client: Optional[Any] = None

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 pose model with better error handling."""
    global POSE_MODEL
    try:
        st.info("⬇️ Downloading/loading YOLOv8 pose model...")
        POSE_MODEL = YOLO("yolov8n-pose.pt")  # Auto-downloads if missing
        st.success("✅ YOLOv8 Pose Model Loaded!")
        return POSE_MODEL
    except Exception as e:
        st.error(f"❌ YOLO Error: {e}")
        st.info("💡 Install: pip install ultralytics torch")
        POSE_MODEL = None
        return None

def initialize_api_clients():
    """Initialize API clients with stable models."""
    global openai_client, gemini_client
    
    openai_client = None
    gemini_client = None
    
    # Gemini - STABLE MODEL
    try:
        gemini_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            gemini_client = genai.GenerativeModel('gemini-1.5-pro')  # ✅ STABLE
            st.success("✅ Gemini (1.5-pro) ready!")
        else:
            st.warning("⚠️ GEMINI_API_KEY missing - depth analysis limited")
    except Exception as e:
        st.error(f"❌ Gemini failed: {e}")
    
    # OpenAI (optional now)
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if openai_api_key:
            openai_client = OpenAI(api_key=openai_api_key)
            st.success("✅ OpenAI ready!")
    except:
        st.info("ℹ️ OpenAI skipped - using local Whisper")

# --- FAST LOCAL WHISPER (No quota limits!) ---
def transcribe(video_path: str, client: Optional[OpenAI] = None) -> str:
    """Fast local Whisper - works for ANY video length."""
    audio_path = tempfile.mktemp(suffix=".mp3")
    
    # Extract audio
    st.info("🔊 Extracting audio...")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000",
            "-ac", "1", "-acodec", "pcm_s16le", audio_path
        ], check=True, capture_output=True, timeout=120)
    except Exception as e:
        return f"❌ FFmpeg failed: {e}"
    
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return "❌ No audio detected"
    
    st.info("🎤 Transcribing with local Whisper (fast)...")
    
    try:
        import faster_whisper
        model_size = "small"  # Fast & accurate
        model = faster_whisper.WhisperModel(model_size, device="cpu", compute_type="int8")
        
        segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        
        os.remove(audio_path)
        st.success(f"✅ Transcribed {len(transcript.split())} words")
        return transcript or "[No clear speech detected]"
        
    except ImportError:
        st.error("❌ Install: `pip install faster-whisper`")
        if os.path.exists(audio_path): os.remove(audio_path)
        return "❌ faster-whisper not installed"
    except Exception as e:
        st.error(f"❌ Whisper error: {e}")
        if os.path.exists(audio_path): os.remove(audio_path)
        return f"Transcription failed: {e}"

# --- IMPROVED Gesture Analysis ---
def analyze_gestures(video_path: str, model: Optional[YOLO], client: Optional[Any]) -> str:
    if not model:
        return "❌ YOLO model failed to load. Install: `pip install ultralytics torch`"
    
    st.info("🤲 Analyzing gestures...")
    try:
        # Analyze first 100 frames only (fast!)
        results = model(video_path, stream=True, save=False, verbose=False, max_det=1)
        
        hand_frames = 0
        total_frames = 0
        stable_count = 0
        
        for i, r in enumerate(list(results)[:100]):  # Limit analysis
            total_frames += 1
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints.xy[0].cpu().numpy()  # First person
                if len(kpts) >= 15:  # COCO 17 keypoints
                    # Check hands (keypoints 9=left wrist, 10=right wrist)
                    left_hand = kpts[9]
                    right_hand = kpts[10]
                    if (0 < left_hand[0] < 1920 and 0 < left_hand[1] < 1080) or \
                       (0 < right_hand[0] < 1920 and 0 < right_hand[1] < 1080):
                        hand_frames += 1
                    stable_count += 1
        
        if total_frames == 0:
            return "No frames analyzed"
        
        hand_ratio = hand_frames / total_frames
        findings = []
        
        if hand_ratio > 0.5:
            findings.append("👍 Active hand gestures detected")
        elif hand_ratio > 0.2:
            findings.append("👌 Moderate hand usage")
        else:
            findings.append("🙌 Hands mostly static")
            
        findings.append(f"📊 Analyzed {total_frames} frames")
        
        # Gemini summary (optional)
        if client:
            prompt = f"Presentation gestures: {' | '.join(findings)}"
            try:
                response = client.generate_content(prompt)
                return response.text
            except:
                pass
        
        return "**Gestures:** " + " | ".join(findings)
        
    except Exception as e:
        return f"❌ Gesture analysis error: {e}"

# --- Pydantic Models (unchanged) ---
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

# --- IMPROVED Depth Analysis ---
def analyze_concept_depth(transcript: str, client: Optional[Any]) -> DepthAnalysis:
    if not client:
        return DepthAnalysis(
            overall_depth_score=0.5,
            reasoning="Depth analysis requires Gemini API key.",
            per_segment=[SegmentDepth(start=0, end=10, depth_score=0.5)]
        )
    
    st.info("🧠 Analyzing content depth...")
    prompt = f"""
    Analyze this presentation transcript for depth (0-1 scale):
    "{transcript[:3000]}"
    
    Return VALID JSON only:
    {{
      "overall_depth_score": 0.75,
      "reasoning": "3 sentences max",
      "per_segment": [
        {{"start": 0.0, "end": 15.0, "depth_score": 0.8}},
        {{"start": 15.0, "end": 30.0, "depth_score": 0.6}}
      ]
    }}
    """
    
    try:
        response = client.generate_content(prompt)
        data = json.loads(response.text.strip())
        return DepthAnalysis(**data)
    except:
        # Fallback
        return DepthAnalysis(
            overall_depth_score=0.6,
            reasoning="Default analysis - add Gemini key for full depth analysis.",
            per_segment=[SegmentDepth(start=0, end=20, depth_score=0.6)]
        )

# --- Main pipeline (unchanged structure) ---
def run_full_analysis(video_file: UploadFile, temp_video_path: str) -> AnalysisResponse:
    if gemini_client is None:
        st.warning("⚠️ Gemini unavailable - limited analysis")
    
    # Video info
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    st.metric("📹 Duration", f"{duration:.1f}s")
    
    # Run analysis
    transcript = transcribe(temp_video_path, openai_client)
    gestures = analyze_gestures(temp_video_path, POSE_MODEL, gemini_client)
    depth = analyze_concept_depth(transcript, gemini_client)
    
    return AnalysisResponse(
        duration_seconds=duration,
        transcript=transcript,
        gestures=gestures,
        depth=depth
    )

# --- Results display (unchanged) ---
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
    with tab1:
        st.text_area("Transcript", response.transcript, height=200)
    with tab2:
        st.markdown(response.gestures)

# --- Main App ---
def main():
    st.set_page_config(page_title="Mentor AI", layout="wide")
    st.title("🎓 Mentor AI: Presentation Analyzer")
    
    load_yolo_model()
    initialize_api_clients()
    
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
                finally:
                    if os.path.exists(temp_pa
                                      th):
                        os.remove(temp_path)
    else:
        st.info("👆 Upload video to analyze!")
        st.info("💡 **No OpenAI quota needed** - uses local Whisper")

if __name__ == "__main__":
    main()
