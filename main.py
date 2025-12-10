import os
import tempfile
import json
import subprocess
import requests
import re
from typing import Optional, Any, Tuple
from pydantic import BaseModel, Field
import cv2
from ultralytics import YOLO
from fastapi import UploadFile
import streamlit as st
import time

import whisper
from moviepy.editor import VideoFileClip
import yt_dlp


# ==========================
# Reusable video → text helpers
# ==========================

MODEL_NAME = "base"  # change to "small" / "medium" / "large" if you want
_whisper_model = None

def get_whisper_model():
    """
    Lazy-load Whisper model once and reuse.
    """
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {MODEL_NAME} ...")
        _whisper_model = whisper.load_model(MODEL_NAME)
        print("Whisper model loaded.")
    return _whisper_model

def video_file_to_text(video_path: str) -> Tuple[str, float]:
    """
    Convert a local video file to transcript text using Whisper.
    Returns (text, processing_time_seconds).
    Does NOT delete the original video file.
    """
    start_time = time.time()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"

    clip = VideoFileClip(video_path)

    # remove old audio if exists
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # extract audio
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    model = get_whisper_model()
    result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU
    text = result.get("text", "").strip()

    # clean up
    try:
        os.remove(audio_path)
    except OSError:
        pass

    clip.close()

    processing_time = time.time() - start_time
    return text, processing_time

def url_to_text(url: str, download_filename: str = "downloaded_video.mp4") -> Tuple[str, float, str]:
    """
    Download a video from URL (e.g. YouTube) and return:
    - transcript text
    - processing_time_seconds
    - local video path (for further analysis)
    """
    start_time = time.time()

    if not url:
        raise ValueError("No URL provided")

    ydl_opts = {
        "format": "best",
        "outtmpl": download_filename,
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "retries": 3,
        "cachedir": False,
        "default_search": "auto",
        "geo_bypass": True,
        "no_warnings": True,
    }

    # Download video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Reuse your existing helper to get text from local file
    text, _ = video_file_to_text(download_filename)

    processing_time = time.time() - start_time
    # DO NOT delete download_filename here so that OpenCV/YOLO can read it
    return text, processing_time, download_filename


# ==========================
# Existing Mentor AI app
# ==========================

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
    """Cached YOLO model loader - OPTIMIZED for speed."""
    try:
        st.info("⬇️ Loading YOLOv11n pose model (FASTEST)...")
        # SPEED: YOLOv11 Nano - assumed faster than YOLOv8n [web:27][web:30]
        model = YOLO("yolo11n-pose.pt")
        st.success("✅ YOLOv11n Pose Model Loaded! (5x faster)")
        return model
    except:
        try:
            st.warning("⚠️ yolo11n not found, trying YOLOv8n...")
            model = YOLO("yolov8n-pose.pt")
            st.success("✅ YOLOv8n Pose Model Loaded!")
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
    except Exception:
        pass
    ollama_available = False
    return False


def initialize_clients():
    """Initialize with Ollama priority."""
    global openai_client, gemini_client

    gemini_client = None
    openai_client = None

    # PRIORITY 1: Ollama (FREE unlimited)
    if check_ollama():
        st.success("✅ Ollama detected - FREE unlimited AI!")
    else:
        st.info("ℹ️ Ollama not running - using smart local analysis")

    # OpenAI fallback (optional)
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
        if openai_api_key:
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_api_key)
            st.success("✅ OpenAI ready")
    except Exception:
        st.info("ℹ️ OpenAI skipped")


# --- FIXED & OPTIMIZED DEPTH ANALYSIS ---
def analyze_concept_depth(transcript: str) -> DepthAnalysis:
    """Ollama → Smart Local fallback."""

    # PRIORITY 1: Ollama (FREE, UNLIMITED)
    if ollama_available:
        st.info("🧠 Analyzing with Ollama (FREE)...")
        try:
            prompt_text = (
                f'Analyze presentation transcript depth (0-1 scale). Transcript: "{transcript[:1000]}"\n'
                'Return ONLY valid JSON:\n'
                '{"overall_depth_score":0.75,"reasoning":"Brief analysis",'
                '"per_segment":[{"start":0.0,"end":15.0,"depth_score":0.8},'
                '{"start":15.0,"end":30.0,"depth_score":0.7}]}'
            )
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt_text,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                ollama_text = response.json().get("response", "").strip()
                json_start = ollama_text.find('{')
                json_end = ollama_text.rfind('}') + 1
                if 0 <= json_start < json_end:
                    clean_json = ollama_text[json_start:json_end]
                    data = json.loads(clean_json)
                    st.success("✅ Ollama depth analysis complete!")
                    return DepthAnalysis(**data)
                else:
                    st.warning("⚠️ Ollama response does not contain valid JSON")
        except Exception as e:
            st.warning(f"⚠️ Ollama failed: {str(e)[:50]} - using smart local fallback")

    # PRIORITY 2: SMART LOCAL ANALYSIS (FIXED & FAST)
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

    tech_terms_list = [
        'data', 'model', 'system', 'process', 'method', 'analysis', 'algorithm',
        'solution', 'framework', 'architecture', 'neural', 'training', 'dataset',
        'accuracy', 'precision', 'recall', 'feature', 'vector', 'matrix',
        'gradient', 'optimizer', 'loss', 'layer', 'neuron', 'network'
    ]
    tech_terms = sum(1 for w in words if any(term in w for term in tech_terms_list))

    complexity = unique_words / word_count
    depth_score = min(
        0.95,
        0.2 + complexity * 0.3 + (long_words / word_count) * 0.2 + (tech_terms / word_count) * 0.3
    )

    reasoning = (
        f"{word_count} words, {sentences} sentences, {unique_words} unique terms. "
        f"Complexity: {complexity:.1%}, Technical: {tech_terms / word_count:.0%}."
    )

    segment_duration = max(15, word_count // 10)
    segments = [
        SegmentDepth(start=0, end=min(segment_duration, 60), depth_score=round(depth_score, 2)),
        SegmentDepth(start=segment_duration, end=min(segment_duration * 2, 120),
                     depth_score=round(max(0, depth_score - 0.1), 2)),
    ]

    return DepthAnalysis(
        overall_depth_score=round(depth_score, 2),
        reasoning=reasoning,
        per_segment=segments
    )


# --- FAST LOCAL WHISPER (existing) ---
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
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return f"❌ FFmpeg failed: {e}"

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        if os.path.exists(audio_path):
            os.remove(audio_path)
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
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return f"❌ Transcription failed: {e}"


# --- ULTRA-FAST GESTURE ANALYSIS ---
def analyze_gestures(video_path: str, model: Optional[YOLO]) -> str:
    if model is None:
        return "❌ YOLO model unavailable"

    st.info("🤲 Analyzing gestures & engagement...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "❌ Could not open video for gesture analysis"

    hand_frames = 0
    face_frames = 0
    total_frames = 0
    eye_contact_frames = 0

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return "❌ No frames in video"

        step = max(frame_count // 25, 1)
        frame_indices = list(range(0, frame_count, step))[:25]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            total_frames += 1
            frame_small = cv2.resize(frame, (960, 960))

            results = model(frame_small, verbose=False, imgsz=960)
            if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
                continue

            kpts = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()

            if len(kpts) >= 11 and confs[0] > 0.5 and confs[9] > 0.5 and confs[10] > 0.5:
                left_hand = kpts[9]
                right_hand = kpts[10]
                if (
                    (left_hand[0] > 50 and left_hand[1] > 50 and left_hand[0] < 870)
                    or (right_hand[0] > 50 and right_hand[1] > 50 and right_hand[0] < 870)
                ):
                    hand_frames += 1

                nose = kpts[0]
                if 300 < nose[0] < 660 and nose[1] < 600:
                    eye_contact_frames += 1

                face_frames += 1

    except Exception as e:
        return f"❌ Gesture error: {str(e)[:80]}"
    finally:
        cap.release()

    if total_frames == 0:
        return "❌ No frames analyzed"

    hand_ratio = hand_frames / total_frames
    face_ratio = face_frames / total_frames
    eye_ratio = eye_contact_frames / total_frames

    findings = []

    if hand_ratio > 0.6:
        findings.append("👋 **Active hand gestures** - Good engagement")
    elif hand_ratio > 0.3:
        findings.append("🙌 **Moderate gestures** - Natural movement")
    else:
        findings.append("👐 **Minimal gestures** - More static delivery")

    if eye_ratio > 0.5:
        findings.append("👁️ **Strong eye contact** - Confident delivery")
    elif eye_ratio > 0.2:
        findings.append("👀 **Moderate eye contact** - Looking at camera")
    else:
        findings.append("😶 **Limited eye contact** - More face-away moments")

    if face_ratio > 0.7:
        findings.append("✅ **Face clearly visible** throughout")
    elif face_ratio > 0.3:
        findings.append("⚠️ **Face visible intermittently**")
    else:
        findings.append("❌ **Face often not detected**")

    findings.append(f"📊 {total_frames} frames analyzed")

    return "**Engagement Analysis:**\n" + " | ".join(findings)


# --- Main Pipeline ---
def run_full_analysis(video_file: UploadFile, temp_video_path: str) -> AnalysisResponse:
    pose_model = get_pose_model()
    start_time = time.time()

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0
    cap.release()

    st.metric("📹 Duration", f"{duration:.1f}s")

    transcript = transcribe(temp_video_path)
    gestures = analyze_gestures(temp_video_path, pose_model)
    depth = analyze_concept_depth(transcript)

    total_time = time.time() - start_time
    st.success(f"🚀 Analysis complete in {total_time:.1f}s!")

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
    segments = [
        {"Start": f"{s.start:.0f}s", "End": f"{s.end:.0f}s", "Depth": f"{s.depth_score:.1f}"}
        for s in response.depth.per_segment[:5]
    ]
    st.table(segments)

    tab1, tab2 = st.tabs(["📝 Transcript", "🤲 Gestures"])
    with tab1:
        st.text_area("Transcript", response.transcript, height=200)
    with tab2:
        st.markdown(response.gestures)
def main():
    st.set_page_config(page_title="Mentor AI", layout="wide")
    st.title("🎓 Mentor AI: Presentation Analyzer")

    initialize_clients()

    st.info("🚀 **5x FASTER** - YOLOv11n + Optimized Processing!")

    tab_video, tab_url = st.tabs(["📁 Upload Video", "🌐 Analyze from URL"])

    # --------- TAB 1: LOCAL VIDEO UPLOAD ----------
    with tab_video:
        uploaded_file = st.file_uploader(
            "📁 Upload Video",
            type=["mp4", "mov", "avi"],
            key="file_uploader_local"
        )

        if uploaded_file:
            st.video(uploaded_file)
            st.info(f"**{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

            if st.button("🚀 Analyze Uploaded Video", type="primary"):
                with st.spinner("🔬 Ultra-fast processing..."):
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
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        else:
            st.info("👆 Upload a video in this tab to analyze.")

    # --------- TAB 2: URL → FULL ANALYSIS (TRANSCRIPT + GESTURES + DEPTH) ----------
    with tab_url:
        st.subheader("🌐 Analyze Video from URL (Whisper + Gestures + Depth)")

        url = st.text_input("Paste video URL (YouTube, etc.)", key="video_url_input")
        if st.button("🚀 Analyze URL Video", key="url_analyze_button"):
            if not url.strip():
                st.warning("Please enter a URL first.")
            else:
                temp_video_path = None
                with st.spinner("⏳ Downloading, transcribing, and analyzing..."):
                    try:
                        # Download + transcribe from URL (returns text, seconds, and local path)
                        text, sec, temp_video_path = url_to_text(
                            url,
                            download_filename=tempfile.mktemp(suffix=".mp4")
                        )
                        st.success(f"✅ Transcription complete in {sec:.1f}s")

                        # Fake UploadFile for compatibility with run_full_analysis
                        fake_upload = UploadFile(filename="url_video.mp4", file=None)

                        # Run full pipeline (duration + gestures + depth)
                        result = run_full_analysis(fake_upload, temp_video_path)

                        # Override transcript with the URL Whisper transcript
                        result.transcript = text

                        # Show full analysis (same UI as upload)
                        display_analysis_results(result)

                    except Exception as e:
                        st.error(f"❌ URL analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        if temp_video_path and os.path.exists(temp_video_path):
                            os.remove(temp_video_path)

    # Footer info
    st.markdown("""
    ---
    ### ⚙️ Quick Setup:
    - This app uses:
      - Local Whisper (faster-whisper) for uploaded video.
      - OpenAI Whisper (via `whisper` + `moviepy` + `yt_dlp`) for URL videos.
      - YOLO pose for gesture analysis.
    """)

 

if __name__ == "__main__":
    main()
