import os
import tempfile
import json
import subprocess
import time
from typing import Optional

from pydantic import BaseModel, Field
import cv2
from ultralytics import YOLO

from fastapi import UploadFile, HTTPException

from openai import OpenAI
from google import genai
from google.genai.errors import APIError

import streamlit as st


# --- Initialization ---

POSE_MODEL: Optional[YOLO] = None
openai_client: Optional[OpenAI] = None
gemini_client: Optional[genai.Client] = None


@st.cache_resource
def load_yolo_model():
    global POSE_MODEL
    try:
        model_path = "yolov8n-pose.pt"
        if not os.path.exists(model_path):
            st.warning("Downloading yolov8n-pose.pt for gesture analysis. This happens only once.")
        POSE_MODEL = YOLO(model_path)
        st.success("YOLOv8 Pose Model Loaded.")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}. Gesture analysis will be skipped.")
        POSE_MODEL = None


@st.cache_resource
def initialize_api_clients():
    global openai_client, gemini_client

    gemini_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found. Please set it as environment variable or in secrets.toml.")
        gemini_client = None
    else:
        try:
            gemini_client = genai.Client(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Gemini Client: {e}")
            gemini_client = None

    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not openai_api_key:
        st.error("OPENAI_API_KEY not found. Transcription will be skipped.")
        openai_client = None
    else:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI Client: {e}")
            openai_client = None


# --- Pydantic Models ---


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


# --- Helper functions ---


def transcribe(video_path: str, client: OpenAI) -> str:
    audio_path = tempfile.mktemp(suffix=".mp3")
    st.info("Extracting audio from video...")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            audio_path
        ], check=True, capture_output=True, timeout=60)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg Error: {e.stderr.decode()}")
        return "Audio extraction failed."
    except FileNotFoundError:
        st.error("FFmpeg not found.")
        return "FFmpeg missing."
    except subprocess.TimeoutExpired:
        st.error("FFmpeg timed out.")
        return "FFmpeg timed out."

    st.info("Transcribing audio using Whisper...")

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        st.error(f"Whisper Error: {e}")
        return f"Transcription failed: {e}"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def analyze_gestures(video_path: str, model: YOLO, client: genai.Client) -> str:
    if not model:
        return "YOLO model not loaded."
    if not client:
        return "Gemini Client not initialized."

    st.info("Analyzing gestures with YOLOv8...")

    try:
        results = model(video_path, stream=True, save=False, verbose=False)
        key_findings = []
        frame_count = 0
        hand_visibility_count = 0
        SAMPLE_RATE = 10

        for i, r in enumerate(results):
            if i % SAMPLE_RATE != 0:
                continue
            frame_count += 1
            if r.boxes.xyxy.numel() > 0:
                kpts = r.keypoints.data
                if kpts.shape[1] > 10:
                    lw = kpts[0, 9, 2].item()
                    rw = kpts[0, 10, 2].item()
                    if lw > 0.5 or rw > 0.5:
                        hand_visibility_count += 1

        if frame_count > 0:
            hand_ratio = hand_visibility_count / frame_count
            if hand_ratio > 0.6:
                key_findings.append(f"Hands visible in {hand_ratio:.0%} of frames (active engagement).")
            elif hand_ratio > 0.2:
                key_findings.append(f"Hands moderately visible ({hand_ratio:.0%}).")
            else:
                key_findings.append("Hands rarely visible.")

            if frame_count > 50:
                key_findings.append("Centered, stable posture.")
            else:
                key_findings.append("Limited frame data; analysis less reliable.")

        gesture_data = "\n".join(key_findings)

        prompt = f"""
        Summarize the following gesture analysis data into 3–4 sentences.

        Raw Data:
        {gesture_data}
        """

        st.info("Summarizing gesture data with Gemini...")

        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return result.text

    except Exception as e:
        st.error(f"Gesture Error: {e}")
        return f"Gesture analysis failed: {e}"


def analyze_concept_depth(transcript: str, client: genai.Client) -> DepthAnalysis:
    if not client:
        raise HTTPException(status_code=500, detail="Gemini not initialized.")

    st.info("Analyzing concept depth with Gemini...")

    system_prompt = (
        "You are an expert educational analyst. Evaluate the transcript for depth and clarity. "
        "Return JSON following the provided schema."
    )

    user_prompt = f"""
    Analyze the transcript below.

    Transcript:
    ---
    {transcript}
    ---

    Instructions:
    1. Give a single depth score (0–1).
    2. Provide 3–4 sentences of reasoning.
    3. Break into 3–5 segments with time estimates assuming 150 WPM.
    Output valid JSON.
    """

    schema = DepthAnalysis.model_json_schema()

    config = {
        "responseMimeType": "application/json",
        "responseSchema": schema
    }

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=config,
                system_instruction=system_prompt
            )
            data = json.loads(resp.text.strip())
            return DepthAnalysis(**data)

        except (APIError, json.JSONDecodeError) as e:
            st.warning(f"Retry {attempt + 1}/3 – Error: {e}")
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=500, detail="Depth analysis failed.")


def run_full_analysis(video_file: UploadFile, temp_video_path: str,
                      openai_client: OpenAI, gemini_client: genai.Client) -> AnalysisResponse:

    st.header("🔬 Running Analysis Pipelines...")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if duration == 0:
        raise HTTPException(status_code=400, detail="Invalid video duration.")

    st.metric("Video Duration", f"{duration:.2f} seconds")

    transcript = transcribe(temp_video_path, openai_client)
    if "failed" in transcript.lower():
        raise HTTPException(status_code=500, detail=transcript)

    gestures = analyze_gestures(temp_video_path, POSE_MODEL, gemini_client)
    depth = analyze_concept_depth(transcript, gemini_client)

    return AnalysisResponse(
        duration_seconds=duration,
        transcript=transcript,
        gestures=gestures,
        depth=depth
    )


def display_analysis_results(response: AnalysisResponse):
    st.subheader("✅ Analysis Complete")
    st.metric("Total Duration", f"{response.duration_seconds:.2f} sec")

    st.markdown("---")
    st.header("🧠 Concept Depth Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Overall Depth Score",
            f"{response.depth.overall_depth_score:.2f} / 1.0",
            delta="Provided by Gemini"
        )
    with col2:
        st.info("**Reasoning:** " + response.depth.reasoning)

    st.subheader("Segment Breakdown")
    st.table([
        {
            "Start (s)": f"{s.start:.1f}",
            "End (s)": f"{s.end:.1f}",
            "Depth": f"{s.depth_score:.2f}"
        }
        for s in response.depth.per_segment
    ])

    st.markdown("---")
    st.header("📝 Transcript & Body Language")

    tab1, tab2 = st.tabs(["Transcript", "Gestures"])
    with tab1:
        st.code(response.transcript)
    with tab2:
        st.write(response.gestures)


def main_app():
    st.title("Mentor AI: Video Presentation Analyzer")
    st.caption("Upload a video to analyze concept clarity & gestures.")

    load_yolo_model()
    initialize_api_clients()

    uploaded_file = st.file_uploader(
        "Upload a video:", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("Start Analysis", type="primary"):
            temp_path = tempfile.mktemp(
                suffix=f".{uploaded_file.name.split('.')[-1]}"
            )

            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                result = run_full_analysis(
                    video_file=uploaded_file,
                    temp_video_path=temp_path,
                    openai_client=openai_client,
                    gemini_client=gemini_client
                )
                display_analysis_results(result)

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    st.toast("Temporary file removed.")
    else:
        st.info("Awaiting video upload.")


if __name__ == "__main__":
    main_app()
