import streamlit as st
from main import run_full_analysis, AnalysisResponse, display_analysis_results  # ← CHANGED HERE
import tempfile
import os
import sqlite3
from datetime import datetime
from fastapi import UploadFile  # Add this for fake UploadFile

# --- Database Setup (unchanged) ---
def init_db():
    conn = sqlite3.connect('mentor_scores.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS mentors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            video_name TEXT,
            duration_seconds REAL,
            depth_score REAL,
            gesture_analysis TEXT,
            transcript TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_mentor_analysis(name: str, result: AnalysisResponse):
    conn = sqlite3.connect('mentor_scores.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO mentors (name, video_name, duration_seconds, depth_score, gesture_analysis, transcript)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, "ai_analyzed_video.mp4", result.duration_seconds,
          result.depth.overall_depth_score, result.gestures, result.transcript))
    conn.commit()
    conn.close()

def get_all_mentors():
    conn = sqlite3.connect('mentor_scores.db')
    c = conn.cursor()
    c.execute('SELECT * FROM mentors ORDER BY analyzed_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

# --- Main Dashboard ---
@st.cache_data
def load_dashboard():
    return get_all_mentors()

def main():
    st.set_page_config(page_title="Mentor Dashboard + AI", layout="wide")
    st.title("🎓 Mentor AI: Dashboard & Analyzer")

    init_db()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Select:", ["📊 View Scores", "🚀 AI Analyzer", "➕ Manual Add"])

    if page == "📊 View Scores":
        st.header("📊 All Mentor Analyses")
        mentors = load_dashboard()
        
        if mentors:
            data = []
            for row in mentors:
                data.append({
                    "ID": row[0], "Name": row[1], "Duration": f"{row[3]:.1f}s",
                    "Depth": f"{row[4]:.2f}", "Date": row[7][:16]
                })
            st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No analyses yet. Run AI analyzer first!")

    elif page == "🚀 AI Analyzer":
        st.header("🤖 AI Video Analysis (Auto-Saves)")
        name = st.text_input("👤 Mentor Name", help="Required for dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4", "mov", "avi"])
        with col2:
            url = st.text_input("🌐 YouTube URL")

        if name and (uploaded_file or url):
            if st.button("🚀 Analyze & Save to Dashboard", type="primary"):
                with st.spinner("🔬 Processing with YOLO+Whisper..."):
                    temp_path = None
                    try:
                        if uploaded_file:
                            # File upload path
                            temp_path = tempfile.mktemp(suffix=f".{uploaded_file.name.split('.')[-1]}")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Create fake UploadFile object
                            fake_upload = UploadFile(filename=uploaded_file.name, file=None)
                            result = run_full_analysis(fake_upload, temp_path)
                            
                        # Auto-save immediately
                        save_mentor_analysis(name, result)
                        st.success("✅ Saved to dashboard!")
                        
                        # Show results
                        st.subheader("📋 Analysis Results")
                        display_analysis_results(result)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
        else:
            st.warning("👆 Enter name + video/URL")

    elif page == "➕ Manual Add":
        st.subheader("➕ Manual Entry")
        with st.form("manual_form"):
            name = st.text_input("Name")
            duration = st.number_input("Duration (s)", min_value=0.0)
            depth = st.number_input("Depth Score", 0.0, 1.0)
            gesture = st.text_area("Gestures")
            transcript = st.text_area("Transcript")
            if st.form_submit_button("Save"):
                # Create fake result for manual save
                fake_result = AnalysisResponse(
                    duration_seconds=duration,
                    transcript=transcript,
                    gestures=gesture,
                    depth=type('Depth', (), {'overall_depth_score': depth})()
                )
                save_mentor_analysis(name, fake_result)
                st.success("✅ Saved!")
                st.rerun()

if __name__ == "__main__":
    main()
