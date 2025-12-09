import sqlite3
import streamlit as st
from datetime import datetime

# --- Database Setup ---
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

def save_mentor_analysis(name, video_name, duration, depth_score, gesture, transcript):
    conn = sqlite3.connect('mentor_scores.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO mentors (name, video_name, duration_seconds, depth_score, gesture_analysis, transcript)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, video_name, duration, depth_score, gesture, transcript))
    conn.commit()
    conn.close()

def get_all_mentors():
    conn = sqlite3.connect('mentor_scores.db')
    c = conn.cursor()
    c.execute('SELECT * FROM mentors ORDER BY analyzed_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

# --- Dashboard UI ---
def dashboard():
    st.set_page_config(page_title="Mentor Dashboard", layout="wide")
    st.title("🎓 Mentor AI: Teacher Dashboard")

    init_db()

    st.subheader("📊 All Mentor Scores")
    mentors = get_all_mentors()
    
    if mentors:
        data = []
        for row in mentors:
            data.append({
                "ID": row[0],
                "Name": row[1],
                "Video": row[2],
                "Duration (s)": row[3],
                "Depth Score": row[4],
                "Gesture Analysis": row[5],
                "Analyzed At": row[7]
            })
        st.table(data)
    else:
        st.info("No mentor data available.")

    st.subheader("➕ Add New Mentor Analysis")
    with st.form("add_mentor"):
        name = st.text_input("Mentor Name")
        video_name = st.text_input("Video Name")
        duration = st.number_input("Duration (seconds)", min_value=0.0)
        depth_score = st.number_input("Depth Score", min_value=0.0, max_value=1.0)
        gesture = st.text_area("Gesture Analysis")
        transcript = st.text_area("Transcript")
        submitted = st.form_submit_button("Save Analysis")
        if submitted:
            save_mentor_analysis(name, video_name, duration, depth_score, gesture, transcript)
            st.success("✅ Mentor analysis saved!")

if __name__ == "__main__":
    dashboard()
