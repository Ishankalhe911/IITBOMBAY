from flask import Flask, request, jsonify, send_from_directory
from moviepy.editor import VideoFileClip
import whisper
import tkinter as tk
from tkinter import filedialog
import threading
import os
import signal
import yt_dlp
import time

app = Flask(__name__)

# -------------------- LOAD WHISPER MODEL --------------------
MODEL_NAME = "base"
print(f"Loading Whisper model: {MODEL_NAME} ...")
model = whisper.load_model(MODEL_NAME)
print("Whisper Model Loaded.")

# -------------------- FIXED TKINTER PICKER --------------------
def open_file_dialog(filetypes):
    path_holder = {"path": None}

    def run_dialog():
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        path_holder["path"] = file_path
        root.destroy()

    dialog_thread = threading.Thread(target=run_dialog)
    dialog_thread.start()
    dialog_thread.join()

    return path_holder["path"]

# -------------------- PROCESS VIDEO --------------------
def process_video(video_path):
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"

    clip = VideoFileClip(video_path)

    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except:
            pass

    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # TRANSCRIBE USING WHISPER
    result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU
    text = result.get("text", "").strip()

    # DELETE AUDIO FILE
    try:
        os.remove(audio_path)
    except:
        pass

    # CLOSE VIDEO
    clip.close()

    return text

# -------------------- SELECT LOCAL VIDEO --------------------
@app.route("/select_video", methods=["GET"])
def select_video():
    start_time = time.time()

    try:
        video_path = open_file_dialog([
            ("Video Files", "*.mp4;*.mkv;*.mov;*.avi"),
            ("All Files", "*.*")
        ])

        if not video_path:
            return jsonify({"error": "cancelled"})

        text = process_video(video_path)

        # AUTO DELETE LOCAL VIDEO
        try:
            os.remove(video_path)
        except:
            pass

        processing_time = round(time.time() - start_time, 2)

        return jsonify({
            "video_path": "(auto-deleted)",
            "audio_path": "(auto-deleted)",
            "processing_time": processing_time,
            "text": text
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------- URL → VIDEO → AUDIO → TEXT --------------------
@app.route("/url_to_text", methods=["POST"])
def url_to_text():
    start_time = time.time()

    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"})

    try:
        filename = "downloaded_video.mp4"
        ydl_opts = {
            "format": "best",
            "outtmpl": filename,
            "quiet": True,
            "noplaylist": True,  # prevent playlists unless needed
            "ignoreerrors": True,
            "retries": 3,
            "cachedir": False,
            # -------------------- FULL PLATFORM SUPPORT --------------------
            "default_search": "auto",  # auto-detect URL type
            "geo_bypass": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        text = process_video(filename)

        # AUTO DELETE DOWNLOADED VIDEO
        try:
            os.remove(filename)
        except:
            pass

        processing_time = round(time.time() - start_time, 2)

        return jsonify({
            "video_source": "(downloaded, auto-deleted)",
            "audio_path": "(auto-deleted)",
            "processing_time": processing_time,
            "text": text
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------- EXIT APP --------------------
@app.route("/exit_app", methods=["GET"])
def exit_app():
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"status": "closing"})

# -------------------- CLEAR --------------------
@app.route("/clear", methods=["GET"])
def clear_file():
    return jsonify({"cleared": True})

# -------------------- SERVE HTML --------------------
@app.route("/")
def index_page():
    return send_from_directory("web", "index2.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("web", path)

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
