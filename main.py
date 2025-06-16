# ODIN Web Interface - v2.2 (Final)
# main.py

from flask import Flask, render_template, request, jsonify, session
import os
import subprocess
import requests
import json
from werkzeug.utils import secure_filename
from datetime import datetime

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'mp4', 'mov', 'avi'}
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  # 500 MB upload limit
app.config['SECRET_KEY'] = 'your-super-secret-key-for-sessions' # IMPORTANT: Change this for production

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_video_to_audio(video_path, audio_path):
    """Converts video file to a WAV audio file using ffmpeg."""
    try:
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path, '-y' # -y overwrites output file if it exists
        ]
        app.logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        app.logger.info(f"Successfully converted {video_path} to {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        app.logger.error(f"ffmpeg error stdout: {e.stdout}")
        app.logger.error(f"ffmpeg error stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        app.logger.error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return False


@app.route('/')
def index():
    """Renders the main UI and clears the session for a fresh start."""
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_transcribe():
    """Handles file upload, conversion, transcription, and session initialization."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
        
    session.clear()
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    original_base, ext = os.path.splitext(filepath)
    audio_path = filepath # Assume it's audio by default
    
    # If it's a video, create a new, timestamped path for the audio
    if ext.lower() in ['.mp4', '.mov', '.avi']:
        app.logger.info("Video file detected, converting to audio...")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_path = f"{original_base}_{timestamp}.wav"
        if not convert_video_to_audio(filepath, audio_path):
            return jsonify({"error": "Failed to convert video to audio. Is ffmpeg installed?"}), 500

    try:
        app.logger.info(f"Starting transcription process on {audio_path}...")
        script_path = os.path.join(os.path.dirname(__file__), 'transcribe.py')
        
        subprocess.run(['python', script_path, audio_path], check=True, capture_output=True, text=True)
        
        # Correctly construct the transcript file path from the actual audio file path processed
        transcript_base_name = os.path.splitext(audio_path)[0]
        transcript_path = transcript_base_name + '.txt'
        
        app.logger.info(f"Attempting to read transcript from: {transcript_path}")
        with open(transcript_path, 'r') as f:
            transcript = f.read()

        # Initialize session data
        session['original_transcript'] = transcript
        session['report_history'] = []
        session['edit_log'] = []
        
        return jsonify({"original_transcript": transcript, "edit_log": []})

    except subprocess.CalledProcessError as e:
        app.logger.error(f"Transcription script failed with stderr: {e.stderr}")
        return jsonify({"error": "Transcription failed. The processing script encountered an error. Check server logs."}), 500
    except FileNotFoundError:
        app.logger.error(f"Could not find expected transcript file at: {transcript_path}")
        return jsonify({"error": "Could not find transcript output file. The processing script may have failed silently."}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generates or updates a report using conversation history."""
    data = request.get_json()
    if not all(k in data for k in ['current_report', 'prompt', 'notes']):
        return jsonify({"error": "Missing data for report generation."}), 400

    current_report = data['current_report']
    new_instruction = data['prompt']
    officer_notes = data['notes']
    
    original_transcript = session.get('original_transcript', '')
    edit_log = session.get('edit_log', [])

    # Use a different prompt template for the initial generation vs. an iterative edit.
    if not current_report.strip():
        prompt_template = (
            "You are a police report generation assistant. Your task is to create an objective, chronological, and "
            "formal police report based on the provided body camera audio transcript and officer notes.\n\n"
            "--- BODY CAM TRANSCRIPT ---\n{transcript}\n\n"
            "--- OFFICER'S NOTES ---\n{notes}\n\n"
            "--- INSTRUCTION ---\n{instruction}\n\n"
            "Generate the full report now:"
        )
        full_prompt = prompt_template.format(transcript=original_transcript, notes=officer_notes, instruction=new_instruction)
    else:
        prompt_template = (
            "You are a police report editing assistant. You must modify an existing report based on a new instruction. "
            "Context is provided, including the original transcript and previous edits.\n\n"
            "--- ORIGINAL BODY CAM TRANSCRIPT (for reference) ---\n{transcript}\n\n"
            "--- PREVIOUS EDIT INSTRUCTIONS ---\n{edit_log}\n\n"
            "--- OFFICER'S NOTES ---\n{notes}\n\n"
            "--- CURRENT REPORT DRAFT ---\n{current_report}\n\n"
            "--- NEW INSTRUCTION ---\nApply this instruction to the 'CURRENT REPORT DRAFT': '{instruction}'\n\n"
            "Now, provide the complete, updated text of the new report. Do not just describe the changes."
        )
        full_prompt = prompt_template.format(
            transcript=original_transcript,
            edit_log='\n'.join(f"- {item}" for item in edit_log) or "None",
            notes=officer_notes,
            current_report=current_report,
            instruction=new_instruction
        )
    
    payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}

    try:
        app.logger.info(f"Sending request to Ollama with model {OLLAMA_MODEL}...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        response_data = response.json()
        new_report = response_data.get("response", "Error: No response content from model.").strip()
        
        # Update session history
        session['edit_log'].append(new_instruction)
        session['report_history'].append(new_report)
        session.modified = True
        
        return jsonify({
            "report": new_report,
            "version": len(session['report_history']),
            "edit_log": session['edit_log'] 
        })
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Could not connect to Ollama API: {e}")
        return jsonify({"error": f"Failed to connect to Ollama at {OLLAMA_API_URL}. Is it running?"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred during report generation."}), 500

@app.route('/revert', methods=['POST'])
def revert_report():
    """Reverts the report to the previous version."""
    if 'report_history' not in session or not session['report_history']:
        return jsonify({"error": "No history to revert."}), 400

    session['report_history'].pop()
    if session['edit_log']:
        session['edit_log'].pop()
    session.modified = True

    new_report = session['report_history'][-1] if session['report_history'] else ""
    
    return jsonify({
        "report": new_report,
        "version": len(session['report_history']),
        "edit_log": session.get('edit_log', [])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
