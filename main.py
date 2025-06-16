# ODIN Web Interface - v2.1 with Timestamping and History UI
# main.py

from flask import Flask, render_template, request, jsonify, session
import os
import subprocess
import requests
import json
from werkzeug.utils import secure_filename
from datetime import datetime # Import datetime

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'mp4', 'mov', 'avi'}
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500
app.config['SECRET_KEY'] = 'your-super-secret-key-for-sessions'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_video_to_audio(video_path, audio_path):
    """Converts video file to a WAV audio file using ffmpeg."""
    try:
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path, '-y' # Add -y to auto-overwrite
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
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_transcribe():
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

    base, ext = os.path.splitext(filepath)
    audio_path = filepath
    
    if ext.lower() in ['.mp4', '.mov', 'avi']:
        app.logger.info("Video file detected, converting to audio...")
        # --- MODIFICATION: Add timestamp to prevent filename collision ---
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_path = f"{base}_{timestamp}.wav"
        # --- END MODIFICATION ---
        if not convert_video_to_audio(filepath, audio_path):
            return jsonify({"error": "Failed to convert video to audio. Is ffmpeg installed?"}), 500

    try:
        app.logger.info(f"Starting transcription process on {audio_path}...")
        script_path = os.path.join(os.path.dirname(__file__), 'transcribe.py')
        # Use the base name of the audio file for the transcript output
        transcript_base_name = os.path.splitext(audio_path)[0]
        subprocess.run(['python', script_path, audio_path], check=True, capture_output=True, text=True)
        
        transcript_path = transcript_base_name + '.txt'
        with open(transcript_path, 'r') as f:
            transcript = f.read()

        session['original_transcript'] = transcript
        session['report_history'] = []
        session['edit_log'] = []
        
        # MODIFICATION: Return an empty edit_log for UI consistency
        return jsonify({"original_transcript": transcript, "edit_log": []})

    except subprocess.CalledProcessError as e:
        app.logger.error(f"Transcription script failed with stderr: {e.stderr}")
        return jsonify({"error": "Transcription failed. Check server logs."}), 500
    except FileNotFoundError:
        return jsonify({"error": "Could not find transcript output file."}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    # ... (prompt construction logic is the same)
    # ...
    # The full logic from the previous step remains here. This is a placeholder for brevity.
    original_transcript = session.get('original_transcript', '')
    edit_log = session.get('edit_log', [])
    current_report = data['current_report']
    new_instruction = data['prompt']
    officer_notes = data['notes']
    
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
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        response_data = response.json()
        new_report = response_data.get("response", "Error: No response content from model.").strip()
        
        session['edit_log'].append(new_instruction)
        session['report_history'].append(new_report)
        session.modified = True
        
        # MODIFICATION: Return the updated edit_log
        return jsonify({
            "report": new_report,
            "version": len(session['report_history']),
            "edit_log": session['edit_log'] 
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to Ollama at {OLLAMA_API_URL}. Is it running?"}), 500
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred during report generation."}), 500

@app.route('/revert', methods=['POST'])
def revert_report():
    if 'report_history' not in session or not session['report_history']:
        return jsonify({"error": "No history to revert."}), 400

    session['report_history'].pop()
    if session['edit_log']:
        session['edit_log'].pop()
    session.modified = True

    new_report = session['report_history'][-1] if session['report_history'] else ""
    
    # MODIFICATION: Return the updated edit_log
    return jsonify({
        "report": new_report,
        "version": len(session['report_history']),
        "edit_log": session['edit_log']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
