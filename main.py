# ODIN Web Interface - v3.0 with Auditing, Export, and Enhanced Logging
# main.py

from flask import Flask, render_template, request, jsonify, session, Response
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
OPERATOR_NAME = "ODIN (https://github.com/ruapotato/ODIN) By David Hamner" # For audit logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500
app.config['SECRET_KEY'] = 'your-super-secret-key-for-sessions' # IMPORTANT: Change this

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_video_to_audio(video_path, audio_path):
    try:
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y']
        app.logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        app.logger.info(f"Successfully converted {video_path} to {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        app.logger.error(f"ffmpeg failed. STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        app.logger.error("ffmpeg not found. Please ensure ffmpeg is installed.")
        return False

# --- Core Application Routes ---
@app.route('/')
def index():
    app.logger.info(f"New session started for client {request.remote_addr}. Clearing session data.")
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_transcribe():
    app.logger.info(f"'/upload' endpoint called by {request.remote_addr}")
    if 'file' not in request.files:
        app.logger.warning("Upload attempt failed: 'file' not in request.files")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        app.logger.warning("Upload attempt failed: No selected file")
        return jsonify({"error": "No selected file"}), 400
        
    if not file or not allowed_file(file.filename):
        app.logger.warning(f"Upload attempt failed: File type not allowed for '{file.filename}'")
        return jsonify({"error": "File type not allowed"}), 400
    
    app.logger.info("Clearing session for new file upload.")
    session.clear()
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    app.logger.info(f"File '{filename}' saved to '{filepath}'")

    original_base, ext = os.path.splitext(filepath)
    audio_path = filepath
    
    if ext.lower() in ['.mp4', '.mov', 'avi']:
        app.logger.info(f"Video file '{filename}' detected, initiating conversion.")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_path = f"{original_base}_{timestamp}.wav"
        if not convert_video_to_audio(filepath, audio_path):
            return jsonify({"error": "Failed to convert video to audio. Is ffmpeg installed?"}), 500

    try:
        script_path = os.path.join(os.path.dirname(__file__), 'transcribe.py')
        command = ['python', script_path, audio_path]
        app.logger.info(f"Running transcription command: {' '.join(command)}")
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        app.logger.info("Transcription script finished execution.")
        
        transcript_base_name = os.path.splitext(audio_path)[0]
        transcript_path = transcript_base_name + '.txt'
        
        app.logger.info(f"Attempting to read transcript from: {transcript_path}")
        with open(transcript_path, 'r') as f:
            transcript = f.read()
        app.logger.info("Transcript file read successfully.")

        session['original_transcript'] = transcript
        session['report_history'] = []
        session['edit_log'] = []
        session['upload_filename'] = filename
        app.logger.info("Session initialized with transcript data.")
        
        return jsonify({"original_transcript": transcript, "edit_log": []})

    except subprocess.CalledProcessError as e:
        app.logger.error("The transcription script failed!")
        app.logger.error(f"--- TRANSCRIPT SCRIPT STDERR ---\n{e.stderr}")
        return jsonify({"error": "Transcription process failed. Check server logs for details."}), 500
    except FileNotFoundError:
        app.logger.error(f"Could not find expected transcript file at: {transcript_path}")
        return jsonify({"error": "Could not find transcript output file."}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    app.logger.info(f"'/generate_report' endpoint called by {request.remote_addr}")
    data = request.get_json()
    if not all(k in data for k in ['current_report', 'prompt', 'notes']):
        app.logger.error("Generate report call failed: Missing required data in request.")
        return jsonify({"error": "Missing data for report generation."}), 400
    
    # The new instruction from the user, which may include a note about manual edits
    new_instruction = data['prompt']
    
    # Add the edit to the log
    session.setdefault('edit_log', []).append(new_instruction)
    app.logger.info(f"New instruction added to edit log: '{new_instruction}'")
    
    #... (The rest of the logic is the same as before)
    current_report = data['current_report']
    officer_notes = data['notes']
    original_transcript = session.get('original_transcript', '')
    edit_log = session.get('edit_log', [])
    
    # (The detailed prompt construction logic remains unchanged)
    if not current_report.strip() or "(User manually edited the report before this instruction)" in new_instruction:
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
            edit_log='\n'.join(f"- {item}" for item in edit_log[:-1]) or "None", # Show previous edits
            notes=officer_notes,
            current_report=current_report,
            instruction=new_instruction
        )
    
    payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}

    try:
        app.logger.info(f"Sending request to Ollama with model '{OLLAMA_MODEL}'.")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        response_data = response.json()
        new_report = response_data.get("response", "Error: No response content from model.").strip()
        app.logger.info("Received successful response from Ollama.")
        
        session.setdefault('report_history', []).append(new_report)
        session.modified = True
        app.logger.info(f"Report history updated. Current version: {len(session['report_history'])}")
        
        return jsonify({
            "report": new_report,
            "version": len(session['report_history']),
            "edit_log": session['edit_log'] 
        })
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Could not connect to Ollama API: {e}")
        return jsonify({"error": f"Failed to connect to Ollama at {OLLAMA_API_URL}. Is it running?"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during report generation: {e}")
        return jsonify({"error": "An unexpected error occurred during report generation."}), 500


@app.route('/revert', methods=['POST'])
def revert_report():
    app.logger.info(f"'/revert' endpoint called by {request.remote_addr}")
    if 'report_history' not in session or len(session.get('report_history', [])) < 1:
        app.logger.warning("Revert failed: No history to revert.")
        return jsonify({"error": "No history to revert."}), 400

    session['report_history'].pop()
    if session.get('edit_log'):
        session['edit_log'].pop()
    session.modified = True
    app.logger.info(f"Reverted to previous version. New version count: {len(session['report_history'])}")

    new_report = session['report_history'][-1] if session['report_history'] else ""
    
    return jsonify({
        "report": new_report,
        "version": len(session['report_history']),
        "edit_log": session.get('edit_log', [])
    })

@app.route('/export')
def export_report():
    """Generates a comprehensive text file for download with a full audit trail."""
    app.logger.info(f"'/export' endpoint called by {request.remote_addr}. Generating audit report.")
    
    # --- Gather all data from session ---
    final_report = session.get('report_history', ['No report generated.'])[-1]
    original_transcript = session.get('original_transcript', 'No transcript found.')
    edit_log = session.get('edit_log', [])
    upload_filename = session.get('upload_filename', 'N/A')
    export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    # --- Build the export content ---
    content = []
    content.append("======================================================")
    content.append("          ODIN: OFFICIAL CASE REPORT EXPORT")
    content.append("======================================================")
    content.append(f"\nReport Generated By: {OPERATOR_NAME}")
    content.append(f"Export Timestamp:    {export_time}")
    content.append(f"Original Evidence File: {upload_filename}")
    content.append("\n------------------------------------------------------")
    content.append("                  FINAL REPORT")
    content.append("------------------------------------------------------\n")
    content.append(final_report)
    content.append("\n\n------------------------------------------------------")
    content.append("                AUDIT & EDIT LOG")
    content.append("------------------------------------------------------\n")
    if edit_log:
        for i, entry in enumerate(edit_log, 1):
            content.append(f"{i}. {entry}")
    else:
        content.append("No AI-assisted edits were made.")
    
    content.append("\n\n------------------------------------------------------")
    content.append("              ORIGINAL AI TRANSCRIPT")
    content.append("------------------------------------------------------\n")
    content.append(original_transcript)
    content.append("\n\n======================================================")
    content.append("                    END OF REPORT")
    content.append("======================================================")

    export_data = "\n".join(content)
    
    # --- Create response to trigger download ---
    return Response(
        export_data,
        mimetype="text/plain",
        headers={"Content-disposition":
                 f"attachment; filename=ODIN_Report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"}
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
