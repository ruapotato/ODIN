# ODIN Web Interface
# main.py

from flask import Flask, render_template, request, jsonify
import os
import subprocess
import requests
import json
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'mp4', 'mov', 'avi'}
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API URL
OLLAMA_MODEL = "mistral" # The model to use for report generation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  # 500 MB upload limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_video_to_audio(video_path, audio_path):
    """Converts video file to a WAV audio file using ffmpeg."""
    try:
        # Command to convert video to WAV, mono, 16kHz sample rate
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path
        ]
        app.logger.info(f"Running ffmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        app.logger.info(f"Successfully converted {video_path} to {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        app.logger.error(f"ffmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        app.logger.error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return False


@app.route('/')
def index():
    """Renders the main user interface."""
    return render_template('index.html', default_prompt="Generate a detailed police report from the following audio transcript. The report should be objective, chronological, and suitable for official records. Identify all speakers and summarize the key events and statements.")

@app.route('/upload', methods=['POST'])
def upload_and_transcribe():
    """Handles file upload, conversion (if needed), and transcription."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    app.logger.info(f"File saved to {filepath}")

    base, ext = os.path.splitext(filepath)
    audio_path = filepath
    
    # Convert video to audio if necessary
    if ext.lower() in ['.mp4', '.mov', '.avi']:
        app.logger.info("Video file detected, converting to audio...")
        audio_path = base + '.wav'
        if not convert_video_to_audio(filepath, audio_path):
            return jsonify({"error": "Failed to convert video to audio. Is ffmpeg installed?"}), 500

    # Run transcription as a separate process to manage memory
    try:
        app.logger.info("Starting transcription process...")
        script_path = os.path.join(os.path.dirname(__file__), 'transcribe.py')
        subprocess.run(['python', script_path, audio_path], check=True)
        app.logger.info("Transcription process finished.")
        
        # Read the resulting transcript file
        transcript_path = base + '.txt'
        with open(transcript_path, 'r') as f:
            transcript = f.read()
        
        return jsonify({"transcript": transcript})

    except subprocess.CalledProcessError as e:
        app.logger.error(f"Transcription script failed: {e}")
        return jsonify({"error": "Transcription failed. Check server logs."}), 500
    except FileNotFoundError:
        app.logger.error("Transcript file not found.")
        return jsonify({"error": "Could not find transcript output file."}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generates a report by calling the Ollama API."""
    data = request.get_json()
    if not all(k in data for k in ['transcript', 'prompt', 'notes']):
        return jsonify({"error": "Missing data for report generation."}), 400

    transcript = data['transcript']
    prompt_template = data['prompt']
    officer_notes = data['notes']

    # Construct the full prompt for Ollama
    full_prompt = (
        f"{prompt_template}\n\n"
        f"--- AUDIO TRANSCRIPT ---\n{transcript}\n\n"
        f"--- OFFICER'S NOTES ---\n{officer_notes}\n\n"
        f"--- END OF DATA ---\n\n"
        "Generate the report now:"
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        # "keep_alive": 0 # Use this to unload the model after generation if VRAM is extremely tight
    }

    try:
        app.logger.info(f"Sending request to Ollama with model {OLLAMA_MODEL}...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # The response from Ollama is a JSON object, with the content in the 'response' key
        response_data = response.json()
        report = response_data.get("response", "Error: No response content from model.")
        
        return jsonify({"report": report.strip()})
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Could not connect to Ollama API: {e}")
        return jsonify({"error": f"Failed to connect to Ollama at {OLLAMA_API_URL}. Is it running?"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred during report generation."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
