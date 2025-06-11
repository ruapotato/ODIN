import torch
import torchaudio
import os
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import json # To help with inspecting the output

# --- 1. Setup ---
# Select the most powerful Whisper model
MODEL_SIZE = "large-v3"
# Set your local audio file
ORIGINAL_AUDIO_FILE = "/home/david/Downloads/test.wav"


# --- 2. Preprocess Audio to Ensure it is Mono ---
# This part remains the same as it's a best practice
AUDIO_FILE = os.path.splitext(ORIGINAL_AUDIO_FILE)[0] + "_mono.wav"

print(f"Loading audio from: {ORIGINAL_AUDIO_FILE}")
waveform, sample_rate = torchaudio.load(ORIGINAL_AUDIO_FILE)

if waveform.shape[0] > 1:
    print("Audio is stereo. Converting to mono.")
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save(AUDIO_FILE, waveform, sample_rate)
    print(f"Mono audio saved to: {AUDIO_FILE}")
else:
    print("Audio is already mono.")
    AUDIO_FILE = ORIGINAL_AUDIO_FILE

# --- 3. Load Models ---

# Load Diarization Pipeline
print("Loading diarization pipeline...")
# IMPORTANT: Enter your Hugging Face access token here if prompted or ensure it's in your environment
# For example: diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="YOUR_HF_TOKEN")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

if torch.cuda.is_available():
    diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
    print("Diarization pipeline moved to GPU.")

# Load ASR Model (Whisper)
print(f"Loading Whisper model: {MODEL_SIZE}...")
# The model will be downloaded automatically on the first run.
# To use GPU, set compute_type="float16". For CPU, compute_type="int8"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
asr_model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
print("Whisper model loaded.")


# --- 4. Run Pipelines ---

# A. Run Speaker Diarization
print(f"Running speaker diarization on {AUDIO_FILE}...")
diarization_result = diarization_pipeline(AUDIO_FILE)
print("Diarization complete.")

# B. Run Transcription with Word Timestamps
print(f"Running transcription on {AUDIO_FILE}...")
# Set word_timestamps=True to get word-level timestamps
segments, info = asr_model.transcribe(AUDIO_FILE, word_timestamps=True)

# Flatten the list of all words from all segments
all_words = []
for segment in segments:
    for word in segment.words:
        all_words.append(word._asdict()) # Use ._asdict() to get a dictionary representation
print("Transcription complete.")


# --- 5. Merge Diarization and Transcription ---

def get_words_in_segment(word_list, start_time, end_time):
    """Helper function to find words within a given diarization segment."""
    segment_words = []
    for word_info in word_list:
        # Check if the word's start or end time falls within the segment
        if (word_info['start'] >= start_time and word_info['start'] <= end_time) or \
           (word_info['end'] >= start_time and word_info['end'] <= end_time):
            segment_words.append(word_info['word'])
    
    # We strip leading/trailing spaces from the words themselves
    return ' '.join(word.strip() for word in segment_words)


print("\n--- Full Transcript with Speakers ---")
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end
    
    # Get the transcribed text for the current speaker segment
    segment_text = get_words_in_segment(all_words, start_time, end_time)

    if segment_text: # Only print if there is text in the segment
        print(f"[{start_time:.3f}s --> {end_time:.3f}s] {speaker}:{segment_text}")

print("\n--- End of Transcript ---")
