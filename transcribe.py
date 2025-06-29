# transcribe.py - v3.1 with missing import fixed
# (Updated for sequential model loading/unloading)

import torch
import torchaudio
import os
import sys
import logging
import traceback
import requests
import json
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# --- Configuration ---
OLLAMA_MODEL_TO_UNLOAD = "mistral"
OLLAMA_API_URL = "http://localhost:11434/api/unload"

# Setup logging to output to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unload_ollama_model(model_name):
    # This function seems to be for Ollama, not the PyTorch models themselves.
    # Keep it as is, but it's separate from PyTorch VRAM management.
    os.system(f"ollama stop {model_name}")

def process_audio(input_audio_path):
    """
    Processes a single audio file for speaker diarization and transcription,
    sequentially loading and unloading models to manage VRAM.
    """
    unload_ollama_model(OLLAMA_MODEL_TO_UNLOAD)

    if not os.path.exists(input_audio_path):
        logging.error(f"Input file not found: {input_audio_path}")
        return

    base_name = os.path.splitext(input_audio_path)[0]
    output_transcript_file = base_name + ".txt"
    mono_audio_file = base_name + "_mono.wav"

    logging.info(f"Loading audio from: {input_audio_path}")
    waveform, sample_rate = torchaudio.load(input_audio_path)

    if waveform.shape[0] > 1:
        logging.info("Audio is stereo. Converting to mono.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(mono_audio_file, waveform, sample_rate)
        audio_to_process = mono_audio_file
    else:
        logging.info("Audio is already mono.")
        audio_to_process = input_audio_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # --- Diarization Phase ---
    logging.info("Attempting to load diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if device == "cuda":
        diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        torch.cuda.empty_cache() # Clear cache after moving to GPU
    logging.info("Diarization pipeline loaded successfully.")

    logging.info(f"Running speaker diarization on {audio_to_process}...")
    diarization_result = diarization_pipeline(audio_to_process)
    logging.info("Diarization complete.")

    # Explicitly unload diarization model to free VRAM
    logging.info("Unloading diarization model and clearing CUDA cache...")
    if device == "cuda":
        diarization_pipeline.to(torch.device("cpu")) # Move to CPU
    del diarization_pipeline # Delete the object
    if device == "cuda":
        torch.cuda.empty_cache() # Clear CUDA cache
    logging.info("Diarization model unloaded.")

    # --- Transcription Phase ---
    logging.info("Attempting to load Whisper ASR model...")
    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    if device == "cuda":
        torch.cuda.empty_cache() # Clear cache after loading (might be needed if cache fills up from loading)
    logging.info(f"Whisper model '{model_size}' loaded successfully.")
    
    logging.info(f"Running transcription on {audio_to_process}...")
    segments, _ = asr_model.transcribe(audio_to_process, word_timestamps=True)
    logging.info("Transcription complete.")

    all_words = [word._asdict() for segment in segments for word in segment.words]
    
    def get_words_in_segment(word_list, start_time, end_time):
        segment_words = [
            word_info['word'] for word_info in word_list 
            if (word_info['start'] >= start_time and word_info['start'] < end_time) or 
               (word_info['end'] > start_time and word_info['end'] <= end_time)
        ]
        return ' '.join(word.strip() for word in segment_words)

    logging.info(f"Writing final transcript to {output_transcript_file}...")
    with open(output_transcript_file, "w") as f:
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segment_text = get_words_in_segment(all_words, turn.start, turn.end)
            if segment_text:
                f.write(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}: {segment_text}\n")
    
    logging.info("--- Transcription script finished successfully ---")
    
    if audio_to_process == mono_audio_file:
        os.remove(mono_audio_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <path_to_audio_file>", file=sys.stderr)
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    try:
        process_audio(audio_file_path)
    except Exception as e:
        logging.error("An unhandled exception occurred during transcription!")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
