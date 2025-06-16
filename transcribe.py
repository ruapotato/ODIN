# transcribe.py - v2 with robust error handling

import torch
import torchaudio
import os
import sys
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import logging
import traceback

# Setup logging to output to stderr, which can be captured by the main app
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio(input_audio_path):
    """
    Processes a single audio file for speaker diarization and transcription.
    The output is saved to a text file.
    """
    if not os.path.exists(input_audio_path):
        logging.error(f"Input file not found: {input_audio_path}")
        # No need to sys.exit(1) here, as a missing file is a known failure case
        # that the main app can handle more gracefully if needed. But for an
        # unrecoverable script error, we will exit.
        return

    base_name = os.path.splitext(input_audio_path)[0]
    output_transcript_file = base_name + ".txt"
    mono_audio_file = base_name + "_mono.wav"

    # --- 1. Preprocess Audio to Mono ---
    logging.info(f"Loading audio from: {input_audio_path}")
    waveform, sample_rate = torchaudio.load(input_audio_path)

    if waveform.shape[0] > 1:
        logging.info("Audio is stereo. Converting to mono.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(mono_audio_file, waveform, sample_rate)
        logging.info(f"Mono audio saved to: {mono_audio_file}")
        audio_to_process = mono_audio_file
    else:
        logging.info("Audio is already mono.")
        audio_to_process = input_audio_path
    
    # --- 2. Load Models ---
    logging.info("Attempting to load diarization and ASR models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if device == "cuda":
        diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
    logging.info("Diarization pipeline loaded successfully.")

    model_size = "large-v3"
    asr_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logging.info(f"Whisper model '{model_size}' loaded successfully.")

    # --- 3. Run Pipelines ---
    logging.info(f"Running speaker diarization on {audio_to_process}...")
    diarization_result = diarization_pipeline(audio_to_process)
    
    logging.info(f"Running transcription on {audio_to_process}...")
    segments, _ = asr_model.transcribe(audio_to_process, word_timestamps=True)
    
    all_words = [word._asdict() for segment in segments for word in segment.words]
    logging.info("Diarization and transcription complete.")

    # --- 4. Merge Results and Save ---
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
        # This is the crucial part. Catch ANY exception that wasn't handled.
        logging.error("An unhandled exception occurred during transcription!")
        # Print the full error traceback to stderr so the calling process can see it.
        traceback.print_exc(file=sys.stderr)
        # Exit with a non-zero status code to indicate failure.
        sys.exit(1)
