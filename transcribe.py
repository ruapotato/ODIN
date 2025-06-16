# ODIN Transcription Service
# Usage: python transcribe.py /path/to/your/audio.wav

import torch
import torchaudio
import os
import sys
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio(input_audio_path):
    """
    Processes a single audio file for speaker diarization and transcription.
    The output is saved to a text file.
    """
    if not os.path.exists(input_audio_path):
        logging.error(f"Input file not found: {input_audio_path}")
        return

    base_name = os.path.splitext(input_audio_path)[0]
    output_transcript_file = base_name + ".txt"
    mono_audio_file = base_name + "_mono.wav"

    # --- 1. Preprocess Audio to Mono ---
    try:
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
    except Exception as e:
        logging.error(f"Error during audio preprocessing: {e}")
        return

    # --- 2. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        # Load Diarization Pipeline
        logging.info("Loading diarization pipeline...")
        # Note: Ensure you have authenticated with 'huggingface-cli login' first.
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if device == "cuda":
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        logging.info("Diarization pipeline loaded.")

        # Load ASR Model (Whisper)
        model_size = "large-v3"
        logging.info(f"Loading Whisper model: {model_size}...")
        asr_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info("Whisper model loaded.")
    except Exception as e:
        logging.error(f"Error loading AI models: {e}")
        return

    # --- 3. Run Pipelines ---
    # A. Speaker Diarization
    logging.info(f"Running speaker diarization on {audio_to_process}...")
    diarization_result = diarization_pipeline(audio_to_process)
    logging.info("Diarization complete.")

    # B. Transcription with Word Timestamps
    logging.info(f"Running transcription on {audio_to_process}...")
    segments, _ = asr_model.transcribe(audio_to_process, word_timestamps=True)
    
    all_words = []
    for segment in segments:
        for word in segment.words:
            all_words.append(word._asdict())
    logging.info("Transcription complete.")

    # --- 4. Merge Results and Save ---
    def get_words_in_segment(word_list, start_time, end_time):
        segment_words = []
        for word_info in word_list:
            if (word_info['start'] >= start_time and word_info['start'] <= end_time) or \
               (word_info['end'] >= start_time and word_info['end'] <= end_time):
                segment_words.append(word_info['word'])
        return ' '.join(word.strip() for word in segment_words)

    logging.info(f"Writing transcript to {output_transcript_file}...")
    with open(output_transcript_file, "w") as f:
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            segment_text = get_words_in_segment(all_words, start_time, end_time)
            if segment_text:
                f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {segment_text}\n")
    
    logging.info("--- End of Transcription ---")
    
    # Clean up the mono file if it was created
    if audio_to_process == mono_audio_file:
        os.remove(mono_audio_file)
        logging.info(f"Removed temporary mono file: {mono_audio_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    transcribe_audio(audio_file_path)
