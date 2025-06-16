# **ODIN: Observational Diarization and Insight Network**

ODIN is an open-source pipeline designed to process audio and video recordings, automatically generating speaker-labeled transcripts and narrative summaries. It transforms raw conversational data into structured, actionable reports, streamlining the documentation process for law enforcement and other professional fields.

## **Planned Architecture**

ODIN is designed as a modular system that watches for new media files, processes them through an AI pipeline, and provides an interface for final review and editing. The core components will be:

  * **config.yml**: The central configuration file. This will manage settings such as input/output directories, model selections (e.g., which Ollama model to use for report generation), and other operational parameters.
  * **main.py**: The main processing engine. This script will act as a daemon or service, watching the configured input folder for new audio/video files. It uses `pyannote.audio` for speaker diarization and `faster-whisper` for transcription. It will then pass the final annotated transcript to a large language model for summary generation.
  * **interface.py**: A graphical user interface (GUI) for end-users. This application will allow for easy uploading of media files and will provide a user-friendly editor to review the final output from `main.py`. Users can correct speaker assignments (e.g., map `SPEAKER_01` to "Officer Smith"), edit the transcribed text, and save the final, polished report.

## **Core Dependencies**

This project relies on a modern, robust stack for audio processing:

  * **PyTorch**: The underlying deep learning framework for all AI models.
  * **pyannote.audio**: A state-of-the-art speaker diarization library used to identify *who* spoke and *when*.
  * **faster-whisper**: A highly optimized implementation of OpenAI's Whisper ASR model for fast and accurate transcription to determine *what* was said.

## **Installation**

### Prerequisites

  * Python 3.9+
  * An NVIDIA GPU with the appropriate CUDA drivers installed.

### Step-by-Step Setup
0. `sudo apt update && sudo apt install ffmpeg`

1.  **Create and Activate a Virtual Environment:**

    ```
    # Create a virtual environment in a folder named .venv
    python3 -m venv pyenv

    # Activate the environment
    source pyenv/bin/activate
    ```

2.  **Install PyTorch with CUDA Support:**
    Install the stable version of PyTorch that matches your system's CUDA toolkit. The following command is for a system with CUDA 12.1 drivers.

    ```
    # For CUDA 12.1 - check the PyTorch website for other versions
    pip install torch torchvision torchaudio
    pip install Flask Werkzeug requests moviepy
    ```

3.  **Install Project Libraries:**
    Install `pyannote.audio` and `faster-whisper`.

    ```bash
    pip install pyannote.audio faster-whisper
    ```

4.  **Hugging Face Authentication (Required for Pyannote):**
    The `pyannote/speaker-diarization-3.1` model requires you to accept user terms on its Hugging Face page.

      * Visit the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) page and agree to the terms.
      * Visit the [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) page and agree to the terms.
      * Log in via the command line with a Hugging Face access token to authenticate.

    <!-- end list -->

    ```
    pip install huggingface_cli
    huggingface-cli login
    ```

## **Usage**

*This section will be updated as the project develops.*

Currently, the core processing logic exists in a single Python script.

1.  Place the developed Python code into a file named `main.py`.
2.  Modify the `ORIGINAL_AUDIO_FILE` variable in the script to point to your desired audio file.
3.  Run the script from your terminal:
    ```bash
    source ./pyenv/bin/activate
    #Edit with your install path
    export LD_LIBRARY_PATH=/home/hamner/ODIN/pyenv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
    python main.py
    ```
4.  The script will automatically handle stereo-to-mono conversion and print the final, speaker-labeled transcript to the console.
      * **Note:** The first time you run the script, the required AI models (pyannote and Whisper) will be downloaded. This may take some time and requires an internet connection. Subsequent runs will be much faster.

## **Contributing**

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please refer to the project's issue tracker for areas where you can help.

## **License**

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) file for more details.
