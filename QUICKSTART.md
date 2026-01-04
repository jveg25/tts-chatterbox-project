# 🚀 Quickstart Guide

Follow these steps to get Chatterbox TTS Studio up and running on your machine.

## 📋 Prerequisites

-   **Python 3.10 or 3.11**: Stable versions for the current dependencies.
-   **FFmpeg**: Required for MP3 conversion (via `pydub`).
    *   *Mac*: `brew install ffmpeg`
-   **Git**: To clone the repository.

## 🛠️ Installation

### 1. Environment Setup
Create and activate a virtual environment to keep your system clean:

```bash
# Create venv
python3.10 -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies
We need to install some core libraries first to ensure proper building on MacOS:

```bash
# Setup build tools
pip install "numpy<2" cython setuptools wheel

# Install pkuseg without isolation (recommended for MacOS)
pip install pkuseg --no-build-isolation

# Install everything else
pip install -r requirements.txt
```

### 3. Configure `.env`
Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
```

Open `.env` and configure your settings:

```text
HF_TOKEN=your_huggingface_token_here
PYTORCH_ENABLE_MPS_FALLBACK=1
```
*Note: The `HF_TOKEN` is optional but recommended for downloading models from HuggingFace.*

## 🏃 Usage

### Start the GUI
```bash
python src/frontend/app.py
```
Wait for the message: `Running on local URL: http://127.0.0.1:7860`. Open that link in your browser.

### Typical Workflow

1.  **Voice Cloning**: Upload a clear audio clip (minimum 5 seconds) in the "Voice Cloning" accordion.
2.  **Input Text**: 
    *   Paste text into the "Input Text" box.
    *   **OR** Upload a `.txt` file to have the output inherit its name.
3.  **Adjust Parameters**:
    *   For the Turbo model, most parameters are preset, but you can choose between `.wav` and `.mp3`.
4.  **Generate**: Click **Generate Speech** and wait for the "S3 Token" progress bars in your terminal to finish.

## 🍎 Mac GPU (MPS) Support
The app is configured to automatically use the Apple Silicon GPU. If it feels slow, verify GPU support with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
It should return `True`.

## 📂 Locating Outputs
All generated audio files are stored in the `outputs/` directory in the project root.
