# 🎙️ Chatterbox TTS Studio

Chatterbox TTS Studio is a powerful, local Text-to-Speech (TTS) application built using Resemble AI's **Chatterbox-Turbo** model. It features a modern **Gradio** interface designed for high-quality speech synthesis, zero-shot voice cloning, and batch processing.

## ✨ Key Features

-   **Zero-Shot Voice Cloning**: Clone any voice with just a 5-10 second reference audio clip.
-   **Dual Processing Modes**:
    *   **Single Text**: Type directly or upload a single `.txt` file for immediate playback.
    *   **Batch Processing**: Upload multiple `.txt` files and generate audio for all of them at once.
-   **Natural Pacing**: Custom silence injection algorithm that adds realistic pauses after periods, commas, and semicolons.
-   **Smart Naming**: Generated files automatically inherit the name of your input text files.
-   **GPU Acceleration**: Fully optimized for **Apple Silicon (M1/M2/M3)** via Metal Performance Shaders (MPS).
-   **Multi-Format Support**: Save your audio in high-quality `.wav` or compressed `.mp3`.

## 🛠️ Tech Stack

-   **Model**: [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) by Resemble AI.
-   **Frontend**: Gradio.
-   **Audio Engine**: PyTorch (MPS/CUDA), Librosa, SoundFile, Pydub.
-   **Backend**: Python 3.10+.

## ⚙️ Project Structure

```text
.
├── src/
│   ├── backend/         # TTS logic, chunking, and audio processing
│   └── frontend/        # Gradio Blocks UI
├── outputs/             # Generated audio files
├── tests/               # Unit tests
├── .env                 # Configuration (tokens, GPU settings)
└── requirements.txt     # Dependency list
```

## 🚦 Getting Started

For detailed installation steps, check out the [**Quickstart Guide**](./QUICKSTART.md).

1.  **Clone and Setup**:
    ```bash
    git clone <your-repo-url>
    cd 09_tts_project
    python3.10 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    python src/frontend/app.py
    ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The underlying Chatterbox model is open-sourced by Resemble AI. Please refer to their repository for specific model licensing.

---
Developed with ❤️ by [Your Name/GitHub Handle]
