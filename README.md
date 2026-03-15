# Chatterbox French Voice Cloning Pipeline

This repository contains an optimized pipeline for generating high-quality French voice cloning datasets. It uses the **Chatterbox Multilingual TTS** (0.5B Llama-based) model to clone voices from the English **ACL 60/60** dataset into French.

## 🚀 Quick Start (RunPod / Server)

If you are on a fresh GPU instance, run this one-liner to set up everything and start the generation:

```bash
git clone https://github.com/Aman-byte1/voice-clone-data.git && \
cd voice-clone-data && \
export HF_TOKEN=your_token_here && \
bash run_french_dataset.sh 100 784 8
```

## 🛠️ Key Features

- **High Speed**: Built-in multithreading (`ThreadPoolExecutor`) processes audio segments in parallel (typically ~2-4s per clip on a modern GPU).
- **Accent Mitigation**: Uses `cfg_weight=0.0` (with an internal stability workaround) to ensure the French output has a clean French accent, even when using an English reference clip.
- **Robust Audio Handling**: Specifically tailored for Python 3.12+ and modern GPU environments, bypassing common `torchcodec` and FFmpeg issues.
- **HuggingFace Integration**: Automatically pushes the generated audio and metadata to the HuggingFace Hub as a playable dataset.

## 📁 Project Structure

- `generate_french_dataset.py`: The core generation engine utilizing `ChatterboxMultilingualTTS`.
- `run_french_dataset.sh`: Orchestration script (installs dependencies, authenticates, and runs generation).
- `push_to_hub.py`: Robust uploader that handles metadata and audio feature casting.
- `legacy/`: Folder containing previous iterations and experimental scripts (Scicom, MagpieTTS, etc.).

## ⚙️ Detailed Usage

The `run_french_dataset.sh` script accepts three arguments:
```bash
bash run_french_dataset.sh [TEST_SAMPLES] [TRAIN_SAMPLES] [WORKERS]
```
- **TEST_SAMPLES**: Number of samples for the test split (default: 100).
- **TRAIN_SAMPLES**: Number of samples for the train split (default: 784).
- **WORKERS**: Number of parallel threads (default: 8).

## 📋 Requirements

- **GPU**: NVIDIA GPU (RTX 4000 Ada, L4, A100, etc.)
- **Python**: 3.10+
- **Core Dependencies**:
  - `chatterbox-tts == 0.1.6`
  - `transformers == 4.46.3` (Pinned for model compatibility)
  - `datasets < 3.2.0`
  - `soundfile`, `librosa`

## 🔗 Source Data & Models

- **Source Dataset**: [ymoslem/acl-6060](https://huggingface.co/datasets/ymoslem/acl-6060)
- **Model**: [Chatterbox Multilingual TTS](https://huggingface.co/resemble-ai/chatterbox-multilingual)
