#!/bin/bash
# ============================================================================
# Voice Clone Dataset Pipeline — French Train + Test Splits
#
# Generates French cloned voice audio for the ymoslem/acl-6060 dataset:
#   - Merges both splits (884 total)
#   - Shuffles with seed 0
#   - Test: 100 samples, Train: 784 samples
#   - TTS model: resemble-ai/chatterbox (Chatterbox AI)
#
# Usage (on server):
#   bash run_french_dataset.sh
# ============================================================================

set -e

# ── Auto-detect Python binary ─────────────────────────────────────────────
# Prefer specific versioned python (e.g. python3.13) over generic python3
if command -v python3.13 &> /dev/null; then
    PY=python3.13
elif command -v python3.12 &> /dev/null; then
    PY=python3.12
elif command -v python3.11 &> /dev/null; then
    PY=python3.11
elif command -v python3.10 &> /dev/null; then
    PY=python3.10
elif command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
else
    echo "ERROR: No python found on PATH"
    exit 1
fi
echo "Using Python: $PY ($($PY --version))"

echo "============================================"
echo "  French Voice Cloning — Full Pipeline"
echo "============================================"

# ── 1. Install dependencies ────────────────────────────────────────────────
echo ""
echo "[1/4] Installing system dependencies (ffmpeg)..."
if command -v apt-get &> /dev/null; then
    SUDO_CMD=""
    if command -v sudo &> /dev/null; then SUDO_CMD="sudo"; fi
    $SUDO_CMD apt-get update && $SUDO_CMD apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
fi

echo "[1/4] Installing python dependencies..."
$PY -m pip install six python-dateutil --force-reinstall
# Install chatterbox without its strict numpy pin (incompatible with Python 3.13)
$PY -m pip install chatterbox-tts --no-deps
# Install actual runtime deps separately (uses system-compatible versions)
# NOTE: Pinning datasets < 3.2.0 to avoid mandatory torchcodec dependency
# NOTE: Pinning transformers==4.46.3 as strictly required by chatterbox-tts 0.1.6
$PY -m pip install torch torchaudio numpy pandas huggingface_hub soundfile tqdm "datasets<3.2.0" \
    "transformers==4.46.3" safetensors tokenizers conformer resemble-perth \
    s3tokenizer diffusers pykakasi spacy-pkuseg gradio librosa \
    soundfile omegaconf pyloudnorm

# ── 2. Set HuggingFace token ──────────────────────────────────────────────
echo ""
echo "[2/4] Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Please run:"
    echo "  export HF_TOKEN=hf_YOUR_TOKEN_HERE"
    echo "Then re-run this script."
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
$PY -c "from huggingface_hub import login; login(token='$HF_TOKEN')" || true
echo "✓ HuggingFace token configured"

# ── 3. Generate train + test splits ──────────────────────────────────────
# Default sample counts (Full scale)
NUM_TEST=${1:-100}
NUM_TRAIN=${2:-784}
NUM_WORKERS=${3:-4}

echo "       Test: $NUM_TEST samples, Train: $NUM_TRAIN samples"
echo "       Parallel workers: $NUM_WORKERS"

# Auto-detect device
DEVICE="cuda"
if ! $PY -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "⚠ CUDA not available or NVIDIA drivers missing. Falling back to CPU."
    DEVICE="cpu"
else
    echo "✓ CUDA detected. Using GPU acceleration."
fi

$PY generate_french_dataset.py \
    --output_dir ./output/acl6060_fr \
    --num_test "$NUM_TEST" \
    --num_train "$NUM_TRAIN" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE"

echo "✓ Generation complete"

# ── 4. Push to HuggingFace ──────────────────────────────────────────────
echo ""
echo "[4/4] Pushing dataset to HuggingFace..."
$PY push_to_hub.py \
    --output_dir ./output/acl6060_fr \
    --repo_name amanuelbyte/acl6060-voice-cloning-fr

echo "✓ Push complete"

echo ""
echo "============================================"
echo "  ✓ Pipeline complete!"
echo "  Dataset: https://huggingface.co/datasets/amanuelbyte/acl6060-voice-cloning-fr"
echo "============================================"
