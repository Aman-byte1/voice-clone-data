#!/bin/bash
# ============================================================================
# Voice Clone Dataset Pipeline — French Train + Test Splits
#
# Generates French cloned voice audio for the ymoslem/acl-6060 dataset:
#   - Merges both splits (884 total)
#   - Shuffles with seed 0
#   - Test: 100 samples, Train: 784 samples
#   - TTS model: nvidia/magpie_tts_multilingual_357m (MagpieTTS)
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
echo "[1/4] Installing dependencies..."
$PY -m pip install six python-dateutil --force-reinstall
# Only install what generate_french_dataset.py + push_to_hub.py need
$PY -m pip install chatterbox-tts torchaudio pandas huggingface_hub soundfile tqdm datasets

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
echo ""
echo "[3/4] Generating French dataset (train + test)..."
echo "       TTS Model: nvidia/magpie_tts_multilingual_357m"
echo "       Test: 100 samples, Train: 784 samples"
$PY generate_french_dataset.py \
    --output_dir ./output/acl6060_fr \
    --device cuda

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
