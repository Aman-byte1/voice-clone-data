#!/bin/bash
# ============================================================================
# Voice Clone Dataset Pipeline — Server Run Script
#
# This script does everything end-to-end:
#   1. Clones the repo
#   2. Installs dependencies
#   3. Runs the voice cloning script (Scicom model on ACL 60/60)
#   4. Pushes the resulting dataset to HuggingFace
#
# Usage (copy-paste this entire block to your server):
#
#   git clone https://github.com/Aman-byte1/voice-clone-data.git && \
#   cd voice-clone-data && \
#   bash run.sh
#
# ============================================================================

set -e  # Exit on any error

echo "============================================"
echo "  Voice Clone Dataset Pipeline"
echo "============================================"

# ── 1. Install dependencies ────────────────────────────────────────────────
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

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
huggingface-cli login --token "$HF_TOKEN" || true
echo "✓ HuggingFace token configured"

# ── 3. Run voice cloning ──────────────────────────────────────────────────
echo ""
echo "[3/4] Running voice cloning (Scicom Multilingual-TTS on ACL 60/60)..."
echo "       Target languages: fr, zh, ar"
echo "       This may take a while..."

python clone_with_scicom.py \
    --output_dir ./output/acl6060_scicom \
    --target_languages fr,zh,ar \
    --split dev \
    --device cuda \
    --temperature 0.8

echo "✓ Voice cloning complete"

# ── 4. Push dataset to HuggingFace ─────────────────────────────────────────
echo ""
echo "[4/4] Pushing dataset to HuggingFace..."

python push_to_hub.py \
    --output_dir ./output/acl6060_scicom \
    --repo_name amanuelbyte/acl6060-voice-cloning-multilingual

echo ""
echo "============================================"
echo "  ✓ Pipeline complete!"
echo "  Dataset: https://huggingface.co/datasets/amanuelbyte/acl6060-voice-cloning-multilingual"
echo "============================================"
