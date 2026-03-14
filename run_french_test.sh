#!/bin/bash
# ============================================================================
# Voice Clone Dataset Pipeline — Server Run Script (French 100-sample test)
#
# Usage (on your server):
#   bash run_french_test.sh
#
# ============================================================================

set -e  # Exit on any error

echo "============================================"
echo "  Voice Clone French Test Generation"
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

# ── 3. Run voice cloning for 100 samples ─────────────────────────────────
echo ""
echo "[3/4] Generating French Test Dataset (100 samples)..."
python generate_french_test.py \
    --output_dir ./output/acl6060_test \
    --device cuda 

echo "✓ Generation complete"

# ── 4. Push to HuggingFace ──────────────────────────────────────────────
echo ""
echo "[4/4] Pushing dataset to HuggingFace..."
python push_to_hub.py \
    --output_dir ./output/acl6060_test \
    --repo_name amanuelbyte/acl6060-voice-cloning-fr-test

echo "✓ Push complete"

echo ""
echo "============================================"
echo "  ✓ Pipeline complete!"
echo "  Dataset: https://huggingface.co/datasets/amanuelbyte/acl6060-voice-cloning-fr-test"
echo "============================================"
