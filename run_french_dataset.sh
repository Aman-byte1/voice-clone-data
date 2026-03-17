#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  run.sh — set up environment and generate French cloned dataset
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration (edit these) ────────────────────────────────
OUTPUT_DIR="./output/acl6060_fr"
DEVICE="cuda"                 # "cuda" or "cpu"
LANG="fr"
CFG=0.5                       # CFG weight for accent control
SAVE_EVERY=10                 # checkpoint CSV every N clips
REPO_NAME="amanuelbyte/acl-voice-cloning-fr-data"                  # e.g. "your-user/your-dataset" (leave empty to skip uploads)

# ── Setup ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  Setting up environment"
echo "══════════════════════════════════════════════════════════"

# Create and activate venv (skip if already in one)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment…"
        python3 -m venv .venv
    fi
    echo "Activating virtual environment…"
    source .venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (adjust cu121/cu124 to match your driver)
if [ "$DEVICE" = "cuda" ]; then
    echo "Installing PyTorch with CUDA support…"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install remaining dependencies
echo "Installing dependencies…"
pip install -r requirements.txt

# ── Verify GPU ────────────────────────────────────────────────
if [ "$DEVICE" = "cuda" ]; then
    echo ""
    python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {name} ({mem:.1f} GB)')
else:
    print('  ⚠ CUDA not available — will fall back to CPU')
"
fi

# ── Run ───────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Starting generation"
echo "══════════════════════════════════════════════════════════"

CMD="python3 generate_french_dataset.py \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --lang $LANG \
    --cfg $CFG \
    --save_every $SAVE_EVERY"

# Add repo_name if set
if [ -n "$REPO_NAME" ]; then
    CMD="$CMD --repo_name $REPO_NAME"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✓ Pipeline complete"
echo "  Output: $OUTPUT_DIR"
echo "══════════════════════════════════════════════════════════"
