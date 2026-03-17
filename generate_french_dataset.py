#!/usr/bin/env python3
"""
generate_french_dataset.py
==========================
Generate a French cloned-voice dataset with speaker-based train / test splits.

Source
------
ymoslem/acl-6060  (dev 468 + eval 416 = 884 utterances, 10 speakers)

Splitting strategy
------------------
• TEST  – the 2 speakers with the fewest utterances
          (Letitia 56 + Michał 66 = 122 clips)
• TRAIN – the remaining 8 speakers (762 clips)

Original row order (dev → eval) is preserved within each split.

Key design choices
------------------
• Fully sequential processing — no threading.  This eliminates OOM spikes,
  row-mixing, and the false parallelism that a GPU lock creates anyway.
• Aggressive VRAM cleanup (empty_cache + gc.collect) after every clip.
• Incremental CSV checkpoint every N clips with full resume support.
• Deterministic filenames based on original split + index (e.g. cloned_dev_107_fr.wav)
  so they stay stable across resume runs.

Usage
-----
    python generate_french_dataset.py \
        --output_dir ./output/acl6060_fr \
        --device cuda \
        --cfg 0.5 \
        --save_every 10

    # With incremental HF uploads (requires push_to_hub.py alongside):
    python generate_french_dataset.py \
        --output_dir ./output/acl6060_fr \
        --device cuda \
        --repo_name your-username/your-dataset
"""

import argparse
import gc
import os
import sys
import subprocess
import tempfile
import traceback

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio as ta
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# Constants & Speaker Map
# ══════════════════════════════════════════════════════════════════════════════

DATASET_NAME = "ymoslem/acl-6060"
MODEL_NAME = "resemble-ai/chatterbox-multilingual"

# (original_split, start_idx, end_idx, speaker_num, speaker_name)
# Ranges are inclusive: start_idx..end_idx
SPEAKER_MAP = [
    # ── DEV split ──
    ("dev",    0,  106,  1, "Elena"),        # 107 utterances
    ("dev",  107,  188,  2, "Antoine"),      #  82
    ("dev",  189,  254,  3, "Michał"),       #  66
    ("dev",  255,  362,  4, "Jiawei"),       # 108
    ("dev",  363,  467,  5, "Bhargavi"),     # 105
    # ── EVAL split ──
    ("eval",   0,   99,  6, "Allan"),        # 100
    ("eval", 100,  183,  7, "Antoine_MUV"),  #  84
    ("eval", 184,  239,  8, "Letitia"),      #  56
    ("eval", 240,  329,  9, "Kamezawa"),     #  90
    ("eval", 330,  415, 10, "Asaf"),         #  86
]


# ══════════════════════════════════════════════════════════════════════════════
# Speaker Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sid(num: int) -> str:
    """Formatted speaker ID string."""
    return f"speaker_{num:02d}"


def get_speaker(split: str, idx: int):
    """Return (speaker_id, speaker_name) for a positional index in a split."""
    for s, lo, hi, num, name in SPEAKER_MAP:
        if s == split and lo <= idx <= hi:
            return _sid(num), name
    return "speaker_unknown", "Unknown"


def auto_test_speakers(n: int = 2) -> list[str]:
    """Return the speaker IDs of the N speakers with the fewest utterances."""
    sizes: dict[str, tuple[int, str]] = {}
    for _, lo, hi, num, name in SPEAKER_MAP:
        sid = _sid(num)
        sizes[sid] = (hi - lo + 1, name)

    ranked = sorted(sizes.items(), key=lambda kv: kv[1][0])
    chosen = [sid for sid, _ in ranked[:n]]

    print("  Auto-selected test speakers (smallest):")
    for sid, (cnt, name) in ranked[:n]:
        print(f"    {sid}  {name:<15s}  {cnt} utterances")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# Filesystem Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_key(key: str) -> str:
    """Convert resume key 'dev:107' → 'dev_107' for safe filenames."""
    return key.replace(":", "_")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_with_speakers():
    """Load both splits, annotate with speaker metadata, concatenate in order."""
    print(f"Loading dataset: {DATASET_NAME}")
    ds_dev = load_dataset(DATASET_NAME, split="dev")
    ds_eval = load_dataset(DATASET_NAME, split="eval")
    print(f"  dev={len(ds_dev)}  eval={len(ds_eval)}  total={len(ds_dev) + len(ds_eval)}")

    def annotate(row, idx, split):
        sid, sname = get_speaker(split, idx)
        row["speaker_id"] = sid
        row["speaker_name"] = sname
        row["original_split"] = split
        row["original_index"] = idx
        row["_key"] = f"{split}:{idx}"
        return row

    ds_dev = ds_dev.map(lambda r, i: annotate(r, i, "dev"), with_indices=True)
    ds_eval = ds_eval.map(lambda r, i: annotate(r, i, "eval"), with_indices=True)
    return concatenate_datasets([ds_dev, ds_eval])


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(device: str):
    """Load ChatterboxMultilingualTTS. Returns (model, actual_device_str)."""
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠  CUDA not available — falling back to CPU")
        device = "cpu"

    # ── Perth watermarker workaround (RunPod / Python 3.13) ──
    try:
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            class _Dummy:
                def __init__(self, *a, **k):
                    pass
                def __call__(self, audio, *a, **k):
                    return audio
            perth.PerthImplicitWatermarker = _Dummy
            print("⚠  Patched broken perth watermarker")
    except ImportError:
        pass

    # ── CPU torch.load fix ──
    if device == "cpu":
        _orig_load = torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("map_location", "cpu")
            return _orig_load(*a, **kw)
        torch.load = _patched_load

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"Loading {MODEL_NAME} on {device}…")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("✓ Model loaded\n")
    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# Audio Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(audio_field):
    """Extract (np.float32 array, sample_rate) from a HF audio column value."""
    if audio_field is None:
        return None, None
    try:
        if isinstance(audio_field, dict):
            return (
                np.asarray(audio_field["array"], dtype=np.float32),
                audio_field["sampling_rate"],
            )
        # Object-style access (e.g. AudioDecoder)
        arr = getattr(audio_field, "array", None)
        sr = getattr(audio_field, "sampling_rate", None)
        if arr is not None and sr is not None:
            return np.asarray(arr, dtype=np.float32), sr
    except Exception:
        pass
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Core Generation Loop
# ══════════════════════════════════════════════════════════════════════════════

def generate_split(
    ds,
    split_name: str,
    output_dir: str,
    model,
    device: str,
    lang: str = "fr",
    cfg: float = 0.5,
    save_every: int = 10,
    repo_name: str | None = None,
):
    """Generate cloned audio for one split.  Fully sequential & resumable."""
    split_dir = _ensure(os.path.join(output_dir, split_name))
    en_dir = _ensure(os.path.join(split_dir, "original_audio_en"))
    fr_dir = _ensure(os.path.join(split_dir, "cloned_audio_fr"))
    csv_path = os.path.join(split_dir, "metadata.csv")

    # ── Resume: collect already-completed rows ──────────────────────────
    done: dict[str, dict] = {}  # _key → record
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path, dtype=str).fillna("")
            for _, r in old_df.iterrows():
                rd = r.to_dict()
                fr_rel = rd.get("cloned_audio_fr", "")
                if fr_rel and os.path.isfile(os.path.join(split_dir, fr_rel)):
                    done[rd["_key"]] = rd
            print(f"  ↻ Resume [{split_name}]: {len(done)}/{len(ds)} clips already done")
        except Exception as e:
            print(f"  ⚠ Resume failed ({e}); starting fresh")
            done = {}

    total = len(ds)
    records: list[dict] = []
    new_count = 0

    pbar = tqdm(
        total=total,
        initial=len(done),
        desc=split_name,
        unit="clip",
        ncols=100,
        ascii=True,
    )

    for pos in range(total):
        row = ds[pos]
        key: str = row["_key"]

        # ── Already completed → reuse old record ──
        if key in done:
            records.append(done[key])
            continue

        # ── Build new record ──
        rec = {
            "_key":            key,
            "speaker_id":      row["speaker_id"],
            "speaker_name":    row["speaker_name"],
            "original_split":  row["original_split"],
            "original_index":  int(row["original_index"]),
            "text_en":         row.get("text_en", ""),
            "text_fr":         row.get("text_fr", ""),
            "audio_en":        "",
            "cloned_audio_fr": "",
            "error":           "",
        }

        audio_arr, sr = extract_audio(row.get("audio"))
        text_fr = row.get("text_fr", "")

        if audio_arr is None or sr is None or len(audio_arr) == 0:
            rec["error"] = "missing or empty source audio"
            records.append(rec)
            pbar.update(1)
            continue

        if not text_fr.strip():
            rec["error"] = "missing French text"
            records.append(rec)
            pbar.update(1)
            continue

        # Deterministic filenames based on original identity
        safe = _safe_key(key)
        en_fname = f"original_{safe}_en.wav"
        fr_fname = f"cloned_{safe}_fr.wav"
        en_path = os.path.join(en_dir, en_fname)
        fr_path = os.path.join(fr_dir, fr_fname)

        try:
            # Save English reference audio
            sf.write(en_path, audio_arr, sr)
            rec["audio_en"] = os.path.relpath(en_path, split_dir)

            # Temp file for model prompt
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(tmp, audio_arr, sr)

                # ── Inference (single clip, no parallelism) ──
                with torch.no_grad():
                    wav = model.generate(
                        text_fr,
                        audio_prompt_path=tmp,
                        language_id=lang,
                        cfg_weight=max(cfg, 0.001),  # avoid exact 0.0
                    )

                # Ensure correct shape for torchaudio.save
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)

                ta.save(fr_path, wav, model.sr)
                rec["cloned_audio_fr"] = os.path.relpath(fr_path, split_dir)

                del wav
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

        except Exception:
            rec["error"] = traceback.format_exc()

        # ── Aggressive memory cleanup after every clip ──
        del audio_arr
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        records.append(rec)
        new_count += 1
        pbar.update(1)

        # ── Periodic checkpoint ──
        if save_every and new_count % save_every == 0:
            _save(records, csv_path)
            if repo_name:
                _push(output_dir, repo_name)

    pbar.close()

    # ── Final save ──
    _save(records, csv_path)
    _save_json(records, os.path.join(split_dir, "metadata.json"))

    ok = sum(1 for r in records if r.get("cloned_audio_fr"))
    errors = sum(1 for r in records if r.get("error"))
    print(f"  ✓ [{split_name}] {ok}/{total} clips generated  ({errors} errors)")


def _save(records: list[dict], path: str):
    """Save records to CSV (preserves insertion order = dataset order)."""
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def _save_json(records: list[dict], path: str):
    """Save records to JSON."""
    df = pd.DataFrame(records)
    df.to_json(path, orient="records", force_ascii=False, indent=2)


def _push(output_dir: str, repo_name: str):
    """Call push_to_hub.py if it exists alongside this script."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_to_hub.py")
    if not os.path.isfile(script):
        return
    try:
        subprocess.run(
            [sys.executable, script, "--output_dir", output_dir, "--repo_name", repo_name],
            check=True,
            timeout=300,
        )
        print(f"  ↑ Pushed to {repo_name}")
    except Exception as e:
        print(f"  ⚠ Push failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate French cloned-voice dataset with Chatterbox Multilingual TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", default="./output/acl6060_fr",
                    help="Root output directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Inference device")
    p.add_argument("--lang", default="fr",
                    help="Target language ID")
    p.add_argument("--cfg", type=float, default=0.5,
                    help="CFG weight (0.5 recommended for accent control)")
    p.add_argument("--save_every", type=int, default=10,
                    help="Checkpoint CSV every N new clips")
    p.add_argument("--repo_name", default=None,
                    help="HuggingFace repo for incremental uploads")
    p.add_argument("--n_test_speakers", type=int, default=2,
                    help="Number of smallest speakers to put in test set")
    p.add_argument("--only", choices=["train", "test"], default=None,
                    help="Generate only one split (useful for debugging)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("  French Voice Cloning — Speaker-based Train / Test")
    print("=" * 64)
    print(f"  Dataset:    {DATASET_NAME}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Language:   {args.lang}")
    print(f"  CFG weight: {args.cfg}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 64)

    # 1. Determine which speakers go to test
    test_ids = auto_test_speakers(args.n_test_speakers)

    # 2. Load & annotate full dataset
    ds_all = load_dataset_with_speakers()

    # 3. Split by speaker (preserves original order within each split)
    ds_test = ds_all.filter(lambda x: x["speaker_id"] in test_ids)
    ds_train = ds_all.filter(lambda x: x["speaker_id"] not in test_ids)

    # Print speaker distribution
    print(f"\n  Train: {len(ds_train)} clips  |  Test: {len(ds_test)} clips")
    print()

    # 4. Load model
    model, device = load_model(args.device)

    # 5. Generate splits
    if args.only != "train":
        print(f"{'─' * 64}")
        print(f"  TEST split  ({len(ds_test)} clips)")
        print(f"{'─' * 64}")
        generate_split(
            ds_test, "test", args.output_dir, model, device,
            lang=args.lang, cfg=args.cfg, save_every=args.save_every,
            repo_name=args.repo_name,
        )
        if args.repo_name:
            _push(args.output_dir, args.repo_name)

    if args.only != "test":
        print(f"\n{'─' * 64}")
        print(f"  TRAIN split  ({len(ds_train)} clips)")
        print(f"{'─' * 64}")
        generate_split(
            ds_train, "train", args.output_dir, model, device,
            lang=args.lang, cfg=args.cfg, save_every=args.save_every,
            repo_name=args.repo_name,
        )
        if args.repo_name:
            _push(args.output_dir, args.repo_name)

    print(f"\n{'=' * 64}")
    print("  ✓ All done!")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
