# Voice Clone Dataset Pipeline

Generate multilingual voice cloning datasets using the **ACL 60/60** dataset and **Scicom Multilingual-TTS-1.7B-Base**.

## Quick Start (Server)

```bash
git clone https://github.com/Aman-byte1/voice-clone-data.git && \
cd voice-clone-data && \
bash run.sh
```

This will:
1. Install dependencies
2. Run voice cloning (French, Chinese, Arabic) on ACL 60/60
3. Push the dataset to HuggingFace

## Scripts

| Script | Description |
|--------|-------------|
| `clone_with_scicom.py` | **TRUE voice cloning** using Scicom Multilingual-TTS-1.7B-Base (150+ languages, Arabic support) |
| `generate_synthetic_dataset.py` | Fully synthetic TTS using MagpieTTS 357M (5 speakers × 9 languages) |
| `clone_acl6060_voices.py` | ACL 60/60 with MagpieTTS fixed speakers (de, fr, ja, zh only) |
| `push_to_hub.py` | Push generated datasets to HuggingFace Hub |
| `run.sh` | End-to-end server pipeline |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- HuggingFace account & token

## Manual Usage

```bash
pip install -r requirements.txt

# Voice cloning (recommended)
python clone_with_scicom.py --target_languages fr,zh,ar --split dev

# Push to HuggingFace
export HF_TOKEN=hf_xxxxx
python push_to_hub.py --output_dir ./output/acl6060_scicom --repo_name YOUR_USERNAME/dataset-name
```

## Source Data

- **ACL 60/60**: [ymoslem/acl-6060](https://huggingface.co/datasets/ymoslem/acl-6060) — Multilingual speech translations of ACL 2022 presentations
- **Scicom TTS**: [Scicom-intl/Multilingual-TTS-1.7B-Base](https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base) — 150+ language TTS with voice cloning
- **MagpieTTS**: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) — 9 language TTS with 5 fixed speakers
