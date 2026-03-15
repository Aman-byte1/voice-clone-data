"""
Shared utilities for voice clone dataset creation.
Provides MagpieTTS model loading, audio I/O helpers, and language mappings.
"""

import os
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# ─── Language & Speaker Constants ───────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "vi": "Vietnamese",
    "it": "Italian",
    "zh": "Chinese",
    "hi": "Hindi",
    "ja": "Japanese",
}

# Languages that support text normalization in MagpieTTS
TN_SUPPORTED_LANGUAGES = {"en", "es", "de", "fr", "it", "zh"}

SPEAKER_MAP = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}

SPEAKER_NAMES = list(SPEAKER_MAP.keys())


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_chatterbox_model(device: str = "cuda"):
    """
    Load the ChatterboxTTS model from HuggingFace, with caching.

    Args:
        device: Device to load model onto ('cuda' or 'cpu').

    Returns:
        The loaded ChatterboxTTS instance.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    import torch
    # Handle CPU monkeypatch if needed
    if device == "cpu" and not hasattr(torch.load, "_is_patched"):
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        patched_load._is_patched = True
        torch.load = patched_load

    from chatterbox.tts import ChatterboxTTS

    print("Loading ChatterboxTTS model from HuggingFace...")
    # Fix for SDPA conflict in some environments
    try:
        model = ChatterboxTTS.from_pretrained(device=device, attn_implementation="eager")
    except TypeError:
        model = ChatterboxTTS.from_pretrained(device=device)
    
    _cached_model = model
    print("Model loaded successfully.")
    return model


# ─── TTS Generation ────────────────────────────────────────────────────────────

def generate_speech(
    model,
    text: str,
    language: str,
    audio_prompt_path: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech audio from text using ChatterboxTTS.

    Args:
        model: Loaded ChatterboxTTS instance.
        text: Text transcript to synthesize.
        language: Language code (e.g., 'en', 'fr', 'zh').
        audio_prompt_path: Optional path to audio for voice cloning.

    Returns:
        Tuple of (audio_numpy_array, sample_rate).
    """
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        language_id=language,
    )

    # Convert to numpy if it's a torch tensor
    if hasattr(wav, "cpu"):
        audio_np = wav.cpu().numpy().squeeze()
    else:
        audio_np = np.array(wav).squeeze()

    return audio_np, model.sr


# ─── Audio I/O ──────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 22050  # MagpieTTS uses NanoCodec at 22kHz


def save_audio(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """
    Save audio numpy array to a WAV file.

    Args:
        audio: Audio data as numpy array.
        filepath: Output file path.
        sample_rate: Sample rate in Hz.

    Returns:
        The absolute path of the saved file.
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, audio, sample_rate)
    return os.path.abspath(filepath)


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from a WAV file.

    Args:
        filepath: Path to the audio file.

    Returns:
        Tuple of (audio_numpy_array, sample_rate).
    """
    audio, sr = sf.read(filepath)
    return audio, sr


# ─── Path Helpers ───────────────────────────────────────────────────────────────

def make_audio_filename(
    speaker: str,
    language: str,
    index: int,
    ext: str = "wav",
) -> str:
    """
    Generate a standardized audio filename.

    Args:
        speaker: Speaker name.
        language: Language code.
        index: Sample index.
        ext: File extension.

    Returns:
        Formatted filename string.
    """
    return f"{speaker.lower()}_{language}_{index:05d}.{ext}"


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path
