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

_cached_model = None


def load_magpie_model(device: str = "cuda"):
    """
    Load the MagpieTTS model from HuggingFace, with caching to avoid reloading.

    Args:
        device: Device to load model onto ('cuda' or 'cpu').

    Returns:
        The loaded MagpieTTSModel instance.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    from nemo.collections.tts.models import MagpieTTSModel

    print("Loading MagpieTTS model from HuggingFace...")
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model = model.to(device)
    model.eval()
    _cached_model = model
    print("Model loaded successfully.")
    return model


# ─── TTS Generation ────────────────────────────────────────────────────────────

def generate_speech(
    model,
    text: str,
    language: str,
    speaker: str = "Sofia",
    apply_tn: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech audio from text using MagpieTTS.

    Args:
        model: Loaded MagpieTTSModel instance.
        text: Text transcript to synthesize.
        language: Language code (e.g., 'en', 'fr', 'zh').
        speaker: Speaker name (John, Sofia, Aria, Jason, Leo).
        apply_tn: Whether to apply text normalization (only supported for en, es, de, fr, it, zh).

    Returns:
        Tuple of (audio_numpy_array, audio_length).

    Raises:
        ValueError: If language or speaker is not supported.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language '{language}' not supported. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    if speaker not in SPEAKER_MAP:
        raise ValueError(
            f"Speaker '{speaker}' not supported. "
            f"Supported: {SPEAKER_NAMES}"
        )

    # Only apply TN for languages that support it
    use_tn = apply_tn and (language in TN_SUPPORTED_LANGUAGES)

    speaker_idx = SPEAKER_MAP[speaker]
    audio, audio_len = model.do_tts(
        text,
        language=language,
        apply_TN=use_tn,
        speaker_index=speaker_idx,
    )

    # Convert to numpy
    audio_np = audio.cpu().numpy().squeeze()
    length = audio_len.item() if hasattr(audio_len, "item") else int(audio_len)

    return audio_np[:length], length


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
