import os, glob, io
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf

try:
    from IPython.display import Audio, display 
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False
    class Audio: 
        def __init__(self, *_a, **_k): ...
    def display(*_a, **_k): ...

_KOKORO = None
_SAMPLE_RATE = None


def _resolve_kokoro_assets(
    base: str = "~/.cache/kokoro_assets",
    env_model: str = "KOKORO_MODEL",
    env_voices: str = "KOKORO_VOICES",
) -> Tuple[str, str]:
    """Find model/voices either from env vars or in a cache dir."""
    base_dir = os.path.expanduser(base)

    def pick(patterns):
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(base_dir, pat)))
            if hits:
                return hits[-1]
        return None

    m = os.getenv(env_model) or pick(["*.onnx"])
    v = os.getenv(env_voices) or pick(["*voices*.*", "voices.*", "*.bin", "*.json"])

    if not (m and os.path.exists(m) and v and os.path.exists(v)):
        raise FileNotFoundError(
            "Kokoro assets not found.\n"
            f"Set {env_model} and {env_voices} env vars, "
            f"or put files in {base_dir}/"
        )
    return m, v


def _ensure_kokoro(model_path: str, voices_path: str):
    """Load Kokoro ONNX model once (module-level cache)."""
    global _KOKORO, __SAMPLE_RATE
    if _KOKORO is None:
        from kokoro_onnx import Kokoro, SAMPLE_RATE
        _KOKORO = Kokoro(model_path, voices_path)
        _SAMPLE_RATE = int(SAMPLE_RATE)
    return _KOKORO, _SAMPLE_RATE


def _to_mono_float32(y) -> np.ndarray:
    """Accepts list/tuple/np arrays, returns 1-D float32 mono waveform."""
    if isinstance(y, (list, tuple)) and len(y) > 0:
        y = y[0]
    a = np.asarray(y, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] in (1, 2):
        a = np.mean(a, axis=0).astype(np.float32)
    return a


def tts_kokoro(
    text: str,
    *,
    voice: str = "am_adam",
    speed: float = 1.05,
    sample_rate: Optional[int] = None,
    play: bool = False,
    assets_dir: str = "~/.cache/kokoro_assets",
    env_model: str = "KOKORO_MODEL",
    env_voices: str = "KOKORO_VOICES",
) -> Tuple[bytes, float]:
    """
    Synthesize speech with Kokoro and return (wav_bytes, duration_seconds).

    Args:
        text: Text to speak.
        voice: Kokoro voice id (e.g., 'am_adam').
        speed: Playback speed for generation.
        sample_rate: Output sample rate. If None, use kokoro SAMPLE_RATE.
        play: If True and IPython is present, auto-play in notebook.
        assets_dir: Where to look for model/voices if env vars are unset.
        env_model/env_voices: Env var names for explicit file paths.

    Returns:
        (wav_bytes, duration_seconds)
    """
    model_path, voices_path = _resolve_kokoro_assets(
        base=assets_dir, env_model=env_model, env_voices=env_voices
    )
    tts, sr_default = _ensure_kokoro(model_path, voices_path)
    sr = int(sample_rate or sr_default)

    y = tts.create(text, voice=voice, speed=speed)
    audio = _to_mono_float32(y)

    if play and _HAS_IPY:
        display(Audio(audio, rate=sr, autoplay=True))

    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sr, channels=1,
                      format="WAV", subtype="FLOAT") as f:
        f.write(audio)
    buf.seek(0)
    dur = sf.info(buf).duration
    return buf.getvalue(), float(dur)
