"""
synth.py - make fake sustained "aaah" voices with KNOWN properties
==================================================================
Used by the demo, the Streamlit app and the performance metrics. Because we choose the jitter,
shimmer and noise ourselves, we can check whether the measuring code recovers them (metrics.py).

These are computer-made sounds, not real people. A real voice is more complicated.
"""
import io
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050


def synth_vowel(f0, jitter, shimmer, noise, seconds, rng):
    """
    Build a fake 'aaah' one vocal-fold cycle at a time.
      f0      pitch in Hz
      jitter  random wobble in cycle LENGTH, as a fraction (0.005 = 0.5 percent)
      shimmer random wobble in cycle LOUDNESS, as a fraction
      noise   breathiness: size of random hiss added on top
      rng     a numpy random Generator (pass a seeded one to get the same sound again)
    Returns int16 samples.
    """
    cycles, total = [], 0
    while total < seconds * SAMPLE_RATE:
        period = (1 / f0) * (1 + jitter * rng.standard_normal())
        n = max(int(period * SAMPLE_RATE), 10)
        t = np.arange(n) / n
        amp = 1 + shimmer * rng.standard_normal()
        cycles.append(sum(np.sin(2 * np.pi * k * t) / k for k in range(1, 9)) * amp)   # 8 harmonics
        total += n
    signal = np.concatenate(cycles)[: int(seconds * SAMPLE_RATE)]
    signal = signal / np.max(np.abs(signal)) * 0.6
    signal += noise * rng.standard_normal(len(signal))
    fade = int(0.05 * SAMPLE_RATE)                                   # short fade in/out avoids clicks
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return (np.clip(signal, -1, 1) * 32767).astype(np.int16)


def to_wav_bytes(samples):
    """WAV file bytes, e.g. for st.audio or to save to disk."""
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, samples)
    return buf.getvalue()
