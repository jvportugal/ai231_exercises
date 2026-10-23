"""Log-mel spectrogram feature extractor (scipy + numpy only).

Produces (N_MELS, N_FRAMES) = (40, 150) frames, matching the old model's
input shape (1, 40, 150). Values are log-mel rescaled to [0, 1].
"""
import io
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
from config import SR, N_MELS, N_FFT, HOP, WIN, N_FRAMES, FMIN, FMAX


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_min, mel_max = hz2mel(fmin), hz2mel(fmax)
    pts = mel2hz(np.linspace(mel_min, mel_max, n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            if center > left:
                fbank[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[m, k] = (right - k) / (right - center)
    return fbank


_MEL_FB = None


def _get_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX)
    return _MEL_FB


def wav_to_mel(wav_bytes):
    """Decode 16-bit PCM WAV bytes -> (N_MELS, N_FRAMES) log-mel float32."""
    sr, data = wavfile.read(io.BytesIO(wav_bytes))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        n = int(len(data) * SR / sr)
        idx = np.linspace(0, len(data) - 1, n)
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)

    need = 1 + (N_FRAMES - 1) * HOP
    if len(data) < need:
        data = np.pad(data, (0, need - len(data)))
    else:
        data = data[:need]

    win = get_window("hann", WIN, fftbins=True).astype(np.float32)
    fb = _get_fb()
    mags = np.empty((N_FRAMES, N_FFT // 2 + 1), dtype=np.float32)
    for t in range(N_FRAMES):
        seg = data[t * HOP: t * HOP + WIN]
        if len(seg) < WIN:                 # final partial frame -> zero pad
            seg = np.pad(seg, (0, WIN - len(seg)))
        seg = seg * win
        spec = np.fft.rfft(seg, N_FFT)
        mags[t] = (np.abs(spec) ** 2).astype(np.float32)

    mel = mags @ fb.T
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel).T
    # Normalize so the distribution matches the old model's input
    # (mean ~0.18, p50 ~0.085, p95 ~0.67): anchor at -50 dB .. -20 dB.
    # Percentile-robust anchors for this (loud) HF audio: p5 -> 0, p95 -> 1.
    # Gives full dynamic range with only ~10% clipping at the extremes.
    lo, hi = -10.0, 1.4
    logmel = (logmel - lo) / (hi - lo)
    return np.clip(logmel, 0.0, 1.0).astype(np.float32)
