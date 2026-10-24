#!/usr/bin/env python3
"""Continuous Destiny wake-word detection and automatic command capture.

This module uses NumPy, sounddevice, and onnxruntime so it can run in the same
lightweight Raspberry Pi virtualenv as the VCM. Linux chime playback uses the
``aplay`` command installed by setup.sh, avoiding desktop-audio assumptions in
SSH sessions. One persistent PortAudio stream is used for microphone capture;
the two models never compete for it.
"""
import io
import math
import os
import shutil
import subprocess
import sys
import time
import wave
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SAMPLE_RATE = 16000
BLOCK_MS = 100
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
WINDOW_SAMPLES = 16000                 # 1.0 s, identical to robust-v2 training
N_MELS = 40
N_FFT = 512
HOP = 160
N_FRAMES = 100

DEFAULT_MODEL = os.path.join(_ROOT, "wakeword_model", "destiny.onnx")
DEFAULT_MELBANK = os.path.join(_ROOT, "wakeword_model", "melbank.npy")


def block_rms(audio):
    """Return a stable RMS level for a mono audio block."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(x):
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def short_speech_level(audio, frame_ms=20):
    """Return a level that does not average a short word out of a 100 ms block.

    Command capture reads 100 ms blocks for efficient streaming.  A stop
    consonant or a quickly spoken one-word command may occupy only one or two
    20 ms frames, making the RMS of the complete block deceptively quiet.  The
    second-loudest short-frame RMS preserves 40 ms of speech evidence without
    making an isolated click look like sustained speech.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_samples = max(1, SAMPLE_RATE * int(frame_ms) // 1000)
    count = len(x) // frame_samples
    if count == 0:
        return block_rms(x)
    frames = x[:count * frame_samples].reshape(count, frame_samples)
    levels = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    if len(levels) == 1:
        return float(levels[0])
    return float(np.partition(levels, -2)[-2])


class AdaptiveSpeechGate:
    """Track the room floor so a steady fan does not start speech capture.

    ``minimum`` remains the absolute lower bound for quiet rooms.  The rolling
    20th percentile follows persistent background noise, while the multiplier
    requires a word-like rise above that floor.  The history is intentionally
    short enough to follow a fan speed change but long enough not to follow a
    spoken syllable.
    """

    def __init__(self, minimum=0.008, multiplier=1.65, history_blocks=40):
        self.minimum = float(minimum)
        self.multiplier = float(multiplier)
        self.history = deque(maxlen=int(history_blocks))

    @property
    def threshold(self):
        if len(self.history) < 5:
            return self.minimum
        floor = float(np.percentile(np.asarray(self.history), 20))
        return max(self.minimum, floor * self.multiplier)

    def update(self, rms, currently_speaking=False):
        threshold = self.threshold
        voiced = float(rms) >= threshold
        # Only the first block of a possible utterance enters the history. A
        # low percentile makes that single speech spike harmless, while still
        # allowing a changed fan speed to become part of the room estimate.
        if not currently_speaking:
            self.history.append(float(rms))
        return voiced, threshold


class LiveLevelMeter:
    """Single-line terminal audio meter used by wake-word and command modes.

    The line is redrawn only on an interactive terminal. State changes are
    still printed as normal lines, so redirected logs remain readable.
    """

    def __init__(self, state, width=32):
        self.state = state
        self.width = width
        self._drawn = False

    def set_state(self, state):
        self.clear()
        self.state = state

    def update(self, audio):
        rms = block_rms(audio)
        if sys.stdout.isatty():
            # Speech from typical USB microphones is commonly in the
            # 0.005-0.08 RMS range. A logarithmic meter keeps quieter speech
            # visible without allowing loud sounds to overrun the display.
            db = 20.0 * np.log10(max(rms, 1e-6))
            fraction = min(1.0, max(0.0, (db + 60.0) / 60.0))
            filled = int(round(self.width * fraction))
            bar = "#" * filled + "-" * (self.width - filled)
            line = "  [%-23s] |%s| level %.4f" % (self.state, bar, rms)
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            self._drawn = True
        return rms

    def clear(self):
        if self._drawn:
            sys.stdout.write("\r" + (" " * 86) + "\r")
            sys.stdout.flush()
            self._drawn = False

    def close(self):
        self.clear()


def extract_features(audio, melbank):
    """Reproduce robust-v2 log-mel plus temporal-delta channels in NumPy."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (WINDOW_SAMPLES - len(y), 0))
    elif len(y) > WINDOW_SAMPLES:
        y = y[-WINDOW_SAMPLES:]
    # librosa.stft(center=True, pad_mode="constant") pads n_fft//2 on both
    # sides. A one-second clip produces 101 frames; training kept the first 100.
    padded = np.pad(y, (N_FFT // 2, N_FFT // 2))
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    power = np.empty((N_FRAMES, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_FRAMES):
        frame = padded[i * HOP:i * HOP + N_FFT] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        power[i] = (np.abs(spectrum) ** 2).astype(np.float32)

    mel = np.asarray(melbank, dtype=np.float32).dot(power.T)
    amin = 1e-10
    mel = np.maximum(mel, amin)
    db = 10.0 * np.log10(mel)
    db -= 10.0 * np.log10(max(amin, float(mel.max())))
    db = np.maximum(db, float(db.max()) - 80.0)
    base = (db - float(db.mean())) / (float(db.std()) + 1e-5)
    delta = np.zeros_like(base)
    delta[:, 1:] = base[:, 1:] - base[:, :-1]
    delta /= float(delta.std()) + 1e-5
    feat = np.stack((base, delta)).astype(np.float32)
    return feat[None, :, :, :]


def prepare_wakeword_windows(audio):
    """VAD-crop speech and return its legal start/center/end placements."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(y):
        return [np.zeros(WINDOW_SAMPLES, dtype=np.float32)]
    y = y - float(y.mean())
    frame_len = SAMPLE_RATE * 10 // 1000
    frame_hop = SAMPLE_RATE * 5 // 1000
    crop = y
    voiced_len = 0.0
    if len(y) >= frame_len:
        count = 1 + (len(y) - frame_len) // frame_hop
        starts = np.arange(count) * frame_hop
        idx = starts[:, None] + np.arange(frame_len)[None, :]
        energy = np.sqrt(np.mean(y[idx] ** 2, axis=1))
        floor = float(np.percentile(energy, 20))
        ceiling = float(np.percentile(energy, 95))
        threshold = max(floor * 2.2,
                        floor + (ceiling - floor) * 0.22, 8e-5)
        voiced = energy > threshold
        if voiced.any():
            first = int(np.argmax(voiced))
            last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
            padding = SAMPLE_RATE * 65 // 1000
            start = max(0, first * frame_hop - padding)
            end = min(len(y), (last + 1) * frame_hop + padding)
            crop = y[start:end]
            voiced_len = (end - start) / float(SAMPLE_RATE)
    if voiced_len < 0.18:
        crop = y

    if len(crop) >= WINDOW_SAMPLES:
        # A close-mic plosive or a deliberately slow pronunciation can make
        # the VAD crop longer than the model window. Score start/center/end so
        # the second syllable is not silently discarded.
        gap = len(crop) - WINDOW_SAMPLES
        starts = sorted(set((0, gap // 2, gap)))
        return [crop[start:start + WINDOW_SAMPLES].astype(np.float32)
                for start in starts]

    # Training randomly placed each cropped word in its 1.0-second window.
    # At runtime, score the three representative legal placements and use the
    # maximum. This raises held-out recall at the same 2% OOS constraint while
    # costing only two extra tiny-CNN calls after a speech candidate ends.
    gap = WINDOW_SAMPLES - len(crop)
    offsets = sorted(set((0, gap // 2, gap)))
    windows = []
    for offset in offsets:
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        window[offset:offset + len(crop)] = crop
        windows.append(window)
    return windows


def prepare_wakeword(audio):
    """Return the left-aligned prepared window (compatibility/test helper)."""
    return prepare_wakeword_windows(audio)[0]


class WakeWordDetector:
    """Small ONNX wake-word model with the exact training frontend."""

    def __init__(self, model_path=DEFAULT_MODEL, melbank_path=DEFAULT_MELBANK):
        import onnxruntime as ort
        self.model_path = model_path
        self.melbank = np.load(melbank_path).astype(np.float32)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score(self, audio):
        best = 0.0
        for window in prepare_wakeword_windows(audio):
            feat = extract_features(window, self.melbank)
            logit = float(np.asarray(
                self.session.run(None, {self.input_name: feat})[0]).reshape(-1)[0])
            # Numerically stable sigmoid.
            if logit >= 0:
                probability = 1.0 / (1.0 + np.exp(-logit))
            else:
                exp_logit = np.exp(logit)
                probability = float(exp_logit / (1.0 + exp_logit))
            best = max(best, probability)
        return float(best)

    def warmup(self):
        self.score(np.zeros(WINDOW_SAMPLES, dtype=np.float32))


def _resample_block(audio, source_rate):
    """Resample one fixed-duration mic block to exactly 16 kHz."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == SAMPLE_RATE and len(x) == BLOCK_SAMPLES:
        return x
    if not len(x):
        return np.zeros(BLOCK_SAMPLES, dtype=np.float32)
    old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    new = np.linspace(0.0, 1.0, num=BLOCK_SAMPLES, endpoint=False)
    return np.interp(new, old, x).astype(np.float32)


class Microphone:
    """A persistent microphone stream with Raspberry Pi sample-rate fallback."""

    CANDIDATE_RATES = (48000, 44100, 16000, 32000, 22050, 96000)

    def __init__(self, device=None):
        self.device = device
        self.stream = None
        self.native_rate = None
        self.native_block = None
        self.device_name = "default"

    def open(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "sounddevice is unavailable; run ./setup.sh (%s)" % exc)

        if self.device is not None:
            try:
                self.device_name = sd.query_devices(self.device, "input").get(
                    "name", str(self.device))
            except Exception:
                self.device_name = str(self.device)

        errors = []
        for rate in self.CANDIDATE_RATES:
            stream = None
            try:
                block = int(rate * BLOCK_MS / 1000)
                stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="float32",
                    device=self.device, blocksize=block)
                stream.start()
                self.stream = stream
                self.native_rate = rate
                self.native_block = block
                if self.device is None:
                    try:
                        info = sd.query_devices(stream.device, "input")
                        self.device_name = info.get("name", "default")
                    except Exception:
                        pass
                return self
            except Exception as exc:
                errors.append("%d Hz: %s" % (rate, exc))
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        raise RuntimeError("could not open microphone (" + "; ".join(errors) + ")")

    def read(self):
        data, overflowed = self.stream.read(self.native_block)
        if overflowed:
            print("  [wakeword] microphone overflow; continuing", file=sys.stderr)
        return _resample_block(data[:, 0], self.native_rate)

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
            self.stream = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _generated_wake_chime(volume=0.22):
    """Return a short original two-note acknowledgement chime at 16 kHz."""
    notes = []
    for frequency, seconds in ((880.0, 0.065), (1318.5, 0.085)):
        count = int(SAMPLE_RATE * seconds)
        t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
        # A sine plus quiet octave reads clearly on small Raspberry Pi speakers.
        tone = np.sin(2 * np.pi * frequency * t)
        tone += 0.18 * np.sin(2 * np.pi * frequency * 2.0 * t)
        attack = max(1, int(0.008 * SAMPLE_RATE))
        release = max(1, int(0.025 * SAMPLE_RATE))
        envelope = np.ones(count, dtype=np.float32)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        envelope[-release:] = np.linspace(1.0, 0.0, release, endpoint=True)
        notes.append((tone * envelope).astype(np.float32))
    gap = np.zeros(int(0.018 * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate((notes[0], gap, notes[1])) * float(volume)


def _load_pcm_wav(path, volume):
    """Load an optional uncompressed PCM WAV without adding dependencies."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError("wake chime must be 8-, 16-, or 32-bit PCM WAV")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(audio):
        duration = len(audio) / float(rate)
        old = np.linspace(0.0, duration, len(audio), endpoint=False)
        new = np.linspace(0.0, duration, int(duration * SAMPLE_RATE), endpoint=False)
        audio = np.interp(new, old, audio).astype(np.float32)
    return np.clip(audio * float(volume), -1.0, 1.0).astype(np.float32)


def _resample_audio(audio, source_rate, target_rate):
    """Resample a complete mono cue while preserving its duration."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(source_rate) == int(target_rate) or not len(x):
        return x
    count = max(1, int(round(len(x) * float(target_rate) / source_rate)))
    old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    new = np.linspace(0.0, 1.0, num=count, endpoint=False)
    return np.interp(new, old, x).astype(np.float32)


def _play_audio_compatible(sd, audio, source_rate=SAMPLE_RATE):
    """Play mono float audio using formats commonly accepted by ALSA devices.

    ``sounddevice.play`` inherits the array's float32 format and the supplied
    16 kHz rate.  Many Raspberry Pi USB, HDMI, and analogue outputs only expose
    signed 16-bit PCM at 44.1 or 48 kHz.  Prefer the device's advertised native
    rate and int16, resampling and duplicating to stereo when necessary.

    Return a small description of the successful stream.  Raise the final
    error only after trying the practical ALSA combinations.
    """
    info = sd.query_devices(None, "output")
    default_rate = int(round(float(info.get("default_samplerate", 0) or 0)))
    max_channels = int(info.get("max_output_channels", 0) or 0)
    if max_channels < 1:
        raise RuntimeError("default audio device has no output channels")

    rates = []
    for rate in (default_rate, 48000, 44100, source_rate, 32000, 22050):
        rate = int(rate)
        if rate > 0 and rate not in rates:
            rates.append(rate)
    # Stereo is the most widely accepted hardware layout; retain mono as a
    # fallback for small USB speakers that expose exactly one channel.
    channels = [2, 1] if max_channels >= 2 else [1]
    errors = []
    # Exhaust the hardware-friendly int16 combinations before attempting
    # float32, avoiding the noisy ALSA sample-format failure in the common
    # case where only the channel layout needs to fall back from stereo.
    for dtype in ("int16", "float32"):
        for rate in rates:
            samples = _resample_audio(audio, source_rate, rate)
            for channel_count in channels:
                if dtype == "int16":
                    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
                else:
                    data = samples.astype(np.float32)
                if channel_count > 1:
                    data = np.repeat(data[:, None], channel_count, axis=1)
                stream = None
                try:
                    stream = sd.OutputStream(
                        samplerate=rate, channels=channel_count, dtype=dtype)
                    stream.start()
                    stream.write(data)
                    stream.stop()
                    return {
                        "rate": rate,
                        "channels": channel_count,
                        "dtype": dtype,
                        "device": info.get("name", "default"),
                    }
                except Exception as exc:
                    errors.append("%d Hz/%d ch/%s: %s" %
                                  (rate, channel_count, dtype, exc))
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
    raise RuntimeError("no compatible output format (" + "; ".join(errors) + ")")


def _play_audio_aplay(audio, source_rate=SAMPLE_RATE, device=None):
    """Play a cue through native ALSA, independent of desktop audio services.

    An SSH login often has no PulseAudio/PipeWire session, and PortAudio can
    consequently expose a nominal default output whose hardware formats all
    fail to open.  ``aplay`` talks to ALSA directly and lets the ``default`` or
    explicitly selected ``plug*`` PCM perform the final hardware conversion.
    """
    executable = shutil.which("aplay")
    if executable is None:
        raise RuntimeError("aplay is unavailable; install the alsa-utils package")

    rate = 48000
    channels = 2
    samples = _resample_audio(audio, source_rate, rate)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    pcm = np.repeat(pcm[:, None], channels, axis=1)
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())

    alsa_device = device or "default"
    command = [executable, "-q", "-D", alsa_device]
    try:
        result = subprocess.run(
            command, input=encoded.getvalue(), stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, timeout=5, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("aplay timed out on ALSA device %s" % alsa_device) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError("aplay could not open ALSA device %s%s" %
                           (alsa_device, (": " + detail) if detail else ""))
    return {
        "rate": rate,
        "channels": channels,
        "dtype": "int16",
        "device": "ALSA " + alsa_device,
        "backend": "aplay",
    }


class WakeWordSession:
    """Wait for Destiny, then capture one command using silence endpointing."""

    def __init__(self, threshold=0.875, device=None, command_timeout=6.0,
                 speech_start_timeout=4.0, silence_seconds=0.5,
                 speech_rms=0.008, guard_seconds=0.0,
                 noise_floor_multiplier=1.65,
                 wake_silence_seconds=0.4, wake_max_seconds=2.0,
                 model_path=DEFAULT_MODEL, melbank_path=DEFAULT_MELBANK,
                 chime=True, chime_volume=0.22, chime_path=None,
                 chime_device=None):
        if not 0.0 < threshold < 1.0:
            raise ValueError("wake-word threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self.command_timeout = float(command_timeout)
        self.speech_start_timeout = float(speech_start_timeout)
        self.silence_blocks = max(1, int(round(silence_seconds * 1000 / BLOCK_MS)))
        self.speech_rms = float(speech_rms)
        self.guard_blocks = max(0, int(round(guard_seconds * 1000 / BLOCK_MS)))
        self.wake_silence_blocks = max(
            1, int(round(wake_silence_seconds * 1000 / BLOCK_MS)))
        self.wake_max_blocks = max(
            self.wake_silence_blocks + 1,
            int(round(wake_max_seconds * 1000 / BLOCK_MS)))
        self.detector = WakeWordDetector(model_path, melbank_path)
        self.microphone = Microphone(device)
        self.chime_enabled = bool(chime)
        self.chime_volume = float(chime_volume)
        self.chime_path = chime_path
        self.chime_device = chime_device
        self.chime_audio = None
        self._chime_failed = False
        self._chime_output = None
        self.speech_gate = AdaptiveSpeechGate(
            self.speech_rms, multiplier=noise_floor_multiplier)
        self.last_timing = {}

    def open(self):
        self.detector.warmup()
        self.microphone.open()
        # Prime the adaptive room floor before accepting candidates. This is
        # less than one second and prevents an already-running fan from being
        # mistaken for speech immediately at startup.
        for _ in range(8):
            block = self.microphone.read()
            self.speech_gate.history.append(block_rms(block))
        if self.chime_enabled:
            try:
                self.chime_audio = (_load_pcm_wav(self.chime_path, self.chime_volume)
                                    if self.chime_path else
                                    _generated_wake_chime(self.chime_volume))
            except Exception as exc:
                print("  [wakeword] could not load wake chime: %s" % exc,
                      file=sys.stderr)
                self.chime_audio = None
        print("  [wakeword] microphone: %s @ %d Hz -> 16000 Hz" % (
            self.microphone.device_name, self.microphone.native_rate))
        print("  [wakeword] model ready: Destiny (threshold %.3f)" % self.threshold)
        return self

    def _play_chime(self):
        """Play the acknowledgement cue and return its duration in seconds."""
        if self.chime_audio is None or self._chime_failed:
            return 0.0
        try:
            started = time.perf_counter()
            if sys.platform.startswith("linux"):
                # In particular, use native ALSA when launched through SSH;
                # there may be no per-user desktop audio server for PortAudio.
                output = _play_audio_aplay(
                    self.chime_audio, SAMPLE_RATE, self.chime_device)
            else:
                import sounddevice as sd
                output = _play_audio_compatible(sd, self.chime_audio, SAMPLE_RATE)
            elapsed = time.perf_counter() - started
            self.last_timing["wake_chime_ms"] = elapsed * 1000.0
            if self._chime_output is None:
                print("  [wakeword] chime output: %s @ %d Hz, %s, %d ch" %
                      (output["device"], output["rate"], output["dtype"],
                       output["channels"]), flush=True)
            self._chime_output = output
            return len(self.chime_audio) / float(SAMPLE_RATE)
        except Exception as exc:
            # A missing/default output device must never disable voice control.
            print("  [wakeword] wake chime unavailable; continuing silently (%s)" % exc,
                  file=sys.stderr)
            self._chime_failed = True
            return 0.0

    def close(self):
        self.microphone.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def wait_for_command(self, quit_flag=None):
        """Return one 16 kHz command waveform, or None after a capture timeout."""
        self.last_timing = {}
        pre_roll = deque(maxlen=2)
        candidate = []
        in_speech = False
        quiet_blocks = 0
        meter = LiveLevelMeter("WAITING FOR DESTINY")
        print("\n  [state] WAITING FOR DESTINY -- microphone is live", flush=True)
        try:
            while quit_flag is None or not quit_flag.is_set():
                block = self.microphone.read()
                rms = meter.update(block)
                voiced, _ = self.speech_gate.update(rms, in_speech)
                if not in_speech:
                    pre_roll.append(block)
                    if voiced:
                        in_speech = True
                        candidate = list(pre_roll)
                        quiet_blocks = 0
                    continue

                candidate.append(block)
                quiet_blocks = 0 if voiced else quiet_blocks + 1
                ended = quiet_blocks >= self.wake_silence_blocks
                capped = len(candidate) >= self.wake_max_blocks
                if not ended and not capped:
                    continue

                meter.clear()
                score_t0 = time.perf_counter()
                probability = self.detector.score(np.concatenate(candidate))
                score_ms = (time.perf_counter() - score_t0) * 1000.0
                endpoint_ms = quiet_blocks * BLOCK_MS if ended else 0
                in_speech = False
                quiet_blocks = 0
                candidate = []
                pre_roll.clear()
                if probability >= self.threshold:
                    self.last_timing = {
                        "wake_endpoint_ms": float(endpoint_ms),
                        "wake_score_ms": score_ms,
                        "wake_probability": probability,
                    }
                    print("  [state] DESTINY DETECTED -- p=%.3f "
                          "(silence wait %d ms, model %.0f ms)" %
                          (probability, endpoint_ms, score_ms), flush=True)
                    chime_seconds = self._play_chime()
                    # The input stream stayed open during playback. Consume
                    # only the blocks that overlap the cue before inviting the
                    # user to speak; normal reaction time supplies the margin.
                    chime_blocks = (int(math.ceil(
                        chime_seconds * 1000 / BLOCK_MS)) + 1) if chime_seconds else 0
                    for _ in range(chime_blocks):
                        chime_block = self.microphone.read()
                        self.speech_gate.update(block_rms(chime_block), False)
                    print("  [state] LISTENING FOR COMMAND -- speak now", flush=True)
                    return self._capture_command(quit_flag)
        finally:
            meter.close()
        return None

    def _capture_command(self, quit_flag):
        for _ in range(self.guard_blocks):
            block = self.microphone.read()
            self.speech_gate.update(block_rms(block), False)

        pre_roll = deque(maxlen=3)       # retain 300 ms before detected speech
        captured = []
        started = False
        voiced_blocks = 0
        quiet_blocks = 0
        t0 = time.monotonic()
        speech_t0 = None
        meter = LiveLevelMeter("LISTENING FOR COMMAND")

        try:
            while quit_flag is None or not quit_flag.is_set():
                block = self.microphone.read()
                rms = meter.update(block)
                # Command mode is explicitly armed by the wake word, so it can
                # use a more sensitive gate than always-on wake-word mode.  A
                # short-frame level prevents quick words such as "stop" and
                # "next" from being averaged away inside a 100 ms read.
                command_threshold = max(
                    0.003, self.speech_gate.threshold * 0.70)
                voiced = short_speech_level(block) >= command_threshold
                self.speech_gate.update(rms, started)
                elapsed = time.monotonic() - t0

                if not started:
                    pre_roll.append(block)
                    if voiced:
                        started = True
                        speech_t0 = time.monotonic()
                        captured.extend(pre_roll)
                        voiced_blocks = 1
                        self.last_timing["command_speech_wait_ms"] = elapsed * 1000.0
                    elif elapsed >= self.speech_start_timeout:
                        meter.clear()
                        print("  [state] COMMAND TIMEOUT -- no speech; returning to wake word",
                              flush=True)
                        return None
                    continue

                captured.append(block)
                if voiced:
                    voiced_blocks += 1
                    quiet_blocks = 0
                else:
                    quiet_blocks += 1

                # One 100 ms block is sufficient.  The VCM's own 20 ms VAD and
                # confidence/margin policy still reject clicks and ambiguity.
                # Requiring two blocks made a valid short command wait for the
                # full command timeout before it could be classified.
                if quiet_blocks >= self.silence_blocks and voiced_blocks >= 1:
                    break
                if time.monotonic() - speech_t0 >= self.command_timeout:
                    meter.clear()
                    print("  [voice] command time limit reached", flush=True)
                    break
        finally:
            meter.close()

        if not captured:
            return None
        audio = np.concatenate(captured).astype(np.float32)
        self.last_timing["command_endpoint_ms"] = float(quiet_blocks * BLOCK_MS)
        self.last_timing["command_audio_s"] = len(audio) / float(SAMPLE_RATE)
        print("  [state] PROCESSING COMMAND -- captured %.2f s "
              "(silence wait %d ms)" %
              (self.last_timing["command_audio_s"],
               self.last_timing["command_endpoint_ms"]), flush=True)
        return audio
