"""Real audio backend for the media intents (PLAY / PAUSE / STOP / NEXT / VOL).

Design
------
This module owns the *actual* sound. It is deliberately isolated so the rest of
the app only ever calls five methods:

    AUDIO.play(track_idx)     # load + play track `track_idx` (or resume if it's
                              # the track already loaded and merely paused)
    AUDIO.pause()             # pause the current track (no-op if not playing)
    AUDIO.stop()              # stop and rewind to the start (track 0)
    AUDIO.next()              # advance to the next track and play it
    AUDIO.set_volume(pct)     # 0..100, applied live without interrupting

It wraps **pygame.mixer**. Why pygame:
  * installs with a plain `pip install pygame` -- no system libraries, no sudo,
    no SWIG, no C build (unlike lgpio). It talks to ALSA/PulseAudio directly,
    so it works headless on the Pi.
  * gives us real pause/resume (mixer.music.pause()/unpause()) rather than
    stop+reload, which is exactly the "pause then play resumes" behaviour the
    spec asks for.

Graceful degradation
--------------------
If pygame is not importable (e.g. running the bundle on a laptop for testing)
or the mixer cannot open an audio device, every method becomes a harmless
no-op and `AUDIO.available` is False. The app's state machine (DEVICE.playing /
paused / track_idx) still runs, so the UI and the dispatch logic are unaffected;
you just get no sound until `pip install pygame` on the Pi.

Track files
-----------
Looked up in this order for track index `i`:
  1. assets/music/track_{i}.mp3   (drop your real songs here)
  2. assets/music/track_{i}.wav
  3. assets/music/song_a.wav / song_b.wav  (the two generated demo tones,
     cycled for i beyond 1)
If nothing exists, a short sine tone is synthesised on the fly so the pipeline
is always audible.
"""
import os
import sys
import threading
import wave
import struct
import math

# ---------------------------------------------------------------- locate files
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # .../vcm_action_runner
ASSET_DIR = os.path.join(_PKG, "assets", "music")

# pygame is optional at import time.
try:
    import pygame
    _HAVE_PYGAME = True
except Exception:                                   # pragma: no cover
    pygame = None
    _HAVE_PYGAME = False


def _candidate_paths(idx):
    """Ordered list of on-disk paths for logical track index `idx`."""
    base = ASSET_DIR
    exts = ("mp3", "wav", "ogg")
    cands = [os.path.join(base, "track_%d.%s" % (idx, e)) for e in exts]
    # fall back to the two demo tones for any index
    demo = {0: "song_a.wav", 1: "song_b.wav"}
    key = idx % 2
    cands.append(os.path.join(base, demo[key]))
    return cands


def _resolve_track(idx):
    """Return an existing file path for `idx`, or None if none is on disk."""
    for p in _candidate_paths(idx):
        if os.path.isfile(p):
            return p
    return None


def _synth_tone(path, freq=440.0, seconds=4.0, rate=22050):
    """Write a simple sine-wave WAV so there is always *something* to play."""
    n = int(rate * seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        amp = 12000
        for i in range(n):
            # gentle fade in/out so there's no click
            env = min(1.0, i / (0.05 * rate), (n - i) / (0.05 * rate))
            val = int(amp * env * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))


class AudioDriver:
    def __init__(self):
        self._lock = threading.RLock()
        self.available = False
        self._loaded_idx = None      # track currently loaded in the mixer
        self._paused = False
        self._playing = False
        self._vol = 50
        if _HAVE_PYGAME:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                self.available = True
            except Exception as exc:               # no audio device / headless
                self.available = False
                sys.stderr.write("[audio] mixer unavailable (%s) -> silent mode\n" % exc)

    # ------------------------------------------------------------- internals
    def _load_and_play(self, idx):
        path = _resolve_track(idx)
        if path is None:
            # synthesize a distinct tone per track so it's always audible
            path = os.path.join(ASSET_DIR, "gen_%d.wav" % idx)
            os.makedirs(ASSET_DIR, exist_ok=True)
            if not os.path.isfile(path):
                _synth_tone(path, freq=(330.0 if idx % 2 == 0 else 523.0))
        with self._lock:
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(loops=0)
                self._loaded_idx = idx
                self._paused = False
                self._playing = True
                self._apply_vol()
            except Exception as exc:
                sys.stderr.write("[audio] load/play failed: %s\n" % exc)

    def _apply_vol(self):
        try:
            pygame.mixer.music.set_volume(max(0.0, min(1.0, self._vol / 100.0)))
        except Exception:
            pass

    # --------------------------------------------------------------- public
    def play(self, idx):
        """Resume if `idx` is the loaded track and merely paused; else load it."""
        if not self.available:
            return
        with self._lock:
            if self._loaded_idx == idx and self._paused and self._playing:
                pygame.mixer.music.unpause()
                self._paused = False
                return
            self._load_and_play(idx)

    def pause(self):
        if not self.available:
            return
        with self._lock:
            if self._playing and not self._paused:
                pygame.mixer.music.pause()
                self._paused = True

    def stop(self):
        if not self.available:
            return
        with self._lock:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self._playing = False
            self._paused = False
            self._loaded_idx = None

    def next(self, current_idx, count):
        nxt = (current_idx + 1) % count
        self.play(nxt)
        return nxt

    def set_volume(self, pct):
        self._vol = max(0, min(100, int(pct)))
        self._apply_vol()

    def progress(self):
        """Playback position as a fraction 0.0-1.0 of the current track.

        Returns 0.0 when nothing is loaded or the track length is unknown
        (some encoders don't expose duration). Used by the persistent music
        UI to show a real elapsed-time clock instead of a fake bar.
        """
        if not self.available:
            return 0.0
        with self._lock:
            if not self._playing or self._paused is None:
                return 0.0
            try:
                pos = pygame.mixer.music.get_pos()      # ms since play()
                dur = pygame.mixer.music.get_length()   # seconds
                if pos is None or dur is None or dur <= 0:
                    return 0.0
                frac = (pos / 1000.0) / dur
                return max(0.0, min(1.0, frac))
            except Exception:
                return 0.0

    def elapsed_seconds(self):
        """Current playback position in SECONDS (real, from the mixer).

        Returns 0.0 when nothing is loaded, the track length is unknown, or
        pygame is unavailable. Used by the music window to show a real
        elapsed clock (m:ss) instead of a guessed one.
        """
        if not self.available:
            return 0.0
        with self._lock:
            if not self._playing or self._paused is None:
                return 0.0
            try:
                pos = pygame.mixer.music.get_pos()      # ms since play()
                if pos is None:
                    return 0.0
                return max(0.0, pos / 1000.0)
            except Exception:
                return 0.0

    def shutdown(self):
        if not self.available:
            return
        try:
            pygame.mixer.quit()
        except Exception:
            pass


# Module-level singleton shared by the media handlers.
AUDIO = AudioDriver()
