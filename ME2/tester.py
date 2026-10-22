#!/usr/bin/env python3
"""VCM Tester -- SPACEBAR-to-listen, voice -> intent.

Spec (exactly what was requested):
  * At startup, list ALL the possible commands (the 19-intent golden schema).
  * Press SPACEBAR  -> start listening.
  * Press SPACEBAR  -> stop listening; the audio between the two presses is the input.
  * Run that clip through the VCM (mel -> ONNX -> classify) and PRINT the intent.
  * Loop. 'q' quits.

Runs on macOS AND Raspberry Pi OS. Microphone uses sounddevice (pip install
sounddevice). If no mic / sounddevice is unavailable, pass a WAV file with
--audio (or just a path as the first argument) to classify a pre-recorded clip
instead of live mic capture.

Usage:
    python3 tester.py                       # live mic (spacebar to record)
    python3 tester.py --audio clip.wav      # classify a file, no mic needed
    python3 tester.py --audio clip.wav --loop   # re-classify the same file each spacebar
    python3 tester.py --model /path/vcm.onnx
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import vcm_infer as V


# ---------------------------------------------------------------------------
# Command / schema listing (printed at startup)
# ---------------------------------------------------------------------------
# Example phrases for the 19 intents (display only). Slotted intents show a
# concrete example; the actual slot value is resolved by head B at inference.
_EXAMPLES = {
    "PLAY_MUSIC":      "Play some music",
    "WEATHER":         "What's the weather?",
    "TIME":            "What time is it?",
    "LIGHT_ON":        "Turn on the light",
    "LIGHT_OFF":       "Turn off the light",
    "PAUSE":           "Pause",
    "STOP":            "Stop",
    "NEXT":            "Next track",
    "VOLUME_UP":       "Volume up",
    "VOLUME_DOWN":     "Volume down",
    "CALL":            "Call mom",
    "MESSAGE":         "Send a message",
    "LIST_REMINDERS":  "List my reminders",
    "TIMER":           "Set a timer for 10 seconds",
    "ALARM":           "Set an alarm for 6:00 AM",
    "TEMPERATURE":     "Set temperature to 22 degrees",
    "BRIGHTNESS":      "Set brightness to 60 percent",
    "COLOR":           "Set color to red",
    "CREATE_REMINDER": "Remind me to drink water",
}


def print_schema():
    names = V.intent_schema()          # list of intent-name strings (incl. REJECT)
    n_real = len(names) - 1            # minus REJECT
    print("=" * 70)
    print("  VCM TESTER  --  voice -> intent   (golden schema: %d intents)" % n_real)
    print("=" * 70)
    print("  %-18s %-11s %s" % ("INTENT", "TYPE", "EXAMPLE PHRASE"))
    print("  " + "-" * 68)
    for name in names:
        if name == "REJECT":
            continue
        kind = "slotted" if name in getattr(V, "_SLOTTED_PREFIX", {}) else "simple"
        print("  %-18s %-11s %s" % (name, kind, _EXAMPLES.get(name, "")))
    print("  " + "-" * 68)
    print("  %-18s %-11s %s" % ("(noise/silence)", "-", "REJECT (ignored)"))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Audio acquisition
# ---------------------------------------------------------------------------
def read_wav(path):
    """Read a mono WAV -> float32 @16kHz. Resamples linearly if needed."""
    import wave
    with wave.open(path, "rb") as w:
        assert w.getnchannels() in (1, 2), "WAV must be mono or stereo"
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
        sr = w.getframerate()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        data = data[0::2]
    if sr != V.SAMPLE_RATE:
        n_out = int(len(data) * V.SAMPLE_RATE / sr)
        idx = np.linspace(0, len(data) - 1, n_out)
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data


def record_spacebar():
    """Record from the mic between two SPACEBAR presses.
    Returns float32 @16kHz mono, or None on cancel/error."""
    try:
        import sounddevice as sd
    except Exception as e:
        print("\n  [!] sounddevice not available (%s)." % e)
        print("      pip install sounddevice   -- or use:  python3 tester.py --audio clip.wav")
        return None

    print("\n  Press SPACE or ENTER to START recording...   ('q' to quit)")
    start_key = _read_key()
    if start_key == "q":
        return None
    if start_key not in (" ", "\n", "\r"):
        # Not a trigger key (stray char / buffered newline). Drain the line so
        # it can't be mis-read, then wait for a real Space/Enter.
        print("  (press SPACE or ENTER)")
        start_key = _read_key()
        if start_key == "q":
            return None

    # --- pick the input device + a sample rate it ACTUALLY supports ---------
    # Many USB mics (e.g. the M-830) only run at 48 kHz. Forcing 16 kHz makes
    # PortAudio fail with paInvalidSampleRate. We do NOT trust the reported
    # default rate (ALSA often lies and says 16k). Instead we OPEN the stream
    # at each candidate rate in turn and keep the first one PortAudio actually
    # accepts, then resample to 16 kHz in Python.
    dev = None
    try:
        d = sd.default.device
        if isinstance(d, dict):
            dev = d.get("input")
        elif isinstance(d, (tuple, list)):
            dev = d[0]
        if dev is None or dev < 0:
            dev = None
    except Exception:
        dev = None

    if dev is not None:
        print("  input device : #%s  %s" % (dev, sd.query_devices(dev, "input").get("name", "?")))

    # Candidate rates to try, most-likely-first. The M-830 typically wants 48k.
    cands = [48000, 44100, 16000, 22050, 32000, 96000]

    chunks = []

    def cb(indata, frames, t_time, status):
        if status:
            print("    [mic status: %s]" % status, file=sys.stderr)
        chunks.append(indata[:, 0].copy())
        lvl = float(np.abs(indata[:, 0]).mean())
        bar = "#" * int(min(40, lvl * 400))
        sys.stdout.write("\r    |%-40s| %.3f   " % (bar, lvl))
        sys.stdout.flush()

    rec = None
    native_sr = None
    for sr in cands:
        try:
            rec = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                 device=dev,
                                 blocksize=int(sr * 0.1), callback=cb)
            rec.start()
            native_sr = sr
            break
        except Exception as e:
            try:
                if rec is not None:
                    rec.stop(); rec.close()
            except Exception:
                pass
            rec = None
            print("    [..] %d Hz rejected -- trying next" % sr)

    if rec is None:
        print("  [!] No sample rate worked for this mic.")
        print("      Fallback:  python3 tester.py --audio clip.wav")
        return None

    if native_sr != V.SAMPLE_RATE:
        print("  capture @ %d Hz -> resample to %d Hz" % (native_sr, V.SAMPLE_RATE))
    else:
        print("  capture @ %d Hz (native, no resample)" % native_sr)
    print("  RECORDING... (live level below)   Press SPACE or ENTER to STOP.")

    t0 = time.time()
    try:
        while True:
            if _read_key() in (" ", "\n"):
                break
            if time.time() - t0 > 60:
                print("\n    [60s cap reached -- stopping]")
                break
    finally:
        rec.stop()
        rec.close()
    sys.stdout.write("\n")
    if not chunks:
        print("  (no audio captured)")
        return None
    raw = np.concatenate(chunks).astype(np.float32)
    # Resample native rate -> 16 kHz (linear interp is fine for speech).
    if native_sr != V.SAMPLE_RATE:
        dur = len(raw) / float(native_sr)
        n_out = int(round(dur * V.SAMPLE_RATE))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=len(raw), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
    return raw


# Input handling.
#
# Consume EXACTLY ONE keypress per action so a buffered newline (e.g. from a
# piped script) can't be mis-read as a second "stop" and cause a runaway loop.
# `_read_key` blocks until one byte is available and returns it:
#   ' '  -> spacebar (start/stop)
#   '\n' -> Enter (also treated as a trigger)
#   'q'  -> quit
#   other-> ignored (still consumed, so it can't pile up)
def _read_key():
    """Block until one keypress, return it.

    After reading the key we drain any *other* bytes already sitting in the
    stdin buffer (non-blocking) so a single physical keypress can't leave a
    stray newline behind that gets mis-read as the NEXT action. This is what
    made Space feel broken on the Pi console: the OS delivered Space + a
    trailing newline, and the newline triggered the following read.
    """
    fd = None
    old_settings = None
    try:
        # A normal Unix terminal is line-buffered, so SPACE is otherwise not
        # delivered to Python until Enter is pressed. cbreak mode makes the
        # documented one-key Space control work over a Pi console or SSH.
        if sys.stdin.isatty():
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        import select
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                _drain_stdin()
                return "q" if ch == "" else ch
            time.sleep(0.005)
    except Exception:
        ch = sys.stdin.read(1)
        _drain_stdin()
        return "q" if ch == "" else ch
    finally:
        if fd is not None and old_settings is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


def _drain_stdin():
    """Consume any bytes already waiting in stdin (non-blocking)."""
    try:
        import select
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            if sys.stdin.read(1) == "":
                break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------
def report(label, conf, probs, count, info=None):
    intent, slot = V.label_to_intent(label, info)
    print("\n  " + "-" * 68)
    if info:
        span = info.get("span")
        if span:
            print("  [vad] speech %.2f-%.2f s  |  window centered on speech, "
                  "gain x%.2f" % (span[0], span[1], info.get("gain", 1.0)))
        else:
            print("  [vad] no clear speech detected (raw window used)")
    if intent is None:
        print("  [%d] REJECTED  (model label: %-28s conf=%.1f%%)" % (count, label, conf * 100))
        top3 = probs.argsort()[::-1][:3]
        alt = ", ".join("%s(%.0f%%)" % (l, p * 100) for l, p in
                        [(V.labels[i], probs[i]) for i in top3])
        print("        top3: %s" % alt)
        print("  [%d] -> no action (ignored)" % count)
    else:
        print("  [%d] INTENT: %s   %s   (model label: %s, conf=%.1f%%)"
              % (count, intent, ("slot=" + slot) if slot else "", label, conf * 100))
        if conf < 0.5:
            top3 = probs.argsort()[::-1][:3]
            alt = ", ".join("%s(%.0f%%)" % (l, p * 100) for l, p in
                            [(V.labels[i], probs[i]) for i in top3])
            print("        LOW CONFIDENCE -- top3: %s" % alt)
        print("  [%d] -> would fire action: %s%s" % (count, intent, (" (%s)" % slot) if slot else ""))
    print("  " + "-" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VCM spacebar-to-listen tester")
    ap.add_argument("--audio", nargs="?", const="__first_arg__", default=None,
                    metavar="WAV", help="classify a WAV file instead of live mic")
    ap.add_argument("--model", default=V.DEFAULT_MODEL, help="path to vcm.onnx")
    ap.add_argument("--labels", default=V.DEFAULT_LABELS, help="path to labels.json")
    ap.add_argument("--loop", action="store_true",
                    help="with --audio: re-classify the same file on each spacebar")
    args, unknown = ap.parse_known_args()
    if args.audio == "__first_arg__" and unknown:
        args.audio = unknown[0]

    # Load model.
    try:
        sess = V.load_model(args.model)
        labels = V.load_labels(args.labels)
    except Exception as e:
        print("  [!] Could not load VCM model: %s" % e)
        print("      Expected model at: %s" % args.model)
        return 1
    V.labels = labels.get("intents", labels)
    V.warmup(sess)

    print_schema()
    print("  model: %s  (%d classes)" % (os.path.basename(args.model), len(labels)))
    print("  engine: onnxruntime CPU   sample rate: %d Hz   window: %.1f s"
          % (V.SAMPLE_RATE, V.WINDOW_SEC))

    # ---- FILE MODE ---------------------------------------------------------
    if args.audio:
        if not os.path.exists(args.audio):
            print("  [!] audio file not found: %s" % args.audio)
            return 1
        data = read_wav(args.audio)
        print("\n  File mode: %s  (%.2f s)" % (args.audio, len(data) / V.SAMPLE_RATE))
        if args.loop:
            print("  (loop: press SPACEBAR to re-classify, 'q' to quit)")
            count = 0
            while True:
                k = _read_key()
                if k == "q":
                    break
                count += 1
                lab, conf, probs, info = V.classify(sess, labels, data)
                report(lab, conf, probs, count, info)
        else:
            lab, conf, probs, info = V.classify(sess, labels, data)
            report(lab, conf, probs, 1, info)
        return 0

    # ---- LIVE MIC MODE (spacebar to record) --------------------------------
    print("\n  Mode: LIVE MIC.")
    print("    SPACEBAR -> start listening, SPACEBAR again -> stop & classify.")
    print("    (Just press the spacebar. No Enter. Type q to quit.)\n")
    count = 0
    while True:
        data = record_spacebar()
        if data is None:
            # record_spacebar returns None when the user pressed 'q' at the
            # START prompt (or sounddevice is missing). Either way, stop.
            print("\n  bye.")
            break
        count += 1
        t0 = time.time()
        lab, conf, probs, info = V.classify(sess, labels, data)
        dt = (time.time() - t0) * 1000
        report(lab, conf, probs, count, info)
        print("  (inference: %.0f ms)   Press SPACEBAR for the next clip. (q to quit)" % dt)
    print("\n  bye.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  bye.")
        sys.exit(0)
