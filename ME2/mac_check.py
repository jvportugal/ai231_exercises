#!/usr/bin/env python3
"""VCM Mac check -- verify the SAME voice->intent pipeline on a laptop.

Runs on macOS (and Linux/Windows) using the exact same model + mel features
as the Pi tester. No ALSA involved, so the Pi's sample-rate quirks don't apply.

Setup (one block):
    cd ai231_exercises/ME2
    ./setup.sh
    source .venv/bin/activate

Run:
    python3 mac_check.py                 # live mic (SPACE or ENTER to start/stop)
    python3 mac_check.py --audio clip.wav   # classify a saved wav instead

Controls:
    SPACE or ENTER  -> start recording
    SPACE or ENTER  -> stop recording + classify
    q               -> quit
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import vcm_infer as V  # noqa: E402  (same inference core the Pi tester uses)


# ---------------------------------------------------------------------------
# Key input -- accepts SPACE or ENTER, drains stray buffered bytes so one
# physical keypress == exactly one action (the bug that confused the Pi run).
# ---------------------------------------------------------------------------
def _drain_stdin():
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


def _read_key():
    try:
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


TRIGGERS = (" ", "\n", "\r")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def _resample(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    dur = len(x) / float(src_sr)
    n_out = int(round(dur * dst_sr))
    if n_out <= 0:
        return np.zeros(dst_sr, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, x).astype(np.float32)


def record_live():
    """Record between two SPACE/ENTER presses. Returns 16kHz mono float32."""
    import sounddevice as sd

    print("\n  Press SPACE or ENTER to START recording...   ('q' to quit)")
    k = _read_key()
    if k == "q":
        return None
    if k not in TRIGGERS:
        print("  (press SPACE or ENTER)")
        k = _read_key()
        if k == "q":
            return None

    # Pick a sample rate the mic actually accepts (mac mics: 48k/44.1k typical).
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
                                 blocksize=int(sr * 0.1), callback=cb)
            rec.start()
            native_sr = sr
            break
        except Exception:
            try:
                if rec is not None:
                    rec.stop(); rec.close()
            except Exception:
                pass
            rec = None
            print("    [..] %d Hz rejected -- trying next" % sr)

    if rec is None:
        print("  [!] No sample rate worked for this mic.")
        print("      Fallback:  python3 mac_check.py --audio clip.wav")
        return None

    if native_sr != V.SAMPLE_RATE:
        print("  capture @ %d Hz -> resample to %d Hz" % (native_sr, V.SAMPLE_RATE))
    else:
        print("  capture @ %d Hz (native)" % native_sr)
    print("  RECORDING... (live level below)   Press SPACE or ENTER to STOP.")

    t0 = time.time()
    try:
        while True:
            if _read_key() in TRIGGERS:
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
    return _resample(raw, native_sr, V.SAMPLE_RATE)


def load_wav(path):
    """Load a wav as 16kHz mono float32."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported sample width %d" % sw)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return _resample(data, sr, V.SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(conf, probs, label, info=None):
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
        print("  REJECTED  (model label: %-28s conf=%.1f%%)" % (label, conf * 100))
    else:
        line = "  INTENT : %-14s  (conf=%.1f%%)" % (intent, conf * 100)
        if slot:
            line += "   slot=%s" % slot
        print(line)
        print("  label  : %s" % label)
    top = np.argsort(probs)[::-1][:3]
    print("  top-3  :")
    for i in top:
        print("           %-28s %5.1f%%" % (labels["intents"][i], probs[i] * 100))
    print("-" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global labels
    ap = argparse.ArgumentParser(description="VCM Mac verification")
    ap.add_argument("--audio", help="classify a wav file instead of the live mic")
    ap.add_argument("--loop", action="store_true",
                    help="with --audio: re-classify the same file on each SPACE/ENTER")
    args = ap.parse_args()

    print("=" * 68)
    print("  VCM MAC CHECK -- voice -> intent (same model as the Pi)")
    print("=" * 68)

    sess = V.load_model()
    labels = V.load_labels()
    print("  model loaded : %s" % V.DEFAULT_MODEL)
    print("  output heads : %d intent classes / %d slot classes" %
          (len(labels.get("intents", [])), len(labels.get("value_tokens", []))))
    V.warmup(sess)

    schema = [name for name in V.intent_schema() if name != "REJECT"]
    print("\n  Actions (%d):" % len(schema))
    for name in schema:
        prefix = V._SLOTTED_PREFIX.get(name)
        slots = [V._SLOT_TEXT[token]
                 for token in labels.get("value_tokens", [])
                 if prefix and token.startswith(prefix)]
        tag = "slotted" if prefix else "fixed  "
        suffix = "  slots=" + ", ".join(slots) if slots else ""
        print("    - %-16s [%s]%s" % (name, tag, suffix))

    if args.audio:
        x = load_wav(args.audio)
        print("\n  Loaded %s  (%.2fs @ %d Hz)" % (args.audio, len(x) / V.SAMPLE_RATE, V.SAMPLE_RATE))
        while True:
            t0 = time.time()
            label, conf, probs, info = V.classify(sess, labels, x)
            dt = (time.time() - t0) * 1000
            report(conf, probs, label, info)
            print("  (inference: %.0f ms)" % dt)
            if not args.loop:
                break
            print("  Press SPACE or ENTER to re-classify, 'q' to quit.")
            if _read_key() == "q":
                break
        return 0

    # Live mic loop.
    print("\n" + "=" * 68)
    print("  LIVE MIC MODE")
    print("    SPACE or ENTER -> start listening, again -> stop & classify.")
    print("    (Space AND Enter both work. Type q to quit.)\n")
    print("=" * 68)
    count = 0
    while True:
        x = record_live()
        if x is None:
            break
        count += 1
        t0 = time.time()
        label, conf, probs, info = V.classify(sess, labels, x)
        dt = (time.time() - t0) * 1000
        report(conf, probs, label, info)
        print("  (inference: %.0f ms)   Press SPACE or ENTER for the next clip. (q to quit)" % dt)
        if _read_key() == "q":
            break
    print("\n  Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
