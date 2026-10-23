#!/usr/bin/env python3
"""Mic -> intent bridge for the VCM Action Runner.

This is the "mic side" that the runner used to fake by typing codes. It:
  1. Records from the microphone between two SPACE/ENTER presses
     (reuses the proven capture in tester.py: device auto-detect, rate
     fallback, live level meter, resample to 16 kHz).
  2. Runs the clip through the VCM (vcm_infer.predict) -> (intent, slot,
     conf, rejected).
  3. Returns a small result dict the runner feeds straight into _dispatch,
     so the SAME domain hand-off rules apply (an unrelated music command
     stops the light, leaving the light domain turns the bulb off, etc.).

Only numpy + onnxruntime + sounddevice are needed (all already in setup.sh).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # bundle root (holds vcm_infer.py, tester.py)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _import_vcm():
    """Import the inference core (lives at the bundle root, not in the pkg)."""
    import vcm_infer
    return vcm_infer


def _import_capture():
    """Reuse tester.py's record_spacebar() so mic handling stays identical to
    the standalone tester (device detect + rate fallback + resample)."""
    import tester
    return tester.record_spacebar


class Voice:
    """One-shot 'listen -> classify' helper with a cached model."""

    def __init__(self, model=None, labels=None, verbose=False):
        self.verbose = verbose
        self._sess = None
        self._labels = None
        self._model = model
        self._labels_path = labels

    def _ensure_model(self):
        if self._sess is None:
            vcm = _import_vcm()
            # None -> use vcm_infer's bundled E8 GRU model and labels.
            self._sess = vcm.load_model(self._model) if self._model else vcm.load_model()
            self._labels = (vcm.load_labels(self._labels_path)
                            if self._labels_path else vcm.load_labels())
        return self._sess, self._labels

    def warmup(self):
        """Load and warm the VCM before the first wake word is spoken."""
        vcm = _import_vcm()
        sess, _ = self._ensure_model()
        vcm.warmup(sess, n=1)

    def classify_audio(self, x):
        """x = float32 @16 kHz mono -> dict {intent, slot, conf, rejected, ok}."""
        vcm = _import_vcm()
        sess, labels = self._ensure_model()
        t0 = time.perf_counter()
        intent, conf, probs, info = vcm.classify(sess, labels, x,
                                                 verbose=self.verbose)
        t1 = time.perf_counter()
        lat = {"classify_ms": (t1 - t0) * 1000.0}
        # Reject on ANY of:
        #   * model chose REJECT, OR
        #   * VAD found no clear speech (span is None) -> the clip was silence
        #     / noise and the model's guess is meaningless (it can 'hallucinate'
        #     e.g. CALL @0.53 on pure silence), OR
        #   * the winning intent is not in the golden 19 (defensive).
        no_speech = info.get("span") is None
        rejected = (bool(info.get("rejected"))
                    or intent in (None, "REJECT")
                    or no_speech)
        return {
            "ok": (not rejected) and intent is not None,
            "latency": lat,
            "intent": None if rejected else intent,
            "slot": None if rejected else info.get("slot"),
            "conf": conf,
            "rejected": rejected,
            "no_speech": no_speech,
        }

    def listen_and_classify(self):
        """SPACE to record, SPACE to stop, classify. Returns the result dict,
        or {'ok': False, 'cancelled': True} when the user pressed 'q' / no mic."""
        try:
            capture = _import_capture()
        except Exception as e:
            print("\n  [voice] can't import the mic capture (%s)" % e)
            print("          pip install sounddevice   (or run ./setup.sh)")
            return {"ok": False, "cancelled": False, "error": str(e)}

        x = capture()
        if x is None:
            return {"ok": False, "cancelled": True}
        return self.classify_audio(x)


def describe(r):
    """One human-readable line for a result dict (for the terminal)."""
    if r.get("cancelled"):
        return "  [voice] cancelled -- no audio captured"
    if r.get("error"):
        return "  [voice] error: %s" % r["error"]
    if not r.get("ok"):
        if r.get("no_speech"):
            return ("  [voice] no speech detected -- speak closer / louder, then SPACE again"
                    + _latency_line(r))
        conf = r.get("conf")
        c = ("%.2f" % conf) if conf is not None else "?"
        return ("  [voice] no command recognised (REJECT, conf=%s) -- try again" % c
                + _latency_line(r))
    slot = r.get("slot")
    c = ("%.2f" % r["conf"]) if r.get("conf") is not None else "?"
    line = "  [voice] heard: %s" % r["intent"]
    if slot:
        line += "  (slot: %s)" % slot
    line += "   [conf %s]" % c
    return line + _latency_line(r)


def _latency_line(r):
    """Measured delay components, appended to the result line."""
    lat = r.get("latency")
    if not lat:
        return ""
    classify_ms = lat.get("classify_ms", 0.0)
    if "wake_score_ms" not in lat:
        return "   [VCM %.0f ms]" % classify_ms
    return ("\n  [timing] wake silence %.0f ms + wake model %.0f ms; "
            "command silence %.0f ms; VCM %.0f ms" %
            (lat.get("wake_endpoint_ms", 0.0),
             lat.get("wake_score_ms", 0.0),
             lat.get("command_endpoint_ms", 0.0),
             classify_ms))
