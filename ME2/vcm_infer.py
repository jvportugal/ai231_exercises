#!/usr/bin/env python3
"""VCM inference core for the channel-robust calibrated RV model.

Model: CRNN_N (3 conv 32/64/128 + BiGRU 256x3), 3.19M params, 250-frame (2.5 s)
window, DUAL heads -> (logits_a: 20 intents incl REJECT, logits_b: 19 slot
tokens incl REJECT). Runs on onnxruntime (CPU). Only numpy + onnxruntime needed.

Feature path is VERIFIED bit-exact against E8 training:
  16 kHz mono -> VAD-trim + center 2.5 s window on speech midpoint ->
  log-mel (40 mels, N_FFT=400, HOP=160, 250 frames) -> per-utt CMVN -> (x+3)/6.

Public API (same names as before so tester.py/mac_check.py keep working):
    load_model(model_path)          -> InferenceSession
    load_labels(labels_path)        -> {"intents":[...], "value_tokens":[...]}
    audio_to_mel(x)                 -> np.ndarray (1,1,40,250) float32
    classify(sess, labels, x)       -> (label, conf, probs, info)   # x = 16 kHz mono
    label_to_intent(label)          -> (intent_name, slot_or_None)
    predict(x, sess=None, labels=None) -> dict  # convenience wrapper
    intent_schema()                 -> list
"""
import json
import os
from functools import lru_cache
import numpy as np

# ---------------------------------------------------------------------------
# Config (identical to E8 training)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
WINDOW_SEC = 2.5
N_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)      # 40000
N_MELS = 40
N_FRAMES = 250
HOP = 160                                       # 10 ms @ 16 kHz
N_FFT = 400                                     # 25 ms @ 16 kHz
WIN = 400
FMIN = 20.0
FMAX = 8000.0
MEL_LO = -10.0
MEL_HI = 1.4

# --- Voice-activity / placement-invariance tuning --------------------------
VAD_FRAME_MS = 20
VAD_HANGOVER_MS = 250
VAD_MIN_SPEECH_MS = 120
VAD_REL_THRESH = 0.12
VAD_ABS_FLOOR = 0.008
PEAK_TARGET = 0.9

# A wrong action is substantially worse than asking for one repeat.  These
# operating points were checked on the larger 824-clip real-voice test (not the
# small development holdout). Together with the valid-slot gate, this demo-safe
# point substantially reduces wrong accepted actions while retaining most
# otherwise-correct predictions. Slot gates are applied after restricting head
# B to the three values valid for the intent.
MIN_INTENT_CONFIDENCE = 0.80
MIN_INTENT_MARGIN = 0.50
MIN_SLOT_CONFIDENCE = 0.50
MIN_SLOT_MARGIN = 0.10

# A valid wake word substantially changes the prior probability that the next
# short utterance is a command.  The hands-free runner may therefore use this
# second, deliberately bounded operating point for fixed (non-slotted)
# commands.  It is not enabled by default in the generic inference API.
WAKE_SHORT_MIN_CONFIDENCE = 0.70
WAKE_SHORT_MIN_MARGIN = 0.35
WAKE_SHORT_MAX_SPAN_SECONDS = 1.50
WAKE_SHORT_INTENTS = frozenset({
    "PLAY_MUSIC", "TIME", "PAUSE", "STOP", "NEXT", "VOLUME_UP",
    "VOLUME_DOWN", "CALL", "MESSAGE", "LIST_REMINDERS",
})
WAKE_REJECT_RESCUE_INTENTS = WAKE_SHORT_INTENTS | frozenset({"WEATHER"})
WAKE_REJECT_MIN_COMMAND_CONFIDENCE = 0.05
WAKE_REJECT_MIN_CONDITIONAL_CONFIDENCE = 0.60
WAKE_REJECT_MIN_CONDITIONAL_MARGIN = 0.50
WAKE_REJECT_MAX_CONFIDENCE = 0.90

_HERE = os.path.dirname(os.path.abspath(__file__))
# Ship the raw .onnx. load_model() also accepts a .onnx.gz and decompresses it
# to a sibling .onnx on first use, but the bundle ships the plain model so it
# loads with zero extra steps.
DEFAULT_MODEL = os.path.join(_HERE, "vcm_model", "vcm.onnx")
DEFAULT_LABELS = os.path.join(_HERE, "vcm_model", "labels.json")
DEFAULT_MELBANK = os.path.join(_HERE, "vcm_model", "melbank.npy")
DEFAULT_POLICY = os.path.join(_HERE, "vcm_model", "policy.json")

# ---------------------------------------------------------------------------
# Dual-head label -> (golden intent, slot) mapping.
# Head A (intents): 19 real + REJECT.  Head B (value_tokens): 18 + REJECT.
# ---------------------------------------------------------------------------
INTENT_NAMES = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE",
    "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE",
    "LIST_REMINDERS", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS",
    "COLOR", "CREATE_REMINDER", "REJECT",
]
VALUE_TOKENS = [
    "TIMER_10_SECONDS", "TIMER_30_SECONDS", "TIMER_1_MINUTE",
    "ALARM_6:00_AM", "ALARM_8:00_AM", "ALARM_9:00_PM",
    "TEMPERATURE_18_DEGREES", "TEMPERATURE_22_DEGREES", "TEMPERATURE_26_DEGREES",
    "BRIGHTNESS_20_PERCENT", "BRIGHTNESS_60_PERCENT", "BRIGHTNESS_100_PERCENT",
    "COLOR_RED", "COLOR_BLUE", "COLOR_GREEN",
    "CREATE_REMINDER_DRINK_WATER", "CREATE_REMINDER_STUDY",
    "CREATE_REMINDER_EXERCISE", "REJECT",
]
REJECT_IDX_A = len(INTENT_NAMES) - 1            # 19
REJECT_IDX_B = len(VALUE_TOKENS) - 1            # 18

# slot value text per value token (for the 6 slotted intents)
_SLOT_TEXT = {
    "TIMER_10_SECONDS": "10 seconds", "TIMER_30_SECONDS": "30 seconds",
    "TIMER_1_MINUTE": "1 minute",
    "ALARM_6:00_AM": "6:00 AM", "ALARM_8:00_AM": "8:00 AM", "ALARM_9:00_PM": "9:00 PM",
    "TEMPERATURE_18_DEGREES": "18 degrees", "TEMPERATURE_22_DEGREES": "22 degrees",
    "TEMPERATURE_26_DEGREES": "26 degrees",
    "BRIGHTNESS_20_PERCENT": "20 percent", "BRIGHTNESS_60_PERCENT": "60 percent",
    "BRIGHTNESS_100_PERCENT": "100 percent",
    "COLOR_RED": "Red", "COLOR_BLUE": "Blue", "COLOR_GREEN": "Green",
    "CREATE_REMINDER_DRINK_WATER": "Drink water", "CREATE_REMINDER_STUDY": "Study",
    "CREATE_REMINDER_EXERCISE": "Exercise",
}
# which value tokens belong to which intent
_SLOTTED_PREFIX = {
    "TIMER": "TIMER_", "ALARM": "ALARM_", "TEMPERATURE": "TEMPERATURE_",
    "BRIGHTNESS": "BRIGHTNESS_", "COLOR": "COLOR_", "CREATE_REMINDER": "CREATE_REMINDER_",
}


def intent_schema():
    return list(INTENT_NAMES)


def warmup(sess, n=2):
    """Run a couple of dummy inferences so the first real clip isn't slow."""
    for _ in range(n):
        sess.run(None, {"mel": np.zeros((1, 1, N_MELS, N_FRAMES), np.float32)})


class ProbabilityEnsembleSession:
    """Small ONNX Runtime-compatible facade for weighted model averaging.

    Each member emits intent and slot logits.  We average their softmax
    probabilities and return log probabilities, allowing the existing decoder
    and calibration-policy path to remain unchanged.
    """

    def __init__(self, manifest_path):
        import onnxruntime as ort

        manifest_path = os.path.abspath(os.fspath(manifest_path))
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if manifest.get("type") != "probability_ensemble_onnx_v1":
            raise ValueError("unsupported ensemble manifest type")
        root = os.path.dirname(manifest_path)
        members = manifest.get("members", [])
        if not members:
            raise ValueError("ensemble manifest has no members")
        self.members = []
        total_weight = 0.0
        for member in members:
            weight = float(member["weight"])
            if weight <= 0.0:
                raise ValueError("ensemble weights must be positive")
            model_path = os.path.join(root, member["model"])
            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self.members.append((member["name"], weight, session))
            total_weight += weight
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError("ensemble weights must sum to 1")

    @staticmethod
    def _softmax_batch(logits):
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def run(self, output_names, feed):
        mixed = None
        for _, weight, session in self.members:
            outputs = session.run(None, feed)
            if len(outputs) < 2:
                raise RuntimeError("ensemble member must have two outputs")
            probabilities = [self._softmax_batch(outputs[0]),
                             self._softmax_batch(outputs[1])]
            if mixed is None:
                mixed = [weight * value for value in probabilities]
            else:
                for index, value in enumerate(probabilities):
                    mixed[index] += weight * value
        result = [np.log(np.maximum(value, 1e-12)).astype(np.float32)
                  for value in mixed]
        if output_names is None:
            return result
        raise ValueError("named ensemble outputs are not supported")


def load_model(model_path=DEFAULT_MODEL):
    import onnxruntime as ort
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "ensemble.json")
    if model_path.lower().endswith(".json"):
        return ProbabilityEnsembleSession(model_path)
    # The bundle ships a raw .onnx file. For compatibility with compressed
    # copies, accept .onnx.gz and decompress it once next to the archive.
    if model_path.endswith(".gz"):
        import gzip
        import shutil
        raw_path = model_path[:-3]
        if not os.path.exists(raw_path):
            with gzip.open(model_path, "rb") as fin, open(raw_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        model_path = raw_path
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def load_labels(labels_path=DEFAULT_LABELS):
    with open(labels_path) as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_policy(policy_path=DEFAULT_POLICY):
    """Load the fitted deployment decoder, or return None for old bundles."""
    if not policy_path or not os.path.exists(policy_path):
        return None
    with open(policy_path) as f:
        return json.load(f)


def _load_melbank(path=DEFAULT_MELBANK):
    return np.load(path).astype(np.float32)


def label_to_intent(label, info=None):
    """Map a predicted intent name to (golden_intent, slot_or_None).

    `label` is the intent name returned by classify() (or "REJECT"). The slot,
    when present, comes from `info["slot"]` (resolved from head B during
    classify). Falls back to a value-token lookup for robustness.
    """
    if label == "REJECT" or label is None:
        return (None, None)
    slot = None
    if info is not None:
        slot = info.get("slot")
    if slot is None and label in _SLOT_TEXT:
        slot = _SLOT_TEXT[label]
    return (label, slot)


# ---------------------------------------------------------------------------
# Mel spectrogram (VERIFIED bit-exact vs training battery.wav_to_mel)
# ---------------------------------------------------------------------------
_FB = None


def _get_fb():
    global _FB
    if _FB is None:
        _FB = _load_melbank()
    return _FB


def audio_to_mel(x):
    """x = 16 kHz mono float32, length >= N_SAMPLES -> (1,1,40,250) float32."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)))
    else:
        x = x[:N_SAMPLES]
    win = np.hanning(WIN).astype(np.float32)
    mags = np.empty((N_FRAMES, N_FFT // 2 + 1), dtype=np.float32)
    for t in range(N_FRAMES):
        seg = x[t * HOP: t * HOP + WIN]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        seg = seg * win
        spec = np.fft.rfft(seg, N_FFT)
        mags[t] = (np.abs(spec) ** 2).astype(np.float32)
    mel = mags @ _get_fb().T
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel).T                       # (40, 250)
    logmel = (logmel - MEL_LO) / (MEL_HI - MEL_LO)
    logmel = np.clip(logmel, 0.0, 1.0).astype(np.float32)
    # per-utt CMVN (per-mel mean/std) then rescale to ~[0,1]
    mu = logmel.mean(axis=1, keepdims=True)
    sd = logmel.std(axis=1, keepdims=True) + 1e-6
    logmel = (logmel - mu) / sd
    logmel = (logmel + 3.0) / 6.0
    logmel = np.clip(logmel, 0.0, 1.0).astype(np.float32)
    return logmel[None, None]                      # (1,1,40,250)


# ---------------------------------------------------------------------------
# VAD + placement/level invariance (kept from the working bundle)
# ---------------------------------------------------------------------------
def _energy_envelope(x, frame_ms=VAD_FRAME_MS):
    fs = int(SAMPLE_RATE * frame_ms / 1000)
    n = len(x) // fs
    if n == 0:
        return np.zeros(0, np.float32), fs
    seg = x[:n * fs].reshape(n, fs)
    return np.sqrt((seg ** 2).mean(axis=1)).astype(np.float32), fs


def _detect_speech(x, return_raw=False):
    """Return the VAD span, optionally with the unpadded active-speech span."""
    env, fs = _energy_envelope(x)
    if env.size == 0:
        return None
    peak = float(env.max())
    if peak >= VAD_ABS_FLOOR:
        # Preserve the training/evaluation-era placement for ordinary clips.
        thresh = max(VAD_ABS_FLOOR, VAD_REL_THRESH * peak)
    else:
        # Quiet far-field fallback: use the recording's own noise floor, but
        # require a clear sustained rise over it.  This path does not alter
        # normal-level clips and therefore avoids moving their crop.
        floor = float(np.percentile(env, 20))
        robust_peak = float(np.percentile(env, 95))
        if robust_peak < 0.001 or robust_peak < floor * 1.8:
            return None
        thresh = max(0.001, floor * 1.8,
                     floor + 0.25 * (robust_peak - floor))
    voiced = env >= thresh
    # At least four raw 20-ms frames must contain sustained energy.  Hangover
    # is intentionally checked later because it can make one click look like a
    # half-second utterance.
    if int(voiced.sum()) * VAD_FRAME_MS < 80:
        return None
    raw_idx = np.where(voiced)[0]
    raw_span = (raw_idx[0] * fs / SAMPLE_RATE,
                (raw_idx[-1] + 1) * fs / SAMPLE_RATE)
    # hangover: extend voiced region to bridge word gaps
    hang = max(1, int(VAD_HANGOVER_MS / VAD_FRAME_MS))
    ext = voiced.copy()
    for i in range(len(voiced)):
        if voiced[i]:
            lo = max(0, i - hang)
            hi = min(len(voiced), i + hang + 1)
            ext[lo:hi] = True
    idx = np.where(ext)[0]
    if idx.size == 0:
        return None
    start_s = idx[0] * fs / SAMPLE_RATE
    end_s = (idx[-1] + 1) * fs / SAMPLE_RATE
    if (end_s - start_s) * 1000 < VAD_MIN_SPEECH_MS:
        return None
    span = (start_s, end_s)
    return (span, raw_span) if return_raw else span


def prepare_clip(x, verbose=False):
    """VAD-trim speech, center it in a 2.5 s window, and peak-normalize.

    Returns (clip, info). clip is 16 kHz mono float32 of length N_SAMPLES.
    """
    x = np.asarray(x, dtype=np.float32)
    capture_peak = float(np.max(np.abs(x))) if x.size else 0.0
    capture_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    clipping_fraction = float(np.mean(np.abs(x) >= 0.99)) if x.size else 0.0
    detected = _detect_speech(x, return_raw=True)
    if detected is None:
        # no clear speech: use the raw (zero-padded) window, flag it
        if len(x) < N_SAMPLES:
            clip = np.pad(x, (0, N_SAMPLES - len(x)))
        else:
            clip = x[:N_SAMPLES]
        info = dict(span=None, gain=1.0, note="no clear speech detected",
                    capture_peak=capture_peak, capture_rms=capture_rms,
                    clipping_fraction=clipping_fraction)
        if verbose:
            print("[vad] " + info["note"])
        return clip, info

    span, active_span = detected
    start_s, end_s = span
    active_start_s, active_end_s = active_span
    speech_i0 = max(0, int(round(active_start_s * SAMPLE_RATE)))
    speech_i1 = min(len(x), int(round(active_end_s * SAMPLE_RATE)))
    active_samples = max(0, speech_i1 - speech_i0)

    # Training center_crop() symmetrically padded short recordings and chose
    # the loudest 2.5 s region of long recordings. Runtime previously padded
    # only on the right, so a 1.6 s command sat near the beginning of E8's
    # window. Put the detected active-speech midpoint at the exact model
    # midpoint, retaining only the balanced local context that fits around it.
    speech_midpoint = (speech_i0 + speech_i1) // 2
    source_start = speech_midpoint - N_SAMPLES // 2
    source_end = source_start + N_SAMPLES
    copy_i0 = max(0, source_start)
    copy_i1 = min(len(x), source_end)
    destination_i0 = copy_i0 - source_start
    seg = np.zeros(N_SAMPLES, dtype=np.float32)
    if copy_i1 > copy_i0:
        copied = copy_i1 - copy_i0
        seg[destination_i0:destination_i0 + copied] = x[copy_i0:copy_i1]
    centered_start = max(0, speech_i0 - source_start)
    centered_end = min(N_SAMPLES, speech_i1 - source_start)

    # Peak-normalize using only active speech, not silence/background context.
    ref = x[speech_i0:speech_i1]
    ref_peak = float(np.max(np.abs(ref))) if ref.size else 0.0
    gain = 1.0
    if ref_peak > 1e-6:
        gain = PEAK_TARGET / ref_peak
        seg = seg * gain
    active_speech_seconds = active_samples / float(SAMPLE_RATE)
    info = dict(span=(start_s, end_s), active_span=active_span, gain=gain,
                active_speech_seconds=active_speech_seconds,
                centered_start_seconds=centered_start / float(SAMPLE_RATE),
                capture_peak=capture_peak, capture_rms=capture_rms,
                clipping_fraction=clipping_fraction)
    if verbose:
        print(f"[vad] active speech {active_start_s:.2f}-{active_end_s:.2f} s | "
              f"{active_speech_seconds:.2f} s centered in {WINDOW_SEC:.2f} s, "
              f"gain x{gain:.2f}")
    return seg, info


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _top_margin(probs):
    """Return top-1 minus top-2 probability for a one-dimensional vector."""
    if len(probs) < 2:
        return float(probs[0]) if len(probs) else 0.0
    pair = np.partition(np.asarray(probs), -2)[-2:]
    return float(pair.max() - pair.min())


def _wake_short_decision(intent, conf, info):
    """Whether a wake-armed, short fixed command is sufficiently distinct."""
    if intent not in WAKE_SHORT_INTENTS:
        return False
    span = info.get("span")
    if span is None or len(span) != 2:
        return False
    speech_span = max(0.0, float(span[1]) - float(span[0]))
    return (speech_span <= WAKE_SHORT_MAX_SPAN_SECONDS and
            conf >= WAKE_SHORT_MIN_CONFIDENCE and
            info.get("intent_margin", 0.0) >= WAKE_SHORT_MIN_MARGIN)


def wake_reject_rescue(info):
    """Return a fixed intent hidden under REJECT in wake-armed short audio.

    The command probabilities are renormalized without the REJECT class. This
    is only safe after a valid wake word and only when they strongly agree on
    one eligible fixed intent; generic inference keeps the original argmax.
    """
    # The fitted decoder already compares every command with the learned
    # REJECT class. Do not override that decision with the old hand-tuned
    # wake-word rescue heuristic.
    if info.get("decision_calibrated"):
        return None
    span = info.get("span")
    if span is None or len(span) != 2:
        return None
    speech_span = max(0.0, float(span[1]) - float(span[0]))
    candidate = info.get("command_intent")
    if candidate not in WAKE_REJECT_RESCUE_INTENTS:
        return None
    if speech_span > WAKE_SHORT_MAX_SPAN_SECONDS:
        return None
    if info.get("reject_conf", 1.0) > WAKE_REJECT_MAX_CONFIDENCE:
        return None
    if info.get("command_conf", 0.0) < WAKE_REJECT_MIN_COMMAND_CONFIDENCE:
        return None
    if (info.get("command_conditional_conf", 0.0) <
            WAKE_REJECT_MIN_CONDITIONAL_CONFIDENCE):
        return None
    if (info.get("command_conditional_margin", 0.0) <
            WAKE_REJECT_MIN_CONDITIONAL_MARGIN):
        return None
    return candidate


def safe_decision(intent, conf, info, allow_wake_short=False):
    """Return ``(accepted, reason)`` for the demo-safe action policy.

    Classification remains visible to diagnostics, but the action runner only
    fires when the audio, intent, and (when required) slot are unambiguous.
    """
    if intent in (None, "REJECT") or info.get("rejected"):
        return False, "model_reject"
    if info.get("span") is None:
        return False, "no_speech"
    if info.get("decision_calibrated"):
        if intent in _SLOTTED_PREFIX and info.get("slot") is None:
            return False, "missing_slot"
        return True, None
    if intent in _SLOTTED_PREFIX:
        if conf < MIN_INTENT_CONFIDENCE:
            return False, "low_intent_confidence"
        if info.get("intent_margin", 0.0) < MIN_INTENT_MARGIN:
            return False, "low_intent_margin"
        if info.get("slot") is None:
            return False, "missing_slot"
        if info.get("slot_conf", 0.0) < MIN_SLOT_CONFIDENCE:
            return False, "low_slot_confidence"
        if info.get("slot_margin", 0.0) < MIN_SLOT_MARGIN:
            return False, "low_slot_margin"
        return True, None
    if (conf >= MIN_INTENT_CONFIDENCE and
            info.get("intent_margin", 0.0) >= MIN_INTENT_MARGIN):
        return True, None
    if allow_wake_short and _wake_short_decision(intent, conf, info):
        return True, None
    if conf < MIN_INTENT_CONFIDENCE:
        return False, "low_intent_confidence"
    if info.get("intent_margin", 0.0) < MIN_INTENT_MARGIN:
        return False, "low_intent_margin"
    return True, None


def classify(sess, labels, x, verbose=False, policy=None):
    """Run one clip. x = 16 kHz mono float32 (any length).

    Returns (label, conf, probs, info):
      label = predicted intent name (or "REJECT")
      conf  = softmax probability of the intent head
      probs = intent-head softmax array (len 20)
      info  = dict with slot, slot_conf, vad span/gain, rejected flag
    """
    prep, info = prepare_clip(x, verbose=verbose)
    mel = audio_to_mel(prep)
    la, lb = sess.run(None, {"mel": mel})
    # New bundles use a calibration learned on clean speech, procedural
    # speaker/microphone playback, and out-of-scope audio. It replaces the
    # universal confidence/margin thresholds with one calibrated argmax over
    # all 19 commands plus the trained REJECT class. Passing policy=False keeps
    # compatibility with the original raw-logit E8 behavior.
    if policy is None:
        policy = load_policy()
    decoder = policy.get("decoder", {}) if policy else {}
    decision_calibrated = decoder.get("type") == "temperature_bias_argmax"
    intent_logits = la[0]
    if decision_calibrated:
        temperature = max(float(decoder.get("temperature", 1.0)), 1e-6)
        bias_by_name = decoder.get("intent_bias", {})
        bias = np.asarray([bias_by_name.get(name, 0.0)
                           for name in INTENT_NAMES], dtype=np.float32)
        intent_logits = intent_logits / temperature + bias
    pa = _softmax(intent_logits)
    pb = _softmax(lb[0])
    ia = int(np.argmax(pa))
    ib = int(np.argmax(pb))
    intents = labels.get("intents", INTENT_NAMES)
    vtoks = labels.get("value_tokens", VALUE_TOKENS)
    intent = intents[ia]
    rejected = ia == REJECT_IDX_A
    command_probs = pa[:REJECT_IDX_A]
    command_total = float(command_probs.sum())
    command_conditional = (command_probs / command_total if command_total > 0.0
                           else command_probs)
    command_index = int(np.argmax(command_probs))
    command_intent = intents[command_index]
    slot = None
    slot_conf = 0.0
    slot_margin = 0.0
    slot_token = None
    # Head B covers values from all six slotted intents.  Once head A chooses
    # an intent, compare only the three legal values for that intent.  A global
    # argmax here used to turn e.g. COLOR_RED into COLOR_GREEN/None when an
    # unrelated token narrowly won the head.
    prefix = _SLOTTED_PREFIX.get(intent)
    if not rejected and prefix:
        valid = [i for i, token in enumerate(vtoks) if token.startswith(prefix)]
        if valid:
            # Confidence is conditional on the selected intent: only its three
            # legal slot values compete. A valid slot no longer loses mass to
            # the other fifteen unrelated slot tokens.
            valid_probs = _softmax(lb[0][valid])
            local = int(np.argmax(valid_probs))
            ib = valid[local]
            slot_token = vtoks[ib]
            slot = _SLOT_TEXT.get(slot_token)
            slot_conf = float(valid_probs[local])
            slot_margin = _top_margin(valid_probs)
    info = dict(info)
    info.update(slot=slot, slot_token=slot_token, slot_conf=slot_conf,
                slot_margin=slot_margin, intent_margin=_top_margin(pa),
                rejected=rejected, decision_calibrated=decision_calibrated,
                decoder_type=decoder.get("type") if decoder else "raw_logits",
                command_intent=command_intent,
                command_conf=float(command_probs[command_index]),
                command_conditional_conf=float(command_conditional[command_index]),
                command_conditional_margin=_top_margin(command_conditional),
                reject_conf=float(pa[REJECT_IDX_A]))
    return intent, float(pa[ia]), pa, info


def predict(x, sess=None, labels=None, verbose=False):
    """Convenience: returns a flat dict {intent, slot, conf, rejected, vad}."""
    if sess is None:
        sess = load_model()
    if labels is None:
        labels = load_labels()
    intent, conf, probs, info = classify(sess, labels, x, verbose=verbose)
    return dict(intent=intent, slot=info.get("slot"), conf=conf,
                rejected=info.get("rejected", False), vad=info)
