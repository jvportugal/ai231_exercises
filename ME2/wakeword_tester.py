#!/usr/bin/env python3
"""Interactive Raspberry Pi tester for the bundled "Destiny" wake word.

SPACE starts a test and SPACE stops it. While active, this runs the same
microphone, speech segmentation, model, and threshold as runner.py --wakeword.
Every speech candidate is reported so false triggers are visible.
"""
import argparse
import contextlib
import os
import select
import sys
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER_DIR = os.path.join(_HERE, "vcm_action_runner")
if _RUNNER_DIR not in sys.path:
    sys.path.insert(0, _RUNNER_DIR)

import wakeword as W  # noqa: E402


@contextlib.contextmanager
def _single_key_mode():
    """Make Space available immediately (without requiring Enter)."""
    if not sys.stdin.isatty():
        yield
        return
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_key(timeout=0.0):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    return "q" if ch == "" else ch.lower()


def _wait_to_start():
    print("\n  Press SPACE to START listening, or q to quit.", flush=True)
    with _single_key_mode():
        while True:
            key = _poll_key(0.1)
            if key in (" ", "\n", "\r", "q"):
                return key


def _capture_test(detector, microphone, args):
    """Run one Space-to-Space test. Return False when q requests exit."""
    pre_roll = deque(maxlen=2)
    candidate = []
    in_speech = False
    quiet_blocks = 0
    silence_blocks = max(1, int(round(args.wake_silence * 1000 / W.BLOCK_MS)))
    max_blocks = max(silence_blocks + 1,
                     int(round(args.wake_max_seconds * 1000 / W.BLOCK_MS)))
    candidate_count = 0
    detection_count = 0
    best_probability = 0.0
    started_at = time.monotonic()
    quit_requested = False
    meter = W.LiveLevelMeter("TESTING FOR DESTINY")
    speech_gate = W.AdaptiveSpeechGate(
        args.speech_rms, multiplier=args.noise_floor_multiplier)

    def score_candidate(blocks, endpoint_ms):
        nonlocal candidate_count, detection_count, best_probability
        if not blocks:
            return
        meter.clear()
        score_t0 = time.perf_counter()
        probability = detector.score(np.concatenate(blocks))
        score_ms = (time.perf_counter() - score_t0) * 1000.0
        candidate_count += 1
        best_probability = max(best_probability, probability)
        detected = probability >= args.threshold
        if detected:
            detection_count += 1
        elapsed = time.monotonic() - started_at
        verdict = "DESTINY DETECTED" if detected else "not Destiny"
        print("  [%6.2f s] candidate %d: %-16s p=%.3f "
              "(threshold %.3f, silence %d ms, model %.0f ms)" %
              (elapsed, candidate_count, verdict, probability,
               args.threshold, endpoint_ms, score_ms), flush=True)

    print("  [state] LISTENING FOR DESTINY -- microphone is live", flush=True)
    print("  Say Destiny, other words, or make normal room noise. "
          "Press SPACE to STOP.", flush=True)
    try:
        with _single_key_mode():
            while True:
                block = microphone.read()
                rms = meter.update(block)
                key = _poll_key()
                if key in (" ", "\n", "\r", "q"):
                    quit_requested = key == "q"
                    break

                if len(speech_gate.history) < 8:
                    speech_gate.history.append(rms)
                    continue

                voiced, _ = speech_gate.update(rms, in_speech)
                if not in_speech:
                    pre_roll.append(block)
                    if voiced:
                        in_speech = True
                        candidate = list(pre_roll)
                        quiet_blocks = 0
                    continue

                candidate.append(block)
                quiet_blocks = 0 if voiced else quiet_blocks + 1
                ended = quiet_blocks >= silence_blocks
                capped = len(candidate) >= max_blocks
                if ended or capped:
                    score_candidate(candidate,
                                    quiet_blocks * W.BLOCK_MS if ended else 0)
                    candidate = []
                    pre_roll.clear()
                    quiet_blocks = 0
                    in_speech = False
    finally:
        meter.close()

    # If Space was pressed in the middle of an utterance, score what was
    # captured instead of silently throwing away the final test phrase.
    if candidate:
        score_candidate(candidate, quiet_blocks * W.BLOCK_MS)

    duration = time.monotonic() - started_at
    print("\n  " + "=" * 66)
    if detection_count:
        print("  RESULT: YES -- Destiny was detected %d time(s)." % detection_count)
    else:
        print("  RESULT: NO -- Destiny was not detected.")
    print("  duration %.1f s | speech candidates %d | highest p=%.3f | threshold %.3f" %
          (duration, candidate_count, best_probability, args.threshold))
    if candidate_count == 0:
        print("  No sound crossed the adaptive speech gate (minimum RMS %.4f); "
              "speak closer/louder or lower --speech-rms." % args.speech_rms)
    print("  " + "=" * 66)
    return not quit_requested


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spacebar-controlled Destiny wake-word tester")
    parser.add_argument("--threshold", type=float, default=0.875,
                        help="Destiny detection threshold (default: 0.875)")
    parser.add_argument("--mic-device", type=int, default=None,
                        help="sounddevice input device number")
    parser.add_argument("--speech-rms", type=float, default=0.008,
                        help="RMS that begins a speech candidate (default: 0.008)")
    parser.add_argument("--noise-floor-multiplier", type=float, default=1.65,
                        help="speech rise over adaptive room floor (default: 1.65)")
    parser.add_argument("--wake-silence", type=float, default=0.4,
                        help="silence that ends a candidate (default: 0.4 s)")
    parser.add_argument("--wake-max-seconds", type=float, default=2.0,
                        help="maximum candidate length (default: 2.0 s)")
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.speech_rms <= 0.0:
        parser.error("--speech-rms must be positive")
    if args.noise_floor_multiplier < 1.0:
        parser.error("--noise-floor-multiplier must be at least 1.0")

    print("=" * 70)
    print("  DESTINY WAKE-WORD TESTER")
    print("  SPACE starts/stops | q quits | every speech candidate is scored")
    print("=" * 70)
    print("  Loading and warming the exact model used by runner.py --wakeword...")
    detector = W.WakeWordDetector()
    detector.warmup()

    try:
        with W.Microphone(args.mic_device) as microphone:
            print("  microphone: %s @ %d Hz -> 16000 Hz" %
                  (microphone.device_name, microphone.native_rate))
            print("  threshold: %.3f | speech RMS: %.4f" %
                  (args.threshold, args.speech_rms))
            while True:
                key = _wait_to_start()
                if key == "q":
                    break
                if not _capture_test(detector, microphone, args):
                    break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("\n  [!] wake-word tester error: %s" % exc)
        print("      Run ./setup.sh, then try again.")
        return 1

    print("\n  bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
