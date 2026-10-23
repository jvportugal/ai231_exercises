#!/usr/bin/env python3
"""VCM Action Runner -- wake word / voice / typed command to action.

Modes:
  * --wakeword: persistent mic -> "Destiny" -> endpointed command -> action.
  * --voice: original SPACE/ENTER start-stop recording -> command -> action.
  * default: original typed intent code -> action.

All three modes dispatch through the same intent/action and domain hand-off
logic. Actions render in a GUI window or a live terminal panel when headless.

Threading model (fixes the macOS "NSWindow main thread" crash):
  * The MAIN thread owns the single tkinter window (ui/appkit.App.start()).
    tkinter on macOS may only be driven from the main thread, so we never
    create a Tk window anywhere else.
  * A DAEMON control thread owns terminal input or the persistent microphone
    and posts OPEN/CLOSE requests to the App via its thread-safe queue.
  * Each action runs on its own worker thread; in GUI mode it only posts an
    OPEN request and blocks on a cancel event (it never touches tkinter).

In terminal (headless) mode there is no App: the main thread runs the reader
directly and each action renders a live terminal panel on its worker thread.
"""
import argparse
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import intents
from intents import INTENTS, BY_NAME, CODES, resolve_code
import actions.timers
import actions.sims
from state import DEVICE
from actions.light_driver import LIGHT
from actions.audio_driver import AUDIO
from ui.appkit import App, has_display
from ui.timer_display import CountdownTimer  # noqa: F401  (kept for API parity)
import voice as voice_mod
import wakeword as wakeword_mod

# Blocking handlers (worker-thread entry points).
HANDLERS = dict(actions.sims.HANDLERS)
HANDLERS["TIMER"] = actions.timers.run

# GUI renderers (main-thread, called by the App).
RENDERERS = dict(actions.sims.RENDERERS)
RENDERERS["TIMER"] = actions.timers.RENDERER


# --------------------------------------------------------------------------
# Domain grouping -- drives the "smooth hand-off" rule.
#
# Three families. Staying inside one family keeps the action alive (e.g.
# light->brightness, play->next). Moving to a DIFFERENT family closes the
# current action AND stops the underlying hardware so the new one is clean:
#   * LIGHT  -> leaving it switches the bulb OFF (color+brightness kept, so the
#               next light command restores exactly what it was).
#   * MUSIC  -> leaving it STOPS playback (volume/track kept).
#   * OTHERS -> timers/alarms/climate/comms/reminders; closing just clears the
#               window, nothing persistent to stop.
DOMAINS = {
    "LIGHT":   {"LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR"},
    "MUSIC":   {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"},
    "OTHERS":  {"TIME", "WEATHER", "TEMPERATURE", "ALARM", "CALL", "MESSAGE",
                "LIST_REMINDERS", "CREATE_REMINDER", "TIMER"},
}
_DOMAIN_OF = {}
for _fam, _names in DOMAINS.items():
    for _n in _names:
        _DOMAIN_OF[_n] = _fam


def _family(intent):
    return _DOMAIN_OF.get(intent, "OTHERS")


def _stop_family(fam):
    """Stop the hardware owned by a family when we leave it. State is preserved
    (brightness/color/volume/track) so re-entering the family restores it."""
    try:
        if fam == "LIGHT":
            with DEVICE.locked():
                DEVICE.lights_on = False
            LIGHT.apply(DEVICE)
        elif fam == "MUSIC":
            AUDIO.stop()
            with DEVICE.locked():
                DEVICE.playing = False
                DEVICE.paused = False
    except Exception as e:
        sys.stderr.write("[domain] stop %s failed: %s\n" % (fam, e))


# --------------------------------------------------------------------------
class ActiveAction:
    """Wraps a running action thread with a live/cancel handshake.

    `ready` is set by the worker once the action is actually live. `cancel()`
    waits for `ready` (bounded) before asking the action to stop, so a cancel
    issued during startup is never lost.
    """
    def __init__(self, intent, slot, app=None):
        self.intent = intent
        self.slot = slot
        self.app = app
        self.ready = threading.Event()
        self.cancelled = threading.Event()
        self._cancel_req = threading.Event()
        self.thread = None
        self.state = {"current": intent, "handle": None,
                      "_cancel_req": self._cancel_req, "app": app}

    def start(self):
        def worker():
            handler = HANDLERS[self.intent]
            try:
                handler(self.intent, self.slot, self.state)
            except Exception as e:
                print("  [%s] error: %s" % (self.intent, e))
            finally:
                self.ready.set()  # ensure cancel() never hangs
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def cancel(self, timeout=3.0):
        # Wait (bounded) for the action to be live so the cancel isn't lost.
        self.ready.wait(timeout)
        self._cancel_req.set()
        self.cancelled.set()
        if self.app is not None:
            self.app.close()


CURRENT = {"act": None}


def prompt_slot(intent):
    slots = BY_NAME[intent][2]
    if not slots:
        return None
    print("  -- %s slots --" % intent)
    for i, s in enumerate(slots, start=1):
        print("     %d) %s" % (i, s))
    while True:
        raw = input("  pick a slot (1-%d) or blank to skip: " % len(slots)).strip()
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(slots):
            return slots[int(raw) - 1]
        # allow typing the raw slot value directly
        if raw in slots:
            return raw
        print("  (enter a number 1-%d, the exact value, or blank)" % len(slots))


def print_menu():
    print("=" * 58)
    print("  VCM ACTION RUNNER   (type a code + ENTER, 'q' to quit)")
    print("=" * 58)
    for i, entry in enumerate(INTENTS, start=1):
        name = entry[0]
        slots = entry[3]
        code = CODES[name]
        tag = ("  [%s]" % ", ".join(slots)) if slots else ""
        print("  %2d  %-4s  %-16s%s" % (i, code, name, tag))
    print("-" * 58)


def _open_intent(intent, slot, app):
    """Close the current action (with cross-domain hand-off) and open `intent`
    with `slot`. Shared by the typed-code path and the voice path so BOTH get
    identical domain behaviour (unrelated music cmd stops the light, leaving
    the light domain turns the bulb off, etc.)."""
    new_fam = _family(intent)
    cur = CURRENT["act"]
    if cur is not None:
        old_fam = _family(cur.intent)
        print("\n  >> closing running action: %s" % cur.intent)
        cur.cancel()
        CURRENT["act"] = None
        # Cross-domain hand-off: leaving a family stops its hardware so the
        # incoming action starts clean. Same-family keeps it running (smooth).
        if old_fam != new_fam:
            _stop_family(old_fam)
            if old_fam == "LIGHT":
                print("  [domain] lights switched OFF (color+brightness saved)")
            elif old_fam == "MUSIC":
                print("  [domain] music stopped (volume+track saved)")

    print("  >> opening action: %s%s" % (intent, ("  slot=%s" % slot) if slot else ""))
    act = ActiveAction(intent, slot, app=app)
    act.start()
    CURRENT["act"] = act
    return act


def _dispatch(code_or_num, app):
    """Resolve a code/number, prompt for a slot, close current, open new."""
    intent = resolve_code(code_or_num)
    if intent is None:
        print("  (unknown code '%s' -- try the menu)" % code_or_num)
        return

    slot = prompt_slot(intent)
    if slot is None and BY_NAME[intent][2]:
        if intent == "TIMER":
            slot = BY_NAME["TIMER"][2][1]
            print("  (no slot chosen -> defaulting to '%s')" % slot)
        else:
            print("  (skipped %s)" % intent)
            return

    _open_intent(intent, slot, app)
    print("  [running -- type a new code + ENTER to switch, 'q' to quit]")


# --------------------------------------------------------------------------
# VOICE mode: mic -> VCM -> intent -> the SAME _open_intent dispatch above.
# --------------------------------------------------------------------------
def _handle_voice_result(r, app, running_message):
    """Print and dispatch one classification from either microphone mode."""
    print(voice_mod.describe(r))
    if r.get("cancelled"):
        return False
    if not r.get("ok"):
        return True  # REJECT / no command -> just listen again
    intent = r["intent"]
    slot = r.get("slot")
    if BY_NAME.get(intent, (None, None, []))[2] and slot is None:
        print("  [voice] no action: %s requires a clearly recognised value" % intent)
        return True
    # Normalise a slot the model emitted to a golden value when possible
    # (head-B tokens map 1:1, but be tolerant of case/spacing differences).
    if slot is not None:
        gold = BY_NAME.get(intent, (None, None, []))[2]
        if gold and slot not in gold:
            low = {g.lower().strip(): g for g in gold}
            slot = low.get(str(slot).lower().strip(), slot)
    _open_intent(intent, slot, app)
    print(running_message)
    return True


def _voice_dispatch(vcm, app):
    """One keyboard listen->classify->act cycle."""
    return _handle_voice_result(
        vcm.listen_and_classify(), app,
        "  [running -- press SPACE to say a new command, 'q' to quit]")


def print_voice_menu(wakeword=False):
    """The command list shown in voice mode -- the same 19 intents as code
    mode, so the user always sees what can be said."""
    print("=" * 58)
    if wakeword:
        print('  VCM WAKE-WORD MODE   (say "Destiny" before each command)')
    else:
        print("  VCM VOICE MODE   (SPACE = record, SPACE = stop, 'q' = quit)")
    print("  Say ONE of these commands:")
    print("=" * 58)
    for i, entry in enumerate(INTENTS, start=1):
        name = entry[0]
        slots = entry[3]
        tag = ("  [%s]" % ", ".join(slots)) if slots else ""
        print("  %2d  %-16s%s" % (i, name, tag))
    print("-" * 58)


def _voice_loop(vcm, app, quit_flag):
    """Terminal loop for --voice.

    Exactly the flow requested:
      * print the list of commands once,
      * SPACE to record -> SPACE to stop -> VCM classifies,
      * print the RESULT of that listen (heard / rejected / no-speech +
        confidence + which action fired),
      * repeat. 'q' quits.
    """
    print_voice_menu()
    n = 0
    while not quit_flag.is_set():
        n += 1
        print("\n  [listen %d] press SPACE to record..." % n)
        try:
            if not _voice_dispatch(vcm, app):
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("  [voice] error: %s" % e)
    if CURRENT["act"] is not None:
        CURRENT["act"].cancel()
        CURRENT["act"] = None


def _wakeword_loop(vcm, app, quit_flag, args):
    """Hands-free Destiny -> command -> action loop using one mic stream."""
    print_voice_menu(wakeword=True)
    print("  Wake-word control is ON: no SPACE/ENTER input is needed.")
    print("  Say: \"Destiny\", wait for LISTENING, then say one command.")
    print("  Press CTRL+C to quit.")
    try:
        print("  [voice] loading and warming VCM...")
        vcm.warmup()
        with wakeword_mod.WakeWordSession(
                threshold=args.wakeword_threshold,
                device=args.mic_device,
                command_timeout=args.command_timeout,
                speech_start_timeout=args.speech_start_timeout,
                silence_seconds=args.command_silence,
                speech_rms=args.speech_rms,
                noise_floor_multiplier=args.noise_floor_multiplier,
                chime=not args.no_wake_chime,
                chime_volume=args.wake_chime_volume,
                chime_path=args.wake_chime) as session:
            while not quit_flag.is_set():
                audio = session.wait_for_command(quit_flag)
                if audio is None:
                    continue
                result = vcm.classify_audio(audio)
                # Join the wake/endpoint timings to the VCM inference timing
                # so one result line explains where the perceived delay came
                # from on this particular Raspberry Pi.
                result.setdefault("latency", {}).update(session.last_timing)
                _handle_voice_result(
                    result, app,
                    "  [running -- say \"Destiny\" for a new command]")
                print("  [state] RETURNING TO WAKE-WORD MODE", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("  [wakeword] error: %s" % exc)
        print("  Run ./setup.sh, then verify the mic with: python3 tester.py")
    finally:
        quit_flag.set()
        if CURRENT["act"] is not None:
            CURRENT["act"].cancel()
            CURRENT["act"] = None


def _reader_loop(app, quit_flag):
    """Daemon-thread terminal loop (GUI mode) or main-thread loop (TERM mode)."""
    print_menu()
    while not quit_flag.is_set():
        try:
            raw = input("\naction> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue
        _dispatch(raw, app)
    # On exit, close any running action.
    if CURRENT["act"] is not None:
        CURRENT["act"].cancel()
        CURRENT["act"] = None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="VCM command-to-action runner")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="force GUI actions")
    display.add_argument("--term", action="store_true", help="force terminal actions")
    parser.add_argument("--voice", action="store_true",
                        help="keyboard-controlled mic recording")
    parser.add_argument("--wakeword", action="store_true",
                        help='hands-free: say "Destiny" before each command')
    parser.add_argument("--wakeword-threshold", type=float, default=0.875,
                        help="Destiny probability threshold (default: 0.875)")
    parser.add_argument("--no-wake-chime", action="store_true",
                        help="disable the sound played after Destiny is detected")
    parser.add_argument("--wake-chime", default=None, metavar="PCM_WAV",
                        help="use a custom PCM WAV instead of the built-in wake chime")
    parser.add_argument("--wake-chime-volume", type=float, default=0.22,
                        help="wake chime volume multiplier (default: 0.22)")
    parser.add_argument("--mic-device", type=int, default=None,
                        help="sounddevice input device number")
    parser.add_argument("--speech-rms", type=float, default=0.008,
                        help="minimum speech-start RMS (default: 0.008)")
    parser.add_argument("--noise-floor-multiplier", type=float, default=1.65,
                        help="speech must exceed the adaptive room floor by this factor")
    parser.add_argument("--command-timeout", type=float, default=6.0,
                        help="maximum command speech seconds")
    parser.add_argument("--speech-start-timeout", type=float, default=4.0,
                        help="seconds to wait for speech after Destiny")
    parser.add_argument("--command-silence", type=float, default=0.8,
                        help="trailing silence seconds that ends a command")
    args = parser.parse_args(argv)
    if not 0.0 <= args.wake_chime_volume <= 1.0:
        parser.error("--wake-chime-volume must be between 0 and 1")
    if args.speech_rms <= 0.0:
        parser.error("--speech-rms must be positive")
    if args.noise_floor_multiplier < 1.0:
        parser.error("--noise-floor-multiplier must be at least 1.0")
    mode = "gui" if args.gui else ("term" if args.term else "auto")
    wakeword_mode = args.wakeword
    voice_mode = args.voice or wakeword_mode

    # GUI is the DEFAULT. The terminal "box" is only a headless escape hatch
    # (explicit --term). If there's no display, we say so and stop instead of
    # silently drawing a box in the terminal.
    if mode == "term":
        use_gui = False
    elif mode == "gui":
        use_gui = True
    else:  # auto -> prefer GUI
        use_gui = has_display()

    # The light driver already printed its own status line at import time
    # ("gpiozero RGBLED ready on BCM 17/27/22" or "-> SIMULATED mode"), so
    # there is nothing extra to surface here.

    # Build the mic->intent bridge once (model loads lazily on first listen).
    vcm = voice_mod.Voice() if voice_mode else None

    if use_gui:
        # Build the App on the MAIN thread and run its tkinter loop there.
        # The reader runs on a daemon thread and talks to the App via its queue.
        app = App(RENDERERS, title="VCM", width=520, height=430)
        quit_flag = threading.Event()

        def reader():
            try:
                if wakeword_mode:
                    _wakeword_loop(vcm, app, quit_flag, args)
                elif voice_mode:
                    _voice_loop(vcm, app, quit_flag)
                else:
                    _reader_loop(app, quit_flag)
            except Exception as e:
                print("  [reader] %s" % e)
            finally:
                # Reader is done: close any action, then ask the App (main
                # thread) to tear the window down. We schedule the destroy via
                # the App's own queue so it runs on the main thread.
                quit_flag.set()
                try:
                    app.close()
                except Exception:
                    pass
                # Post a special QUIT so the App destroys itself on main thread.
                app.q.put(("quit", None, None, None))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        app.start()  # blocks on the main thread until the window is destroyed
    else:
        if mode == "term":
            print("  [terminal mode -- headless panels in this terminal]")
        else:
            print("  [NO DISPLAY DETECTED -- the GUI window can't open.]")
            print("  You're running over SSH without X11 forwarding.")
            print("  FIX (from your laptop):  ssh -X user@rpi5.local   then run again,")
            print("  OR run this ON the Pi desktop (not over SSH) for the window.")
            print("  Continuing in terminal mode for now...")
        quit_flag = threading.Event()
        if wakeword_mode:
            _wakeword_loop(vcm, None, quit_flag, args)
        elif voice_mode:
            _voice_loop(vcm, None, quit_flag)
        else:
            _reader_loop(None, quit_flag)
    LIGHT.cleanup()
    print("\n  bye.")


if __name__ == "__main__":
    main()
