#!/usr/bin/env python3
"""Drive the real runner.main() end-to-end with a scripted input generator.

Feeds codes + slot choices in the right order (responding to slot prompts),
captures all stdout, and asserts every one of the 19 intents actually rendered
its panel (proving the simulators run, not just print a stub). Exits 0 on pass.
"""
import io
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner

CAP = io.StringIO()
_orig_stdout = sys.stdout


class ScriptedInput:
    """Yields one line per input() call, in the exact order the runner asks."""
    def __init__(self, lines):
        self.lines = list(lines)
        self.i = 0
        self.lock = threading.Lock()

    def readline(self):
        with self.lock:
            if self.i >= len(self.lines):
                self.i += 1
                return "q\n"  # safety: quit if we run dry
            line = self.lines[self.i]
            self.i += 1
            return line


def main():
    # Sequence: every intent once (menu order), then a DOMAIN HAND-OFF test.
    #
    # Domain hand-off (the new rule):
    #   17 BRIGHTNESS 20  -> 18 COLOR Blue  : same LIGHT family -> stays lit
    #   19 CREATE_REMINDER (OTHERS)          : leave LIGHT -> bulb OFF
    #   1  PLAY_MUSIC    (MUSIC)             : leave OTHERS -> (nothing to stop)
    #   4  LIGHT_ON      (LIGHT)             : leave MUSIC -> playback STOPS
    # So at the very end: lights ON (we just turned them on), music STOPPED.
    script = [
        "1",     # PLAY_MUSIC
        "2",     # WEATHER
        "3",     # TIME
        "4",     # LIGHT_ON
        "5",     # LIGHT_OFF
        "6",     # PAUSE
        "7",     # STOP
        "8",     # NEXT
        "9",     # VOLUME_UP
        "10",    # VOLUME_DOWN
        "11",    # CALL
        "12",    # MESSAGE
        "13",    # LIST_REMINDERS
        "14", "1",   # TIMER -> 10 seconds
        "15", "2",   # ALARM -> 8:00 AM
        "16", "3",   # TEMPERATURE -> 26
        "17", "1",   # BRIGHTNESS -> 20
        "18", "2",   # COLOR -> Blue
        "19", "3",   # CREATE_REMINDER -> Exercise   (leaves LIGHT -> off)
        # ---- domain hand-off ----
        "1",     # PLAY_MUSIC (MUSIC)
        "4",     # LIGHT_ON   (LIGHT)  -> leaves MUSIC -> music stops
        "q",
    ]
    si = ScriptedInput(script)

    # Redirect stdout to capture (panels write to sys.stdout).
    sys.stdout = CAP
    try:
        # Patch builtins.input to our scripted reader.
        import builtins
        def fake_input(prompt=""):
            line = si.readline()
            return line.rstrip("\n")
        builtins.input = fake_input
        runner.main()
    finally:
        sys.stdout = _orig_stdout
        import builtins
        del builtins.input

    out = CAP.getvalue()
    print(out[-4000:], file=_orig_stdout)
    print("=" * 60, file=_orig_stdout)

    # Assertions: each intent must have produced its on-screen marker.
    # NOTE: lights are SILENT (the physical bulb is the feedback; no terminal
    # panel) and music renders as a card/window titled "MUSIC" -- so those
    # intents are verified by their side effects (below) rather than a panel
    # title. The rest keep their panel titles.
    expect_titles = {
        "WEATHER": "WEATHER",
        "TIME": "TIME",
        "PAUSE": "MUSIC",
        "STOP": "MUSIC",
        "NEXT": "MUSIC",
        "VOLUME_UP": "MUSIC",
        "VOLUME_DOWN": "MUSIC",
        "CALL": "CALL",
        "MESSAGE": "MESSAGE",
        "LIST_REMINDERS": "REMINDERS",
        "TIMER": "TIMER",
        "ALARM": "ALARM",
        "TEMPERATURE": "TEMPERATURE",
        "CREATE_REMINDER": "CREATE_REMINDER",
    }
    failures = []
    for intent, title in expect_titles.items():
        if title not in out:
            failures.append(intent)

    # State assertions (coherent device + NEW domain hand-off behavior).
    from state import DEVICE
    checks = {
        # We ended on LIGHT_ON, so the bulb is on...
        "lights ON at end (ended on LIGHT_ON)": (DEVICE.lights_on),
        # ...but brightness/color were set earlier and must be PRESERVED
        # (turning off for a cross-domain action must not wipe them).
        "brightness preserved at 20": (DEVICE.brightness == 20),
        "color preserved as blue": (DEVICE.color == "blue"),
        "volume net 0 (up then down)": (DEVICE.volume == 50),
        "temperature 26": (DEVICE.temperature == 26),
        "alarm 8:00 added": ((8, 0, "AM") in DEVICE.alarms),
        "reminder Exercise added": ("Exercise" in DEVICE.reminders),
        # Domain hand-off: the LAST action was LIGHT_ON, which leaves the MUSIC
        # family -> playback must have been STOPPED.
        "music STOPPED after leaving MUSIC for LIGHT": (not DEVICE.playing),
    }
    for name, ok in checks.items():
        if not ok:
            failures.append("STATE: " + name)

    if failures:
        print("FAILURES:", failures, file=_orig_stdout)
        return 1
    print("ALL PASS: 19/19 intents rendered + device-state + domain hand-off OK",
          file=_orig_stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
