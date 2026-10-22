"""Real action simulators for the 18 non-TIMER intents.

Two presentation layers share ONE set of renderers:
  * GUI  (ui/appkit.py)  -- a main-thread tkinter window, correct on macOS and
                            Raspberry Pi. The app calls RENDERERS[intent](...).
  * TERM (ui/panel.py)   -- a headless terminal "window" for machines with no
                            display (this sandbox, or an RPi console).

Each renderer has the signature:
    renderer(intent, slot, state, ctx) -> list[str]
where `ctx` is a small mutable namespace (ctx.tick, ctx.data) the caller hands
back in every tick. Renderers read/write the shared DEVICE singleton so lighting
/ media / volume / reminders behave like one coherent device.

Handlers (the blocking entry points runner.py calls) are:
    handler(intent, slot, state)
They set the device state, then block until the dispatcher cancels them (or, in
terminal mode, their lifetime elapses). In GUI mode they post an OPEN request to
the app and wait on a cancel event.
"""
import datetime as _dt
import time

from intents import SLOT_NORM
from state import DEVICE
from ui.panel import Panel
try:
    from weather import get_weather
except ImportError:
    from .weather import get_weather
try:
    from light_driver import LIGHT
except ImportError:
    from .light_driver import LIGHT
try:
    from audio_driver import AUDIO
except ImportError:
    from .audio_driver import AUDIO


# ================================================================ helpers ====
def _bar(value, width=20, char="#", empty="-"):
    v = max(0, min(width, int(round(value / 100.0 * width))))
    return char * v + empty * (width - v)


def _now():
    return _dt.datetime.now()


def _wait_cancel(state, timeout=None):
    """Block (on a worker thread) until the dispatcher cancels this action.

    `timeout` (seconds) lets fire-and-hold actions auto-close after a while,
    matching the old terminal lifetimes. Returns True if cancelled, False if it
    timed out on its own.
    """
    req = state.get("_cancel_req")
    if req is None:
        time.sleep(0.5)
        return False
    if timeout is None:
        req.wait()
        return True
    return req.wait(timeout)


# ================================================================ TIME ========
def r_time(intent, slot, state, ctx):
    n = _now()
    return [
        "",
        "    " + n.strftime("%A"),
        "",
        "    " + n.strftime("%H:%M:%S"),
        "",
        "    " + n.strftime("%d %B %Y"),
        "",
    ]


# ================================================================ WEATHER =====
def r_weather(intent, slot, state, ctx):
    weather = state.get("_weather") or {"status": "loading"}
    if weather.get("status") == "loading":
        pulse = (".", "..", "...")[ctx.tick % 3]
        return [
            "",
            "  [ WEATHER LINK ]",
            "",
            "  QUEZON CITY, PH",
            "",
            "  RECEIVING CURRENT DATA%s" % pulse,
            "",
            "  API: OPEN-METEO",
        ]

    live = bool(weather.get("live"))
    badge = "[ LIVE // %d-MIN MODEL ]" % weather["interval_minutes"] \
        if live else "[ SIMULATED // OFFLINE ]"
    lines = [
        "",
        "  %s" % badge,
        "  %s" % weather["location"],
        "  --------------------------------",
        "",
        "  %4.1f C   %s" % (weather["temperature"], weather["condition"]),
        "  FEELS     %4.1f C" % weather["apparent_temperature"],
        "  HUMIDITY  %3d %%" % weather["humidity"],
        "  CLOUD     %3d %%" % weather["cloud_cover"],
        "  RAIN      %4.1f mm" % weather["precipitation"],
        "  WIND      %4.1f km/h  %s" % (
            weather["wind_speed"], weather["wind_direction"]),
        "",
        "  OBSERVED  %s" % weather["observed_at"],
        "  RETRIEVED %s" % weather["retrieved_at"],
        "  SOURCE    %s" % weather["source"],
    ]
    if not live:
        lines += ["", "  FALLBACK  %s" % weather.get("error", "OFFLINE")]
    return [
        line[:44] for line in lines
    ]


# ================================================================ LIGHTING ====
_LIGHT_COLORS = {"red": "RED   ", "blue": "BLUE  ", "green": "GREEN ",
                 "white": "WHITE "}


def r_light(intent, slot, state, ctx):
    # Lighting is silent in the terminal: the physical bulb IS the feedback.
    # The handler already called LIGHT.apply(DEVICE) once, so there is nothing
    # to redraw each tick. Returning [] keeps the terminal completely quiet
    # (no per-tick panel, no flicker) for ON / OFF / BRIGHTNESS / COLOR.
    return []


# ================================================================ MEDIA =======
_MEDIA_LEVELS = (0, 25, 50, 75, 100)


def _level_idx(pct):
    """Nearest 0/25/50/75/100 step to `pct` (for the UI level indicator)."""
    return min(range(len(_MEDIA_LEVELS)),
               key=lambda k: abs(_MEDIA_LEVELS[k] - pct))


def _fmt_clock(secs):
    """Render a seconds value as m:ss."""
    s = max(0, int(round(secs)))
    return "%02d:%02d" % (s // 60, s % 60)


def r_media(intent, slot, state, ctx):
    """Persistent music-player UI (no box border, no per-tick redraw).

    Shows the real track, the live status (PLAYING / PAUSED / STANDBY), the
    real elapsed time (from the audio engine), and the volume snapped to the
    0/25/50/75/100 steps. In GUI mode this renders inside the dedicated
    music WINDOW; in terminal mode it is printed once per command.
    """
    with DEVICE.locked():
        playing = DEVICE.playing
        paused = DEVICE.paused
        vol = DEVICE.volume
        idx = DEVICE.track_idx
        track = DEVICE.playlist[idx]
        ntracks = len(DEVICE.playlist)

    # Real playback position from the audio engine (seconds + 0.0-1.0 frac).
    elapsed = AUDIO.elapsed_seconds()
    frac = AUDIO.progress()
    if not playing:
        status = "STANDBY"
        elapsed = 0.0
        frac = 0.0
    elif paused:
        status = "PAUSED"
    else:
        status = "PLAYING"

    li = _level_idx(vol)
    lv = _MEDIA_LEVELS[li]
    return [
        "",
        "  MUSIC",
        "  ------",
        "  Status : %s" % status,
        "  Track  : %d/%d  %s" % (idx + 1, ntracks, track),
        "  Time   : %s" % _fmt_clock(elapsed),
        "  Vol    : %s  (%d%%)" % ("|" + "#" * li + "-" * (4 - li) + "|", lv),
        "",
    ]


# ================================================================ CLIMATE =====
def r_climate(intent, slot, state, ctx):
    with DEVICE.locked():
        target = DEVICE.temperature
    cur = target - max(0, 3 - ctx.tick)
    mode = "HEATING" if cur < target else ("COOLING" if cur > target else "HOLD")
    return [
        "",
        "    Thermostat",
        "",
        "    Set point : %d degrees" % target,
        "    Current   : %d degrees" % cur,
        "    Mode      : %s" % mode,
        "",
        "    [%s]" % _bar((cur - 15) * 10, char="#"),
        "",
    ]


# ================================================================ ALARM =======
def r_alarm(intent, slot, state, ctx):
    with DEVICE.locked():
        alarms = list(DEVICE.alarms)
    lines = ["", "    Alarms", ""]
    if not alarms:
        lines.append("    (none set)")
    for a in alarms:
        lines.append("    * %02d:%02d %s" % a)
    lines += ["", "    (next ring simulated)"]
    return lines


# ================================================================ COMMS =======
def r_call(intent, slot, state, ctx):
    secs = ctx.tick
    mm, ss = divmod(secs, 60)
    return [
        "",
        "    Phone",
        "",
        "    Status : Calling...",
        "    Elapsed: %02d:%02d" % (mm, ss),
        "",
        "    (type a new code to hang up)",
    ]


_MESSAGES = [
    "Hey, are we still on for tonight?",
    "Running 10 min late, heads up.",
    "Did you see the game last night?",
    "Can you pick up milk on the way home?",
]


def r_message(intent, slot, state, ctx):
    msg = _MESSAGES[ctx.tick % len(_MESSAGES)]
    return [
        "",
        "    Messages",
        "",
        "    Draft :",
        '     "',
        "     %s" % msg,
        '     "',
        "",
        "    (type a new code to send & close)",
    ]


# ================================================================ REMINDERS ===
def r_list_reminders(intent, slot, state, ctx):
    with DEVICE.locked():
        rem = list(DEVICE.reminders)
    lines = ["", "  REMINDERS", "  ---------", ""]
    if not rem:
        lines += [
            "  (none yet)",
            "",
            "  Say:  Remind me to <task>",
            "  to add one, then Show reminders again.",
            "",
        ]
        return lines
    for i, r in enumerate(rem, start=1):
        lines.append("  %d.  %s" % (i, r))
    lines += ["", "  (%d total)" % len(rem)]
    return lines


def r_create_reminder(intent, slot, state, ctx):
    with DEVICE.locked():
        rem = list(DEVICE.reminders)
    added = rem[-1] if rem else slot
    lines = [
        "",
        "  REMINDERS",
        "  ---------",
        "",
        '  + "%s"' % added,
        "    (saved)",
        "",
    ]
    if len(rem) > 1:
        lines.append("  Now on your list:")
        for i, r in enumerate(rem, start=1):
            lines.append("  %d.  %s" % (i, r))
    else:
        lines += ["", "  Say:  Show reminders   to see your list."]
    return lines


# ================================================================ RENDERERS ===
RENDERERS = {
    "TIME": r_time,
    "WEATHER": r_weather,
    "LIGHT_ON": r_light,
    "LIGHT_OFF": r_light,
    "BRIGHTNESS": r_light,
    "COLOR": r_light,
    "PLAY_MUSIC": r_media,
    "PAUSE": r_media,
    "STOP": r_media,
    "NEXT": r_media,
    "VOLUME_UP": r_media,
    "VOLUME_DOWN": r_media,
    "TEMPERATURE": r_climate,
    "ALARM": r_alarm,
    "CALL": r_call,
    "MESSAGE": r_message,
    "LIST_REMINDERS": r_list_reminders,
    "CREATE_REMINDER": r_create_reminder,
}

# Per-intent terminal lifetime (seconds) -- mirrors the old panel behavior so a
# fire-and-hold action doesn't run forever in headless mode.
LIFETIMES = {
    "TIME": 12, "WEATHER": 15, "LIGHT_ON": 8, "LIGHT_OFF": 8, "BRIGHTNESS": 8,
    "COLOR": 8, "PLAY_MUSIC": None, "PAUSE": None, "STOP": None, "NEXT": None,
    "VOLUME_UP": None, "VOLUME_DOWN": None, "TEMPERATURE": 8, "ALARM": 8,
    "CALL": 20, "MESSAGE": 10, "LIST_REMINDERS": 10, "CREATE_REMINDER": 8,
}

# Per-intent terminal repaint interval (seconds).
INTERVALS = {
    "TIME": 1.0, "WEATHER": 2.0, "LIGHT_ON": 1.0, "LIGHT_OFF": 1.0,
    "BRIGHTNESS": 1.0, "COLOR": 1.0, "PLAY_MUSIC": 1.0, "PAUSE": 1.0,
    "STOP": 1.0, "NEXT": 1.0, "VOLUME_UP": 1.0, "VOLUME_DOWN": 1.0,
    "TEMPERATURE": 1.0, "ALARM": 1.0, "CALL": 1.0, "MESSAGE": 1.5,
    "LIST_REMINDERS": 1.0, "CREATE_REMINDER": 1.0,
}


# ================================================================ HANDLERS ====
def _apply_side_effect(intent, slot, state):
    """Mutate the shared DEVICE for this intent (idempotent, safe to re-call)."""
    if intent == "LIGHT_ON":
        with DEVICE.locked():
            DEVICE.lights_on = True
        LIGHT.apply(DEVICE)
    elif intent == "LIGHT_OFF":
        with DEVICE.locked():
            DEVICE.lights_on = False
        LIGHT.apply(DEVICE)
    elif intent == "BRIGHTNESS":
        pct = SLOT_NORM["BRIGHTNESS"].get(slot)
        if pct is None:
            return False
        with DEVICE.locked():
            DEVICE.brightness = pct
            if not DEVICE.lights_on:
                DEVICE.lights_on = True
        LIGHT.apply(DEVICE)
    elif intent == "COLOR":
        col = (slot or "").lower()
        if col not in _LIGHT_COLORS:
            return False
        with DEVICE.locked():
            DEVICE.color = col
            if not DEVICE.lights_on:
                DEVICE.lights_on = True
        LIGHT.apply(DEVICE)
    elif intent == "PLAY_MUSIC":
        with DEVICE.locked():
            DEVICE.playing = True
            DEVICE.paused = False
            idx = DEVICE.track_idx
        # Real audio: resume if this track is already loaded & paused, else load.
        AUDIO.play(idx)
    elif intent == "PAUSE":
        with DEVICE.locked():
            if DEVICE.playing:
                DEVICE.paused = True
        AUDIO.pause()
    elif intent == "STOP":
        with DEVICE.locked():
            DEVICE.playing = False
            DEVICE.paused = False
            DEVICE.track_idx = 0
        AUDIO.stop()
    elif intent == "NEXT":
        with DEVICE.locked():
            DEVICE.track_idx = (DEVICE.track_idx + 1) % len(DEVICE.playlist)
            if not DEVICE.playing:
                DEVICE.playing = True
                DEVICE.paused = False
            idx = DEVICE.track_idx
        AUDIO.play(idx)
    elif intent == "VOLUME_UP":
        with DEVICE.locked():
            cur = DEVICE.volume
            # Snap to the next 0/25/50/75/100 step above the current level.
            nxt = min([lv for lv in _MEDIA_LEVELS if lv > cur], default=100)
            DEVICE.volume = nxt
            vol = DEVICE.volume
        AUDIO.set_volume(vol)
    elif intent == "VOLUME_DOWN":
        with DEVICE.locked():
            cur = DEVICE.volume
            # Snap to the next 0/25/50/75/100 step below the current level.
            prv = max([lv for lv in _MEDIA_LEVELS if lv < cur], default=0)
            DEVICE.volume = prv
            vol = DEVICE.volume
        AUDIO.set_volume(vol)
    elif intent == "TEMPERATURE":
        deg = SLOT_NORM["TEMPERATURE"].get(slot)
        if deg is None:
            return False
        with DEVICE.locked():
            DEVICE.temperature = deg
    elif intent == "ALARM":
        spec = SLOT_NORM["ALARM"].get(slot)
        if spec is None:
            return False
        with DEVICE.locked():
            DEVICE.add_alarm(spec)
    elif intent == "CREATE_REMINDER":
        with DEVICE.locked():
            DEVICE.add_reminder(slot)
    return True


# Intents that produce NO terminal UI at all. Their feedback is physical (the
# RGB bulb) or instant (a state change), so drawing a live panel would only add
# per-tick noise. For these we apply the effect once and then quietly wait for
# the dispatcher to cancel us (i.e. until the next command arrives).
_SILENT_INTENTS = {"LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR"}
# Media intents draw a PERSISTENT, box-free player card: painted once per
# command and left on screen (no per-tick redraw, no bordered box) until the
# next command replaces it.
_MEDIA_INTENTS = {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT",
                  "VOLUME_UP", "VOLUME_DOWN"}


def _make_handler(intent):
    def handler(i, slot, state):
        ok = _apply_side_effect(intent, slot, state)
        if not ok:
            print("  [%s] invalid slot '%s'" % (intent, slot))
            return
        app = state.get("app")
        if intent == "WEATHER":
            state["_weather"] = {"status": "loading"}
            if app is not None:
                # Show a loading card immediately. This handler already runs
                # on a worker thread, so the HTTPS request cannot freeze Tk.
                app.open(intent, slot, state)
                state["_weather"] = get_weather(timeout=3.5)
                req = state.get("_cancel_req")
                if req is not None:
                    req.wait()
                return
            # Headless mode has no event loop to keep animated during the
            # request; fetch first, then render live or simulated data.
            state["_weather"] = get_weather(timeout=3.5)
        if app is not None:
            # GUI mode: post OPEN, then block until cancelled.
            app.open(intent, slot, state)
            req = state.get("_cancel_req")
            if req is not None:
                req.wait()
        elif intent in _SILENT_INTENTS:
            # No terminal UI: the effect is already applied (e.g. LIGHT.apply
            # drove the bulb). Print nothing and simply wait until the
            # dispatcher cancels us when the next command arrives.
            cancel = state.get("_cancel_req")
            if cancel is not None:
                cancel.wait()
        elif intent in _MEDIA_INTENTS:
            # Persistent music UI: paint the player card ONCE (no box, no
            # per-tick redraw) and wait until the dispatcher cancels us.
            lines = RENDERERS[intent](
                intent, slot, state,
                type("C", (), {"tick": 0, "data": {}})())
            for ln in lines:
                print(ln)
            cancel = state.get("_cancel_req")
            if cancel is not None:
                cancel.wait()
        else:
            # Terminal mode: run a live panel until cancelled or lifetime ends.
            p = Panel(intent, lambda st, tick: RENDERERS[intent](
                intent, slot, state, type("C", (), {"tick": tick, "data": {}})()),
                interval=INTERVALS.get(intent, 1.0),
                lifetime=LIFETIMES.get(intent, 10),
                cancel_event=state.get("_cancel_req"))
            state["handle"] = p
            try:
                p.start()
            finally:
                state["handle"] = None
    handler.__name__ = "run_" + intent.lower()
    return handler


HANDLERS = {name: _make_handler(name) for name in RENDERERS}
