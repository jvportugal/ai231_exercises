"""TIMER action: a live countdown (window on a display, bar in a terminal).

Only the frozen CSV slot values are valid: 10 seconds / 30 seconds / 1 minute.

The countdown is driven by wall-clock time (end_at), so it stays accurate
regardless of the renderer's tick rate. The end-time is stashed in
state["_timer"]["end_at"]; both the GUI renderer and the terminal bar read it.
"""
import os
import sys
import time

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from intents import SLOT_NORM
from ui.timer_display import CountdownTimer, _fmt


def _end_at(state):
    return state.get("_timer", {}).get("end_at")


# ---- GUI renderer (called on the MAIN thread by ui/appkit.App) ----
def r_timer(intent, slot, state, ctx):
    end_at = _end_at(state)
    if end_at is None:
        return ["", "    (timer not started)", ""]
    remain = end_at - time.time()
    if remain <= 0:
        return [
            "",
            "    Timer",
            "",
            "    0:00",
            "",
            "    DONE",
            "",
        ]
    total = _total(state)
    filled = int(20 * (remain / total)) if total else 0
    bar = "#" * filled + "-" * (20 - filled)
    return [
        "",
        "    Timer",
        "",
        "    %s" % _fmt(remain),
        "",
        "    [%s]" % bar,
        "    %d%% remaining" % int(100 * (remain / total)) if total else "    --",
        "",
    ]


def _total(state):
    return state.get("_timer", {}).get("total", 0)


# ---- terminal renderer (called on the worker thread by ui/panel.Panel) ----
def _make_term_render(state):
    """Panel calls renderer(None, tick) -- so capture the real state here."""
    def _term_render(st, tick):
        end_at = _end_at(state)
        total = _total(state)
        if end_at is None:
            return ["", "    (timer not started)", ""]
        remain = max(0.0, end_at - time.time())
        filled = int(20 * (remain / total)) if total else 0
        bar = "#" * filled + "-" * (20 - filled)
        return [
            "",
            "    %s" % _fmt(remain),
            "",
            "    [%s]" % bar,
            "",
        ]
    return _term_render


def run(intent, slot, state):
    """slot = one of the TIMER slot strings. Blocks until done/cancelled."""
    norm = SLOT_NORM["TIMER"].get(slot)
    if norm is None:
        print("  [TIMER] invalid duration '%s' (allowed: %s)"
              % (slot, ", ".join(SLOT_NORM["TIMER"])))
        return
    print("  [TIMER] starting %ss countdown (%s)" % (norm, slot))

    state.setdefault("_timer", {})
    state["_timer"]["total"] = norm
    state["_timer"]["end_at"] = time.time() + norm

    app = state.get("app")
    timer = CountdownTimer(norm, title="TIMER")
    timer._state = state
    if app is not None:
        timer._app = app
        state["handle"] = timer
        try:
            timer.start()  # posts OPEN to the main-thread App, blocks until cancel
        finally:
            state["handle"] = None
    else:
        # Terminal mode: live bar until the countdown ends or we're cancelled.
        from ui.panel import Panel
        p = Panel("TIMER", _make_term_render(state), interval=0.5,
                  lifetime=norm + 2, cancel_event=state.get("_cancel_req"))
        state["handle"] = p
        try:
            p.start()
        finally:
            state["handle"] = None

    if timer.done:
        print("  [TIMER] finished.")
    else:
        print("  [TIMER] stopped.")


# Expose the GUI renderer so runner.py can register it with the App.
RENDERER = r_timer
