"""Countdown timer display.

Two presentation paths, SAME logic and SAME public API:
    t = CountdownTimer(seconds)
    t.start()          # blocks until done OR cancelled
    t.cancel()         # called from another thread to stop early
    t.done             # True if it counted to zero, False if cancelled

GUI path (machine WITH a display -- RPi desktop, macOS, Windows):
    The window is owned by the MAIN-thread App (ui/appkit.py). The timer only
    posts an OPEN request and stashes its end-time in the renderer context.
    This is what fixes the macOS "NSWindow should only be instantiated on the
    main thread!" crash -- we never call tk.Tk() off the main thread anymore.

Terminal path (HEADLESS -- no DISPLAY): a live countdown bar printed in place.
"""
import os
import sys
import threading
import time


def _has_display():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # mac/windows assume a display


def _fmt(sec):
    sec = int(max(0, round(sec)))
    m, s = divmod(sec, 60)
    return "%d:%02d" % (m, s)


class CountdownTimer:
    def __init__(self, seconds, title="TIMER"):
        self.total = int(seconds)
        self.title = title
        self._cancel = threading.Event()
        self._done_event = threading.Event()
        self.done = False  # True => reached zero, False => cancelled
        self._thread = None

    # ---- public ----
    def start(self):
        """Block the calling thread until the countdown finishes or is cancelled."""
        self._done_event.clear()
        self._cancel.clear()
        self.done = False
        if _has_display():
            try:
                self._run_gui()
            except Exception:
                # Display vanished / tkinter failed mid-run -> terminal fallback
                self._run_terminal()
        else:
            self._run_terminal()
        return self.done

    def cancel(self):
        """Thread-safe: request an early stop."""
        self._cancel.set()

    def wait_done(self, timeout=None):
        return self._done_event.wait(timeout)

    # ---- terminal fallback ----
    def _run_terminal(self):
        remaining = self.total
        width = 30
        sys.stdout.write("\n")
        while remaining >= 0 and not self._cancel.is_set():
            frac = remaining / self.total if self.total else 0
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            sys.stdout.write("\r  [%s] %2ds left " % (bar, max(remaining, 0)))
            sys.stdout.flush()
            if remaining == 0:
                break
            deadline = time.time() + 1.0
            while time.time() < deadline and not self._cancel.is_set():
                time.sleep(0.1)
            remaining -= 1
        sys.stdout.write("\n")
        if self._cancel.is_set():
            self.done = False
            print("  [timer CANCELLED]")
        else:
            self.done = True
            print("  [timer DONE]")
        self._done_event.set()

    # ---- GUI (via main-thread App) ----
    def _run_gui(self):
        app = getattr(self, "_app", None)
        if app is None:
            # No App wired in (standalone use) -> fall back to terminal.
            self._run_terminal()
            return
        state = getattr(self, "_state", {})
        # Stash the end-time so the renderer can compute remaining each tick.
        state.setdefault("_timer", {})["end_at"] = time.time() + self.total
        app.open(self.title, "%ss" % self.total, state)
        # Block until the dispatcher cancels us (typing a new code / closing).
        self._cancel.wait()
        self.done = False
        self._done_event.set()
