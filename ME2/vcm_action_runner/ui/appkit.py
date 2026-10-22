"""Main-thread tkinter application for the action simulators.

WHY THIS EXISTS (the macOS bug)
-------------------------------
On macOS, a Tk window may ONLY be created and driven from the process's MAIN
thread. The old design ran each action on its own worker thread and called
tk.Tk() there, which on Mac raises:

    NSInternalInconsistencyException: NSWindow should only be instantiated
    on the main thread!

and aborts the whole process. (Linux / Raspberry Pi has no such rule, which is
why it only blew up on the Mac.)

THE FIX
-------
A SINGLE tkinter app lives on the MAIN thread and owns the one window. Worker
threads (one per running action) NEVER touch tkinter. Instead they post
messages onto a thread-safe queue:

    APP.open(intent, slot, state)   # post an OPEN request
    APP.close()                     # post a CLOSE request

The main thread drains that queue every ~100 ms inside a tkinter `after` loop,
then redraws the current intent every `interval` ms using a per-intent
renderer:

    renderer(intent, slot, state, ctx) -> list[str]

`ctx` is a small mutable namespace the app passes back in (e.g. for the timer
to stash its end-time). Because the window and its refresh loop are always on
the main thread, this is correct on BOTH macOS and Raspberry Pi.

One window at a time: posting a new OPEN while one is showing simply replaces
the current intent; the app tears the old view down and builds the new one.
Closing the window (red X) or posting CLOSE both stop the current action.

Headless machines (no DISPLAY) never construct this class -- runner.py picks
the terminal fallback instead. So importing this module is safe anywhere; only
instantiating it needs a display.
"""
import os
import sys
import queue
import threading


# Four-shade dot-matrix palette inspired by the original handheld LCD.  The
# interface uses no downloaded fonts or image assets, so it remains portable
# across Raspberry Pi OS, macOS, and Windows.
INK = "#0f380f"
DARK = "#306230"
MID = "#8bac0f"
LCD = "#9bbc0f"


def has_display():
    """True when a GUI window can be shown on this machine.

    Over SSH, the Pi's own $DISPLAY is unset, but an X11-forwarded display
    (``ssh -X``) shows up as ``DISPLAY=localhost:<n>.0``. We accept that too,
    so the window pops up on the laptop even when driven over SSH.
    """
    if sys.platform.startswith("linux"):
        disp = os.environ.get("DISPLAY") or ""
        wayland = os.environ.get("WAYLAND_DISPLAY") or ""
        if wayland:
            return True
        if not disp:
            return False
        # A real local display (":0", ":1", ...) or an X11-forwarded one
        # ("localhost:10.0"). Anything that looks like a socket/display is OK.
        return disp.startswith(":") or disp.startswith(".") or "localhost" in disp or "/" in disp
    return True  # macOS / Windows assume a display is present


class _Ctx:
    """Scratch namespace handed to each renderer tick (per active intent)."""
    def __init__(self):
        self.tick = 0
        self.data = {}


class App:
    """Owns the single action window on the main thread.

    Usage (from the MAIN thread):
        app = App(renderers)
        app.start()        # blocks; runs the tkinter mainloop
    From ANY worker thread:
        app.open(intent, slot, state)
        app.close()
    """

    def __init__(self, renderers, title="VCM", width=440, height=320):
        self.renderers = renderers          # {intent: fn(intent,slot,state,ctx)}
        self.title = title
        self.width = width
        self.height = height
        self.q = queue.Queue()              # thread-safe message queue
        self._closed_by_user = threading.Event()
        self._root = None

    # ---- worker-thread API (thread-safe) ----
    def open(self, intent, slot, state):
        self.q.put(("open", intent, slot, state))

    def close(self):
        self.q.put(("close", None, None, None))

    # ---- main-thread lifecycle ----
    def start(self):
        import tkinter as tk
        self._root = tk.Tk()
        root = self._root
        root.title("%s // POCKET TERMINAL" % self.title)
        root.configure(bg=INK)
        root.geometry("%dx%d" % (self.width, self.height))
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Dark outer shell and recessed green LCD. The geometry deliberately
        # resembles a compact handheld rather than a desktop application.
        bezel = tk.Frame(
            root, bg=DARK, highlightbackground=MID, highlightthickness=2,
            bd=5, relief="ridge")
        bezel.pack(fill="both", expand=True, padx=12, pady=(12, 7))

        header = tk.Frame(bezel, bg=DARK)
        header.pack(fill="x", padx=8, pady=(5, 4))
        self._lbl_title = tk.Label(
            header, text="%s // READY" % self.title.upper(), bg=DARK, fg=LCD,
            anchor="w", font=("DejaVu Sans Mono", 13, "bold"))
        self._lbl_title.pack(side="left", fill="x", expand=True)
        self._power = tk.Label(
            header, text="● LINK", bg=DARK, fg=LCD,
            font=("DejaVu Sans Mono", 9, "bold"))
        self._power.pack(side="right")

        screen = tk.Frame(
            bezel, bg=LCD, bd=5, relief="sunken",
            highlightbackground=INK, highlightthickness=2)
        screen.pack(fill="both", expand=True, padx=7, pady=(0, 7))

        # Body: one read-only dot-matrix-style LCD surface.
        self._body = tk.Text(
            screen, width=48, height=17, bg=LCD, fg=INK,
            insertbackground=INK, selectbackground=DARK, selectforeground=LCD,
            relief="flat", bd=0, cursor="arrow",
            font=("DejaVu Sans Mono", 11, "bold"), state="disabled",
            wrap="none", padx=10, pady=8, spacing1=1)
        self._body.pack(fill="both", expand=True)
        self._body.tag_configure(
            "badge", foreground=INK, background=MID,
            font=("DejaVu Sans Mono", 11, "bold"))

        self._status = tk.Label(
            screen, text="STANDBY // SAY DESTINY", anchor="w",
            bg=MID, fg=INK, padx=8, pady=2,
            font=("DejaVu Sans Mono", 9, "bold"))
        self._status.pack(fill="x")

        self._hint = tk.Label(
            root, text="VCM-01  •  VOICE LINK  •  CTRL+C QUITS",
            bg=INK, fg=MID, font=("DejaVu Sans Mono", 9, "bold"))
        self._hint.pack(pady=(0, 10))

        self._current = None   # (intent, slot, state)
        self._ctx = None
        self._poll()
        root.mainloop()

    # ---- internals (main thread) ----
    def _on_window_close(self):
        # User clicked the red X: stop the current action, then quit the window.
        self._closed_by_user.set()
        self.q.put(("close", None, None, None))
        try:
            self._root.destroy()
        except Exception:
            pass

    def _poll(self):
        root = self._root
        if root is None:
            return
        # Drain all queued messages (a close may follow an open, etc.).
        quit_requested = False
        try:
            while True:
                msg, intent, slot, state = self.q.get_nowait()
                if msg == "open":
                    self._set_current(intent, slot, state)
                elif msg == "close":
                    self._clear_current()
                elif msg == "quit":
                    quit_requested = True
        except Exception:
            pass

        # Redraw if something is showing.
        if self._current is not None:
            self._draw()

        # If the window was closed by the user OR the reader asked to quit,
        # tear everything down on the main thread.
        if self._closed_by_user.is_set() or quit_requested:
            try:
                root.destroy()
            except Exception:
                pass
            return

        root.after(100, self._poll)

    def _set_current(self, intent, slot, state):
        self._current = (intent, slot, state)
        self._ctx = _Ctx()
        self._lbl_title.config(text=self._display_title(intent, slot))
        self._status.config(text="ACTIVE LINK // %s" % intent.replace("_", " "))
        self._draw()

    def _clear_current(self):
        self._current = None
        self._ctx = None
        self._lbl_title.config(text="%s // READY" % self.title.upper())
        self._status.config(text="STANDBY // SAY DESTINY")
        self._set_body(["", "  SYSTEM READY", "", "  SAY DESTINY TO BEGIN", ""])

    def _display_title(self, intent, slot):
        names = {
            "TIME": "Clock", "WEATHER": "Weather", "LIGHT_ON": "Lights",
            "LIGHT_OFF": "Lights", "BRIGHTNESS": "Brightness",
            "COLOR": "Light Color", "PLAY_MUSIC": "Music", "PAUSE": "Music",
            "STOP": "Music", "NEXT": "Next Track", "VOLUME_UP": "Volume",
            "VOLUME_DOWN": "Volume", "TEMPERATURE": "Climate",
            "ALARM": "Alarm", "CALL": "Call", "MESSAGE": "Message",
            "LIST_REMINDERS": "Reminders", "CREATE_REMINDER": "Reminder",
            "TIMER": "Timer",
        }
        base = names.get(intent, intent.replace("_", " ").title()).upper()
        if slot:
            return "■ %s // %s" % (base, str(slot).upper())
        return "■ %s" % base

    def _set_body(self, lines):
        body = self._body
        body.config(state="normal")
        body.delete("1.0", "end")
        for ln in lines:
            tag = "badge" if ln.strip().startswith("[") else None
            body.insert("end", ln + "\n", tag)
        body.config(state="disabled")
        body.yview_moveto(0)

    def _draw(self):
        intent, slot, state = self._current
        renderer = self.renderers.get(intent)
        if renderer is None:
            self._set_body(["", "  (no renderer for %s)" % intent, ""])
            return
        try:
            lines = renderer(intent, slot, state, self._ctx)
        except Exception as e:
            lines = ["", "  (render error: %s)" % e, ""]
        if not isinstance(lines, (list, tuple)):
            lines = [str(lines)]
        self._set_body(list(lines))
