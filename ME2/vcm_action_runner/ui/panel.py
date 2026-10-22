"""Terminal "window" framework for the action simulators.

Each action renders as a bordered, self-updating box that lives in its own
thread and repaints every tick (clearing the screen so it looks like a live
app). The public contract matches what runner.py expects of an action handle:

    handle = Panel(title, renderer, interval, cancel_event)
    handle.start()      # blocks until cancelled OR a max lifetime elapses
    handle.cancel()     # called from another thread (dispatcher) to stop

`renderer(state, tick)` returns a list of lines to draw. `state` is the shared
DeviceState. `interval` is the repaint period in seconds.

Headless-safe by construction: no DISPLAY, no tkinter required, so it runs and
is testable anywhere (this sandbox included).
"""
import sys
import time


class Panel:
    def __init__(self, title, renderer, interval=1.0, cancel_event=None,
                 lifetime=None, header=None):
        """
        title       : short name shown in the frame border.
        renderer    : callable(state, tick) -> list[str]  (body lines).
        interval    : seconds between repaints.
        cancel_event: optional external threading.Event; if set, the panel stops
                      (lets a dispatcher kill it independently of handle.cancel()).
        lifetime    : optional max seconds; the panel auto-closes after this so a
                      "fire-and-hold" action doesn't run forever.
        header      : optional fixed line(s) drawn above the body (list[str]).
        """
        self.title = title
        self.renderer = renderer
        self.interval = interval
        self._cancel = cancel_event  # may be None
        self.lifetime = lifetime
        self.header = header or []
        self._stop = False
        self.cancelled = False
        self.ticks = 0

    # ---- public ----
    def cancel(self):
        self._stop = True
        self.cancelled = True

    def _should_stop(self):
        if self._stop:
            return True
        if self._cancel is not None and self._cancel.is_set():
            self.cancelled = True
            return True
        if self.lifetime is not None and self.ticks * self.interval >= self.lifetime:
            return True
        return False

    def start(self):
        """Block the calling thread until cancelled or lifetime elapses."""
        self._stop = False
        self.cancelled = False
        self.ticks = 0
        first = True
        while not self._should_stop():
            lines = self._render()
            self._paint(lines, first)
            first = False
            self.ticks += 1
            # sleep in small slices so cancel() is responsive
            deadline = time.time() + self.interval
            while time.time() < deadline:
                if self._should_stop():
                    break
                time.sleep(0.05)
        # final clear so the next action's panel starts clean
        sys.stdout.write("\n")
        if self.cancelled:
            print("  [%s] stopped." % self.title)
        else:
            print("  [%s] closed." % self.title)
        return self.cancelled

    # ---- rendering ----
    def _render(self):
        body = self.renderer(None, self.ticks)
        if not isinstance(body, (list, tuple)):
            body = [str(body)]
        all_lines = list(self.header) + list(body)
        width = max([len(self.title) + 6] + [len(l) for l in all_lines] + [20])
        width = max(width, 34)
        top = "+" + "-" * (width - 2) + "+"
        out = [top]
        out.append("|  " + self.title.ljust(width - 5) + " |")
        out.append("+" + "-" * (width - 2) + "+")
        for l in all_lines:
            out.append("|  " + l.ljust(width - 5) + " |")
        out.append(top)
        return out

    def _paint(self, lines, first):
        # ANSI: hide cursor, move home, clear, redraw. Falls back gracefully if
        # the stream isn't a tty (we still emit the lines, just no clearing).
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")
        else:
            if not first:
                sys.stdout.write("\n")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
