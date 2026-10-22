"""Shared device state for the action simulators.

One module-level singleton so that lighting / media / volume / reminders behave
like ONE coherent device rather than 19 unrelated popups:
  - turning lights on then setting brightness composes (brightness only applies
    while the light is on);
  - volume up then down accumulates;
  - CREATE_REMINDER adds to a list that LIST_REMINDERS shows.

Every field is guarded by a single lock so the action threads (each action runs
on its own thread) and the dispatcher never race.
"""
import threading


class DeviceState:
    def __init__(self):
        self._lock = threading.RLock()

        # ---- lighting ----
        self.lights_on = False
        self.brightness = 100      # percent, applied only while lights_on
        self.color = "white"       # red / blue / green / white

        # ---- media ----
        self.playing = False
        self.paused = False
        self.volume = 50           # 0..100
        self.track_idx = 0
        # Exactly two tracks (Song A / Song B). NEXT wraps 0 -> 1 -> 0.
        # Override the filenames below to match whatever you drop into
        # assets/music/ (the audio driver looks for track_0.*, track_1.*).
        self.playlist = ["Song A", "Song B"]

        # ---- climate ----
        self.temperature = 22      # degrees

        # ---- reminders ----
        self.reminders = []        # list of task strings

        # ---- alarms ----
        self.alarms = []           # list of (hour, minute, meridiem)

        # ---- comms ----
        self.last_call = None
        self.last_message = None

    # ---- locking helper ----
    def locked(self):
        return self._lock

    # ---- reminders ----
    def add_reminder(self, task):
        with self._lock:
            if task not in self.reminders:
                self.reminders.append(task)
            return list(self.reminders)

    def list_reminders(self):
        with self._lock:
            return list(self.reminders)

    # ---- alarms ----
    def add_alarm(self, spec):
        with self._lock:
            if spec not in self.alarms:
                self.alarms.append(spec)
            return list(self.alarms)


# Module-level singleton shared by every action module.
DEVICE = DeviceState()
