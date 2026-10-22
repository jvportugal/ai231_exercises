"""Golden-truth intent registry.

Single source of truth = "Dataset Schema - Option B Final Final.csv".
19 intents (13 fixed + 6 slotted), exactly 3 variations each, exactly the
slot values listed. NOTHING here is invented; if it's not in the CSV it
does not exist. Do not add intents, variations, or slot values.
"""

# Ordered exactly as they appear in the CSV.
INTENTS = [
    # ---- Fixed (13) ----
    ("PLAY_MUSIC",     "Fixed",    ["Play music", "Start music", "Play some music"], []),
    ("WEATHER",        "Fixed",    ["Weather", "What's the weather?", "Tell me the weather"], []),
    ("TIME",           "Fixed",    ["Time", "What time is it?", "Tell me the time"], []),
    ("LIGHT_ON",       "Fixed",    ["Lights on", "Power on the lights", "Turn on the lights"], []),
    ("LIGHT_OFF",      "Fixed",    ["Lights out", "Kill the lights", "Shut off the lights"], []),
    ("PAUSE",          "Fixed",    ["Pause", "Pause audio", "Pause song"], []),
    ("STOP",           "Fixed",    ["Stop", "Stop playing", "End playback"], []),
    ("NEXT",           "Fixed",    ["Next song", "Skip song", "Play next song"], []),
    ("VOLUME_UP",      "Fixed",    ["Volume up", "Increase the volume", "Turn the volume up"], []),
    ("VOLUME_DOWN",    "Fixed",    ["Volume down", "Lower the volume", "Turn the volume down"], []),
    ("CALL",           "Fixed",    ["Call", "Make a call", "Make a phone call"], []),
    ("MESSAGE",        "Fixed",    ["Message", "Send a message", "Send my message"], []),
    ("LIST_REMINDERS", "Fixed",    ["Reminders", "Show my reminders", "List my reminders"], []),
    # ---- Slotted (6) ----
    ("TIMER",          "Slotted",  ["Timer {duration}", "Countdown for {duration}", "Start a timer for {duration}"],
                                ["10 seconds", "30 seconds", "1 minute"]),
    ("ALARM",          "Slotted",  ["Alarm {time}", "Wake me up at {time}", "Set an alarm for {time}"],
                                ["6:00 AM", "8:00 AM", "9:00 PM"]),
    ("TEMPERATURE",    "Slotted",  ["Temperature {degrees}", "Change the temperature to {degrees}", "Set the temperature to {degrees}"],
                                ["18 degrees", "22 degrees", "26 degrees"]),
    ("BRIGHTNESS",     "Slotted",  ["Brightness {percent}", "Adjust brightness to {percent}", "Brightness level {percent}"],
                                ["20 percent", "60 percent", "100 percent"]),
    ("COLOR",          "Slotted",  ["Change color to {color}", "Switch color to {color}", "Set color to {color}"],
                                ["Red", "Blue", "Green"]),
    ("CREATE_REMINDER","Slotted",  ["Reminder {task}", "Remind me to {task}", "Create a reminder to {task}"],
                                ["Drink water", "Study", "Exercise"]),
]

BY_NAME = {name: (typ, vars_, vals) for name, typ, vars_, vals in INTENTS}
NAMES   = [name for name, _, _, _ in INTENTS]

# Short letter codes for the terminal menu (numbers 1..N also accepted).
CODES = {
    "PLAY_MUSIC":     "P",
    "WEATHER":        "W",
    "TIME":           "T",
    "LIGHT_ON":       "ON",
    "LIGHT_OFF":      "OFF",
    "PAUSE":          "PA",
    "STOP":           "S",
    "NEXT":           "N",
    "VOLUME_UP":      "VU",
    "VOLUME_DOWN":    "VD",
    "CALL":           "C",
    "MESSAGE":        "M",
    "LIST_REMINDERS": "LR",
    "TIMER":          "TM",
    "ALARM":          "AL",
    "TEMPERATURE":    "TE",
    "BRIGHTNESS":     "BR",
    "COLOR":          "CO",
    "CREATE_REMINDER":"CR",
}
CODE_TO_INTENT = {code: name for name, code in CODES.items()}


def resolve_code(token: str):
    """Map a user-typed token (number or code) to an intent name, or None."""
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        i = int(token)
        if 1 <= i <= len(INTENTS):
            return NAMES[i - 1]
        return None
    return CODE_TO_INTENT.get(token.upper())


# Slot-value normalization (ONLY the frozen CSV values are valid).
SLOT_NORM = {
    "TIMER": {
        "10 seconds": 10,
        "30 seconds": 30,
        "1 minute":   60,
    },
    "ALARM": {
        "6:00 AM": (6, 0, "AM"),
        "8:00 AM": (8, 0, "AM"),
        "9:00 PM": (21, 0, "PM"),
    },
    "TEMPERATURE": {
        "18 degrees": 18,
        "22 degrees": 22,
        "26 degrees": 26,
    },
    "BRIGHTNESS": {
        "20 percent":  20,
        "60 percent":  60,
        "100 percent": 100,
    },
    "COLOR": {
        "Red":   "red",
        "Blue":  "blue",
        "Green": "green",
    },
    "CREATE_REMINDER": {
        "Drink water": "Drink water",
        "Study":       "Study",
        "Exercise":    "Exercise",
    },
}
