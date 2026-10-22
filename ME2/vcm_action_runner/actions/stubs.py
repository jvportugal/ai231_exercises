"""Placeholder actions for the 18 non-TIMER intents.

Each just prints what it WOULD do, so the dispatcher/menu is fully exercisable
end-to-end. Replace the body of any function with real device code later.
Slot-bearing intents validate against the frozen CSV values and reject others.
"""
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from intents import BY_NAME, SLOT_NORM


def _slot_ok(intent, slot):
    allowed = BY_NAME[intent][2]
    return slot in allowed, allowed


def _stub(intent, slot, state):
    tag = intent
    if slot:
        ok, allowed = _slot_ok(intent, slot)
        if not ok:
            print("  [%s] invalid slot '%s' (allowed: %s)" % (tag, slot, ", ".join(allowed)))
            return
        print("  [%s] (stub) would perform: slot=%s" % (tag, slot))
    else:
        print("  [%s] (stub) would perform this action" % tag)


HANDLERS = {
    "PLAY_MUSIC":     lambda i, s, st: _stub(i, s, st),
    "WEATHER":        lambda i, s, st: _stub(i, s, st),
    "TIME":           lambda i, s, st: _stub(i, s, st),
    "LIGHT_ON":       lambda i, s, st: _stub(i, s, st),
    "LIGHT_OFF":      lambda i, s, st: _stub(i, s, st),
    "PAUSE":          lambda i, s, st: _stub(i, s, st),
    "STOP":           lambda i, s, st: _stub(i, s, st),
    "NEXT":           lambda i, s, st: _stub(i, s, st),
    "VOLUME_UP":      lambda i, s, st: _stub(i, s, st),
    "VOLUME_DOWN":    lambda i, s, st: _stub(i, s, st),
    "CALL":           lambda i, s, st: _stub(i, s, st),
    "MESSAGE":        lambda i, s, st: _stub(i, s, st),
    "LIST_REMINDERS": lambda i, s, st: _stub(i, s, st),
    "ALARM":          lambda i, s, st: _stub(i, s, st),
    "TEMPERATURE":    lambda i, s, st: _stub(i, s, st),
    "BRIGHTNESS":     lambda i, s, st: _stub(i, s, st),
    "COLOR":          lambda i, s, st: _stub(i, s, st),
    "CREATE_REMINDER":lambda i, s, st: _stub(i, s, st),
}
