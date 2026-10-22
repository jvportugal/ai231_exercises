"""Physical RGB light driver for the 4-pin common-cathode bulb.

Hardware
--------
A 4-pin RGB bulb has ONE shared ground plus three colour channels:

    pin 1  GND   -> Pi GND (physical pin 6)
    pin 2  R     -> BCM 17  (physical pin 11)
    pin 3  G     -> BCM 27  (physical pin 13)
    pin 4  B     -> BCM 22  (physical pin 15)

Backend
-------
Uses **gpiozero** (the same library that reliably drives the bulb in the
standalone test). Each colour channel is a PWMLED at 1 kHz; brightness is the
PWM value (0.0-1.0) and colour is which channels are lit. Channels are
active_high, matching the common-cathode wiring.

Why gpiozero
------------
RPi.GPIO's hardware-PWM path failed to initialise on this image ("Cannot
determine SOC peripheral base address"), which silently dropped the driver
into SIMULATED mode and lit nothing. gpiozero abstracts the PWM setup and is
known-good on the Pi, so we drive the bulb through it instead.

Safe degradation
----------------
If gpiozero is not importable (e.g. running the bundle on a Mac/laptop for
testing, or before the Pi's GPIO is available), the driver switches to
SIMULATED mode: it still computes the exact RGB values it *would* write and
prints them, so the whole intent->action pipeline keeps working end to end.
On the real Pi it drives the actual pins.

The driver is a module-level singleton (LIGHT). sims.py calls
LIGHT.apply(DEVICE) after every lighting side effect.
"""
import time

# ---- wiring (EDIT THESE if your bulb's pin order differs) -----------------
# BCM numbering, so the code survives board revisions. These are the SAME
# numbers your standalone gpiozero test used (red=17, green=27, blue=22).
PIN_R = 17    # physical pin 11
PIN_G = 27    # physical pin 13
PIN_B = 22    # physical pin 15
# ----------------------------------------------------------------------------

_PWM_FREQ = 1000        # Hz -- 1 kHz is smooth, above flicker perception

# Map the logical colour names used by the VCM schema to (r, g, b) weights.
# Values are 0-100 "full intensity" per channel; brightness scales them.
_COLOUR_RGB = {
    "red":   (100, 0, 0),
    "green": (0, 100, 0),
    "blue":  (0, 0, 100),
    "white": (100, 100, 100),
}


class LightDriver:
    """Drives a common-cathode 4-pin RGB bulb via gpiozero PWMLEDs.

    Falls back to SIMULATED mode when gpiozero is unavailable, so the rest of
    the bundle (runner, sims, selftest) runs unchanged on non-Pi machines.
    """

    def __init__(self, pin_r=PIN_R, pin_g=PIN_G, pin_b=PIN_B,
                 freq=_PWM_FREQ, force_sim=False):
        self.pin_r = pin_r
        self.pin_g = pin_g
        self.pin_b = pin_b
        self.freq = freq
        self.simulated = force_sim
        self._led = None         # gpiozero.RGBLED (single object, like the test)
        self._last = {"on": False, "brightness": 0, "color": "white"}

        if not force_sim:
            try:
                from gpiozero import RGBLED
            except Exception as e:
                self.simulated = True
                print("  [light] gpiozero not usable (%s) -> SIMULATED mode "
                      "(no physical bulb driven)" % e)
                return
            # --- create the RGBLED -----------------------------------------
            # Let gpiozero AUTO-SELECT its pin backend (lgpio -> rpigpio ->
            # pigpio -> native). This is exactly what the standalone bulb
            # test that already works on this Pi does -- we do NOT poke at
            # gpiozero.pins internals, because those symbols (PinFactory,
            # get_pin_factory) come and go between versions and were the
            # source of the crashes. If ANYTHING fails -- wrong version, no
            # GPIO, permission denied -- we fall through to SIMULATED mode
            # instead of raising, so the runner always starts.
            try:
                self._led = RGBLED(red=pin_r, green=pin_g, blue=pin_b,
                                   active_high=True)
                self._led.value = (0.0, 0.0, 0.0)
                try:
                    self._pin_backend = getattr(self._led.red, "_pin", None)
                    self._pin_backend = (getattr(self._pin_backend, "pin_class", None)
                                         or getattr(self._pin_backend, "interface", None)
                                         or "auto")
                except Exception:
                    self._pin_backend = "auto"
                print("  [light] gpiozero RGBLED ready on BCM %d/%d/%d (R/G/B), "
                      "backend=%s" % (pin_r, pin_g, pin_b, self._pin_backend))
            except Exception as e:
                self._close_quietly()
                self.simulated = True
                print("  [light] gpiozero unavailable (%s) -> SIMULATED mode "
                      "(no physical bulb driven)" % e)

    # ------------------------------------------------------------------ core
    def _rgb_for(self, on, brightness_pct, color):
        """Return (r, g, b) values 0.0-1.0 for the requested state."""
        if not on:
            return (0.0, 0.0, 0.0)
        base = _COLOUR_RGB.get(color, _COLOUR_RGB["white"])
        scale = max(0, min(100, brightness_pct)) / 100.0
        return (base[0] * scale / 100.0,
                base[1] * scale / 100.0,
                base[2] * scale / 100.0)

    def apply(self, device):
        """Push the current DEVICE lighting state to the bulb (or simulate).

        `device` is the shared DEVICE singleton exposing .lights_on,
        .brightness, .color. Reads them under the device lock so it never
        races the action threads.
        """
        with device.locked():
            on = device.lights_on
            bright = device.brightness
            color = device.color
        # Skip redundant writes (e.g. two identical ticks).
        if (on, bright, color) == (self._last["on"],
                                   self._last["brightness"],
                                   self._last["color"]):
            return
        self._last = {"on": on, "brightness": bright, "color": color}

        r, g, b = self._rgb_for(on, bright, color)
        if self.simulated:
            print("  [light:SIM] on=%s bright=%3d%% color=%-5s -> R=%.2f G=%.2f B=%.2f"
                  % (on, bright, color, r, g, b))
            return
        # RGBLED.value takes a (r, g, b) tuple of floats 0.0-1.0.
        self._led.value = (r, g, b)

    # ------------------------------------------------------------- lifecycle
    def _close_quietly(self):
        led = getattr(self, "_led", None)
        if led is not None:
            try:
                led.value = (0.0, 0.0, 0.0)
                led.close()
            except Exception:
                pass
        self._led = None

    def cleanup(self):
        """Turn the bulb off and release the pins. Call on shutdown."""
        if self.simulated or getattr(self, "_led", None) is None:
            return
        self._close_quietly()


# Module-level singleton imported by sims.py.
LIGHT = LightDriver()
