#!/usr/bin/env python3
"""One-shot light diagnosis. Run ON THE PI:
    source .venv/bin/activate
    python3 vcm_action_runner/diagnose_light.py
Prints exactly what is blocking the bulb, then STOPS (does not touch pins).
"""
import os, sys, stat, glob

print("=" * 62)
print("  LIGHT DIAGNOSIS  (safe -- reads only, no pins touched)")
print("=" * 62)

# 1. Who are we?
try:
    with open("/proc/self/loginuid") as f: pass
except Exception:
    pass
print("[1] user          : %s (uid=%d, gid=%d)" % (os.environ.get("USER","?"), os.getuid(), os.getgid()))
groups = os.getgroups()
print("    groups        : %s" % ", ".join(str(g) for g in groups))
have_gpio_grp = 1001 in groups or 1002 in groups or 4 in groups
print("    in gpio grp?  : %s" % ("YES" if have_gpio_grp else "NO  <- likely the problem"))

# 2. GPIO node permissions
print("\n[2] GPIO nodes:")
for node in ("/dev/gpiomem", "/dev/gpiochip0"):
    p = node
    if os.path.exists(p):
        st = os.stat(p)
        mode = stat.filemode(st.st_mode)
        uid, gid = st.st_uid, st.st_gid
        # can current user read?
        can = os.access(p, os.R_OK)
        print("    %-16s EXISTS  mode=%s owner=%d:%d  readable_by_you=%s"
              % (p, mode, uid, gid, "YES" if can else "NO  <- PROBLEM"))
    else:
        print("    %-16s MISSING  <- big problem (kernel/overlay)" % p)

# 3. Is this a Pi? (BCM peripheral base)
print("\n[3] Pi detection:")
model = ""
for f in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
    if os.path.exists(f):
        try:
            model = open(f,"rb").read().split(b"\0")[0].decode(errors="replace")
        except Exception:
            pass
print("    device-tree model : %s" % (model or "(not found)"))
rev = ""
if os.path.exists("/proc/cpuinfo"):
    for line in open("/proc/cpuinfo"):
        if line.lower().startswith("hardware") or line.lower().startswith("model name"):
            rev = line.split(":",1)[1].strip()
print("    cpuinfo hardware  : %s" % (rev or "(none)"))

# 4. Which gpiozero factories are importable?
print("\n[4] gpiozero backends (which can actually be imported):")
for mod, label in [("lgpio","lgpio"), ("RPi","RPi.GPIO"), ("pigpio","pigpio")]:
    try:
        m = __import__(mod)
        ok = "OK"
        extra = ""
        if mod == "lgpio":
            extra = " Chip=%s" % ("YES" if hasattr(m,"Chip") else "NO  <- empty/broken build")
        if mod == "RPi":
            extra = " version=%s" % getattr(m,"__version__","?")
        print("    %-10s %s%s" % (label, ok, extra))
    except Exception as e:
        print("    %-10s NOT IMPORTABLE (%s)" % (label, type(e).__name__))

# 5. Try the EXACT native factory gpiozero would use, capture the real error
print("\n[5] Direct native-factory probe (the one that must succeed):")
try:
    from gpiozero.pins.rpigpio import RPiGPIOPinFactory
    pf = RPiGPIOPinFactory()
    print("    RPiGPIOPinFactory OK -> native backend AVAILABLE")
    pf.close()
except Exception as e:
    print("    RPiGPIOPinFactory FAILED: %s: %s" % (type(e).__name__, e))
    print("    ^^^ THIS is the real blocker. Fix is below.")

print("\n" + "=" * 62)
print("  VERDICT / NEXT COMMAND")
print("=" * 62)
if not os.path.exists("/dev/gpiomem") and not os.path.exists("/dev/gpiochip0"):
    print("  /dev/gpiomem & /dev/gpiochip0 BOTH MISSING -> kernel/overlay issue.")
    print("  Fix:  sudo modprobe bcm2835-gpio   (or reboot). Then re-run.")
elif os.path.exists("/dev/gpiomem") and not os.access("/dev/gpiomem", os.R_OK):
    print("  /dev/gpiomem exists but you CANNOT read it -> PERMISSIONS.")
    print("  Fix NOW:  sudo chmod 666 /dev/gpiomem")
    print("  (one-time; if it resets on reboot, add to /etc/udev/rules.d/99-gpio.rules:)")
    print("       KERNEL==\"gpiomem\", MODE=\"0666\"")
else:
    print("  Node readable. If [5] still failed, the blocker is the")
    print("  peripheral-base lookup (Pi 5 / new kernel). Fix:")
    print("    sudo apt-get install -y python3-rpi.gpio   # or")
    print("    sudo apt-get install -y pigpio && sudo systemctl enable pigpio")
    print("  Then re-run this script; [5] should say OK.")
print()
