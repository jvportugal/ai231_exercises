#!/usr/bin/env python3
"""One-shot light diagnosis. Run ON THE PI:
    source .venv/bin/activate
    python3 vcm_action_runner/diagnose_light.py
Prints what is blocking the bulb, then stops without changing GPIO outputs.
"""
import glob
import grp
import os
import stat

print("=" * 62)
print("  LIGHT DIAGNOSIS  (safe -- no GPIO outputs are changed)")
print("=" * 62)

# 1. Who are we?
try:
    with open("/proc/self/loginuid") as f: pass
except Exception:
    pass
print("[1] user          : %s (uid=%d, gid=%d)" % (os.environ.get("USER","?"), os.getuid(), os.getgid()))
groups = os.getgroups()
print("    groups        : %s" % ", ".join(str(g) for g in groups))
try:
    gpio_gid = grp.getgrnam("gpio").gr_gid
    have_gpio_grp = gpio_gid in groups or os.getgid() == gpio_gid
    print("    in gpio grp?  : %s (gid=%d)" %
          ("YES" if have_gpio_grp else "NO  <- likely the problem", gpio_gid))
except KeyError:
    have_gpio_grp = False
    print("    in gpio grp?  : NO (the gpio group does not exist)")

# 2. GPIO node permissions
print("\n[2] GPIO nodes:")
gpio_nodes = (["/dev/gpiomem"] if os.path.exists("/dev/gpiomem") else [])
gpio_nodes += sorted(glob.glob("/dev/gpiochip*"))
if not gpio_nodes:
    print("    (no /dev/gpiomem or /dev/gpiochip* nodes found)")
for node in gpio_nodes:
    p = node
    if os.path.exists(p):
        st = os.stat(p)
        mode = stat.filemode(st.st_mode)
        uid, gid = st.st_uid, st.st_gid
        can = os.access(p, os.R_OK | os.W_OK)
        print("    %-16s EXISTS  mode=%s owner=%d:%d  accessible_by_you=%s"
              % (p, mode, uid, gid, "YES" if can else "NO  <- PROBLEM"))

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
        if mod == "RPi":
            extra = " version=%s" % getattr(m,"__version__","?")
        print("    %-10s %s%s" % (label, ok, extra))
    except Exception as e:
        print("    %-10s NOT IMPORTABLE (%s)" % (label, type(e).__name__))

# 5. Probe the lgpio factory installed by setup.sh for Raspberry Pi 5.
print("\n[5] Direct Pi 5 lgpio-factory probe:")
factory_ok = False
try:
    from gpiozero.pins.lgpio import LGPIOFactory
    pf = LGPIOFactory()
    print("    LGPIOFactory OK -> Pi 5 backend AVAILABLE")
    factory_ok = True
    pf.close()
except Exception as e:
    print("    LGPIOFactory FAILED: %s: %s" % (type(e).__name__, e))
    print("    ^^^ THIS is the real blocker. Fix is below.")

print("\n" + "=" * 62)
print("  VERDICT / NEXT COMMAND")
print("=" * 62)
if "Raspberry Pi" not in model:
    print("  Raspberry Pi hardware was not detected. SIMULATED light mode is")
    print("  expected here; run this diagnostic on the Pi to inspect real GPIO.")
elif factory_ok:
    print("  lgpio is usable. If the light still fails, verify common-cathode")
    print("  wiring, resistors, and BCM pins 17/27/22.")
elif not gpio_nodes:
    print("  No GPIO character-device nodes were found. Reboot, then confirm")
    print("  that the Raspberry Pi kernel and device-tree configuration are active.")
elif not all(os.access(node, os.R_OK | os.W_OK) for node in gpio_nodes):
    print("  At least one GPIO node is not readable/writable by this user.")
    print("  Re-run ./setup.sh, then log out and back in for gpio-group changes.")
else:
    print("  Device nodes are accessible, but gpiozero's lgpio factory failed.")
    print("  Re-run ./setup.sh inside ME2 and confirm that its lgpio import check")
    print("  succeeds; then re-run this diagnostic from the activated .venv.")
print()
