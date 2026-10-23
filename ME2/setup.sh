#!/usr/bin/env bash
# Complete setup for a newly installed Raspberry Pi OS environment.
# System libraries come from apt; Python packages stay in a local virtualenv
# so this also works on PEP-668 systems without modifying system Python.
set -euo pipefail
cd "$(dirname "$0")"

echo "== VCM bundle setup =="

# SUDO helper must exist before system prerequisites are installed.
if [ "$(id -u)" = "0" ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  echo "ERROR: setup needs root access to install Raspberry Pi OS packages."
  echo "Install sudo or run this script as root."
  exit 1
fi

# A fresh Raspberry Pi OS image does not necessarily include venv, Tk, the
# PortAudio runtime/devel files, or build tools needed by Pi GPIO packages.
# Install everything before creating the venv. Re-running this is safe.
if command -v apt-get >/dev/null 2>&1; then
  echo "-- installing Raspberry Pi OS system prerequisites --"
  $SUDO apt-get update
  $SUDO apt-get install -y \
    python3 python3-venv python3-pip python3-dev python3-tk \
    libportaudio2 portaudio19-dev alsa-utils \
    build-essential swig unzip \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libasound2-dev libfreetype6-dev
elif command -v brew >/dev/null 2>&1; then
  echo "-- installing macOS audio prerequisite --"
  brew list portaudio >/dev/null 2>&1 || brew install portaudio
fi

PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
  echo "ERROR: python3 is unavailable and this operating system has no supported package manager."
  exit 1
fi
echo "python: $PY ($($PY --version 2>&1))"

VENV_DIR=".venv"

# --- Create a virtualenv (skip if already present) -------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "-- creating virtualenv in $VENV_DIR --"
  "$PY" -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
VPY_VER=$("$VPY" --version 2>&1)
echo "venv python: $VPY ($VPY_VER)"
"$VPY" -m pip install --quiet --upgrade pip

# --- Complete Python runtime ------------------------------------------------
echo "-- installing inference, microphone, and music dependencies into venv --"
"$VPY" -m pip install --upgrade -r requirements.txt

# --- Raspberry Pi: physical RGB bulb driver --------------------------------
# The light driver uses gpiozero plus the lgpio backend on Raspberry Pi 5.
# Gate these installs on Pi detection so laptop/Mac runs stay clean.
# Detect a Raspberry Pi. Primary signal: /proc/device-tree/model. Fallback:
# the RPi.GPIO package itself (present on stock Pi OS images), so a Pi whose
# device-tree is masked still gets the driver instead of silently simulating.
IS_PI=0
if [ -f /proc/device-tree/model ] && grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  IS_PI=1
elif "$PY" -c "import RPi.GPIO" >/dev/null 2>&1; then
  IS_PI=1
fi

if [ "$IS_PI" = "1" ]; then
  echo "-- Raspberry Pi detected: installing the physical RGB bulb driver --"
  echo "  installing gpiozero and the Pi 5 lgpio backend"

  # ---- install gpiozero into the venv ------------------------------------
  # NOTE: the driver no longer depends on any specific gpiozero version -- it
  # lets gpiozero auto-select its pin backend and degrades to SIMULATED on
  # any failure. So we install the latest stable gpiozero and simply VERIFY
  # the import works. No more silent version drift causing crashes.
  echo "  [gpiozero] installing gpiozero into venv ..."
  "$VPY" -m pip install --upgrade gpiozero

  # Report the ACTUAL installed version so any future issue is diagnosable.
  "$VPY" -c "from importlib.metadata import version; print('    gpiozero version:', version('gpiozero'))"

  # Confirm the import the driver actually uses (plain 'from gpiozero import RGBLED').
  "$VPY" -c "from gpiozero import RGBLED"
  echo "    'from gpiozero import RGBLED' OK"

  # ---- install lgpio (REQUIRED on Raspberry Pi 5) -------------------------
  # gpiozero on a Pi 5 needs a pin backend that can actually talk to the
  # hardware. On modern Pi OS the RPi.GPIO backend is gone and gpiozero
  # falls back to the 'lgpio' GPIO character-device backend -- a separate pip
  # package. WITHOUT it, gpiozero can import fine but cannot open a single
  # pin, so the driver silently drops to SIMULATED and the bulb never lights.
  # This is the one dependency that has been missing from every prior bundle.
  echo "  [lgpio] installing Pi 5 GPIO backend ..."
  "$VPY" -m pip install --upgrade lgpio
  "$VPY" -c "import lgpio"
  echo "    'import lgpio' OK -> gpiozero can drive real pins on this Pi"

  # GUARANTEE the current user can open the GPIO device nodes (the #1 cause of
  # 'unable to open /dev/gpiomem' or '/dev/gpiochip0' -> SIMULATED mode).
  #
  # (a) IMMEDIATE fix: make the nodes world-readable RIGHT NOW so the bulb
  #     works in this very session without a re-login.
  TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
  for node in /dev/gpiomem /dev/gpiochip0 /dev/gpiochip4; do
    if [ -e "$node" ]; then
      if [ -r "$node" ]; then
        echo "  $node already readable by '$TARGET_USER'"
      else
        echo "-- making $node readable (immediate fix, no re-login needed) --"
        $SUDO chmod 666 "$node" 2>/dev/null && \
          echo "  $node now readable" || \
          echo "  [!] could not chmod $node. Run manually: sudo chmod 666 $node"
      fi
    fi
  done
  #
  # (b) PERMANENT fix: add the user to the 'gpio' group so it survives reboots.
  if getent group gpio >/dev/null 2>&1; then
    id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx gpio || {
      echo "-- adding user '$TARGET_USER' to the 'gpio' group (persists across reboots) --"
      $SUDO usermod -aG gpio "$TARGET_USER" 2>/dev/null && \
        echo "  [!] LOG OUT AND BACK IN (or reboot) for the group change to stick." || \
        echo "  [!] could not add to gpio group. Run manually: sudo usermod -aG gpio \$USER"
    }
  else
    echo "  [no 'gpio' group on this system -- nodes should be world-readable]"
  fi
else
  echo "-- not a Raspberry Pi: skipping gpiozero (light driver will simulate) --"
fi

# --- Verify the microphone runtime -----------------------------------------
echo "-- verifying PortAudio + sounddevice --"
"$VPY" -c "import sounddevice as sd; sd.query_devices(); print('  sounddevice/PortAudio: OK')"

# --- Raspberry Pi: make sure the mic is visible (ALSA). Best-effort. -------
if [ "$IS_PI" = "1" ]; then
  echo "-- Raspberry Pi: checking ALSA audio --"
  command -v arecord >/dev/null
  command -v aplay >/dev/null
  echo "  test your mic:  arecord -f cd -d 3 /tmp/mic_test.wav && aplay /tmp/mic_test.wav"
fi

# --- tkinter check (for the action-runner windows) -------------------------
echo "-- checking tkinter (action-runner windows) --"
"$VPY" -c "import tkinter; print('  tkinter OK')"

echo "-- checking bundled Python programs --"
"$VPY" -m py_compile \
  tester.py wakeword_tester.py vcm_infer.py \
  vcm_action_runner/runner.py vcm_action_runner/voice.py \
  vcm_action_runner/wakeword.py \
  vcm_action_runner/actions/weather.py

# --- Verify both bundled ONNX models before declaring setup complete --------
echo "-- verifying bundled wake-word + VCM models --"
for asset in \
  wakeword_model/destiny.onnx wakeword_model/melbank.npy wakeword_model/metadata.json \
  vcm_model/vcm.onnx vcm_model/melbank.npy vcm_model/labels.json \
  vcm_model/policy.json; do
  if [ ! -r "$asset" ]; then
    echo "ERROR: required model asset is missing: $asset"
    exit 1
  fi
done
if command -v sha256sum >/dev/null 2>&1; then
  echo "-- verifying bundled model/media checksums --"
  sha256sum -c checksums.sha256
fi
"$VPY" - <<'PY'
import os
import sys

root = os.getcwd()
sys.path.insert(0, os.path.join(root, "vcm_action_runner"))
sys.path.insert(0, root)
import wakeword
import vcm_infer

ww = wakeword.WakeWordDetector()
ww.warmup()
vcm = vcm_infer.load_model()
vcm_infer.warmup(vcm, n=1)
print("  Destiny wake-word ONNX: OK")
print("  command VCM ONNX:       OK")
PY

echo ""
echo "== setup complete =="
echo "Activate the venv, then run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 vcm_action_runner/runner.py --wakeword  # Destiny -> command -> action"
echo "  python3 vcm_action_runner/runner.py --voice     # keyboard-controlled mic mode"
echo "  python3 wakeword_tester.py                      # Space-to-Space Destiny tester"
echo ""
echo "(Or skip activating:  $VPY vcm_action_runner/runner.py --wakeword)"
