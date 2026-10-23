#!/usr/bin/env bash
# Install/verify ME2, then start the Destiny wake-word action runner.
# Invoke from any directory with: bash /path/to/ME2/start.sh
set -euo pipefail

ME2_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ME2_DIR"

echo "== AI 231 ME2 setup and launch =="
chmod +x setup.sh
./setup.sh

HAS_DISPLAY=0
if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  HAS_DISPLAY=1
fi

REQUESTED_TERM=0
REQUESTED_GUI=0
for arg in "$@"; do
  case "$arg" in
    --term) REQUESTED_TERM=1 ;;
    --gui) REQUESTED_GUI=1 ;;
  esac
done

if [ "$HAS_DISPLAY" = "0" ] && [ "$REQUESTED_TERM" = "0" ]; then
  echo ""
  echo "SETUP COMPLETE, BUT THE GUI CANNOT START IN THIS SSH SESSION."
  echo "X11 forwarding must be requested by the laptop when it connects."
  echo ""
  echo "On the laptop, make sure an X server is running, then use:"
  echo "  ssh -Y $(id -un)@$(hostname).local"
  echo ""
  echo "After reconnecting:"
  echo "  cd $ME2_DIR"
  echo "  ./start.sh"
  echo ""
  echo "To intentionally use the terminal interface now, run:"
  echo "  ./start.sh --term"
  exit 2
fi

RUNNER_DISPLAY=()
if [ "$REQUESTED_TERM" = "0" ] && [ "$REQUESTED_GUI" = "0" ]; then
  RUNNER_DISPLAY=(--gui)
fi

echo ""
echo "== starting Destiny wake-word runner =="
exec "$ME2_DIR/.venv/bin/python" \
  "$ME2_DIR/vcm_action_runner/runner.py" \
  --wakeword "${RUNNER_DISPLAY[@]}" "$@"
