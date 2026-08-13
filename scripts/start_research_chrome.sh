#!/usr/bin/env bash
# Headed Chrome for browser research. Persistent profile. Port 9224.
# Sign in to Kaggle once in this window. The agent never uses daily Chrome.
set -euo pipefail

PORT="${RESEARCH_CHROME_PORT:-9224}"
PROFILE="${RESEARCH_CHROME_PROFILE:-$HOME/.local/share/kaggle-agent/chrome}"
CDP="http://127.0.0.1:${PORT}"
CHROME="${GOOGLE_CHROME_BIN:-}"

if [[ -z "$CHROME" ]]; then
  for c in /opt/google/chrome/chrome /usr/bin/google-chrome-stable /usr/bin/google-chrome; do
    if [[ -x "$c" ]]; then
      CHROME="$c"
      break
    fi
  done
fi
if [[ -z "$CHROME" ]]; then
  echo "google-chrome not found" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No DISPLAY. This must be a real window so you can finish Google sign-in." >&2
  exit 1
fi

# Inherit the desktop session. Chrome started from cron/agent has DISPLAY
# but no D-Bus and XMODIFIERS=@im=ibus with no IBus client — keys go nowhere.
uid="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR}/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
fi
# Do not use IBus from this launcher. The daemon is on the desktop session;
# this process is not an IBus client, so GTK eats keystrokes.
export GTK_IM_MODULE=gtk-im-context-simple
export QT_IM_MODULE=simple
export XMODIFIERS=

port_open() {
  command -v ss >/dev/null && ss -ltn | grep -q ":${PORT} " && return 0
  return 1
}

cdp_ok() {
  curl -fsS --max-time 2 "${CDP}/json/version" >/dev/null 2>&1
}

# Stop only the leftover headless automation Chrome that used to own 9224.
stop_stale_headless() {
  local pid cmd
  for pid in $(pgrep -f "remote-debugging-port=${PORT}" || true); do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "$cmd" == *"/tmp/kaggle-agent-chrome"* && "$cmd" == *"--headless"* ]]; then
      echo "stopping leftover headless Chrome pid=${pid} (old /tmp profile)"
      kill "$pid" 2>/dev/null || true
      sleep 1
    fi
  done
}

already_our_profile() {
  local pid cmd
  for pid in $(pgrep -f "remote-debugging-port=${PORT}" || true); do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "$cmd" == *"${PROFILE}"* && "$cmd" != *"--type="* ]]; then
      echo "$pid"
      return 0
    fi
  done
  return 1
}

mkdir -p "$PROFILE"
stop_stale_headless

if [[ "${1:-}" == "--restart" ]]; then
  old="$(already_our_profile || true)"
  if [[ -n "$old" ]]; then
    echo "restarting research Chrome pid=${old}"
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
elif port_open && cdp_ok && already_our_profile >/dev/null; then
  echo "research Chrome already on ${CDP}"
  echo "profile: ${PROFILE}"
  echo "If the window shows Sign In, log into Kaggle there once (Google SSO)."
  echo "If typing does nothing, rerun: bash scripts/start_research_chrome.sh --restart"
  exit 0
fi

if port_open && ! already_our_profile >/dev/null; then
  echo "port ${PORT} is in use by another process. Pick another port:" >&2
  echo "  RESEARCH_CHROME_PORT=9225 bash scripts/start_research_chrome.sh" >&2
  exit 1
fi

echo "starting headed Chrome"
echo "  profile ${PROFILE}"
echo "  cdp     ${CDP}"
echo "Sign in to Kaggle in this window once. Leave it running for daily research."

nohup "$CHROME" \
  --user-data-dir="$PROFILE" \
  --remote-debugging-port="$PORT" \
  --remote-allow-origins="*" \
  --no-first-run \
  --no-default-browser-check \
  "https://www.kaggle.com/" \
  >/tmp/kaggle-agent-research-chrome.log 2>&1 &

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if cdp_ok; then
    echo "CDP ready at ${CDP}"
    echo "export BU_CDP_URL=${CDP}"
    exit 0
  fi
  sleep 0.4
done

echo "Chrome started but ${CDP} is not answering yet. Check /tmp/kaggle-agent-research-chrome.log" >&2
exit 1
