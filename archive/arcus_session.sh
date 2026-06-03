#!/usr/bin/env bash
# Interact with the Arcus challenge TUI in YOUR real terminal and log the session.
# The raw log keeps ANSI codes; run `./arcus_session.sh clean` afterwards to print
# a de-ANSI'd, readable transcript you can paste back here.
#
# Usage:
#   ./arcus_session.sh            # connect + log to arcus_session.log
#   ./arcus_session.sh clean      # print readable text from arcus_session.log

set -euo pipefail
cd "$(dirname "$0")"
LOG="arcus_session.log"

if [[ "${1:-}" == "clean" ]]; then
  python3 - "$LOG" <<'PY'
import re, sys
data = open(sys.argv[1], 'rb').read().decode('utf-8', 'replace')
data = re.sub(r'\x1b\][^\x07\x1b]*(\x07|\x1b\\)', '', data)        # OSC
data = re.sub(r'\x1bP[^\x1b]*\x1b\\', '', data)                    # DCS
data = re.sub(r'\x1b[\[\?][0-9;:<>=$"\' ]*[A-Za-z@`~]', '', data)  # CSI
data = re.sub(r'\x1b[()][AB0]', '', data)
data = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', data)
lines = [ln.rstrip() for ln in data.splitlines()]
print("\n".join(ln for ln in lines if ln.strip()))
PY
  exit 0
fi

# `script` gives the TUI a real PTY while logging every byte.
script -q -c "ssh -tt -o StrictHostKeyChecking=accept-new augustalabs.ai" "$LOG"
echo "Session logged to $LOG — run './arcus_session.sh clean' for a readable transcript."
