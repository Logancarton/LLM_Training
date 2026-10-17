#!/bin/bash
# Optional: keep Sonara's process running across logins and reboots (macOS launchd).
# The On/Off button still decides whether the loop runs; launchd only keeps the panel alive.
#
#   bash launchd/install.sh      # install and start
#   bash launchd/uninstall.sh    # stop and remove

set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3)"
PLIST="$HOME/Library/LaunchAgents/com.sonara.loop.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$DIR/../learning_data/adaptive_loop"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.sonara.loop</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$DIR/run.py</string><string>--no-browser</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DIR/../learning_data/adaptive_loop/sonara.log</string>
  <key>StandardErrorPath</key><string>$DIR/../learning_data/adaptive_loop/sonara.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Installed. Sonara's panel will stay available at http://127.0.0.1:8765"
