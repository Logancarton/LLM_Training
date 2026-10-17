#!/bin/bash
# Stop Sonara's background process and remove it from launchd.
PLIST="$HOME/Library/LaunchAgents/com.sonara.loop.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Removed. Start Sonara by hand with: python3 run.py"
