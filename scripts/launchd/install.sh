#!/bin/zsh
# install + start both services: scripts/launchd/install.sh <dash-token>
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TOKEN="${1:?usage: install.sh <dashboard token>}"
mkdir -p "$REPO/logs"
for name in dashboard worker; do
  plist=~/Library/LaunchAgents/com.originpower.$name.plist
  sed "s|__REPO__|$REPO|g; s|__TOKEN__|$TOKEN|g" \
    "$REPO/scripts/launchd/com.originpower.$name.plist" > "$plist"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
done
echo "services up: open http://$(hostname -s).local:8170/login and enter your token"
