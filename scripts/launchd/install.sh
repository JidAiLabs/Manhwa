#!/bin/zsh
# install + start both services: scripts/launchd/install.sh <dash-token>
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TOKEN="${1:?usage: install.sh <dashboard token>}"
mkdir -p "$REPO/logs"
# NOTE: mlxshim (the MLX gemma-4 vision shim on :11500) was RETIRED 2026-07-23 —
# it pinned 2x16GB model copies resident 24/7 (never evicts) and starved the box.
# The pipeline now routes to real ollama :11434 (gemma4:26b GGUF: one copy, native
# OLLAMA_NUM_PARALLEL, evicts when idle). To A/B it back: add "mlxshim" below AND
# flip the worker plist's OLLAMA_HOST to :11500.
for name in dashboard worker; do
  plist=~/Library/LaunchAgents/com.originpower.$name.plist
  # the plist embeds STUDIO_DASH_TOKEN — create it private (umask covers the
  # `>` redirect) so it is never world/group-readable, even briefly
  ( umask 077
    sed "s|__REPO__|$REPO|g; s|__TOKEN__|$TOKEN|g" \
      "$REPO/scripts/launchd/com.originpower.$name.plist" > "$plist" )
  chmod 600 "$plist"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
done
echo "services up: open http://$(hostname -s).local:8170/login and enter your token"
