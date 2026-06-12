#!/bin/zsh
# Self-hosted WireGuard pair for mini <-> MacBook Air. Run ON EACH machine:
#   scripts/wireguard/setup.sh mini   (on the Mac mini   -> 10.88.0.1)
#   scripts/wireguard/setup.sh air    (on the MacBook Air -> 10.88.0.2)
# Keys are generated LOCALLY and never leave the machine; the script prints
# the [Peer] stanza to paste into the OTHER machine's config.
set -e
ROLE="${1:?usage: setup.sh mini|air}"
brew list wireguard-tools >/dev/null 2>&1 || brew install -q wireguard-tools
CONF_DIR=/usr/local/etc/wireguard
sudo mkdir -p "$CONF_DIR"
PRIV=$(wg genkey); PUB=$(echo "$PRIV" | wg pubkey)

if [ "$ROLE" = "mini" ]; then
  ADDR=10.88.0.1/24; PORT=51820
  sudo tee "$CONF_DIR/wg0.conf" >/dev/null <<CONF
[Interface]
PrivateKey = $PRIV
Address = $ADDR
ListenPort = $PORT

# paste the Air's [Peer] block below:
# [Peer]
# PublicKey = <air public key>
# AllowedIPs = 10.88.0.2/32
CONF
else
  ADDR=10.88.0.2/24
  sudo tee "$CONF_DIR/wg0.conf" >/dev/null <<CONF
[Interface]
PrivateKey = $PRIV
Address = $ADDR

# paste the mini's [Peer] block below (Endpoint = mini's LAN IP at home, or
# router-forwarded WAN/DDNS address when roaming):
# [Peer]
# PublicKey = <mini public key>
# AllowedIPs = 10.88.0.1/32
# Endpoint = <mini-address>:51820
# PersistentKeepalive = 25
CONF
fi
sudo chmod 600 "$CONF_DIR/wg0.conf"
echo "== $ROLE configured at $CONF_DIR/wg0.conf"
echo "== paste this [Peer] block on the OTHER machine:"
echo "[Peer]"
echo "PublicKey = $PUB"
echo "AllowedIPs = ${ADDR%/24}/32"
[ "$ROLE" = "mini" ] && echo "Endpoint = <this-machine's-address>:51820"
echo ""
echo "start:  sudo wg-quick up wg0     (stop: sudo wg-quick down wg0)"
echo "verify: wg show"
