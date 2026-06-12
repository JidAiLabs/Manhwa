# Migrating the studio to the M4 Pro mini (64 GB)

## On the old Mac (this one)
    # 1. everything in git is already portable (weights now in-repo).
    #    Sync the repo INCLUDING .git, EXCLUDING rebuildable dirs:
    rsync -a --exclude .eval_venv --exclude .qwen_venv --exclude .kokoro_venv \
          --exclude remotion/node_modules --exclude logs \
          ~/repos/Manhwa/  mini.local:~/repos/Manhwa/
    # 2. secrets + data that are NOT in git:
    rsync -a ~/repos/Manhwa/keys/     mini.local:~/repos/Manhwa/keys/
    rsync -a ~/repos/Manhwa/ongoing/  mini.local:~/repos/Manhwa/ongoing/   # artifacts (big)
    rsync -a ~/repos/Manhwa/studio.db mini.local:~/repos/Manhwa/studio.db
    # 3. put YOUTUBE_API_KEY into keys/creds.env (it currently lives only in
    #    the old shell profile): echo 'YOUTUBE_API_KEY=...' >> keys/creds.env

## On the mini
    cd ~/repos/Manhwa && scripts/bootstrap_mac.sh      # brew, ollama+gemma, venvs, npm, tests
    scripts/launchd/install.sh <pick-a-secret-token>   # dashboard+worker as services

## Private tunnel (WireGuard, self-hosted — no third parties)
    # on the mini:  scripts/wireguard/setup.sh mini
    # on the Air:   scripts/wireguard/setup.sh air
    # paste each machine's printed [Peer] block into the other's wg0.conf,
    # then on both: sudo wg-quick up wg0
    # Bind the dashboard to the TUNNEL ONLY (unreachable from LAN/Wi-Fi):
    #   studio dashboard --host 10.88.0.1
    #   (in launchd: change --host 0.0.0.0 -> 10.88.0.1 in the plist)

## From the MacBook Air
    open http://10.88.0.1:8170/login    # through the tunnel; enter the token
    # roaming: forward UDP 51820 on the home router to the mini (or DDNS)
    # and set that as Endpoint in the Air's wg0.conf — no other changes.

## Notes
- 64 GB means Gemma (17 GB) stays resident next to Qwen + renders; later the
  gpu lane can split into llm/tts lanes (jobs.LANES) for another ~30% throughput.
- The old Mac can keep running its own worker on a COPY of the db, but do NOT
  point two workers at one studio.db over network shares (sqlite + NFS = no).
