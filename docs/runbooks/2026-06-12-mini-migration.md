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

## From the MacBook Air
    open http://mini.local:8170/login?token=<that token>
    # away from home: install Tailscale on both machines and use the mini's
    # tailnet name instead of mini.local — same token gate applies.

## Notes
- 64 GB means Gemma (17 GB) stays resident next to Qwen + renders; later the
  gpu lane can split into llm/tts lanes (jobs.LANES) for another ~30% throughput.
- The old Mac can keep running its own worker on a COPY of the db, but do NOT
  point two workers at one studio.db over network shares (sqlite + NFS = no).
