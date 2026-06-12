#!/bin/zsh
# Bootstrap a fresh Apple-Silicon Mac (the M4 Pro mini) for OriginPower Studio.
# Run from the repo root AFTER rsyncing the repo (see the migration runbook).
set -e
cd "$(dirname "$0")/.."

echo "== brew deps"
brew install -q ffmpeg espeak-ng ollama node python@3.12 || true

echo "== ollama service (RESTART is load-bearing: a pre-existing server keeps"
echo "   running the OLD binary after brew upgrades it — gemma4 then hangs)"
brew services restart ollama; sleep 3
curl -s -m 5 http://127.0.0.1:11434/api/version || { echo "ollama not up"; exit 1; }

echo "== ollama model (17GB, one-time)"
ollama list | grep -q gemma4:26b || ollama pull gemma4:26b

echo "== python venvs (exact lockfiles)"
[ -d .eval_venv ]   || python3.12 -m venv .eval_venv
[ -d .qwen_venv ]   || python3.12 -m venv .qwen_venv
[ -d .kokoro_venv ] || python3.12 -m venv .kokoro_venv
.eval_venv/bin/pip install -q -r requirements-eval.lock.txt
.qwen_venv/bin/pip install -q -r requirements-qwen.lock.txt
.kokoro_venv/bin/pip install -q -r requirements-kokoro.lock.txt

echo "== remotion"
(cd remotion && npm install --silent)

echo "== sanity"
.eval_venv/bin/python -m pytest -q | tail -1
.eval_venv/bin/python -c "from studio.config import load; w=load().yolo_weights; print('weights:', w, w.exists())"
ls keys/gcp-vision.json keys/creds.env 2>/dev/null || echo '!! copy keys/ from the old machine (NOT in git)'
ls assets/voice/narrator_ref.wav || echo '!! narrator ref missing'
echo "== done. start: scripts/launchd/install.sh  (or two terminals: studio dashboard --host 0.0.0.0 / studio worker)"
