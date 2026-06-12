import os
import tomllib
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class SiteCfg:
    base_url: str

@dataclass(frozen=True)
class Config:
    sites: dict[str, SiteCfg]
    yolo_weights: Path
    detect_backend: str          # "yolo" | "gemini"
    gallerydl_sleep: float
    beats_model: str = "gemini-2.5-flash"   # writer model (vertex id or ollama tag)
    beats_backend: str = "vertex"            # "vertex" | "ollama" (local Gemma)
    script_model: str = "gpt-4.1-mini"      # OpenAI model for the script stage
    tts_backend: str = "elevenlabs"         # "elevenlabs" | "chatterbox" | "kokoro"
    tts_voice_ref: str = ""                 # optional reference wav for voice cloning
    tts_python: str = ""                     # python for the local-TTS venv (deps
                                             # conflict with YOLO's torch); "" = pipeline python
    tts_kokoro_voice: str = "af_heart"      # kokoro voice pack (e.g. am_puck male)
    narration_source: str = "gemini_verbatim"  # scripted stage: "gemini_verbatim"
                                             # (voice the image-grounded beats
                                             # narration verbatim — A/B winner,
                                             # no OpenAI) | "legacy" | "openai_polish"
    punchup: str = "full"                   # persona pass over beats narration:
                                             # "full" | "light" | "off"
                                             # (grounded line kept as
                                             # narration_plain; captions protected)

def load_creds_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from keys/creds.env into os.environ.

    creds.env is AUTHORITATIVE: it overwrites any pre-existing environment
    value, so a stale key left in the user's shell (e.g. an old
    OPENAI_API_KEY) can't shadow the project's intended secret. Secrets live in
    this gitignored file so the CLI works without manual `export`.
    """
    import os
    p = path or (REPO_ROOT / "keys" / "creds.env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k:
            os.environ[k] = v


def load(path: Path | None = None) -> Config:
    p = path or (REPO_ROOT / "studio.toml")
    data = tomllib.loads(p.read_text())
    sites = {k: SiteCfg(**v) for k, v in data.get("sources", {}).items()}
    d = data.get("detect", {})
    g = data.get("gallerydl", {})
    m = data.get("models", {})
    t = data.get("tts", {})
    return Config(
        sites=sites,
        yolo_weights=(lambda _w: _w if _w.is_absolute()
                      else REPO_ROOT / _w)(
            Path(d.get("yolo_weights", "")).expanduser()),
        detect_backend=d.get("backend", "yolo"),
        gallerydl_sleep=float(g.get("sleep", 2.0)),
        beats_model=m.get("beats_model", "gemini-2.5-flash"),
        beats_backend=m.get("beats_backend", "vertex"),
        script_model=m.get("script_model", "gpt-4.1-mini"),
        tts_backend=t.get("backend", "elevenlabs"),
        tts_voice_ref=t.get("voice_ref", ""),
        tts_python=t.get("python", ""),
        tts_kokoro_voice=t.get("kokoro_voice", "af_heart"),
        narration_source=m.get("narration_source", "gemini_verbatim"),
        punchup=(os.environ.get("STUDIO_PUNCHUP")
                 or m.get("punchup", "full")),
    )
