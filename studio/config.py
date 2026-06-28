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
    vision_backend: str = "apple"           # OCR/visioned stage: "apple"
                                             # (on-device macOS Vision, FREE $0,
                                             # 97% token-F1 vs Google) | "google"
                                             # (Cloud Vision, paid per panel)
    semantic_heal: bool = False             # opt-in QA-eyes -> auto-heal: a
                                             # grounding judge flags weak/mis-
                                             # grounded lines (prep_qa
                                             # --semantic-heal), the heal loop
                                             # regenerates them, and the strictly-
                                             # better safeguard gates acceptance.
                                             # OFF = current pipeline byte-for-byte
                                             # unchanged. Env STUDIO_SEMANTIC_HEAL
                                             # wins (per-run toggle).
    teaser_enabled: bool = False            # bundle-level arc teaser: select a
                                             # high-stakes window from a bundle's
                                             # chapters and prepend a short
                                             # teaser.mp4 to the concat. Env
                                             # STUDIO_TEASER_ENABLED wins.
    teaser_shortlist_n: int = 4             # # of non-overlapping candidate
                                             # windows handed to the model pick.
    teaser_min_panels: int = 4              # smallest eligible teaser window.
    teaser_max_hook_panels: int = 10        # largest eligible teaser window.
    teaser_max_hook_scan_chapters: int = 12  # cost guard: only scan the first N
                                             # chapters of the bundle (0 = all).
    teaser_max_seconds: int = 90            # reserved soft cap (narration stays
                                             # uncapped; a future duration trim
                                             # may use it).
    teaser_payoff_tail_frac: float = 0.20   # spoiler guard: never pull a window
                                             # from the last fraction of the
                                             # bundle's reading order.
    teaser_model: str = "gemini-2.5-flash"  # mirrors the beats backend (Vertex
                                             # Gemini / ollama Gemma) — NOT an
                                             # OpenAI id; the model call reuses
                                             # _call_model_with_backoff.
    narration_sanitize: bool = True         # advertiser-safety pass over the
                                             # FINAL narration before TTS: the
                                             # scripted stage runs
                                             # narration_sanitize_pass (swap +
                                             # LLM-reframe flags/blocks +
                                             # re-sanitize) on manifest.script.json
                                             # and writes manifest.sanitize.json;
                                             # the voiced stage REFUSES to voice a
                                             # chapter with unresolved blocks. ON
                                             # by default (safety); env
                                             # STUDIO_NARRATION_SANITIZE wins.

def _env_bool(name: str, default: bool) -> bool:
    """Tri-state env override for a boolean flag. Unset → *default*; otherwise
    truthy ('1'/'true'/'yes'/'on') → True, anything else ('0'/'false'/'no'/
    'off') → False. Unlike the on-only env toggles above, this lets an env var
    DISABLE a flag that defaults ON (the sanitizer is a safety default, so a run
    must be able to turn it off explicitly)."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_tts_python(val: str) -> str:
    """Host-agnostic local-TTS interpreter. STUDIO_TTS_PYTHON env wins (per-host
    override); a RELATIVE path resolves against the repo root so one committed
    studio.toml works on every host (no hardcoded /Users/<name> — that broke the
    voiced stage after the Air->Mini move). An absolute path is honored as-is."""
    env = os.environ.get("STUDIO_TTS_PYTHON")
    if env:
        return env
    if not val:
        return ""
    p = Path(val).expanduser()
    return str(p if p.is_absolute() else REPO_ROOT / p)


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
    tr = data.get("teaser", {})
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
        tts_python=_resolve_tts_python(t.get("python", "")),
        tts_kokoro_voice=t.get("kokoro_voice", "af_heart"),
        narration_source=m.get("narration_source", "gemini_verbatim"),
        punchup=(os.environ.get("STUDIO_PUNCHUP")
                 or m.get("punchup", "full")),
        vision_backend=m.get("vision_backend", "apple"),
        semantic_heal=(os.environ.get("STUDIO_SEMANTIC_HEAL", "").lower()
                       in ("1", "true", "yes")
                       or bool(m.get("semantic_heal", False))),
        narration_sanitize=_env_bool("STUDIO_NARRATION_SANITIZE",
                                     bool(m.get("narration_sanitize", True))),
        teaser_enabled=_env_bool("STUDIO_TEASER_ENABLED",
                                 bool(tr.get("enabled", False))),
        teaser_shortlist_n=int(os.environ.get("STUDIO_TEASER_SHORTLIST_N")
                               or tr.get("shortlist_n", 4)),
        teaser_min_panels=int(tr.get("min_panels", 4)),
        teaser_max_hook_panels=int(tr.get("max_hook_panels", 10)),
        teaser_max_hook_scan_chapters=int(tr.get("max_hook_scan_chapters", 12)),
        teaser_max_seconds=int(tr.get("max_seconds", 90)),
        teaser_payoff_tail_frac=float(tr.get("payoff_tail_frac", 0.20)),
        teaser_model=(os.environ.get("STUDIO_TEASER_MODEL")
                      or tr.get("model", "gemini-2.5-flash")),
    )
