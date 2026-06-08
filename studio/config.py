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

def load_creds_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from keys/creds.env into os.environ (without
    overwriting variables already set). Secrets live in this gitignored file so
    the CLI works without the user manually exporting them."""
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
        if k and k not in os.environ:
            os.environ[k] = v


def load(path: Path | None = None) -> Config:
    p = path or (REPO_ROOT / "studio.toml")
    data = tomllib.loads(p.read_text())
    sites = {k: SiteCfg(**v) for k, v in data.get("sources", {}).items()}
    d = data.get("detect", {})
    g = data.get("gallerydl", {})
    return Config(
        sites=sites,
        yolo_weights=Path(d.get("yolo_weights", "")).expanduser(),
        detect_backend=d.get("backend", "yolo"),
        gallerydl_sleep=float(g.get("sleep", 2.0)),
    )
