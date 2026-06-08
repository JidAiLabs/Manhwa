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
