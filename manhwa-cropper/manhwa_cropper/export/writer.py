from pathlib import Path
import json
import cv2

def write_scenes(out_dir: Path, stem: str, crops, meta=None):
    page_dir = out_dir / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    for i, crop in enumerate(crops):
        outp = page_dir / f"scene_{i:04d}.png"
        cv2.imwrite(str(outp), crop)

    if meta is not None:
        with open(page_dir / "scenes.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
