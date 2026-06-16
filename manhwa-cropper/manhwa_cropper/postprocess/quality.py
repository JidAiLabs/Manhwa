from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class SceneScore:
    edge_density: float
    std: float
    text_area_ratio: float

def _edges(gray: np.ndarray) -> np.ndarray:
    # robust edges for comics
    e = cv2.Canny(gray, 40, 120)
    return e

def score_scene(img_bgr: np.ndarray, text_boxes: list[tuple[int,int,int,int]] | None) -> SceneScore:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    e = _edges(gray)
    edge_density = float((e > 0).mean())

    std = float(gray.std())

    text_area_ratio = 0.0
    if text_boxes:
        area = 0
        for x1,y1,x2,y2 in text_boxes:
            area += max(0, x2-x1) * max(0, y2-y1)
        text_area_ratio = float(area / (h * w + 1e-9))

    return SceneScore(edge_density=edge_density, std=std, text_area_ratio=text_area_ratio)
