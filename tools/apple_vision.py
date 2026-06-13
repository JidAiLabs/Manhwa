"""
tools/apple_vision.py

On-device OCR + detections via Apple's Vision framework (macOS, free, runs on
the Neural Engine). A $0 drop-in for the Google Cloud Vision features that
vision_extract.py uses: TEXT (word boxes), FACE, LABEL, and a saliency-based
OBJECT target. Proven equal OCR to Google (97% token-F1 across a full chapter)
at ~78ms/panel.

COORDINATES: Apple Vision returns NORMALIZED boxes with origin at the
BOTTOM-LEFT; the pipeline (and Google Vision) use TOP-LEFT, [x0,y0,x1,y1].
Every box is flipped here, so downstream bboxes line up with the image.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _flip(bx: float, by: float, bw: float, bh: float) -> List[float]:
    """(x, y, w, h) bottom-left normalized -> [x0, y0, x1, y1] top-left."""
    return [float(bx), float(1.0 - (by + bh)), float(bx + bw), float(1.0 - by)]


def available() -> bool:
    try:
        import Vision  # noqa: F401
        import AppKit  # noqa: F401
        from ocrmac import ocrmac  # noqa: F401
        return True
    except Exception:
        return False


# --- OCR (text) -------------------------------------------------------------

def ocr_words(path: str, *, langs: Tuple[str, ...] = ("en-US", "ko-KR", "zh-Hans"),
              max_words: int = 500) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (full_text, ocr_words). ocrmac yields LINE-level regions; each is
    split into approximate per-word boxes (proportional to token length) so
    text_coverage and ocr_words match Google's word-level granularity."""
    from ocrmac import ocrmac
    res = ocrmac.OCR(path, language_preference=list(langs)).recognize()
    lines: List[str] = []
    words: List[Dict[str, Any]] = []
    for text, conf, (bx, by, bw, bh) in res:
        x0, y0, x1, y1 = _flip(bx, by, bw, bh)
        toks = (text or "").split()
        if not toks:
            continue
        lines.append(text)
        total = sum(len(t) for t in toks) or 1
        cx, width = x0, (x1 - x0)
        for t in toks:
            nx1 = cx + width * (len(t) / total)
            words.append({"t": t,
                          "bbox": [round(cx, 4), round(y0, 4),
                                   round(nx1, 4), round(y1, 4)],
                          "conf": round(float(conf), 3)})
            cx = nx1
            if len(words) >= max_words:
                return "\n".join(lines), words
    return "\n".join(lines), words


# --- Vision-framework requests (faces / labels / saliency) ------------------

def _cgimage(path: str):
    import AppKit
    img = AppKit.NSImage.alloc().initWithContentsOfFile_(path)
    if img is None:
        return None
    data = img.TIFFRepresentation()
    if data is None:
        return None
    return AppKit.NSBitmapImageRep.imageRepWithData_(data).CGImage()


def _run(path: str, request):
    import Vision
    cg = _cgimage(path)
    if cg is None:
        return []
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    handler.performRequests_error_([request], None)
    return list(request.results() or [])


def _box(obs) -> List[float]:
    bb = obs.boundingBox()
    return _flip(bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)


def faces(path: str, *, max_faces: int = 6) -> List[Dict[str, Any]]:
    """Apple face rectangles; falls back to human-body rectangles for the
    anime faces the photo-trained face model misses."""
    import Vision
    out: List[Dict[str, Any]] = []
    for obs in _run(path, Vision.VNDetectFaceRectanglesRequest.alloc().init()):
        x0, y0, x1, y1 = _box(obs)
        out.append({"bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                    "confidence": round(float(obs.confidence()), 3)})
        if len(out) >= max_faces:
            return out
    if not out:
        for obs in _run(path, Vision.VNDetectHumanRectanglesRequest.alloc().init()):
            x0, y0, x1, y1 = _box(obs)
            out.append({"bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                        "confidence": round(float(obs.confidence()), 3)})
            if len(out) >= max_faces:
                break
    return out


def labels(path: str, *, max_labels: int = 15, min_score: float = 0.10
           ) -> List[Dict[str, Any]]:
    import Vision
    out: List[Dict[str, Any]] = []
    for obs in _run(path, Vision.VNClassifyImageRequest.alloc().init()):
        s = float(obs.confidence())
        if s >= min_score:
            out.append({"desc": str(obs.identifier()), "score": round(s, 3)})
        if len(out) >= max_labels:
            break
    return out


def objects(path: str) -> List[Dict[str, Any]]:
    """Attention-saliency salient region(s) as object-like targets — works on
    any art style (replaces Google OBJECT_LOCALIZATION for camera targeting)."""
    import Vision
    res = _run(path, Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init())
    out: List[Dict[str, Any]] = []
    if res:
        for so in (res[0].salientObjects() or []):
            bb = so.boundingBox()
            x0, y0, x1, y1 = _flip(bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)
            out.append({"name": "salient", "score": round(float(so.confidence()), 3),
                        "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]})
    return out
