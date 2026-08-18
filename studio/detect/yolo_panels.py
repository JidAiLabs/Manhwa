"""
studio/detect/yolo_panels.py

YOLO-based panel detector — drop-in replacement for tools/gemini_panel_boxes.py.

Output schema is schema-compatible with gemini_panel_boxes.py:
  {"chunks": [{"chunk_file": "<basename>", "panels_norm": [[ymin,xmin,ymax,xmax], ...]}, ...]}

Boxes are normalized 0..1, sorted top-to-bottom by ymin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from studio.paths import resolve_rel


# ---------------------------------------------------------------------------
# Pure conversion helper
# ---------------------------------------------------------------------------

def boxes_to_panels_norm(
    px_boxes: Sequence[Tuple[float, float, float, float]],
    *,
    w: float,
    h: float,
) -> List[List[float]]:
    """Convert pixel xyxy boxes to normalised [ymin, xmin, ymax, xmax] lists.

    Args:
        px_boxes: Sequence of (x1, y1, x2, y2) pixel coordinates.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        List of [ymin, xmin, ymax, xmax] float lists sorted by ymin ascending.
        Values are rounded to 6 decimal places.
    """
    result: List[List[float]] = []
    for x1, y1, x2, y2 in px_boxes:
        ymin = round(float(y1) / h, 6)
        xmin = round(float(x1) / w, 6)
        ymax = round(float(y2) / h, 6)
        xmax = round(float(x2) / w, 6)
        result.append([ymin, xmin, ymax, xmax])
    result.sort(key=lambda b: b[0])
    return result


def snap_panels_to_elements(
    panels_norm: Sequence[Sequence[float]],
    element_boxes_norm: Sequence[Sequence[float]],
    *,
    min_inside_frac: float = 0.55,
    attach_gap: float = 0.0,
) -> List[List[float]]:
    """Grow panel boxes to swallow speech bubbles/system boxes they slice.

    A panel edge cutting through a bubble leaves a bubble remnant in the crop
    (and the rest in a neighbour). Each element box is assigned to the ONE
    panel containing the largest share of its area; when that share is at
    least *min_inside_frac* but not total, that panel grows to the union.
    Elements fully inside (nothing to fix) or mostly outside every panel
    (floating in the gutter) are left alone. Output stays sorted by ymin.
    Boxes are normalized [ymin, xmin, ymax, xmax].

    *attach_gap* (normalized, default off): art-only panel boxes (v4 weights)
    leave gutter dialogue OUTSIDE the box. An element partly OVER a panel
    (>= _ATTACH_MIN_OVER of its area, below the slice rule) is attached to
    that panel; otherwise one that sits over a panel's x-span (>= half its
    width) and is vertically within *attach_gap* of its top/bottom edge is
    attached to the NEAREST such panel (tie -> the one above). Attachment
    takes the WHOLE element and is skipped when the grown box would intersect
    another panel (never bisect, never overlap crops). The materialized scene
    thus keeps art + its dialogue for OCR/understanding while the raw art box
    (panels_norm_art) stays available for the shown frame. Corner touches and
    side floaters are not attached.
    """
    panels = [[float(v) for v in p] for p in panels_norm]
    # Snapshot the ORIGINAL boxes — clamp targets must stay fixed as panels grow,
    # so the result is independent of element-box iteration order.
    orig = [list(p) for p in panels]
    for b in element_boxes_norm:
        by0, bx0, by1, bx1 = (float(v) for v in b)
        barea = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
        if barea <= 0.0:
            continue
        best_i, best_frac = -1, 0.0
        for i, (py0, px0, py1, px1) in enumerate(panels):
            iy = max(0.0, min(py1, by1) - max(py0, by0))
            ix = max(0.0, min(px1, bx1) - max(px0, bx0))
            frac = (iy * ix) / barea
            if frac > best_frac:
                best_frac, best_i = frac, i
        sliced = best_i >= 0 and min_inside_frac <= best_frac < 1.0 - 1e-9
        attached = False
        if not sliced and attach_gap > 0.0 and best_frac < 1.0 - 1e-9:
            # attach: (1) a bubble PARTLY over a panel (>= _ATTACH_MIN_OVER of
            # its area — a corner touch is not "over") belongs to that panel;
            # (2) otherwise the nearest panel by vertical gap over its x-span.
            near_i = best_i if best_frac >= _ATTACH_MIN_OVER else -1
            if near_i < 0:
                near_gap = attach_gap + 1e-9
                for i, (py0, px0, py1, px1) in enumerate(orig):
                    ix = max(0.0, min(px1, bx1) - max(px0, bx0))
                    if ix < 0.5 * (bx1 - bx0):
                        continue                    # not over this panel's x-span
                    gap = max(py0 - by1, by0 - py1)  # <0 = overlaps vertically
                    if gap < near_gap:
                        near_gap, near_i = gap, i   # strict < keeps the UPPER on ties
            if near_i >= 0:
                # never bisect: the grown box must not intersect any OTHER
                # panel's original box, else leave the element alone (gap
                # recovery downstream still gets a shot at it)
                p = panels[near_i]
                gy0, gx0 = min(p[0], by0), min(p[1], bx0)
                gy1, gx1 = max(p[2], by1), max(p[3], bx1)
                clash = any(
                    j != near_i and min(gy1, qy1) > max(gy0, qy0) + 1e-9
                    and min(gx1, qx1) > max(gx0, qx0) + 1e-9
                    for j, (qy0, qx0, qy1, qx1) in enumerate(orig))
                if not clash:
                    panels[near_i] = [gy0, gx0, gy1, gx1]
                    attached = True
        if sliced:
            p = panels[best_i]
            ny0, nx0 = min(p[0], by0), min(p[1], bx0)
            ny1, nx1 = max(p[2], by1), max(p[3], bx1)
            # Clamp each grown edge to the gutter MIDPOINT with the nearest
            # neighbour so a snap can never cross into a neighbour's band (which
            # used to produce overlapping crops). Midpoint splits the gutter
            # evenly; for touching panels it degenerates to the shared edge.
            oy0, ox0, oy1, ox1 = orig[best_i]
            for j, (qy0, qx0, qy1, qx1) in enumerate(orig):
                if j == best_i:
                    continue
                if qy1 <= oy0:               # neighbour above
                    ny0 = max(ny0, (qy1 + oy0) / 2.0)
                if qy0 >= oy1:               # neighbour below
                    ny1 = min(ny1, (qy0 + oy1) / 2.0)
                if qx1 <= ox0:               # neighbour left
                    nx0 = max(nx0, (qx1 + ox0) / 2.0)
                if qx0 >= ox1:               # neighbour right
                    nx1 = min(nx1, (qx0 + ox1) / 2.0)
            panels[best_i] = [ny0, nx0, ny1, nx1]
    panels.sort(key=lambda p: p[0])
    return [[round(v, 6) for v in p] for p in panels]


# Element classes that, when they float in a gutter overlapping NO panel, are
# a story card of their own (system message, narrative caption, radio/jagged
# notice, a lone bubble) — promoted to panels so each becomes a tight scene.
# sfx never (effects lettering is not a beat).
_CARD_CLASSES = ("system_box", "caption_box", "radio", "speech_bubble",
                 "speech_background")


def floating_cards(
    panels_norm: Sequence[Sequence[float]],
    elements_norm: Dict[str, Sequence[Sequence[float]]],
) -> List[Dict[str, Any]]:
    """[{box:[ymin,xmin,ymax,xmax], classes:[...]}] — every _CARD_CLASSES
    element that intersects no panel, overlapping floaters unioned (classes
    merged, sorted). The classes are what panel_understand needs later: a
    radio/system card is an on-screen system message, a caption card folds."""
    panels = [[float(v) for v in p] for p in panels_norm]

    def _hit(a, b):
        return (min(a[2], b[2]) > max(a[0], b[0]) + 1e-9
                and min(a[3], b[3]) > max(a[1], b[1]) + 1e-9)

    floating = [([float(v) for v in b], cls) for cls in _CARD_CLASSES
                for b in (elements_norm.get(cls) or [])
                if not any(_hit(b, p) for p in panels)]
    cards: List[Dict[str, Any]] = []
    for b, cls in sorted(floating, key=lambda z: (z[0][0], z[0][1])):
        for c in cards:
            if _hit(b, c["box"]):
                bb = c["box"]
                c["box"] = [min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3])]
                c["classes"] = sorted(set(c["classes"]) | {cls})
                break
        else:
            cards.append({"box": list(b), "classes": [cls]})
    changed = True
    while changed:
        changed = False
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                if _hit(cards[i]["box"], cards[j]["box"]):
                    a, b = cards[i]["box"], cards[j]["box"]
                    cards[i] = {"box": [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                "classes": sorted(set(cards[i]["classes"]) | set(cards[j]["classes"]))}
                    del cards[j]
                    changed = True
                    break
            if changed:
                break
    for c in cards:
        c["box"] = [round(v, 6) for v in c["box"]]
    return cards


def promote_floating_cards(
    panels_norm: Sequence[Sequence[float]],
    elements_norm: Dict[str, Sequence[Sequence[float]]],
) -> List[List[float]]:
    """panels + every floating card box (see floating_cards). v4's art-only
    panel class does not box text cards (v3 did — that was the text-line
    poison), so without this a chapter's closing system cards only survive as
    one recovered strip mixed with the logo/credits, which the visual judge
    rightly drops — orphaning the narrated ending. Boxes normalized
    [ymin, xmin, ymax, xmax]; output sorted by ymin."""
    out = [[float(v) for v in p] for p in panels_norm]
    out += [list(c["box"]) for c in floating_cards(panels_norm, elements_norm)]
    out.sort(key=lambda p: p[0])
    return out


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

# Class resolution is NAME-driven from the checkpoint's own names dict so both
# model generations drop in without code edits:
#   legacy 6-class: panel, system_box, speech_bubble, text, sfx, character
#   v3 8-class:     panel, speech_bubble, radio, speech_background, sfx_text,
#                   system_ui, caption_box, free_text
# _NAME_TRANSLATE maps a model class name -> the manifest elements_norm key
# downstream consumers already speak (system_box / speech_bubble / sfx);
# unmapped names (text, character) stay discarded.
_NAME_TRANSLATE = {
    "system_box": "system_box",
    "speech_bubble": "speech_bubble",
    "sfx": "sfx",
    "radio": "radio",
    "speech_background": "speech_background",
    "sfx_text": "sfx",
    "system_ui": "system_box",
    "caption_box": "caption_box",
    "free_text": "free_text",
}
# Element keys whose sliced boxes should pull the panel boundary outward
# (voice containers + system windows must never be bisected by a crop).
_SNAP_CLASSES = ("speech_bubble", "system_box", "radio", "speech_background",
                 "caption_box")
# gutter dialogue farther than this from a panel edge stays a floating element
# (panels_to_scenes' gap recovery still gets a shot at it)
_ATTACH_GAP_PX = 300
# an element with at least this share of its area OVER a panel is that panel's
# (a corner touch ~1% is not); below the 0.55 slice rule it is ATTACHED whole
_ATTACH_MIN_OVER = 0.10


def resolve_classes(names: Dict[int, str]) -> Tuple[int, Dict[int, str]]:
    """(panel_class_id, {class_id: elements_norm key}) from a model names dict."""
    panel_id = next((i for i, n in names.items() if n == "panel"), 0)
    elements = {i: _NAME_TRANSLATE[n] for i, n in names.items()
                if n in _NAME_TRANSLATE}
    return panel_id, elements


def default_weights() -> str:
    """The ONE detector weights path: studio.toml [detect].yolo_weights (the
    same file the detect stage runs) — so a swap is a single config line for
    detect, render_prep's on-crop element pass and panel_understand's system-
    card override alike. Fail-soft to the committed v3 file when the config
    can't load (unit tests / stripped checkouts)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fallback = os.path.join(repo, "assets", "models", "webtoon_panels_v3.pt")
    try:
        from studio.config import load as _load
        p = str(_load().yolo_weights)
        return p if p and os.path.exists(p) else fallback
    except Exception:
        return fallback


def system_class_ids(names) -> set:
    """Class ids meaning 'in-world system window', resolved by NAME so every
    model generation works (legacy 6-class: system_box=1; v3 8-class:
    system_ui=5). Consumers (panel_understand system-card override,
    render_prep _sys_boxes) must NOT hardcode class 1. A ckpt with no names
    dict falls back to the legacy id {1}; names WITHOUT any system class
    yield an empty set (the protection turns off, fail-soft)."""
    if not names:
        return {1}
    return {i for i, n in dict(names).items()
            if n in ("system_box", "system_ui")}


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def _dedup_iou(boxes, thr: float = 0.5, contain_thr: float = 0.85):
    """Drop near-duplicate panel boxes: same panel seen twice (IoU > thr) or a
    fragment mostly contained in a larger box (containment > contain_thr — the
    prep_qa panel_double_covered signature). Larger boxes win; reading order
    otherwise preserved."""
    ordered = sorted(boxes, key=lambda z: (z[1], z[0]))
    areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in ordered]
    dropped = [False] * len(ordered)
    for i in range(len(ordered)):
        if dropped[i]:
            continue
        for j in range(len(ordered)):
            if i == j or dropped[j] or areas[j] < areas[i]:
                continue
            # j is the equal-or-larger box; test i against it
            if areas[j] == areas[i] and j > i:
                continue
            inter_x = max(0.0, min(ordered[i][2], ordered[j][2]) - max(ordered[i][0], ordered[j][0]))
            inter_y = max(0.0, min(ordered[i][3], ordered[j][3]) - max(ordered[i][1], ordered[j][1]))
            inter = inter_x * inter_y
            if inter <= 0 or areas[i] <= 0:
                continue
            if _iou(ordered[i], ordered[j]) > thr or inter / areas[i] > contain_thr:
                dropped[i] = True
                break
    return [b for b, d in zip(ordered, dropped) if not d]


def _under_segmented(px_boxes, img_h: int, *, min_h: int = 8000) -> bool:
    """True when a TALL chunk was under-detected: no panels at all, ONE box
    spanning most of it (a chunk-as-panel), or too few panels for its height —
    the signature of YOLO downscaling a giant chunk until panels vanish."""
    if img_h <= min_h:
        return False
    if not px_boxes:
        return True
    if any((y2 - y1) > 0.7 * img_h for (_x1, y1, _x2, y2) in px_boxes):
        return True
    return len(px_boxes) < img_h / 4000.0


def _retile_panels(model, img_path, img_w, img_h, conf, device, imgsz,
                   panel_class_id, *, win: int = 6000, overlap: int = 600):
    """Re-detect panels in an under-segmented chunk by slicing it into vertical
    windows YOLO resolves at proper scale, offsetting boxes back to chunk coords,
    and de-duplicating the window overlaps. Returns panel boxes (x1,y1,x2,y2)."""
    import numpy as _np
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
    im = _Image.open(img_path).convert("RGB")
    found: List[Tuple[float, float, float, float]] = []
    y = 0
    while True:
        y1 = min(img_h, y + win)
        arr = _np.asarray(im.crop((0, y, img_w, y1)))
        res = model.predict(source=arr, conf=conf, device=device, imgsz=imgsz,
                            verbose=False)[0]
        b = res.boxes
        if b is not None and len(b) > 0:
            for (x1, ty1, x2, ty2), c in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy()):
                if int(c) == panel_class_id:
                    found.append((float(x1), float(ty1) + y, float(x2), float(ty2) + y))
        if y1 >= img_h:
            break
        y += win - overlap
    return _dedup_iou(found)


def _tile_elements(model, img_path, img_w, img_h, conf, device, imgsz,
                   element_class_ids, *, win: int = 2400, overlap: int = 300):
    """Element boxes (bubbles/captions/system/sfx) in CHUNK coords from vertical
    windows YOLO resolves at element scale. The full-chunk pass downscales a
    ~10k-px strip until every bubble vanishes (elements_norm was empty on real
    chapters), so snap had nothing to snap. Returns {elements_norm key: [xyxy]}."""
    import numpy as _np
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
    im = _Image.open(img_path).convert("RGB")
    found: Dict[str, List[Tuple[float, float, float, float]]] = {
        name: [] for name in element_class_ids.values()}
    y = 0
    while True:
        y1 = min(img_h, y + win)
        arr = _np.asarray(im.crop((0, y, img_w, y1)))
        res = model.predict(source=arr, conf=conf, device=device, imgsz=imgsz,
                            verbose=False)[0]
        b = res.boxes
        if b is not None and len(b) > 0:
            for (x1, ty1, x2, ty2), c in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy()):
                name = element_class_ids.get(int(c))
                if name is not None:
                    found[name].append((float(x1), float(ty1) + y, float(x2), float(ty2) + y))
        if y1 >= img_h:
            break
        y += win - overlap
    return {k: _dedup_iou(v) for k, v in found.items() if v}


def detect_panels(
    stitch_manifest_path: str,
    out_path: str,
    weights: str,
    conf: float = 0.25,
    device: Optional[str] = None,
    snap: bool = True,
    imgsz: Optional[int] = None,
) -> Dict[str, Any]:
    """Run YOLO panel detection over all chunks listed in a stitch manifest.

    Args:
        stitch_manifest_path: Path to manifest.stitch.json.
        out_path: Where to write manifest.panels.json.
        weights: Path to YOLO .pt weights file.
        conf: Confidence threshold (default 0.25).
        device: Inference device ("mps", "cpu", "cuda", …).
                Defaults to "mps" if available, else "cpu".
        imgsz: Inference size. Defaults to the size the checkpoint was
               TRAINED at (train_args.imgsz) — the legacy model is 640, the
               v3 8-class model is 960; running a model off its native scale
               costs recall. Falls back to 640 if the ckpt has no record.

    Returns:
        The output dict that was written to out_path.
    """
    # Lazy import so the module can be imported without ultralytics installed
    # (pure tests must not require it).
    import torch
    from ultralytics import YOLO

    # Resolve device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load manifest
    manifest_path = Path(stitch_manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        stitch = json.load(f)

    chunks = stitch.get("chunks") or []

    # Load model once
    model = YOLO(weights)
    panel_class_id, element_class_ids = resolve_classes(model.names or {})
    if imgsz is None:
        imgsz = int(((getattr(model, "ckpt", None) or {}).get("train_args")
                     or {}).get("imgsz") or 640)

    out_chunks: List[Dict[str, Any]] = []

    for ch in chunks:
        chunk_file: str = ch.get("chunk_file") or ""
        chunk_path_stored: str = ch.get("chunk_path") or chunk_file

        # Resolve image path via resolve_rel (absolute paths pass through unchanged)
        img_path = str(resolve_rel(manifest_path, chunk_path_stored))

        basename = os.path.basename(img_path) if not chunk_file else chunk_file

        results = model.predict(
            source=img_path,
            conf=conf,
            device=device,
            imgsz=imgsz,
            verbose=False,
        )

        result = results[0]
        img_h, img_w = result.orig_shape  # (H, W)

        boxes = result.boxes
        px_boxes: List[Tuple[float, float, float, float]] = []
        el_px: Dict[str, List[Tuple[float, float, float, float]]] = {
            name: [] for name in element_class_ids.values()
        }

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()   # shape (N, 4)
            cls = boxes.cls.cpu().numpy()      # shape (N,)
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                ci = int(c)
                box = (float(x1), float(y1), float(x2), float(y2))
                if ci == panel_class_id:
                    px_boxes.append(box)
                elif ci in element_class_ids:
                    el_px[element_class_ids[ci]].append(box)

        # NMS dedup: the same panel emitted twice (high IoU) or a fragment box
        # inside a larger one becomes a double-covered crop downstream.
        px_boxes = _dedup_iou(px_boxes)

        # RE-TILE GUARD: a tall chunk the full-chunk pass under-segmented (one box
        # spanning most of it, or too sparse) means YOLO's downscale ate the
        # panels. Re-run on vertical sub-tiles so each panel is seen at full scale.
        if _under_segmented(px_boxes, img_h):
            retiled = _retile_panels(model, img_path, img_w, img_h, conf, device,
                                     imgsz, panel_class_id)
            if len(retiled) > len(px_boxes):
                px_boxes = retiled

        panels_norm = boxes_to_panels_norm(px_boxes, w=img_w, h=img_h)
        # RAW detector boxes, pre-snap / pre-gutter-expansion. With the v4
        # art-only weights this IS the art region; render_prep's art_only shown
        # frame crops to it (materialization stays full-panel for OCR).
        panels_norm_art = [list(b) for b in panels_norm]
        # elements at ELEMENT scale (windowed) — the full-chunk pass never
        # resolves a bubble on a 10k-px strip; union both, dedup per class.
        for name, bx in _tile_elements(model, img_path, img_w, img_h, conf, device,
                                       imgsz, element_class_ids).items():
            el_px[name] = _dedup_iou(el_px.get(name, []) + list(bx))
        elements_norm = {
            name: boxes_to_panels_norm(bx, w=img_w, h=img_h)
            for name, bx in el_px.items()
            if bx
        }
        if snap:
            snap_boxes = [b for name in _SNAP_CLASSES for b in elements_norm.get(name, [])]
            if snap_boxes:
                # attach gutter dialogue within ~300px of a panel edge
                panels_norm = snap_panels_to_elements(
                    panels_norm, snap_boxes, attach_gap=_ATTACH_GAP_PX / float(img_h))
            # floating cards (system / caption / radio / lone bubble in a gap
            # that no panel touches) are story panels of their own
            prev = {tuple(b) for b in panels_norm}
            cards_norm = floating_cards(panels_norm, elements_norm)
            panels_norm = promote_floating_cards(panels_norm, elements_norm)
            cards = [list(b) for b in panels_norm if tuple(b) not in prev]
            if cards:
                # the card IS its own art box (consumers intersect the two
                # lists geometrically, never by index)
                panels_norm_art = sorted(panels_norm_art + cards, key=lambda b: b[0])
        else:
            cards_norm = []

        out_chunks.append(
            {
                "chunk_file": basename,
                "panels_norm": panels_norm,
                # additive: raw art boxes (same order as panels_norm; snap only
                # grows a box, never reorders/merges) + chunk-space boxes of
                # the model's non-panel classes
                "panels_norm_art": panels_norm_art,
                # promoted floating cards with their element classes —
                # panel_understand stamps radio/system cards panel_kind=system
                "cards_norm": cards_norm,
                "elements_norm": elements_norm,
            }
        )

    out_obj: Dict[str, Any] = {
        "chunks": out_chunks,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    return out_obj
