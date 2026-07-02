#!/usr/bin/env python3
"""
reconcile_seam_panels.py

A chunk seam can bisect ONE tall panel into two near-duplicate slices (bottom of
chunk N + top of chunk N+1). Detection runs per-chunk, so each slice becomes its
own scene -> the same art shows twice -> the `cross_dup` QA ERROR.

This tool runs at the SCENE level (after panels_to_scenes, before vision). It:
  1. detects seam-bisected slice CHAINS geometrically (find_seam_chains),
  2. re-crops each chain into ONE merged panel from the chunk images
     (trimming the shared overlap band at the TRUE seam), and
  3. rewrites manifest.scenes.json + scenes/ in place, deleting orphan slices
     and stamping `reconciled_seam: true` on the survivor.

Idempotent: a re-run finds no chains and changes nothing.

See docs/plans/specs/2026-07-02-cross-chunk-panel-reconciliation-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reuse the EXACT dhash the scene records were hashed with.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from panels_to_scenes import dhash64  # noqa: E402

# Tolerances — small ABSOLUTE pixel values (loose fractions re-introduce false
# merges; see spec §3.2, §8). Chunk pixels.
# NOTE: geometry alone decides. A dhash veto was tried and REMOVED: two slices
# of ONE tall panel share only the stitch overlap band (~14% of the crop), so
# their whole-crop hashes legitimately differ (real ch1 seam measured
# Hamming 29). Do not re-add a similarity gate here.
EDGE_TOL_PX = 24    # A.y1 within this of chunk_h[N]; B.y0 within this of 0
SEAM_TOL_PX = 48    # EDGE_TOL_PX * 2; contiguity slack in stacked global-y


def _chunk_meta(scenes: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """chunk_file -> {chunk_h, gy0} (chunk_h/gy0 are constant within a chunk)."""
    meta: Dict[str, Dict[str, int]] = {}
    for s in scenes:
        cf = str(s.get("chunk_file") or "")
        if cf and cf not in meta:
            meta[cf] = {"chunk_h": int(s.get("chunk_h") or 0),
                        "gy0": int(s.get("chunk_global_y0") or 0)}
    return meta


def _y0(s: Dict[str, Any]) -> int:
    return int(s["box_px_xyxy"][1])


def _y1(s: Dict[str, Any]) -> int:
    return int(s["box_px_xyxy"][3])


def find_seam_chains(
    scenes: List[Dict[str, Any]],
    *,
    edge_tol: int = EDGE_TOL_PX,
    seam_tol: int = SEAM_TOL_PX,
) -> List[List[str]]:
    """Return seam-bisected panel CHAINS as lists of panel_id in stitch order.

    Only chains of length >= 2 are returned (a merge target). A chain is a
    connected component of pairwise seam links between adjacent chunks.

    A pair (A = bottommost panel of chunk N, B = topmost of chunk N+1) links iff:
      1. A touches the forced bottom edge:  chunk_h[N] - A.y1 <= edge_tol
      2. B touches the top edge:            B.y0 <= edge_tol
      3. contiguous stacked global-y:       |(A.gy0+A.y1) - (B.gy0+B.y0)| <= seam_tol
    """
    meta = _chunk_meta(scenes)
    # chunks in stitch order = by naive global-y offset
    order = sorted(meta, key=lambda cf: meta[cf]["gy0"])
    by_chunk: Dict[str, List[Dict[str, Any]]] = {cf: [] for cf in order}
    for s in scenes:
        cf = str(s.get("chunk_file") or "")
        if cf in by_chunk:
            by_chunk[cf].append(s)

    # union-find over panel_ids
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    linked: set = set()
    for n in range(len(order) - 1):
        cf_n, cf_m = order[n], order[n + 1]
        rows_n, rows_m = by_chunk[cf_n], by_chunk[cf_m]
        if not rows_n or not rows_m:
            continue
        a = max(rows_n, key=_y1)          # bottommost of chunk N
        b = min(rows_m, key=_y0)          # topmost of chunk N+1
        if meta[cf_n]["chunk_h"] - _y1(a) > edge_tol:        # cond 1
            continue
        if _y0(b) > edge_tol:                                 # cond 2
            continue
        ga = meta[cf_n]["gy0"] + _y1(a)
        gb = meta[cf_m]["gy0"] + _y0(b)
        if abs(ga - gb) > seam_tol:                           # cond 3
            continue
        union(a["panel_id"], b["panel_id"])
        linked.add(a["panel_id"])
        linked.add(b["panel_id"])

    # gather components, ordered by each member's stacked global-y
    def stack_y(pid: str) -> int:
        s = next(x for x in scenes if x["panel_id"] == pid)
        return meta[str(s["chunk_file"])]["gy0"] + _y0(s)

    comps: Dict[str, List[str]] = {}
    for pid in linked:
        comps.setdefault(find(pid), []).append(pid)
    chains = [sorted(members, key=stack_y) for members in comps.values()]
    chains.sort(key=lambda ch: stack_y(ch[0]))
    return [ch for ch in chains if len(ch) >= 2]


def reassemble_slices(slices: List[Image.Image], top_trims: List[int]) -> Image.Image:
    """Vertically stack *slices*, trimming top_trims[k] px off slice k's top.

    Slices share a source column (same width), aligned by construction. The trim
    on each interior seam removes the duplicated overlap band at the TRUE seam
    (top_trims[k] = OVERLAP_PX - B.y0 for the k-th slice; 0 for the first).
    """
    assert len(slices) == len(top_trims) and slices, "slices/top_trims mismatch"
    parts: List[Image.Image] = []
    for im, trim in zip(slices, top_trims):
        t = max(0, min(int(trim), im.height))
        parts.append(im.crop((0, t, im.width, im.height)) if t else im)
    width = max(p.width for p in parts)
    height = sum(p.height for p in parts)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for p in parts:
        out.paste(p.convert("RGB"), (0, y))
        y += p.height
    return out


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconcile_episode(scenes_manifest: str, stitch_manifest: str,
                      scenes_dir: str) -> int:
    """Detect + merge seam chains in place. Returns the number of chains merged."""
    manifest = _load(scenes_manifest)
    scenes: List[Dict[str, Any]] = manifest.get("scenes") or []
    if not scenes:
        return 0
    overlap_px = int(((_load(stitch_manifest).get("adaptive") or {})
                      .get("overlap_px")) or 700)

    chains = find_seam_chains(scenes)
    if not chains:
        return 0

    by_id = {s["panel_id"]: s for s in scenes}
    meta = _chunk_meta(scenes)
    drop_ids: set = set()

    for chain in chains:
        members = [by_id[pid] for pid in chain]
        head = members[0]                       # survivor = top slice (chunk N)
        x0 = min(int(m["box_px_xyxy"][0]) for m in members)
        x1 = max(int(m["box_px_xyxy"][2]) for m in members)

        slices: List[Image.Image] = []
        top_trims: List[int] = []
        for i, m in enumerate(members):
            with Image.open(m["chunk_path"]) as im:
                im = im.convert("RGB")
                if i == len(members) - 1:
                    y_lo, y_hi = int(m["box_px_xyxy"][1]), int(m["box_px_xyxy"][3])
                else:
                    # interior/head slices run down to the forced cut edge
                    y_lo, y_hi = int(m["box_px_xyxy"][1]), int(m["chunk_h"])
                slices.append(im.crop((x0, y_lo, x1, y_hi)))
            top_trims.append(0 if i == 0 else max(0, overlap_px - int(m["box_px_xyxy"][1])))

        merged = reassemble_slices(slices, top_trims)
        out_path = os.path.join(scenes_dir, head["out_file"])
        merged.save(out_path, "JPEG", quality=92, optimize=True)

        head["w"] = merged.width
        head["h"] = merged.height
        head["box_px_xyxy"] = [x0, int(head["box_px_xyxy"][1]),
                               x1, int(head["box_px_xyxy"][1]) + merged.height]
        head["box_norm"] = [
            head["box_px_xyxy"][1] / max(1, int(head["chunk_h"])),
            x0 / max(1, int(head["chunk_w"])),
            head["box_px_xyxy"][3] / max(1, int(head["chunk_h"])),
            x1 / max(1, int(head["chunk_w"])),
        ]
        head["dhash64"] = int(dhash64(merged))
        head["reconciled_seam"] = True
        head["merged_from"] = list(chain)
        if isinstance(head.get("split"), dict):
            head["split"]["enabled"] = False

        for m in members[1:]:
            drop_ids.add(m["panel_id"])
            slice_path = os.path.join(scenes_dir, m["out_file"])
            if os.path.exists(slice_path):
                os.remove(slice_path)

    manifest["scenes"] = [s for s in scenes if s["panel_id"] not in drop_ids]
    manifest["count_scenes"] = len(manifest["scenes"])
    stats = manifest.setdefault("stats", {})
    stats["reconciled_seams"] = int(stats.get("reconciled_seams", 0)) + len(chains)
    with open(scenes_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return len(chains)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes-manifest", required=True)
    ap.add_argument("--stitch-manifest", required=True)
    ap.add_argument("--scenes-dir", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.scenes_manifest):
        print(f"[reconcile] no scenes manifest at {args.scenes_manifest} — skip")
        return 0
    n = reconcile_episode(args.scenes_manifest, args.stitch_manifest, args.scenes_dir)
    print(f"[ok] reconcile_seam_panels: merged {n} seam chain(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
