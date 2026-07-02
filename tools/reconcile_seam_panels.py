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

# Reuse the EXACT dhash/hamming the scene records were hashed with.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from panels_to_scenes import dhash64, hamming64  # noqa: E402

# Tolerances — small ABSOLUTE pixel values (loose fractions re-introduce false
# merges; see spec §3.2, §8). Chunk pixels.
EDGE_TOL_PX = 24    # A.y1 within this of chunk_h[N]; B.y0 within this of 0
SEAM_TOL_PX = 48    # EDGE_TOL_PX * 2; contiguity slack in stacked global-y
DHASH_VETO = 20     # VETO-ONLY: reject a match whose slice dhashes are FARTHER
                    # apart than this. Never triggers a merge.


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
    dhash_veto: int = DHASH_VETO,
) -> List[List[str]]:
    """Return seam-bisected panel CHAINS as lists of panel_id in stitch order.

    Only chains of length >= 2 are returned (a merge target). A chain is a
    connected component of pairwise seam links between adjacent chunks.

    A pair (A = bottommost panel of chunk N, B = topmost of chunk N+1) links iff:
      1. A touches the forced bottom edge:  chunk_h[N] - A.y1 <= edge_tol
      2. B touches the top edge:            B.y0 <= edge_tol
      3. contiguous stacked global-y:       |(A.gy0+A.y1) - (B.gy0+B.y0)| <= seam_tol
      4. NOT vetoed by a HIGH dhash distance: hamming(A,B) <= dhash_veto
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
        if hamming64(int(a["dhash64"]), int(b["dhash64"])) > dhash_veto:  # cond 4 veto
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


def main() -> int:  # pragma: no cover  (filled in Task 1.3)
    raise SystemExit("main() implemented in Task 1.3")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
