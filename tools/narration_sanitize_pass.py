#!/usr/bin/env python3
"""narration_sanitize_pass — wire the advertiser-safety sanitizer + LLM reframe
over a chapter's FINAL narration, right before TTS.

Model (per the denylist's three actions):
  1. Run the Sanitizer (scope="spoken", seeded by the chapter key) on each
     narration line. report.text already has the deterministic safe SWAPS.
  2. If that line has FLAGS or BLOCKS, reframe_line() it (one small LLM rewrite
     softening per the hits' notes), then re-run the Sanitizer on the rewritten
     line (the reframe could introduce a new term, or fail to clear a block).
  3. If a BLOCK still remains after reframe → record it as UNRESOLVED (the
     chapter must NOT be voiced).
  4. Write the cleaned text back into the manifest the TTS stage consumes
     (manifest.script.json sections), and return a summary.

The TTS stage reads ``manifest.script.json`` →
``sections[].tts_paragraphs_v3[]`` (each item is ``"[mood] <line>"``; the mood
tag is stripped before voicing). ``script_paragraphs[]`` is the display/fallback
text. We sanitize BOTH, preserving the leading mood tag on tts_paragraphs_v3 so
the audio↔narration text_sha gate still lines up and the cleaned words are what
actually get voiced.

The model call is INJECTED (``call_fn``) so the whole pass is testable with a
stub. ``call_fn=None`` runs the deterministic layer only (swaps + flag/block
DETECTION + gating) with zero model calls — flags then pass through (a soft
swap is applied where possible; a residual block still gates) so the pass is
safe to run even when no LLM backend is wired.

CLI (used by studio/pipeline.py _stage_scripted; no LLM by default):
    python narration_sanitize_pass.py --script manifest.script.json \
        --seed ch0001 --marker manifest.sanitize.json
    # exit 0 = clean / reframed-clean, 2 = UNRESOLVED blocks remain
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from narration_sanitize import Sanitizer  # noqa: E402
from narration_reframe import reframe_line, ReframeCallFn  # noqa: E402

_DENYLIST = str(Path(__file__).with_name("narration_denylist.json"))
_LEADING_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")


@dataclass
class PassSummary:
    changed: int = 0                                    # lines whose text changed
    reframed: int = 0                                   # lines sent to the LLM reframe
    unresolved_blocks: List[Tuple[str, str]] = field(default_factory=list)  # (segment_id, matched)

    @property
    def has_unresolved(self) -> bool:
        return bool(self.unresolved_blocks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "reframed": self.reframed,
            "unresolved_blocks": [
                {"segment_id": sid, "matched": m} for sid, m in self.unresolved_blocks
            ],
        }


def _split_tag(text: str) -> Tuple[str, str]:
    """Split a leading ``[mood]`` tag. Returns (tag_or_empty, rest).

    ``tag`` keeps its brackets (e.g. ``"[tense]"``) so callers can re-prefix it
    verbatim; ``""`` when there's no leading tag.
    """
    s = str(text or "")
    m = _LEADING_TAG_RE.match(s)
    if not m:
        return "", s.strip()
    return f"[{m.group(1)}]", s[m.end():].lstrip()


def _clean_one(
    body: str,
    *,
    sanitizer: Sanitizer,
    seed: str,
    call_fn: Optional[ReframeCallFn],
) -> Tuple[str, bool, List[Any]]:
    """Sanitize+reframe ONE tag-stripped narration body.

    Returns (clean_body, was_reframed, residual_blocks).
      - clean_body      : swaps applied, reframed if it had flags/blocks.
      - was_reframed    : the LLM reframe was actually invoked.
      - residual_blocks : Hit list of blocks STILL present after reframe (empty
                          when clean). Flags that remain are tolerated (the
                          deterministic swap is the floor); only blocks gate.
    """
    rep = sanitizer.run(body, scope="spoken", seed=seed)
    clean = rep.text                               # safe swaps already applied
    if not (rep.flagged or rep.blocked):
        return clean, False, []

    # Has flags/blocks → try one LLM reframe (softening per the hit notes),
    # then RE-sanitize the rewrite. Reframe from the swap-applied text so the
    # safe swaps are preserved if the model leaves those spans alone.
    was_reframed = False
    if call_fn is not None:
        hits = list(rep.flags) + list(rep.blocks)
        reframed = reframe_line(clean, hits, call_fn)
        was_reframed = True
        if reframed and reframed != clean:
            rep2 = sanitizer.run(reframed, scope="spoken", seed=seed)
            clean = rep2.text
            return clean, was_reframed, list(rep2.blocks)
        # reframe returned the same line (model declined / failed): re-judge the
        # swap-applied text so the block set is accurate for gating.
        rep2 = sanitizer.run(clean, scope="spoken", seed=seed)
        return clean, was_reframed, list(rep2.blocks)

    # No model wired: keep the swap-applied text; report any residual blocks so
    # the gate still fires (we never publish a block, reframe or not).
    return clean, was_reframed, list(rep.blocks)


def _iter_section_segments(section: Dict[str, Any]) -> List[Tuple[int, str]]:
    """(paragraph_index, segment_id) for a section, mirroring how the TTS
    adapter keys clips (g{group_id:04d}_p{paragraph_index:02d}). Falls back to a
    section/index id when shots are absent (rare; keeps the summary readable)."""
    shots = section.get("shots") or []
    sec_idx = int(section.get("section_index") or 0)
    out: List[Tuple[int, str]] = []
    paras = section.get("script_paragraphs") or section.get("tts_paragraphs_v3") or []
    for i in range(len(paras)):
        gid = 0
        if i < len(shots) and isinstance(shots[i], dict):
            gid = int(shots[i].get("group_id") or 0)
        seg = f"g{gid:04d}_p{i:02d}" if gid > 0 else f"s{sec_idx:02d}_p{i:02d}"
        out.append((i, seg))
    return out


def sanitize_script(
    script_obj: Dict[str, Any],
    *,
    seed: str,
    call_fn: Optional[ReframeCallFn] = None,
) -> PassSummary:
    """Sanitize+reframe every narration line in a manifest.script.json object,
    IN PLACE. Returns a PassSummary.

    Both ``script_paragraphs`` (display/fallback) and ``tts_paragraphs_v3``
    (voiced; leading mood tag preserved) are cleaned. They carry the same words
    by construction (script_expander aligns them), so each paragraph index is
    cleaned ONCE and the result written to both — guaranteeing the voiced audio
    and the displayed text never diverge after sanitizing.
    """
    sanitizer = Sanitizer(_DENYLIST)
    summary = PassSummary()

    for section in script_obj.get("sections") or []:
        if not isinstance(section, dict):
            continue
        script_paras = section.get("script_paragraphs")
        tts_paras = section.get("tts_paragraphs_v3")
        has_script = isinstance(script_paras, list)
        has_tts = isinstance(tts_paras, list)
        if not has_script and not has_tts:
            continue

        seg_ids = {i: seg for i, seg in _iter_section_segments(section)}
        n = max(len(script_paras) if has_script else 0,
                len(tts_paras) if has_tts else 0)

        for i in range(n):
            seg_id = seg_ids.get(i, f"p{i:02d}")
            # Prefer the script paragraph as the source body (no tag); fall back
            # to the tag-stripped TTS line.
            tag = ""
            if has_tts and i < len(tts_paras):
                tag, tts_body = _split_tag(str(tts_paras[i]))
            else:
                tts_body = ""
            if has_script and i < len(script_paras):
                body = str(script_paras[i])
            else:
                body = tts_body

            clean, was_reframed, residual_blocks = _clean_one(
                body, sanitizer=sanitizer, seed=seed, call_fn=call_fn)

            if was_reframed:
                summary.reframed += 1
            if clean != body:
                summary.changed += 1

            if has_script and i < len(script_paras):
                script_paras[i] = clean
            if has_tts and i < len(tts_paras):
                # re-attach the original mood tag; if there was none, keep the
                # line tagless exactly as it was.
                tts_paras[i] = f"{tag} {clean}".strip() if tag else clean

            for hb in residual_blocks:
                summary.unresolved_blocks.append((seg_id, getattr(hb, "matched", "")))

    return summary


def write_marker(marker_path: str | Path, summary: PassSummary, *, seed: str) -> None:
    """Persist the pass result to manifest.sanitize.json. The voiced gate reads
    ``unresolved_blocks`` from here and refuses to voice when it's non-empty."""
    payload = {
        "schema_version": "sanitize_marker_v1",
        "seed": seed,
        "ok": not summary.has_unresolved,
        **summary.as_dict(),
    }
    Path(marker_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


def read_unresolved_blocks(marker_path: str | Path) -> List[Dict[str, str]]:
    """Read the unresolved-block list from a marker file. Missing/unreadable
    marker → [] (treated as 'no recorded blocks'; the gate only HALTS on a
    marker that explicitly lists unresolved blocks)."""
    p = Path(marker_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    ub = data.get("unresolved_blocks") or []
    return [b for b in ub if isinstance(b, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sanitize+reframe a chapter's narration before TTS.")
    ap.add_argument("--script", required=True, help="manifest.script.json (edited in place)")
    ap.add_argument("--seed", default="", help="chapter key for deterministic swap rotation")
    ap.add_argument("--marker", default="",
                    help="write the pass result here (default: <script dir>/manifest.sanitize.json)")
    # The pipeline injects the reframe backend via these flags; without them the
    # pass runs the deterministic layer only (swaps + block gating, no LLM).
    ap.add_argument("--reframe-backend", choices=["none", "vertex", "ollama"], default="none")
    ap.add_argument("--reframe-model", default="")
    ap.add_argument("--project", default="")
    ap.add_argument("--location", default="")
    args = ap.parse_args()

    script_path = Path(args.script)
    script_obj = json.loads(script_path.read_text(encoding="utf-8"))

    call_fn: Optional[ReframeCallFn] = None
    if args.reframe_backend != "none":
        call_fn = _build_backend_call_fn(
            backend=args.reframe_backend, model=args.reframe_model,
            project=args.project, location=args.location)

    summary = sanitize_script(script_obj, seed=args.seed, call_fn=call_fn)

    script_path.write_text(json.dumps(script_obj, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    marker = args.marker or str(script_path.with_name("manifest.sanitize.json"))
    write_marker(marker, summary, seed=args.seed)

    print(f"[sanitize] changed={summary.changed} reframed={summary.reframed} "
          f"unresolved_blocks={len(summary.unresolved_blocks)} marker={marker}")
    if summary.has_unresolved:
        for sid, matched in summary.unresolved_blocks:
            print(f"  UNRESOLVED BLOCK [{sid}] '{matched}'")
        return 2
    return 0


def _build_backend_call_fn(
    *, backend: str, model: str, project: str, location: str
) -> ReframeCallFn:
    """Wrap gemini_narrative_pass._call_model_with_backoff into the injected
    ``call_fn`` shape. Mirrors how _stage_beated resolves the Gemma/Vertex
    backend. Only constructed on the LLM path (CLI --reframe-backend != none).
    """
    from gemini_narrative_pass import _call_model_with_backoff  # noqa: E402
    from google import genai  # noqa: E402

    client: Optional[Any] = None
    if backend == "vertex":
        client = genai.Client(vertexai=True, project=project, location=location)

    def call_fn(system: str, user_payload: Dict[str, Any],
                schema: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        obj, _raw, _usage = _call_model_with_backoff(
            client=client,
            model=model,
            system_instruction=system,
            user_payload=user_payload,
            image_paths=[],
            response_schema=schema,
            max_output_tokens=max_tokens,
            temperature=0.3,
            backoff_max=60.0,
            backend=backend,
        )
        return obj

    return call_fn


if __name__ == "__main__":
    raise SystemExit(main())
