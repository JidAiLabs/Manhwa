#!/usr/bin/env python3
import argparse, json, os
from typing import List, Dict, Any

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def looks_like_separator(text: str) -> bool:
    """
    Heuristic for narration/title cards:
    - short/medium text, often with ellipses
    - not a normal sentence with lots of punctuation variety
    """
    t = (text or "").strip()
    if not t:
        return False
    # common separators: "HOWEVER...", "MEANWHILE...", "LATER...", etc.
    up = t.upper()
    if up.endswith("...") and len(t) <= 80:
        return True
    if len(t) <= 40 and up in {"HOWEVER...", "MEANWHILE...", "LATER...", "SOON...", "THEN..."}:
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True, help="manifest.vision.json")
    ap.add_argument("--out", required=True, help="manifest.filtered.json")
    ap.add_argument("--text-only-min-coverage", type=float, default=0.12,
                    help="extra guard in addition to vision['text_only']")
    ap.add_argument("--carry-mode", default="forward", choices=["forward", "backward"],
                    help="forward=attach text-only to next visual; backward=attach to previous visual")
    ap.add_argument("--max-carry-lines", type=int, default=6)
    ap.add_argument("--max-carry-chars", type=int, default=220)

    args = ap.parse_args()

    with open(args.vision, "r", encoding="utf-8") as f:
        man = json.load(f)

    items: List[Dict[str, Any]] = list(man.get("items", []))
    items.sort(key=lambda x: int(x.get("scene_id", 0)))

    carried_buf: List[str] = []
    out_items: List[Dict[str, Any]] = []

    def flush_to_previous():
        nonlocal carried_buf, out_items
        if not carried_buf or not out_items:
            carried_buf = []
            return
        prev = out_items[-1]
        prev.setdefault("carried_text_after", [])
        prev["carried_text_after"].extend(carried_buf)
        carried_buf = []

    def flush_to_next(cur: Dict[str, Any]):
        nonlocal carried_buf
        if not carried_buf:
            return
        cur.setdefault("carried_text_before", [])
        cur["carried_text_before"].extend(carried_buf)
        carried_buf = []

    for it in items:
        ocr = norm(it.get("ocr_clean", ""))
        cov = float(it.get("text_coverage", 0.0) or 0.0)
        vision_text_only = bool(it.get("text_only", False))

        # Decide "text-only / discard" using vision + a coverage guard.
        is_text_scene = (vision_text_only and cov >= args.text_only_min_coverage and len(ocr) >= 4) or looks_like_separator(ocr)

        # Build output item (don’t destroy your existing structure)
        out_it = dict(it)
        out_it["use_for_video"] = (not is_text_scene)

        if is_text_scene:
            # Keep only meaningful carry text; cap to avoid runaway long scenes
            if ocr:
                carried_buf.append(ocr)
                # cap
                carried_buf = carried_buf[: args.max_carry_lines]
                # cap chars
                total = 0
                capped = []
                for s in carried_buf:
                    if total + len(s) > args.max_carry_chars:
                        break
                    capped.append(s)
                    total += len(s)
                carried_buf = capped

            if args.carry_mode == "backward":
                flush_to_previous()

            # keep the item in manifest (for traceability), but it won’t be used later
            out_items.append(out_it)
            continue

        # visual scene
        if args.carry_mode == "forward":
            flush_to_next(out_it)

        out_items.append(out_it)

    # if anything remains and carry_mode=backward already handled;
    # if forward and ends with text-only, just drop carry (or you can store it globally)
    man2 = dict(man)
    man2["items"] = out_items
    man2["filter"] = {
        "text_only_min_coverage": args.text_only_min_coverage,
        "carry_mode": args.carry_mode,
        "max_carry_lines": args.max_carry_lines,
        "max_carry_chars": args.max_carry_chars,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(man2, f, ensure_ascii=False, indent=2)

    kept = sum(1 for x in out_items if x.get("use_for_video"))
    dropped = len(out_items) - kept
    print(f"[ok] wrote={args.out} keep_for_video={kept} text_only={dropped}")

if __name__ == "__main__":
    main()
