#!/usr/bin/env python3
"""
narration_sanitize — advertiser-friendly pass for manhwa-recap narration & metadata.

Design: a SAFETY NET, not a rewriter.
  - replace  -> swap grammatically-safe tokens silently
  - flag     -> surface for a regenerate/review pass (NOT auto-rewritten)
  - block    -> halt; must not publish as-is (slurs, sexual violence, explicit anatomy)

Primary cleanliness should come from instructing your narration generator to
write clean in the first place (see narration_safe_rules.py for the shared
source-of-truth rules). This module catches leaks and gates publish.

Library:
    from narration_sanitize import Sanitizer
    s = Sanitizer("narration_denylist.json")            # optional: slur_list_file in config
    report = s.run(text, scope="spoken", seed="ch0040")
    if report.blocked:        # do not publish
        ...
    if report.flagged:        # route to rewrite/review
        ...
    clean = report.text

CLI:
    python narration_sanitize.py --in script.txt --scope spoken --seed ch0040 > clean.txt
    # exit 0 = clean, 3 = flags present (review), 2 = blocks present (halt)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Hit:
    category: str
    entry_id: str
    matched: str
    action: str          # replace | flag | block
    replacement: Optional[str] = None
    note: Optional[str] = None


@dataclass
class Report:
    text: str
    replacements: List[Hit] = field(default_factory=list)
    flags: List[Hit] = field(default_factory=list)
    blocks: List[Hit] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    @property
    def blocked(self) -> bool:
        return bool(self.blocks)

    @property
    def exit_code(self) -> int:
        if self.blocks:
            return 2
        if self.flags:
            return 3
        return 0


def _match_case(template: str, replacement: str) -> str:
    """Carry the casing of the matched text onto the replacement."""
    if template.isupper() and len(template) > 1:
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class Sanitizer:
    def __init__(self, config_path: str | Path):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.meta = cfg.get("meta", {})
        self.categories = cfg.get("categories", {})

        # Optional external slur list -> hard blocks. Not enumerated in the config.
        self.slur_pattern: Optional[re.Pattern] = None
        slur_file = self.meta.get("slur_list_file")
        if slur_file:
            p = (Path(config_path).parent / slur_file).resolve()
            if p.exists():
                terms = [t.strip() for t in p.read_text(encoding="utf-8").splitlines()
                         if t.strip() and not t.startswith("#")]
                if terms:
                    alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
                    self.slur_pattern = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

        # Pre-compile every entry.
        self._compiled: List[dict] = []
        for category, entries in self.categories.items():
            for e in entries:
                action = e["action"]
                scope = e.get("scope", "both")
                if action == "replace":
                    forms = e["forms"]
                    alt = "|".join(re.escape(f) for f in
                                   sorted(forms.keys(), key=len, reverse=True))
                    self._compiled.append({
                        "category": category, "id": e["id"], "action": "replace",
                        "scope": scope, "forms": forms,
                        "regex": re.compile(rf"\b(?:{alt})\b", re.IGNORECASE),
                    })
                else:  # flag | block
                    self._compiled.append({
                        "category": category, "id": e["id"], "action": action,
                        "scope": scope, "note": e.get("note"),
                        "regex": re.compile(e["pattern"], re.IGNORECASE),
                    })

    @staticmethod
    def _in_scope(entry_scope: str, scope: str) -> bool:
        return entry_scope == "both" or entry_scope == scope

    def run(self, text: str, scope: str = "spoken", seed: str = "") -> Report:
        report = Report(text=text)
        base = int(hashlib.sha1(seed.encode()).hexdigest(), 16) if seed else 0

        # 1) Detect flags & blocks on the ORIGINAL text (don't mutate these).
        #    Record their spans so the replace pass below never swaps a token
        #    INSIDE a flagged/blocked phrase (e.g. don't turn "killed herself"
        #    into "took out herself" -- the suicide reframe owns that span).
        protected: List[tuple] = []
        for c in self._compiled:
            if c["action"] == "replace" or not self._in_scope(c["scope"], scope):
                continue
            for m in c["regex"].finditer(text):
                hit = Hit(c["category"], c["id"], m.group(0), c["action"], note=c.get("note"))
                (report.blocks if c["action"] == "block" else report.flags).append(hit)
                protected.append((m.start(), m.end()))

        # External slur list -> blocks.
        if self.slur_pattern:
            for m in self.slur_pattern.finditer(text):
                report.blocks.append(Hit("slurs", "slur_list", m.group(0), "block",
                                         note="Term from maintained slur list. Remove entirely."))
                protected.append((m.start(), m.end()))

        # Mask protected spans with non-word placeholders so the \b-anchored
        # replace regexes cannot match inside them; restored verbatim after.
        out = text
        ph_map: Dict[str, str] = {}
        if protected:
            merged: List[List[int]] = []
            for s, e in sorted(set(protected)):
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            for i, (s, e) in enumerate(sorted(merged, reverse=True)):
                ph = "\x00%d\x00" % i
                ph_map[ph] = out[s:e]
                out = out[:s] + ph + out[e:]

        # 2) Apply replacements, producing clean text. Rotate per occurrence.
        counters: Dict[str, int] = {}
        for c in self._compiled:
            if c["action"] != "replace" or not self._in_scope(c["scope"], scope):
                continue
            forms = c["forms"]
            eid = c["id"]

            def _sub(m: re.Match) -> str:
                surface = m.group(0)
                opts = forms.get(surface.lower())
                if not opts:
                    return surface
                idx = (base + counters.get(eid, 0)) % len(opts)
                counters[eid] = counters.get(eid, 0) + 1
                repl = _match_case(surface, opts[idx])
                report.replacements.append(
                    Hit(c["category"], eid, surface, "replace", replacement=repl))
                return repl

            out = c["regex"].sub(_sub, out)

        # restore protected (flagged/blocked) spans verbatim
        for ph, orig in ph_map.items():
            out = out.replace(ph, orig)

        report.text = out
        return report


def _format_report(r: Report) -> str:
    lines: List[str] = []
    if r.blocks:
        lines.append(f"BLOCKS ({len(r.blocks)}) - do not publish:")
        for h in r.blocks:
            lines.append(f"  [{h.category}/{h.entry_id}] '{h.matched}'"
                         + (f" - {h.note}" if h.note else ""))
    if r.flags:
        lines.append(f"FLAGS ({len(r.flags)}) - review/rewrite:")
        for h in r.flags:
            lines.append(f"  [{h.category}/{h.entry_id}] '{h.matched}'"
                         + (f" - {h.note}" if h.note else ""))
    if r.replacements:
        lines.append(f"REPLACED ({len(r.replacements)}):")
        for h in r.replacements:
            lines.append(f"  [{h.category}/{h.entry_id}] '{h.matched}' -> '{h.replacement}'")
    if not lines:
        lines.append("Clean - no hits.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Advertiser-friendly sanitizer for recap narration/metadata.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("narration_denylist.json")))
    ap.add_argument("--in", dest="infile", help="Input text file (default: stdin)")
    ap.add_argument("--scope", choices=["spoken", "metadata"], default="spoken")
    ap.add_argument("--seed", default="", help="Video id, for deterministic replacement rotation")
    ap.add_argument("--report", action="store_true", help="Print report to stderr")
    args = ap.parse_args()

    text = Path(args.infile).read_text(encoding="utf-8") if args.infile else sys.stdin.read()
    report = Sanitizer(args.config).run(text, scope=args.scope, seed=args.seed)

    sys.stdout.write(report.text)
    if args.report or report.flagged or report.blocked:
        sys.stderr.write("\n" + _format_report(report) + "\n")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
