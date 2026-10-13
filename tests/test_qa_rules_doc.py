"""docs/qa/RULES.md is the owner's QA checklist (2026-10-13): "whenever you update
the code you must never break a fixed issue". This test makes the table bite:
a rule can't be silently un-enforced, and a blocking code can't be added
without its row."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import narration_heal as nh  # noqa: E402
from studio import worker  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("qa_rules_sweep",
                                               ROOT / "tools" / "qa_rules_sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

STATUSES = {"BLOCKED", "HEALED", "DROPPED", "REPORTED", "FEATURE", "OPEN"}


def _rows():
    text = (ROOT / "docs" / "qa" / "RULES.md").read_text()
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and re.fullmatch(r"R\d+", cells[0]):
            rid, rule, decided, codes, blocks, tests, sweep_name, status = cells
            rows.append({"id": rid, "codes": [c.strip() for c in codes.split(",") if c.strip()],
                         "blocks": blocks, "tests": [t.strip() for t in tests.split(",")
                                                     if t.strip()],
                         "sweep": sweep_name, "status": status})
    return rows


def _emitted(code: str) -> bool:
    src = "".join((ROOT / "tools" / f).read_text()
                  for f in ("prep_qa.py", "manifest_freshness.py"))
    return f'"{code}"' in src


def test_the_rulebook_has_rows():
    rows = _rows()
    assert len(rows) >= 20
    assert len({r["id"] for r in rows}) == len(rows)


def test_every_named_test_exists():
    for r in _rows():
        assert r["tests"], f"{r['id']} names no test"
        for t in r["tests"]:
            path, _, func = t.partition("::")
            src = (ROOT / path).read_text()
            assert re.search(rf"^def {re.escape(func)}\(", src, re.M), \
                f"{r['id']}: {t} does not exist — a test was deleted or renamed"


def test_every_code_is_emitted_and_its_blocking_matches_the_worker():
    for r in _rows():
        assert r["status"] in STATUSES, r
        for c in r["codes"]:
            assert _emitted(c), f"{r['id']}: QA code {c} is not emitted by prep_qa"
            blocking = c in worker._CRITICAL_QA_CODES
            assert blocking == (r["blocks"] == "yes"), \
                f"{r['id']}: {c} blocking={blocking} but RULES.md says {r['blocks']}"


def test_healed_and_dropped_codes_are_really_handled():
    heal_src = (ROOT / "tools" / "narration_heal.py").read_text()
    for r in _rows():
        for c in r["codes"]:
            if r["status"] == "HEALED":
                assert c in nh.HEALABLE or f'code == "{c}"' in heal_src, (r["id"], c)
            if r["status"] == "DROPPED":
                assert c in worker._VISUAL_DROPPABLE, (r["id"], c)


def test_blocked_and_feature_rows_are_measured_by_the_sweep():
    for r in _rows():
        if r["status"] in ("BLOCKED", "FEATURE"):
            assert r["sweep"] in sweep.CHECKS, \
                f"{r['id']} is {r['status']} but has no sweep check ({r['sweep']!r})"
        elif r["sweep"] != "—":
            assert r["sweep"] in sweep.CHECKS, r


def test_every_blocking_code_in_the_worker_has_a_rule():
    covered = {c for r in _rows() for c in r["codes"]}
    missing = set(worker._CRITICAL_QA_CODES) - covered
    assert not missing, f"blocking codes without a RULES.md row: {sorted(missing)}"
