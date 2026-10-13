"""tools/qa_rules_sweep.py — owner 2026-10-13: "find which episodes are fucked
up", then repair them. Every check judges SHIPPED artifacts with the pipeline's
own functions; --apply queues the heaviest repair a chapter needs, once."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from studio.catalog.db import connect

_SPEC = importlib.util.spec_from_file_location(
    "qa_rules_sweep", Path(__file__).resolve().parents[1] / "tools" / "qa_rules_sweep.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _chapter(con, tmp_path, cid, lines, vision, cast=None, rows=False, codes=()):
    ep = tmp_path / f"Chapter_{cid}"
    ep.mkdir()
    timeline = [{"segment_id": f"g{i:04d}_p00", "tts_text": t,
                 "cuts": [{"file": f, "dur": 3.0}], **({"row": [{}]} if rows else {})}
                for i, (t, f) in enumerate(lines)]
    (ep / "render.plan.clean.json").write_text(json.dumps({"timeline": timeline}))
    (ep / "manifest.vision.json").write_text(json.dumps({"items": vision}))
    (ep / "manifest.cast.json").write_text(json.dumps(cast or {"cast": []}))
    (ep / "prep_qa.json").write_text(json.dumps({"flags": [
        {"code": c, "severity": "ERROR"} for c in codes]}))
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, status, ep_dir, "
                "updated_at) VALUES (?,1,?,?,'u','rendered',?,'t')",
                (cid, cid, f"Chapter {cid}", str(ep)))


def _seed(tmp_path):
    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, added_at) "
                "VALUES (1,'asura','u','tt','The Tutorial Tower of the Advanced Player','t')")
    story = {"scene_file": "p1.jpg", "panel_kind": "story", "ocr_clean": "KIM SHI HYUN!"}
    ad = {"scene_file": "p9.jpg", "panel_kind": "system",
          "ocr_clean": "READ AT ASURASCANS.COM FOR THE FASTEST RELEASES"}
    # 1: TT ch177 as shipped — the ad shown AND voiced
    _chapter(con, tmp_path, 1, [("He runs.", "p1.jpg"),
                                ("Read at. Asurascans.com for the fastest releases.", "p9.jpg")],
             [story, ad], rows=True)
    # 2: clean, with rows
    _chapter(con, tmp_path, 2, [("He runs.", "p1.jpg")], [story], rows=True)
    # 3: BN ch2 — the hero named after the watermark
    _chapter(con, tmp_path, 3, [("Asura scrambles away.", "p1.jpg")],
             [{"scene_file": "p1.jpg", "panel_kind": "story", "ocr_clean": "STOP IT...! ASURA.GG"}],
             cast={"cast": [{"canonical_name": "Asura", "aliases": []}]}, rows=True)
    # 4: shipped before truncated_line became blocking
    _chapter(con, tmp_path, 4, [("He runs.", "p1.jpg")], [story], rows=True,
             codes=["truncated_line"])
    con.commit()
    return db, con


def test_each_rule_finds_its_chapters_and_the_heaviest_repair_wins(tmp_path):
    _, con = _seed(tmp_path)
    rows = {r["chapter_id"]: r for r in tool.sweep(con, checks=[
        "publication_shown", "publication_voiced", "watermark_name", "blocking_now"])}
    assert set(rows[1]["findings"]) == {"publication_shown", "publication_voiced"}
    assert rows[1]["repair"] == "redo"
    assert rows[2]["findings"] == {} and rows[2]["repair"] is None
    assert set(rows[3]["findings"]) == {"watermark_name"} and rows[3]["repair"] == "redo"
    assert rows[4]["findings"] == {"blocking_now": ["truncated_line"]}
    assert rows[4]["repair"] == "reprepare"


def test_apply_queues_each_repair_once_and_skips_live_chapters(tmp_path, capsys):
    db, con = _seed(tmp_path)
    con.execute("INSERT INTO job (type, chapter_id, state, priority) "
                "VALUES ('prepare', 3, 'running', 100)")
    con.commit()
    assert tool.main(["--db", str(db), "--apply", "--out", str(tmp_path / "r.json"),
                      "--check", "publication_voiced", "--check", "watermark_name",
                      "--check", "blocking_now"]) == 0
    out = capsys.readouterr().out
    assert "publication_voiced" in out and "repairs:" in out
    jobs = con.execute("SELECT chapter_id, type, payload_json, priority FROM job "
                       "WHERE state='queued' ORDER BY chapter_id").fetchall()
    assert [(j[0], j[1]) for j in jobs] == [(1, "prepare"), (4, "prepare")]
    assert json.loads(jobs[0][2]) == {"auto_to": "video", "narrate": True}
    assert json.loads(jobs[1][2]) == {"auto_to": "video"}
    assert all(j[3] == 100 for j in jobs)
    assert not (tmp_path / "Chapter_1" / "render.plan.clean.json").exists()   # rewound
    assert json.loads((tmp_path / "r.json").read_text())                       # report kept


def test_rows_backfill_is_only_queued_when_asked(tmp_path, monkeypatch):
    _, con = _seed(tmp_path)
    monkeypatch.setattr(tool, "check_rows_missing", lambda ctx: ["2 row(s) would be made"])
    monkeypatch.setitem(tool.CHECKS, "rows_missing", (tool.check_rows_missing, "rerender"))
    rows = [r for r in tool.sweep(con, chapter_ids=[2], checks=["rows_missing"])]
    assert rows[0]["repair"] == "rerender"
    assert tool.apply(con, rows, rows_backfill=False, priority=100)["rerender"] == 0
    done = tool.apply(con, rows, rows_backfill=True, priority=100, log=lambda m: None)
    assert done["rerender"] == 1
    assert con.execute("SELECT type FROM job WHERE chapter_id=2").fetchone()[0] == "voiceover"
