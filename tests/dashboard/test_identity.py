"""Chapter identity: does a chapter row agree with the files on disk?"""
import json
import os
from pathlib import Path

import pytest

from studio.catalog import identity, repair
from studio.catalog.db import connect


@pytest.fixture()
def cat(tmp_path, monkeypatch):
    """A catalog whose ongoing/ tree lives under tmp_path."""
    import studio.config as cfg
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(identity.studio_config, "REPO_ROOT", tmp_path)
    db = tmp_path / "s.db"
    con = connect(db)
    con.execute("INSERT INTO series (id, source, series_url, slug, title, "
                "added_at) VALUES (1,'webtoon','https://w.example/orv',"
                "'orv','Omniscient Reader','t')")
    con.commit()
    return con, tmp_path, db


def _chapter(con, cid, number, label, url, ep_dir=None, status="downloaded"):
    con.execute("INSERT INTO chapter (id, series_id, number, label, url, "
                "status, ep_dir, updated_at) VALUES (?,1,?,?,?,?,?,'t')",
                (cid, number, label, url, status,
                 str(ep_dir) if ep_dir else None))
    con.commit()


def test_canonical_dirname_matches_the_fetchers_rule():
    """The audit and the fetcher MUST agree, or the audit reports healthy
    directories as drifted."""
    from studio.cli import _sanitize_label
    for label in ("Episode 0 (Prologue)", "Chapter 1", "Ch. 12.5",
                  "Season 2 — Finale!", "", "///"):
        assert identity.canonical_dirname(label) == _sanitize_label(label)


def test_clean_catalog_reports_nothing(cat):
    con, root, _ = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    _chapter(con, 1, 1, "Episode 1", "https://w.example/e1", d)
    assert identity.audit_series(con, 1)["issues"] == []


def test_detects_directory_named_for_a_stale_label(cat):
    con, root, _ = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    # the row was later renumbered/retitled; the directory kept the old name
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "https://w.example/e0", d)
    codes = [i["code"] for i in identity.audit_series(con, 1)["issues"]]
    assert "ep_dir_name_drift" in codes


def test_detects_two_rows_sharing_one_source_url(cat):
    """The real Omniscient Reader defect: the prologue and 'Episode 1' were
    both discovered from the same page."""
    con, root, _ = cat
    (root / "ongoing" / "orv").mkdir(parents=True)
    u = "https://w.example/episode-0-prologue?episode_no=1"
    _chapter(con, 1, 0, "Episode 0 (Prologue)", u)
    _chapter(con, 2, 1, "Episode 1", u)
    issues = identity.audit_series(con, 1)["issues"]
    assert sum(1 for i in issues if i["code"] == "url_duplicate") == 2
    assert all(i["severity"] == "error"
               for i in issues if i["code"] == "url_duplicate")


def test_detects_shared_and_missing_directories(cat):
    con, root, _ = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    _chapter(con, 1, 1, "Episode 1", "u1", d)
    _chapter(con, 2, 2, "Episode 1", "u2", d)          # same dir
    _chapter(con, 3, 3, "Episode 3", "u3",
             root / "ongoing" / "orv" / "Episode_3")   # never created
    codes = {i["code"] for i in identity.audit_series(con, 1)["issues"]}
    assert "ep_dir_shared" in codes and "ep_dir_missing" in codes


def test_detects_a_db_copied_from_another_machine(cat):
    con, root, _ = cat
    (root / "ongoing" / "orv").mkdir(parents=True)
    _chapter(con, 1, 1, "Episode 1", "u1", "/Users/someone-else/x/Episode_1")
    codes = [i["code"] for i in identity.audit_series(con, 1)["issues"]]
    assert codes == ["ep_dir_outside_tree"]


def test_label_number_mismatch_only_fires_when_provable(cat):
    con, root, _ = cat
    (root / "ongoing" / "orv").mkdir(parents=True)
    _chapter(con, 1, 4, "Episode 5", "u1")      # provably inconsistent
    _chapter(con, 2, 7, "Prologue", "u2")       # no number claimed -> silent
    codes = [i["code"] for i in identity.audit_series(con, 1)["issues"]]
    assert codes == ["label_number_mismatch"]
    assert identity.label_number("Prologue") is None
    assert identity.label_number("Ch. 12.5") == 12.5
    assert identity.label_number("Episode 0 (Prologue)") == 0.0


def test_never_flags_a_chapter_that_was_never_fetched(cat):
    con, root, _ = cat
    (root / "ongoing" / "orv").mkdir(parents=True)
    _chapter(con, 1, 1, "Episode 1", "u1", None, status="discovered")
    assert identity.audit_series(con, 1)["issues"] == []


# --- repair -------------------------------------------------------------------

def test_repair_renames_and_is_exactly_reversible(cat):
    con, root, db = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    (d / "001.jpg").write_bytes(b"page one")
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", d)

    plan = repair.plan_series(con, 1)
    assert len(plan["actions"]) == 1
    assert not plan["blockers"]

    j = root / "journal.json"
    repair.apply_plan(con, [plan], j, db_path=str(db))

    moved = root / "ongoing" / "orv" / "Episode_0_Prologue"
    assert moved.is_dir() and (moved / "001.jpg").read_bytes() == b"page one"
    assert not d.exists()
    assert con.execute("SELECT ep_dir FROM chapter WHERE id=1"
                       ).fetchone()[0] == str(moved)
    assert identity.audit_series(con, 1)["issues"] == []

    repair.undo(con, j)
    assert d.is_dir() and (d / "001.jpg").read_bytes() == b"page one"
    assert not moved.exists()
    assert con.execute("SELECT ep_dir FROM chapter WHERE id=1"
                       ).fetchone()[0] == str(d)


def test_repair_refuses_when_the_target_already_exists(cat):
    con, root, _ = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    (root / "ongoing" / "orv" / "Episode_0_Prologue").mkdir()
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", d)
    plan = repair.plan_series(con, 1)
    assert plan["actions"] == [] and plan["blockers"]


def test_repair_refuses_while_a_job_is_live_on_that_chapter(cat):
    """Never move a directory out from under a running process."""
    con, root, _ = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", d)
    con.execute("INSERT INTO job (type, chapter_id, state) VALUES "
                "('prepare', 1, 'running')")
    con.commit()
    plan = repair.plan_series(con, 1)
    assert plan["actions"] == []
    assert any("live job" in b for b in plan["blockers"])


def test_cli_refuses_to_apply_over_an_ambiguous_catalog(cat, capsys):
    """url_duplicate means two rows disagree about which chapter is which.
    No rule settles that from metadata — the tool must stop and say so."""
    con, root, db = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    u = "https://w.example/episode-0-prologue"
    _chapter(con, 1, 0, "Episode 0 (Prologue)", u, d)
    _chapter(con, 2, 1, "Episode 1", u)
    rc = repair.main(["--db", str(db), "--apply", "--journal",
                      str(root / "j.json")])
    assert rc == 3
    assert not (root / "j.json").exists(), "nothing may be written on refusal"
    assert d.is_dir(), "the tree must be untouched"


def test_cli_dry_run_is_the_default_and_mutates_nothing(cat, capsys):
    con, root, db = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", d)
    assert repair.main(["--db", str(db)]) == 0
    assert d.is_dir()
    assert "dry run" in capsys.readouterr().out


def test_cli_apply_requires_a_journal(cat):
    con, root, db = cat
    d = root / "ongoing" / "orv" / "Episode_1"
    d.mkdir(parents=True)
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", d)
    assert repair.main(["--db", str(db), "--apply"]) == 2
    assert d.is_dir()


def test_repair_orders_a_chained_rename(cat):
    """An off-by-one drift is a CHAIN, not independent moves. Checked
    independently the second rename looks impossible ('Episode_1 already
    exists') though it becomes trivial once the first runs. This is the exact
    shape of the live Omniscient Reader data."""
    con, root, db = cat
    base = root / "ongoing" / "orv"
    (base / "Episode_1").mkdir(parents=True)
    (base / "Episode_1" / "p.jpg").write_bytes(b"prologue pages")
    (base / "Episode_2").mkdir()
    (base / "Episode_2" / "p.jpg").write_bytes(b"episode one pages")
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", base / "Episode_1")
    _chapter(con, 2, 1, "Episode 1", "u1", base / "Episode_2")

    plan = repair.plan_series(con, 1)
    assert not plan["blockers"], plan["blockers"]
    assert len(plan["actions"]) == 2
    # the vacating rename must be ordered FIRST
    assert Path(plan["actions"][0]["to"]).name == "Episode_0_Prologue"

    repair.apply_plan(con, [plan], root / "j.json", db_path=str(db))
    assert (base / "Episode_0_Prologue" / "p.jpg").read_bytes() == b"prologue pages"
    assert (base / "Episode_1" / "p.jpg").read_bytes() == b"episode one pages"
    assert not (base / "Episode_2").exists()
    assert identity.audit_series(con, 1)["issues"] == []

    repair.undo(con, root / "j.json")
    assert (base / "Episode_1" / "p.jpg").read_bytes() == b"prologue pages"
    assert (base / "Episode_2" / "p.jpg").read_bytes() == b"episode one pages"
    assert not (base / "Episode_0_Prologue").exists()


def test_repair_breaks_a_true_rename_cycle(cat):
    """A -> B and B -> A cannot be done in any order without staging."""
    con, root, db = cat
    base = root / "ongoing" / "orv"
    (base / "Episode_1").mkdir(parents=True)
    (base / "Episode_1" / "p.jpg").write_bytes(b"content A")
    (base / "Episode_2").mkdir()
    (base / "Episode_2" / "p.jpg").write_bytes(b"content B")
    _chapter(con, 1, 1, "Episode 2", "u1", base / "Episode_1")   # wants Ep_2
    _chapter(con, 2, 2, "Episode 1", "u2", base / "Episode_2")   # wants Ep_1

    plan = repair.plan_series(con, 1)
    assert not plan["blockers"], plan["blockers"]
    repair.apply_plan(con, [plan], root / "j.json", db_path=str(db))
    assert (base / "Episode_2" / "p.jpg").read_bytes() == b"content A"
    assert (base / "Episode_1" / "p.jpg").read_bytes() == b"content B"
    assert not list(base.glob("*.__swap")), "staging dir must not survive"
    # the DRIFT is gone. (This fixture's rows deliberately carry contradictory
    # labels — number 1 labelled "Episode 2" — to force a cycle at all, so a
    # label_number_mismatch legitimately remains and is not this test's
    # concern.)
    codes = [i["code"] for i in identity.audit_series(con, 1)["issues"]]
    assert "ep_dir_name_drift" not in codes


def test_repair_still_blocks_on_a_genuine_collision(cat):
    """A destination occupied by something that is NOT itself moving stays
    blocked — the ordering fix must not become a licence to overwrite."""
    con, root, db = cat
    base = root / "ongoing" / "orv"
    (base / "Episode_1").mkdir(parents=True)
    (base / "Episode_0_Prologue").mkdir()     # squatter, no chapter row
    (base / "Episode_0_Prologue" / "x.jpg").write_bytes(b"someone else")
    _chapter(con, 1, 0, "Episode 0 (Prologue)", "u0", base / "Episode_1")
    plan = repair.plan_series(con, 1)
    assert plan["actions"] == []
    assert plan["blockers"]
    assert (base / "Episode_0_Prologue" / "x.jpg").read_bytes() == b"someone else"
