"""Writer-final arbitration (2026-07-16): a detector-triggered semantic code
(impact_mismatch) that survived an informed heal re-roll stops BLOCKING while
the beats bytes stay identical — and never demotes anything structural."""
import io
import json

from studio.worker import (_qa_arbitrated, _qa_verdict, _write_qa_arbitration,
                           _WRITER_ARBITRATED_CODES)


def _ep(tmp_path, codes=("impact_mismatch", "cut_gap")):
    (tmp_path / "manifest.beats.json").write_text('{"beats": [1]}')
    (tmp_path / "prep_qa.json").write_text(json.dumps({
        "flags": [{"code": c, "severity": "ERROR"} for c in codes]}))
    return tmp_path


def test_no_marker_blocks_normally(tmp_path):
    v = _qa_verdict(_ep(tmp_path), started_at=0)
    assert v.blocking == {"impact_mismatch", "cut_gap"}


def test_valid_marker_demotes_only_arbitrated_code(tmp_path):
    ep = _ep(tmp_path)
    _write_qa_arbitration(ep, {"impact_mismatch"}, io.StringIO())
    v = _qa_verdict(ep, started_at=0)
    assert v.blocking == {"cut_gap"}
    assert "impact_mismatch" in v.codes        # still visible for review


def test_beats_change_expires_arbitration(tmp_path):
    ep = _ep(tmp_path)
    _write_qa_arbitration(ep, {"impact_mismatch"}, io.StringIO())
    (ep / "manifest.beats.json").write_text('{"beats": [1, 2]}')
    assert _qa_arbitrated(ep) == set()
    assert "impact_mismatch" in _qa_verdict(ep, started_at=0).blocking


def test_marker_never_demotes_structural_codes(tmp_path):
    ep = _ep(tmp_path)
    from studio.worker import _beats_sha
    (ep / "qa_arbitration.json").write_text(json.dumps(
        {"beats_sha": _beats_sha(ep), "codes": ["cut_gap", "empty_item"]}))
    assert _qa_arbitrated(ep) == set()
    assert "cut_gap" in _qa_verdict(ep, started_at=0).blocking


def test_corrupt_marker_is_ignored(tmp_path):
    ep = _ep(tmp_path)
    (ep / "qa_arbitration.json").write_text("not json")
    assert _qa_arbitrated(ep) == set()


def test_arbitrated_set_is_impact_only():
    # widening this set is a policy decision — the test makes it deliberate
    assert _WRITER_ARBITRATED_CODES == {"impact_mismatch"}
