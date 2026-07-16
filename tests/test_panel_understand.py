"""panel_understand (Pass 1): per-panel understanding = full coverage by
construction. Tests the pure payload/record logic + the ordered loop with a
stubbed model call (no Gemma needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "panel_understand",
    Path(__file__).resolve().parent.parent / "tools" / "panel_understand.py")
pu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pu)  # type: ignore[union-attr]


def test_build_payload_pulls_ocr_signals_and_rolling_context():
    panel = {"scene_file": "p5.jpg", "ocr_clean": "WHO ARE YOU?",
             "vision": {"labels": [{"desc": "sword"}], "objects": [{"name": "Person"}]}}
    p = pu.build_payload(panel, ["he draws his blade", "the train shakes"])
    assert p["scene_file"] == "p5.jpg"
    assert p["ocr"] == "WHO ARE YOU?"
    assert p["labels"] == ["sword"] and p["objects"] == ["Person"]
    assert p["previous_panels"] == ["he draws his blade", "the train shakes"]


def test_assemble_record_normalizes_and_flags_parse_failure():
    good = pu.assemble_record("p1.jpg", {
        "description": " A monster looms. ", "subjects": ["monster"],
        "action": "it roars", "dialogue": "ROAR", "setting": "train",
        "intensity": "EXPLOSIVE", "panel_kind": "story"})
    assert good["description"] == "A monster looms." and good["intensity"] == "explosive"
    assert good["panel_kind"] == "story" and "error" not in good
    bad = pu.assemble_record("p2.jpg", None)
    assert bad["error"] == "parse_failed" and bad["intensity"] == "unknown"
    assert bad["panel_kind"] == "empty"          # unparsed -> filtered out of grouping
    # invalid intensity -> 'unknown', never crash
    assert pu.assemble_record("p3.jpg", {"intensity": "epic"})["intensity"] == "unknown"
    # chrome/empty/caption pass through; missing/invalid kind defaults to 'story'
    assert pu.assemble_record("p4.jpg", {"panel_kind": "chrome"})["panel_kind"] == "chrome"
    assert pu.assemble_record("p6.jpg", {"panel_kind": "caption"})["panel_kind"] == "caption"
    assert pu.assemble_record("p5.jpg", {})["panel_kind"] == "story"


def test_understand_panels_is_ordered_threads_context_and_covers_all():
    items = [{"scene_file": f"p{i}.jpg", "scene_path": f"/s/p{i}.jpg"} for i in range(3)]
    seen = []

    def stub(payload, image_path):
        seen.append((payload["scene_file"], list(payload["previous_panels"]), image_path))
        return {"description": f"desc {payload['scene_file']}", "action": "x",
                "intensity": "calm"}

    out = pu.understand_panels(items, stub)
    assert [r["scene_file"] for r in out] == ["p0.jpg", "p1.jpg", "p2.jpg"]  # full coverage
    # rolling context threads the prior descriptions, image path passed through
    assert seen[0][1] == [] and seen[1][1] == ["desc p0.jpg"]
    assert seen[2][1] == ["desc p0.jpg", "desc p1.jpg"] and seen[2][2] == "/s/p2.jpg"


def test_resume_skips_already_understood_panels():
    # resume acceptance is content-keyed (Task 12): a cached record must carry
    # a scene_sha matching the CURRENT scene file bytes + the current
    # PROMPT_VERSION, not just a scene_file name match. "a" doesn't exist on
    # disk, so its current sha is "" (see _scene_sha) — the cached record
    # mirrors that so the match holds and p0 is genuinely reused.
    items = [{"scene_file": "p0.jpg", "scene_path": "a"},
             {"scene_file": "p1.jpg", "scene_path": "b"}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm",
                        "scene_sha": "", "prompt_version": pu.PROMPT_VERSION}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == ["p1.jpg"]                       # p0 reused, only p1 called
    assert out[0]["description"] == "kept" and out[1]["description"] == "new"
    assert out[1]["scene_sha"] == "" and out[1]["prompt_version"] == pu.PROMPT_VERSION


def test_resume_legacy_record_without_provenance_reruns():
    # a record from before Task 12 has no scene_sha/prompt_version at all —
    # name match + non-empty description + no error used to be enough to
    # reuse it, but that let new pixels under an old filename (or a prompt
    # rewrite) silently reuse stale understanding. It must now re-run once
    # (and get stamped), the intended one-time migration cost.
    items = [{"scene_file": "p0.jpg", "scene_path": "a"}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm"}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == ["p0.jpg"]                       # re-run despite the name match
    assert out[0]["description"] == "new"
    assert out[0]["scene_sha"] == "" and out[0]["prompt_version"] == pu.PROMPT_VERSION


def test_resume_rejects_changed_scene_bytes(tmp_path):
    # same scene_file name, but the pixels underneath changed -> the stored
    # scene_sha no longer matches the current file's sha, so it must re-run.
    scene = tmp_path / "p0.jpg"
    scene.write_bytes(b"NEW PIXELS")
    items = [{"scene_file": "p0.jpg", "scene_path": str(scene)}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm",
                        "scene_sha": "stale-sha-from-old-pixels",
                        "prompt_version": pu.PROMPT_VERSION}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == ["p0.jpg"]
    assert out[0]["description"] == "new"


def test_resume_rejects_prompt_version_bump(tmp_path):
    # same scene bytes, but the record was produced under an older prompt
    # version -> must re-run so the new prompt's output is picked up.
    scene = tmp_path / "p0.jpg"
    scene.write_bytes(b"STABLE PIXELS")
    items = [{"scene_file": "p0.jpg", "scene_path": str(scene)}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm",
                        "scene_sha": pu._scene_sha(str(scene)),
                        "prompt_version": "pu_v0_old"}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == ["p0.jpg"]
    assert out[0]["description"] == "new"


def test_resume_accepts_full_content_match(tmp_path):
    # same scene bytes AND same prompt_version -> genuinely reused, no re-run.
    scene = tmp_path / "p0.jpg"
    scene.write_bytes(b"STABLE PIXELS")
    items = [{"scene_file": "p0.jpg", "scene_path": str(scene)}]
    prior = {"p0.jpg": {"scene_file": "p0.jpg", "description": "kept", "intensity": "calm",
                        "scene_sha": pu._scene_sha(str(scene)),
                        "prompt_version": pu.PROMPT_VERSION}}
    calls = []

    def stub(payload, image_path):
        calls.append(payload["scene_file"])
        return {"description": "new", "action": "y", "intensity": "tense"}

    out = pu.understand_panels(items, stub, prior=prior)
    assert calls == []                                # fully reused, no call
    assert out[0]["description"] == "kept"


def test_batched_parallel_shares_prebatch_context_and_preserves_order():
    # 5 panels at batch size 3 -> batch A {p0,p1,p2}, batch B {p3,p4}. Parallel
    # execution must still yield ordered, complete output, and every panel in a
    # batch must see the SAME context (the descriptions emitted BEFORE the batch),
    # never its batch-mates. That's the only behavioral diff vs sequential, and
    # it's safe because the window is just the last 2 panels.
    items = [{"scene_file": f"p{i}.jpg", "scene_path": f"/s/p{i}.jpg"} for i in range(5)]
    seen = {}                                         # keyed by file: thread order is nondet

    def stub(payload, image_path):
        seen[payload["scene_file"]] = list(payload["previous_panels"])
        return {"description": f"d{payload['scene_file']}", "action": "x",
                "intensity": "calm"}

    out = pu.understand_panels(items, stub, concurrency=3)
    assert [r["scene_file"] for r in out] == [f"p{i}.jpg" for i in range(5)]  # ordered + complete
    # batch A: all share the empty pre-batch context (batch-mates invisible to each other)
    assert seen["p0.jpg"] == seen["p1.jpg"] == seen["p2.jpg"] == []
    # batch B: all see the last-2 descriptions from BEFORE the batch (p1, p2)
    assert seen["p3.jpg"] == seen["p4.jpg"] == ["dp1.jpg", "dp2.jpg"]


# --- in-world screen rescue (chrome -> story via a real speech balloon) ------

def test_inworld_balloon_promotes_confident_compact():
    # ORV ep1 p000003: the masterpiece comment balloon (conf 0.96, ~0.14 area)
    dets = [(56, 768, 499, 1054, 0.96)]
    assert pu._is_inworld_balloon(dets, 736, 1169) is True


def test_inworld_balloon_rejects_screen_sized_false_positive():
    # ORV ep1 p000004 (stats card): low-conf, ~0.6-area boxes = whole panel
    dets = [(162, 41, 753, 423, 0.47), (216, 38, 781, 423, 0.36)]
    assert pu._is_inworld_balloon(dets, 800, 480) is False


def test_inworld_balloon_rejects_no_detection():
    # ORV ep1 p000033 (publisher credit): no balloon at all
    assert pu._is_inworld_balloon([], 800, 600) is False


def test_inworld_balloon_needs_both_confidence_and_compactness():
    # confident but huge -> rejected; compact but low-conf -> rejected
    assert pu._is_inworld_balloon([(0, 0, 760, 560, 0.95)], 800, 600) is False
    assert pu._is_inworld_balloon([(50, 50, 200, 180, 0.45)], 800, 600) is False


# --- system panel_kind (in-world game/system UI cards) -----------------------

def test_norm_panel_kind_accepts_system():
    assert pu._norm_panel_kind("system") == "system"
    assert pu._norm_panel_kind("SYSTEM") == "system"
    assert pu._norm_panel_kind("garbage") == "story"   # unknown -> never-drop side


def test_panel_schema_enumerates_system():
    enum = pu.PANEL_SCHEMA["properties"]["panel_kind"]["enum"]
    assert "system" in enum
    assert set(enum) == {"story", "chrome", "empty", "caption", "system"}


# --- bubble-on-plain reclassification (the "husk" root cause) -----------------
# A panel that is ONLY a speech/shout/caption bubble or text on a plain/blank/
# white background, with NO drawn scene, must be 'caption' — its words ride the
# narration and the bubble is never shown. The model non-deterministically labels
# it 'story'/'system', which protects an empty-bubble husk on screen. A
# deterministic rule is the guarantee. A real in-world system/stat/HUD window is a
# STORY VISUAL and must NOT be reclassified.

def test_bubble_on_plain_background_is_reclassified_caption():
    f = pu._is_caption_bubble_on_plain
    # Nano ch1 p000020: a shout bubble on a plain white background, no scene art
    assert f("A single white speech bubble containing text centered against a "
             "plain white background", [])
    assert f("A lone shout bubble on a blank white background.", [])
    assert f("A caption box of text over an empty black background.",
             ["a text box"])
    # subjects only describe the bubble/text itself -> still a caption
    assert f("A speech balloon on a plain background.",
             ["speech bubble", "text"])


def test_real_system_window_is_not_reclassified():
    f = pu._is_caption_bubble_on_plain
    # an IN-WORLD status/stat/HUD window is a story visual — never demote it
    assert not f("An in-world status window showing the character's stats on a "
                 "plain background.", [])
    assert not f("A blue SYSTEM notification window on a plain dark background.",
                 ["system window"])
    assert not f("A quest window on a blank background.", [])


def test_real_scene_with_a_bubble_is_not_reclassified():
    f = pu._is_caption_bubble_on_plain
    # a drawn character/scene that happens to carry a bubble is NOT a husk
    assert not f("A man shouts in a speech bubble against a plain white background.",
                 ["man"])
    assert not f("Two warriors clash, a speech bubble between them.",
                 ["warrior", "sword"])
    # no plain/blank-background signal at all -> not a husk
    assert not f("A speech bubble over a busy city street at night.", [])


def test_assemble_record_overrides_story_husk_to_caption():
    # the model mislabeled the plain shout bubble 'story' -> deterministic caption
    rec = pu.assemble_record("p000020.jpg", {
        "description": "A single white speech bubble containing text centered "
                       "against a plain white background",
        "subjects": [], "dialogue": "PEASANT BLOOD... THEY SAY...?",
        "action": "someone speaks", "intensity": "tense", "panel_kind": "story"})
    assert rec["panel_kind"] == "caption"
    assert rec["dialogue"] == "PEASANT BLOOD... THEY SAY...?"   # words preserved
    # a mislabeled 'system' husk is also corrected
    assert pu.assemble_record("p1.jpg", {
        "description": "A plain speech bubble on a blank white background.",
        "subjects": [], "action": "x", "intensity": "calm",
        "panel_kind": "system"})["panel_kind"] == "caption"


def test_assemble_record_keeps_real_system_window():
    rec = pu.assemble_record("p2.jpg", {
        "description": "An in-world STATUS window listing HP, MP and level on a "
                       "plain background.",
        "subjects": ["status window"], "dialogue": "STATUS",
        "action": "the system appears", "intensity": "calm",
        "panel_kind": "system"})
    assert rec["panel_kind"] == "system"           # real system UI stays shown


def test_assemble_record_keeps_real_story_scene():
    rec = pu.assemble_record("p3.jpg", {
        "description": "A man stands on a rooftop at dusk, speaking.",
        "subjects": ["man"], "dialogue": "It's over.",
        "action": "he speaks", "intensity": "tense", "panel_kind": "story"})
    assert rec["panel_kind"] == "story"


# --- vision write-back ordering (the mtime-inversion root fix, Task 8) --------

def test_writeback_before_understood_dump_mtime_order(tmp_path, monkeypatch):
    """The panel_kind/subjects write-back onto manifest.vision.json must land
    BEFORE the understood.json dump, so (a) vision.mtime <= understood.mtime
    always holds going forward (no more mtime inversion) and (b) understood's
    _meta input-sha stamp hashes the FINAL vision bytes — the sha_only
    understood<-vision freshness edge matches on a healthy chapter."""
    import hashlib
    import json
    import os
    import sys

    vision_path = tmp_path / "manifest.vision.json"
    out_path = tmp_path / "manifest.panels.understood.json"
    vision_path.write_text(json.dumps({
        "items": [{"scene_file": "p0.jpg",
                   "scene_path": str(tmp_path / "p0.jpg")}]}))

    def fake_call(**kwargs):
        # panel_kind 'story' differs from the vision item's (absent) kind, so
        # the write-back fires (changed=True); no detector candidates.
        return ({"description": "a scene", "subjects": ["man"], "action": "x",
                 "intensity": "calm", "panel_kind": "story"}, "", None)

    monkeypatch.setattr(pu, "_call_model_with_backoff",
                        lambda **kw: fake_call(**kw))
    monkeypatch.setattr(sys, "argv", [
        "panel_understand.py", "--vision-manifest", str(vision_path),
        "--out", str(out_path), "--backend", "ollama", "--concurrency", "1",
        "--panel-weights", str(tmp_path / "missing.pt")])   # fail-soft: no YOLO
    assert pu.main() == 0

    assert os.path.getmtime(vision_path) <= os.path.getmtime(out_path), (
        "vision was rewritten AFTER understood.json — the write-back must "
        "come first")
    understood = json.loads(out_path.read_text())
    stamped = understood["_meta"]["inputs"]["manifest.vision.json"]
    current = hashlib.sha1(vision_path.read_bytes()).hexdigest()
    assert stamped == current, (
        "understood stamped a pre-write-back vision sha — the sha_only "
        "freshness edge would false-flag every fresh run")
    # and the write-back content itself is preserved
    vision = json.loads(vision_path.read_text())
    assert vision["items"][0]["panel_kind"] == "story"
    assert vision["items"][0]["subjects"] == ["man"]


# ---------------------------------------------------------------------------
# Impact-SFX fusion (eyes wave): the deterministic detector's verdict is
# stamped on every record (DETECTOR-owned, never model-claimed), the panel
# prompt gains ONE context block when lettering is present, and the schema
# grows strikes_or_weapons + sfx_text without touching existing fields.
# ---------------------------------------------------------------------------

def test_prompt_version_bumped_for_impact_fields():
    # pu_v4 invalidates every pu_v3 resume record (INTENDED — chapters
    # re-understand under the evidence-discipline + uncertain-flag prompt;
    # pu_v3 did the same for appearance-aware subjects over pu_v2, pu_v2 for
    # the impact-aware prompt over pu_v1).
    assert pu.PROMPT_VERSION == "pu_v4"


def test_panel_schema_adds_impact_fields_backward_compatibly():
    props = pu.PANEL_SCHEMA["properties"]
    assert props["strikes_or_weapons"]["enum"] == ["none", "visible", "in_use"]
    assert props["sfx_text"]["type"] == "STRING"
    # existing required set unchanged — old consumers keep parsing
    assert pu.PANEL_SCHEMA["required"] == [
        "description", "action", "intensity", "panel_kind"]


def test_build_payload_appends_impact_notice_only_when_regions():
    panel = {"scene_file": "p5.jpg"}
    base = pu.build_payload(panel, [])
    assert "impact_sfx_notice" not in base     # byte-compatible when no signal
    regions = [{"bbox": [10, 20, 64, 75], "area_frac": 0.006,
                "mean_hue_deg": 356.0}]
    p = pu.build_payload(panel, [], impact_regions=regions)
    notice = p["impact_sfx_notice"]
    assert "Large impact-style SFX lettering" in notice
    assert "strike, stab, blow, or crash" in notice
    assert "transcribe" in notice
    assert "10" in notice and "75" in notice   # bbox summary is included


def test_assemble_record_normalizes_strikes_and_sfx_text():
    rec = pu.assemble_record("p1.jpg", {
        "description": "x", "action": "y", "intensity": "calm",
        "panel_kind": "story", "strikes_or_weapons": "IN_USE",
        "sfx_text": " Puk "})
    assert rec["strikes_or_weapons"] == "in_use"
    assert rec["sfx_text"] == "Puk"
    # missing / invalid enum -> safe defaults; parse failure carries them too
    assert pu.assemble_record("p2.jpg", {})["strikes_or_weapons"] == "none"
    assert pu.assemble_record("p2.jpg", {})["sfx_text"] == ""
    assert pu.assemble_record(
        "p3.jpg", {"strikes_or_weapons": "everywhere"}
    )["strikes_or_weapons"] == "none"
    bad = pu.assemble_record("p4.jpg", None)
    assert bad["strikes_or_weapons"] == "none" and bad["sfx_text"] == ""


def test_understand_panels_stamps_detector_owned_impact_sfx():
    items = [{"scene_file": "p0.jpg", "scene_path": "/s/p0.jpg"},
             {"scene_file": "p1.jpg", "scene_path": "/s/p1.jpg"}]
    regions = {"/s/p1.jpg": [{"bbox": [1, 2, 30, 40], "area_frac": 0.01,
                              "mean_hue_deg": 350.0}]}
    payloads = {}

    def stub(payload, image_path):
        payloads[payload["scene_file"]] = payload
        # the model CLAIMS an impact stamp — the detector's verdict must win
        return {"description": "d", "action": "a", "intensity": "calm",
                "impact_sfx": {"present": True, "regions": 9}}

    out = pu.understand_panels(items, stub,
                               impact_fn=lambda sp: regions.get(sp, []))
    assert out[0]["impact_sfx"] == {"present": False, "regions": 0}
    assert out[1]["impact_sfx"] == {"present": True, "regions": 1}
    # the notice reaches ONLY the flagged panel's prompt
    assert "impact_sfx_notice" not in payloads["p0.jpg"]
    assert "impact_sfx_notice" in payloads["p1.jpg"]


def test_default_impact_fn_is_fail_soft_on_missing_images():
    # scene paths that don't exist (unit-test items) must never crash the
    # loop — the default detector path returns [] and stamps absent.
    items = [{"scene_file": "p0.jpg", "scene_path": "/nonexistent/p0.jpg"}]
    out = pu.understand_panels(
        items, lambda payload, image_path: {
            "description": "d", "action": "a", "intensity": "calm"})
    assert out[0]["impact_sfx"] == {"present": False, "regions": 0}


# ---- pu_v4: uncertain flag + evidence discipline -----------------------------

def test_uncertain_flag_passthrough_and_regex_backstop():
    # model-set flag survives normalization
    rec = pu.assemble_record("p1.jpg", {
        "description": "d", "action": "a", "intensity": "calm",
        "panel_kind": "story", "uncertain": True})
    assert rec["uncertain"] is True
    # hedged wording forces the flag even when the model forgot it —
    # the exact production failure string (p000003, "limb or object")
    rec = pu.assemble_record("p2.jpg", {
        "description": "d", "action": "a", "intensity": "calm",
        "panel_kind": "story",
        "subjects": ["a pale pinkish limb or object with a motion trail"]})
    assert rec["uncertain"] is True
    # confident subjects stay unflagged
    rec = pu.assemble_record("p3.jpg", {
        "description": "d", "action": "a", "intensity": "calm",
        "panel_kind": "story", "subjects": ["a masked figure in a dark cloak"]})
    assert rec["uncertain"] is False


def test_forced_choice_reask_commits_or_keeps_hedged():
    items = [{"scene_file": "p0.jpg", "scene_path": "/nonexistent/p0.jpg"}]
    calls = []

    def committing(payload, image_path):
        calls.append("forced_choice_notice" in payload)
        if len(calls) == 1:
            return {"description": "d", "action": "a", "intensity": "calm",
                    "panel_kind": "story", "subjects": ["a limb or object"]}
        return {"description": "d2", "action": "a", "intensity": "calm",
                "panel_kind": "story", "subjects": ["a tree branch"]}

    out = pu.understand_panels(items, committing)
    assert calls == [False, True]              # exactly ONE re-ask, marked
    assert out[0]["subjects"] == ["a tree branch"]
    assert out[0]["uncertain"] is False and out[0].get("reask") is True

    def stubborn(payload, image_path):
        return {"description": "d", "action": "a", "intensity": "calm",
                "panel_kind": "story", "subjects": ["a limb or object"]}

    out = pu.understand_panels(items, stubborn)
    assert out[0]["uncertain"] is True         # hedged record stands
    assert "reask" not in out[0]


def test_tall_strip_merges_uncertain_any_window(monkeypatch, tmp_path):
    from PIL import Image
    p = tmp_path / "strip.jpg"
    Image.new("RGB", (100, 3400), "white").save(p)
    monkeypatch.setattr(pu, "_TALL_MIN_H_PX", 3000)
    monkeypatch.setattr(pu, "_TALL_MIN_RATIO", 3.0)
    responses = iter([
        {"description": "top", "action": "a", "intensity": "calm",
         "panel_kind": "story"},
        {"description": "mid", "action": "a", "intensity": "calm",
         "panel_kind": "story", "uncertain": True},
        {"description": "bot", "action": "a", "intensity": "calm",
         "panel_kind": "story"},
    ])
    merged, meta = pu.understand_tall_strip(
        {"scene_file": "strip.jpg", "scene_path": str(p)}, [],
        lambda payload, image_path: next(responses, {}), (100, 3400))
    assert merged["uncertain"] is True         # any-window OR


def test_prompt_version_is_pu_v4():
    assert pu.PROMPT_VERSION == "pu_v4"
    assert pu.TALL_WINDOWS_VERSION.startswith("pu_v4")
