"""thumbnail_build: concept (style+hook+refs) -> text-free art -> overlay.

The series path (render_thumbnail) must resolve ref scene files against the
CLIMAX chapter dir while writing output to an independent series dir, and must
drive a TEXT-FREE art prompt (the licensed name is never baked into the image).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "thumbnail_build",
    Path(__file__).resolve().parent.parent / "tools" / "thumbnail_build.py")
tb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tb)  # type: ignore[union-attr]


def _patch(monkeypatch, calls):
    def fake_generate(episode_dir, *, hook_text, refs, models,
                      aspect, size, out_path, prompt_override=""):
        calls.update(episode_dir=episode_dir, refs=refs, out_path=out_path,
                     prompt=prompt_override, aspect=aspect)
        with open(out_path, "wb") as f:
            f.write(b"PNG")
        return models[0]

    def fake_overlay(art, out, *, hook, style_overlay, speech,
                     badge="", tags=None, card=None):
        calls.update(overlay_hook=hook, overlay_art=art,
                     overlay_badge=badge, overlay_tags=tags,
                     overlay_card=card)
        with open(out, "wb") as f:
            f.write(b"JPG")

    monkeypatch.setattr(tb.tg, "generate", fake_generate)
    monkeypatch.setattr(tb, "render_overlay", fake_overlay)
    monkeypatch.setattr(tb, "art_text_words", lambda path: [])


def test_render_thumbnail_series_mode_resolves_refs_at_ref_dir(tmp_path, monkeypatch):
    ref_ep = tmp_path / "ch_climax"
    ref_ep.mkdir()
    out_dir = tmp_path / "series_1"
    concept = {"style": "power_reveal", "hook": "HE SNAPS",
               "style_overlay": {"label_xy": [1, 2]},
               "refs": ["scenes/s1.jpg", "scenes/s2.jpg"]}
    calls: dict = {}
    _patch(monkeypatch, calls)

    rep = tb.render_thumbnail(concept, ref_episode_dir=str(ref_ep),
                              out_dir=str(out_dir), models=["m1"])

    assert calls["refs"] == ["scenes/s1.jpg", "scenes/s2.jpg"]  # from concept
    assert calls["episode_dir"] == str(ref_ep)                  # refs resolve here
    assert calls["aspect"] == "16:9"
    assert calls["overlay_hook"] == "HE SNAPS"
    # art + final jpg land in the SERIES dir, not the chapter dir
    assert Path(calls["out_path"]).parent == out_dir
    assert rep["thumbnail"] == str(out_dir / "thumbnail_yt.jpg")
    assert os.path.exists(rep["thumbnail"])
    # the art prompt must forbid text (copyright safety)
    assert "no text" in calls["prompt"].lower()


def test_render_thumbnail_explicit_refs_override_concept(tmp_path, monkeypatch):
    ref_ep = tmp_path / "ch"
    ref_ep.mkdir()
    concept = {"style": "vs_monster", "hook": "X", "refs": ["a.jpg"]}
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(concept, ref_episode_dir=str(ref_ep),
                        out_dir=str(tmp_path / "o"), models=["m1"],
                        refs=["override.jpg"])
    assert calls["refs"] == ["override.jpg"]


# ---- the art must be text-free, and the prompt alone does not make it so ----
# series_1's live triptych art (2026-09-17) carried ORDINARY / TRANSFORMATION /
# POWERFUL and "LEFT (ordinary)" etc. -- the model drew the layout description
# as captions despite "render NO text". Our overlay painted over half of it.
# Measured with Apple Vision over all 6 arts on the Mini: those 9 words at conf
# 1.0 and 4.4-6% of the image height; the only other hit was "BOX" at 1.8%, a
# corner glyph on a drawn UI panel.

def test_drawn_words_flags_readable_text_only():
    words = [
        {"t": "ORDINARY", "conf": 1.0, "bbox": [0.1, 0.02, 0.3, 0.0668]},
        {"t": "(ordinary)", "conf": 1.0, "bbox": [0.1, 0.9, 0.3, 0.9585]},
        {"t": "BOX", "conf": 1.0, "bbox": [0.9477, 0.2812, 0.9666, 0.2995]},
        {"t": "FLARE", "conf": 0.5, "bbox": [0.83, 0.70, 0.87, 0.75]},
        {"t": "18", "conf": 1.0, "bbox": [0.1, 0.1, 0.2, 0.2]},
    ]
    assert tb.drawn_words(words) == ["ORDINARY", "(ordinary)"]


def test_render_thumbnail_refuses_art_with_drawn_text(tmp_path, monkeypatch):
    import pytest
    ref_ep = tmp_path / "ch"; ref_ep.mkdir()
    calls: dict = {}
    _patch(monkeypatch, calls)
    monkeypatch.setattr(tb, "art_text_words", lambda path: ["ORDINARY", "LEFT"])
    with pytest.raises(RuntimeError, match="ORDINARY"):
        tb.render_thumbnail({"style": "power_reveal", "hook": "X",
                             "refs": ["a.jpg"]}, ref_episode_dir=str(ref_ep),
                            out_dir=str(tmp_path / "o"), models=["m1"])
    assert "overlay_hook" not in calls           # never overlaid, never shipped
    assert not (tmp_path / "o" / "thumbnail_yt.jpg").exists()


# --- a hook design reaches the art prompt and the overlay -------------------

def _design_concept(**extra):
    return {"style": "power_reveal", "hook": "OUTCAST", "refs": ["s1.jpg"],
            **extra}


def test_render_thumbnail_passes_the_quoted_card_to_the_overlay(tmp_path, monkeypatch):
    """The card is drawn by the overlay from concept.json; a build that drops
    it ships a system-window design with an empty left third."""
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(
        _design_concept(design="system_window",
                        card=["Main scenario has begun."]),
        ref_episode_dir=str(tmp_path), out_dir=str(tmp_path / "o"), models=["m"])
    assert calls["overlay_card"] == ["Main scenario has begun."]


def test_render_thumbnail_art_prompt_carries_the_designs_clause(tmp_path, monkeypatch):
    """The arrow lands on the lead BY CONSTRUCTION: the art must be told where
    the lead stands, or arrow_to aims at nothing."""
    from thumbnail_styles import HOOK_DESIGNS
    clause = HOOK_DESIGNS["system_window"]["art_clause"]
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(_design_concept(design="system_window"),
                        ref_episode_dir=str(tmp_path),
                        out_dir=str(tmp_path / "a"), models=["m"])
    assert clause in calls["prompt"]
    tb.render_thumbnail(_design_concept(), ref_episode_dir=str(tmp_path),
                        out_dir=str(tmp_path / "b"), models=["m"])
    assert clause not in calls["prompt"]


# --- the story's scene reaches the painter ----------------------------------
# Owner, 2026-09-22: "we dont see a story on the thumbnail". The understanding
# was good ("his commute turns into a death game; he alone has read the script")
# and never reached the image model, which got the style's fixed paragraph: a
# hero with an aura and recoiling onlookers, for every series.

_SCENE = ("A crowded subway car; a blank glowing window hangs in the air; one "
          "calm man reads his phone while passengers panic.")


def test_a_scene_replaces_the_generic_composition(tmp_path, monkeypatch):
    from thumbnail_styles import HOOK_DESIGNS, style_for
    generic = style_for("power_reveal")["art_prompt"]
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(_design_concept(design="nametag", scene=_SCENE),
                        ref_episode_dir=str(tmp_path),
                        out_dir=str(tmp_path / "a"), models=["m"])
    assert _SCENE in calls["prompt"]
    assert generic[:60] not in calls["prompt"]          # no lightning-man default
    assert HOOK_DESIGNS["nametag"]["art_clause"] in calls["prompt"]   # placement stays
    assert "blank" in calls["prompt"].lower()           # windows painted without letters


def test_without_a_scene_the_style_composition_is_still_used(tmp_path, monkeypatch):
    from thumbnail_styles import style_for
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(_design_concept(design="nametag"),
                        ref_episode_dir=str(tmp_path),
                        out_dir=str(tmp_path / "b"), models=["m"])
    assert style_for("power_reveal")["art_prompt"][:60] in calls["prompt"]


def test_the_exact_painter_prompt_is_saved_beside_the_art(tmp_path, monkeypatch):
    """Owner, 2026-09-22: "what is the ... prompt you plan? how do i see them
    properly?" The prompt was built and thrown away; nothing on disk said what
    the painter had been told."""
    calls: dict = {}
    _patch(monkeypatch, calls)
    out = tmp_path / "o"
    tb.render_thumbnail(_design_concept(design="nametag", scene=_SCENE),
                        ref_episode_dir=str(tmp_path), out_dir=str(out), models=["m"])
    saved = (out / "art_prompt.txt").read_text()
    assert saved == calls["prompt"] and _SCENE in saved


def test_a_scene_is_painted_face_forward(tmp_path, monkeypatch):
    """The lead's face is the thumbnail; the scene is context behind it."""
    calls: dict = {}
    _patch(monkeypatch, calls)
    tb.render_thumbnail(_design_concept(design="nametag", scene=_SCENE),
                        ref_episode_dir=str(tmp_path),
                        out_dir=str(tmp_path / "f"), models=["m"])
    low = calls["prompt"].lower()
    assert "face" in low and "half" in low and "close" in low
