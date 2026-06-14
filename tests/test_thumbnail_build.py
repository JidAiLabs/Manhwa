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
    def fake_generate(episode_dir, *, hook_text, refs, models, location,
                      aspect, size, out_path, prompt_override=""):
        calls.update(episode_dir=episode_dir, refs=refs, out_path=out_path,
                     prompt=prompt_override, aspect=aspect)
        with open(out_path, "wb") as f:
            f.write(b"PNG")
        return models[0]

    def fake_overlay(art, out, *, hook, style_overlay, speech):
        calls.update(overlay_hook=hook, overlay_art=art)
        with open(out, "wb") as f:
            f.write(b"JPG")

    monkeypatch.setattr(tb.tg, "generate", fake_generate)
    monkeypatch.setattr(tb, "render_overlay", fake_overlay)


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
