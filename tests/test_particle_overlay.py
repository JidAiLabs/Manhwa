"""
tests/test_particle_overlay.py

TDD for tools/particle_overlay.py — generates the channel's ambient particle
overlay (snow-dust with bokeh) as a SEAMLESSLY LOOPING video that Remotion
screen-blends over the story. Seamlessness = every motion parameter must be
exactly periodic in the loop duration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "particle_overlay",
    Path(__file__).resolve().parent.parent / "tools" / "particle_overlay.py",
)
po = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(po)  # type: ignore[union-attr]


def test_fall_speed_is_loop_periodic():
    # vy * T must be an integer multiple of the wrap height, else the loop pops
    T, wrap = 16.0, 1140.0
    for depth in (0.0, 0.3, 0.7, 1.0):
        vy = po.quantized_fall_speed(depth, loop_sec=T, wrap_h=wrap)
        cycles = vy * T / wrap
        assert abs(cycles - round(cycles)) < 1e-9
        assert vy > 0


def test_deeper_particles_fall_faster():
    T, wrap = 16.0, 1140.0
    near = po.quantized_fall_speed(1.0, loop_sec=T, wrap_h=wrap)
    far = po.quantized_fall_speed(0.0, loop_sec=T, wrap_h=wrap)
    assert near > far


def test_sprite_is_soft_normalized_disc():
    s = po.gaussian_sprite(sigma=6.0)
    assert s.max() <= 1.0 + 1e-6
    assert s.max() > 0.95                      # bright core
    c = s.shape[0] // 2
    assert s[c, c] >= s[c, 0] * 5              # soft falloff to the edge
    assert abs(float(s[c, 0]) - float(s[0, c])) < 1e-6   # radially symmetric


def test_render_loop_is_seamless():
    # frame at t=0 and frame at t=T must be identical
    f0 = po.render_frame(0.0, width=320, height=180, loop_sec=4.0, count=12, seed=7)
    fT = po.render_frame(4.0, width=320, height=180, loop_sec=4.0, count=12, seed=7)
    assert np.abs(f0.astype(int) - fT.astype(int)).max() <= 1
    assert f0.max() > 30                       # particles actually visible
