"""
tests/test_usage_cost.py

TDD for tools/usage_cost.py — exact-token + estimated-dollar accounting for the
paid LLM stages. Token counts come straight from the provider response
(usage_metadata / usage), so they are exact; the dollar figure is tokens x a
configurable rate table (rates are estimates the user can update).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "usage_cost",
    Path(__file__).resolve().parent.parent / "tools" / "usage_cost.py",
)
uc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(uc)  # type: ignore[union-attr]


def test_estimate_cost_known_model():
    # gemini-2.5-flash: input 0.30/1M, output 2.50/1M
    cost = uc.estimate_cost("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.30 + 2.50)


def test_estimate_cost_prefix_match():
    # versioned model ids should match their family
    a = uc.estimate_cost("gemini-2.5-flash-001", 1_000_000, 0)
    b = uc.estimate_cost("gemini-2.5-flash", 1_000_000, 0)
    assert a == b == pytest.approx(0.30)


def test_estimate_cost_unknown_model_is_zero():
    assert uc.estimate_cost("totally-unknown-model", 1000, 1000) == 0.0


def test_accumulator_sums_tokens_and_cost():
    acc = uc.UsageAccumulator("gemini-2.5-flash")
    acc.add(input_tokens=100, output_tokens=50)
    acc.add(input_tokens=200, output_tokens=25)
    assert acc.calls == 2
    assert acc.input_tokens == 300
    assert acc.output_tokens == 75
    assert acc.cost() == pytest.approx(uc.estimate_cost("gemini-2.5-flash", 300, 75))


def test_accumulator_summary_mentions_tokens_and_model():
    acc = uc.UsageAccumulator("gpt-4.1-mini")
    acc.add(input_tokens=1234, output_tokens=567)
    s = acc.summary()
    assert "1234" in s and "567" in s
    assert "gpt-4.1-mini" in s
    assert "$" in s


def test_accumulator_unknown_pricing_flagged():
    acc = uc.UsageAccumulator("mystery-model")
    acc.add(input_tokens=10, output_tokens=10)
    s = acc.summary()
    # cost is zero/unknown but tokens still reported
    assert "10" in s
    assert "unknown" in s.lower() or "n/a" in s.lower()
