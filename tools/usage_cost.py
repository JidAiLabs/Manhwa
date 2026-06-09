"""
tools/usage_cost.py

Exact-token + estimated-dollar accounting for the paid LLM stages (Gemini beats,
OpenAI script). Token counts are pulled from the provider response and are
EXACT; the dollar figure is tokens x a rate table below, which are ESTIMATES the
user should verify against current provider pricing.

Usage:
    acc = UsageAccumulator("gemini-2.5-flash")
    acc.add(input_tokens=..., output_tokens=...)   # once per API call
    print(acc.summary())                           # at end of the run

Gemini's prompt_token_count already INCLUDES image tokens, so input_tokens x the
input rate captures multimodal cost correctly — no separate image pricing needed.
"""

from __future__ import annotations

# USD per 1,000,000 tokens. ESTIMATES as of 2026-06 — update to match current
# provider pricing. Keys are model-family prefixes (matched by startswith).
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4-nano": {"input": 0.20, "output": 0.80},
    "gpt-5.4-mini": {"input": 0.75, "output": 3.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def _match_pricing(model: str) -> dict[str, float] | None:
    """Find the rate entry whose family prefix matches *model* (longest first)."""
    if not model:
        return None
    for prefix in sorted(PRICING, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICING[prefix]
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD estimate for the given token counts. 0.0 when pricing is unknown."""
    rates = _match_pricing(model)
    if rates is None:
        return 0.0
    return (
        input_tokens / 1_000_000.0 * rates["input"]
        + output_tokens / 1_000_000.0 * rates["output"]
    )


def format_cost(usd: float) -> str:
    return f"${usd:.4f}"


class UsageAccumulator:
    """Accumulates token usage across many API calls for one run/model."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, *, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def cost(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    def has_pricing(self) -> bool:
        return _match_pricing(self.model) is not None

    def summary(self) -> str:
        tot = self.input_tokens + self.output_tokens
        if self.has_pricing():
            cost_str = format_cost(self.cost())
        else:
            cost_str = "unknown (no rate for model)"
        return (
            f"[cost] model={self.model} calls={self.calls} "
            f"tokens: in={self.input_tokens} out={self.output_tokens} total={tot} "
            f"est_cost={cost_str}"
        )
