"""studio/deps.py — THE dependency table.

One declarative map of every pipeline artifact: which stage produces it and
which upstream manifests it is derived from. Freshness (tools/
manifest_freshness.py), rewind delete-lists (studio/catalog/reset.py),
reconcile markers and pipeline skip-guards all DERIVE from this table instead
of hand-maintaining four drifting copies.

stdlib only; no imports from tools/. The stage order is the canonical
catalog STATUS_ORDER plus the worker-only "prepped" rung (render_prep runs
after planned but is a stage_run stage, not a chapter status — same ordering
manifest_freshness and reconcile._STAGE_ARTIFACT already use).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from studio.catalog.models import STATUS_ORDER

# "prepped" is not a chapter status; it is the render_prep stage_run rung that
# follows planned. Appended (not redeclared) so artifact ordering covers it.
ORDER: Tuple[str, ...] = tuple(STATUS_ORDER) + ("prepped",)


@dataclass(frozen=True)
class A:
    """One artifact: producing stage + the manifests it is derived from.

    inputs    — required freshness edges (output must not predate these).
    optional  — freshness edges checked only when the input file exists.
    sha_only  — edges enforced ONLY via _meta input-sha stamps (never mtime);
                exists for understood←vision, where the producer re-stamps the
                input after the fact (mtime is inverted by design).
    required  — participates in the freshness completeness check
                (missing_manifest/corrupt_manifest) for statuses at/after its
                stage. Matches the historical STATUS_REQUIRED contract.
    """
    stage: str
    inputs: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    sha_only: bool = False
    required: bool = False


ARTIFACTS: Dict[str, A] = {
    "manifest.stitch.json":            A(stage="stitched"),
    # detect writes BOTH the raw panel boxes and the gutter-expanded ones
    "manifest.panels.json":            A(stage="detected"),
    "manifest.panels.expanded.json":   A(stage="detected"),
    "manifest.scenes.json":            A(stage="scened"),
    "manifest.vision.json":            A(stage="visioned", required=True,
                                         inputs=("manifest.scenes.json",)),
    # panel_understand stamps panel_kind BACK onto vision after understanding,
    # so this edge is sha_only: enforced via _meta stamps, never mtime.
    "manifest.panels.understood.json": A(stage="grouped", required=True,
                                         inputs=("manifest.vision.json",),
                                         sha_only=True),
    "manifest.groups.json":            A(stage="grouped", required=True,
                                         inputs=("manifest.panels.understood.json",)),
    "manifest.story.json":             A(stage="grouped",
                                         inputs=("manifest.panels.understood.json",)),
    "manifest.cast.json":              A(stage="beated",
                                         inputs=("manifest.groups.json",
                                                 "manifest.vision.json")),
    "manifest.beats.json":             A(stage="beated", required=True,
                                         inputs=("manifest.groups.json",),
                                         optional=("manifest.cast.json",)),
    "manifest.script.json":            A(stage="scripted", required=True,
                                         inputs=("manifest.beats.json",)),
    "manifest.sanitize.json":          A(stage="scripted",
                                         inputs=("manifest.script.json",)),
    # transient-ish: absence is normal (estimate-phase artifact), so NOT
    # required — but when present it must not predate the script it paced.
    "render.plan.json":                A(stage="planned",
                                         inputs=("manifest.script.json",)),
    "tts/tts_index.json":              A(stage="voiced"),
    # NO tts_index←script edge by design: estimate-phase leftover clips are
    # expected; text_sha covers audio↔script drift content-precisely.
    "render.plan.clean.json":          A(stage="prepped", required=True,
                                         inputs=("manifest.script.json",
                                                 "manifest.beats.json"),
                                         optional=("tts/tts_index.json",)),
}


def stage_of(artifact: str) -> str:
    """Producing stage of *artifact* (KeyError on unknown names — loud)."""
    return ARTIFACTS[artifact].stage


def dag() -> Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], bool]]:
    """{output: (required_inputs, optional_inputs, sha_only)} for every
    artifact that has at least one declared input — the freshness DAG."""
    return {name: (a.inputs, a.optional, a.sha_only)
            for name, a in ARTIFACTS.items() if a.inputs or a.optional}


def artifacts_beyond(stage: str) -> Tuple[str, ...]:
    """Artifact paths produced by stages strictly AFTER *stage* in ORDER,
    in table order. ValueError on an unknown stage — loud, like ORDER.index."""
    i = ORDER.index(stage)
    return tuple(name for name, a in ARTIFACTS.items()
                 if ORDER.index(a.stage) > i)


def required_chain(status: str) -> Tuple[str, ...]:
    """Completeness-required artifacts for a chapter at *status*: every
    required=True artifact whose producing stage is at or before *status*."""
    i = ORDER.index(status)
    return tuple(name for name, a in ARTIFACTS.items()
                 if a.required and ORDER.index(a.stage) <= i)


def freshness_statuses() -> Tuple[str, ...]:
    """Statuses that carry a completeness contract: every stage (in ORDER)
    that produces at least one required artifact. Historically:
    visioned, grouped, beated, scripted, prepped ('planned' aliases prepped
    in manifest_freshness — its persistent sentinel is the clean plan)."""
    stages = {a.stage for a in ARTIFACTS.values() if a.required}
    return tuple(s for s in ORDER if s in stages)
