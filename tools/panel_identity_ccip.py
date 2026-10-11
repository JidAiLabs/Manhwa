#!/usr/bin/env python3
"""tools/panel_identity_ccip.py — WHO is drawn in a panel, from the picture.

Text-vs-text identity (a panel's "dark-haired young man" against a cast
member's guessed look) could not separate look-alikes: when the identity gate
changed who a line was about it was right ~1 time in 4 (54 graded,
2026-10-09). This compares pictures to pictures, for every series, with no
owner curation (spike 2026-10-10, memory ccip-character-identity-spike):

  index     an anime head detector finds heads on every panel with people;
            CCIP turns each head into a 768-d character fingerprint. Cached per
            chapter in ongoing/<slug>/.identity/<chapter>.{json,npz}.
  profile   the head with the most look-alikes across the series IS the
            protagonist ("the most visible character"; true for ORV and
            Tutorial Tower). References = that seed + its closest heads from
            distinct chapters spanning the series — FIXED, never grown by
            chaining (chaining merged every dark-haired man). Written to
            .identity/profile.json; active only once the evidence is there.
  identify  a panel is the protagonist's iff exactly one of its heads is
            within CUT of a reference -> manifest.identity.json, the shape
            panel_identity.py writes, plus mc/heads/diff.

Measured on the spike data (tests/fixtures/ccip_spike_fixture.*), which is
why the constants are what they are:
  * CCIP's difference is exactly (1 - cos)/2 of the features (max error 0 vs
    ccip_batch_differences over 300 heads), so no metric model is needed.
  * CUT is FIXED at 0.15. By eye: ORV ~88% right at 0.20, Tutorial Tower ~75%
    at 0.20 and ~90%+ at 0.15. Nothing unsupervised separates those: an Otsu
    valley lands at 0.175/0.21 and reads the same for RANDOM references.
  * Leads of different series are look-alikes: ORV's and Tutorial Tower's
    protagonists are 0.165 apart, inside CCIP's own "same character" 0.178.
    So every "same character?" check here uses CUT, never 0.178: the owner's
    exemplars sit 0.091/0.102 from ORV's automatic references, TT's lead 0.164.
  * Two heads matching in one panel is mostly the protagonist next to a
    look-alike talking to him (24-46% of matched multi-head panels), so such a
    panel is never named — and that rate is not a usable calibration signal.

Runs in its own venv (.identity_venv, requirements-identity.lock.txt);
the model imports are lazy and heads_fn/embed_fn are seams, so the tests run
without onnxruntime.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cast_identity import subject_person_count  # noqa: E402
from manifest_io import input_sha, write_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

HEAD_MODEL = "head_detect_v2.0_s"
CCIP_MODEL = "ccip-caformer-24-randaug-pruned"
MARGIN = 0.25           # grow the head box so hair and outline stay in the crop
MODEL_TAG = f"{HEAD_MODEL}|{CCIP_MODEL}|m{MARGIN}"   # fingerprints compare only within one tag

CUT = 0.10              # CANDIDATE cut: below it a head may be the protagonist
                        # and gemma is asked (graded 2026-10-10: every wrong gemma
                        # confirmation on ORV had a score >= 0.10; TT < 0.10 ~88%)
SAME_CHAR = 0.13        # "same character?" (owner exemplars, refresh continuity):
                        # owner's ORV exemplars sit 0.091/0.102 from the auto refs,
                        # TT's lead 0.164 — never CCIP's own 0.178
SEED_RADIUS = 0.20      # look-alike radius for the density that picks the seed
REF_TIGHT = 0.12        # a reference must be this close to the seed
MAX_REFS = 8
SAMPLE_MAX = 3000       # heads in the pairwise density matrix (36 MB float32)
MIN_CHAPTERS = 5
MIN_MATCHED = 100       # matched heads over the series (at CUT; ~5-6 chapters)
MIN_SPREAD = 0.60       # share of indexed chapters with the protagonist in them
MIN_DOMINANCE = 1.5     # seed density vs the densest head of everyone else
REFRESH_EVERY = 10      # profile rebuild cadence once 10 chapters are indexed

_SKIP_KINDS = frozenset({"system", "caption", "empty"})


def ccip_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CCIP's character difference, |a| x |b|: (1 - cos) / 2."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (1.0 - a @ b.T) / 2.0


def chapter_order(name: str) -> Tuple:
    return tuple((0, float(t)) if re.fullmatch(r"\d+(?:\.\d+)?", t) else (1, t)
                 for t in re.findall(r"\d+(?:\.\d+)?|[^\d]+", name))


def people_panels(panels: Sequence[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for p in panels:
        fn = os.path.basename(str(p.get("scene_file") or ""))
        if (fn and (p.get("subjects") or [])
                and str(p.get("panel_kind") or "").lower() not in _SKIP_KINDS):
            out.append((fn, p))
    return out


def _unit(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True) if f.size else f


# ---------------------------------------------------------------- models
def _crop(im, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return im.crop((max(0, int(x0 - MARGIN * w)), max(0, int(y0 - MARGIN * h)),
                    min(im.width, int(x1 + MARGIN * w)), min(im.height, int(y1 + MARGIN * h))))


def _open(path: str):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as im:
        return im.convert("RGB")


def _real_heads(path: str) -> List[Tuple[Tuple[int, int, int, int], float]]:
    from imgutils.detect import detect_heads
    return [(tuple(int(v) for v in box), float(score))
            for box, _label, score in detect_heads(_open(path), model_name=HEAD_MODEL)]


def _real_embed(items: Sequence[Tuple[str, Sequence[int]]]) -> np.ndarray:
    from imgutils.metrics import ccip_batch_extract_features
    crops, cur, im = [], None, None
    for path, box in items:
        if path != cur:
            cur, im = path, _open(path)
        crops.append(_crop(im, box))
    return np.asarray(ccip_batch_extract_features(crops, model=CCIP_MODEL), dtype=np.float32)


# ---------------------------------------------------------------- index
def _cache_dir(series_dir) -> Path:
    return Path(series_dir) / ".identity"


def _read_index(chapter: str, cache: Path) -> Optional[Dict[str, Any]]:
    try:
        meta = json.loads((cache / f"{chapter}.json").read_text())
        with np.load(cache / f"{chapter}.npz") as z:
            feats = z["feats"].astype(np.float32)
    except (OSError, ValueError, KeyError):
        return None
    if meta.get("model") != MODEL_TAG or len(feats) != len(meta.get("heads") or []):
        return None
    return {"chapter": chapter, "key": meta.get("key"), "heads": meta["heads"], "feats": feats}


def index_chapter(ep_dir, *, heads_fn: Optional[Callable] = None,
                  embed_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Heads + fingerprints of every panel with people. Cached; the key is the
    model tag + each people-panel's scene bytes, so a re-detected chapter (new
    crops) re-indexes and a rewritten understanding of the same panels does not."""
    ep = Path(ep_dir)
    cache = _cache_dir(ep.parent)
    understood = json.loads((ep / "manifest.panels.understood.json").read_text())
    paths = [(fn, ep / "scenes" / fn) for fn, _ in people_panels(understood.get("panels") or [])]
    paths = [(fn, p) for fn, p in paths if p.exists()]
    h = hashlib.sha1(MODEL_TAG.encode())
    for fn, p in paths:
        h.update(fn.encode())
        h.update(input_sha(p).encode())
    key = h.hexdigest()
    cached = _read_index(ep.name, cache)
    if cached and cached["key"] == key:
        return cached
    heads_fn = heads_fn or _real_heads
    embed_fn = embed_fn or _real_embed
    heads, items = [], []
    for fn, p in paths:
        for box, score in heads_fn(str(p)):
            heads.append({"panel": fn, "box": [int(v) for v in box], "score": round(float(score), 3)})
            items.append((str(p), tuple(int(v) for v in box)))
    chunks = [np.asarray(embed_fn(items[i:i + 32]), dtype=np.float32)
              for i in range(0, len(items), 32)]
    feats = _unit(np.concatenate(chunks)) if chunks else np.zeros((0, 0), np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"{ep.name}.npz.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, feats=feats.astype(np.float16))
    os.replace(tmp, cache / f"{ep.name}.npz")
    write_manifest(cache / f"{ep.name}.json",            # json last = commit marker
                   {"key": key, "model": MODEL_TAG, "heads": heads}, tool="panel_identity_ccip")
    return {"chapter": ep.name, "key": key, "heads": heads, "feats": feats}


def load_indexes(series_dir) -> List[Dict[str, Any]]:
    cache = _cache_dir(series_dir)
    names = sorted((p.stem for p in cache.glob("*.json") if p.stem != "profile"),
                   key=chapter_order)
    return [ix for ix in (_read_index(n, cache) for n in names) if ix]


# ---------------------------------------------------------------- profile
def _sample(indexes: List[Dict[str, Any]], rng) -> List[Tuple[int, int]]:
    pool = [(c, h) for c, ix in enumerate(indexes) for h in range(len(ix["heads"]))]
    if len(pool) <= SAMPLE_MAX:
        return pool
    quota = max(1, SAMPLE_MAX // len(indexes))           # stratified by chapter
    out = []
    for c, ix in enumerate(indexes):
        n = len(ix["heads"])
        out += [(c, int(h)) for h in sorted(rng.choice(n, min(n, quota), replace=False))]
    return out


def _ref(ix: Dict[str, Any], h: int) -> Dict[str, Any]:
    head = ix["heads"][h]
    return {"chapter": ix["chapter"], "panel": head["panel"], "box": head["box"],
            "key": ix["key"], "feat": [round(float(v), 6) for v in _unit(ix["feats"][h:h + 1])[0]]}


def _feats(refs: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([r["feat"] for r in refs], dtype=np.float32)


def live_refs(profile: Dict[str, Any], indexes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """References whose source chapter was not re-indexed since (a reset
    re-crops the panels: the ref would point at pixels that no longer exist)."""
    keys = {ix["chapter"]: ix["key"] for ix in indexes}
    return [r for r in profile.get("refs") or []
            if r.get("pin") or keys.get(r["chapter"]) == r.get("key")]


def exemplar_feats(indexes: Sequence[Dict[str, Any]], exemplars: Dict[str, Any],
                   key: str = "protagonist") -> Optional[np.ndarray]:
    """Fingerprints of an exemplars file's faces, read from the index cache (no
    detection): by the recorded head boxes when present, else the panel's
    single head. None when none is found."""
    multi = {}
    for ix in indexes:
        for i, h in enumerate(ix["heads"]):
            multi.setdefault((ix["chapter"], h["panel"]), []).append((ix, i))
    out = []
    heads = (exemplars.get("heads") or {}).get(key)
    if heads:
        for h in heads:
            for ix, i in multi.get((h["chapter"], h["panel"]), []):
                if list(ix["heads"][i]["box"]) == list(h["box"]):
                    out.append(ix["feats"][i])
    else:
        for path in exemplars.get(key) or []:
            parts = Path(str(path)).parts
            hit = multi.get((parts[-3], parts[-1]), []) if len(parts) >= 3 else []
            if len(hit) == 1:
                out.append(hit[0][0]["feats"][hit[0][1]])
    return _unit(np.asarray(out, dtype=np.float32)) if out else None


def build_profile(indexes: Sequence[Dict[str, Any]], *, pins: Optional[np.ndarray] = None,
                  prior: Optional[Dict[str, Any]] = None, rng_seed: int = 0,
                  seed_feats: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """The series' protagonist references. Default: the most-drawn face (seed)
    + its closest heads from distinct chapters. With *seed_feats* (the faces of
    an exemplars file marked seed: exemplars — picked by eye or by
    identity_exemplars --auto) the references grow from THOSE faces instead and
    the most-drawn check is skipped: the most-drawn face was the lead on only 6
    of 9 series."""
    every = sorted(indexes, key=lambda ix: chapter_order(ix["chapter"]))
    ixs = [ix for ix in every if len(ix["heads"])]
    reasons: List[str] = []
    if not ixs:
        return {"status": "provisional", "reasons": ["no heads indexed"], "alarms": [],
                "version": (prior or {}).get("version", 0), "model": MODEL_TAG, "cut": CUT,
                "chapters_indexed": len(every), "seed": None, "refs": [], "stats": {}}
    samp = _sample(ixs, np.random.default_rng(rng_seed))
    S = np.stack([ixs[c]["feats"][h] for c, h in samp])
    D = ccip_diff(S, S)
    np.fill_diagonal(D, np.inf)
    dens = (D < SEED_RADIUS).sum(axis=1)
    seeded = seed_feats is not None and len(seed_feats) > 0
    if seeded:
        SF = _unit(seed_feats)
        dist = ccip_diff(S, SF).min(axis=1)       # every sampled head -> the picked lead
        s = int(dist.argmin())
        pool = np.argsort(dist)
        cut_at, keep = REF_TIGHT, MAX_REFS - len(SF)
    else:
        s = int(dens.argmax())
        dist, pool = D[s], np.argsort(D[s])
        cut_at, keep = REF_TIGHT, MAX_REFS - 1
    best: Dict[int, int] = {}                     # chapter -> its closest head to the seed
    for j in pool:
        if dist[j] >= cut_at:
            break
        c = samp[j][0]
        if (seeded or c != samp[s][0]) and c not in best and (seeded or j != s):
            best[c] = int(j)
    chosen = sorted(best)                         # chapter order: spread over the series
    if len(chosen) > keep:
        at = np.linspace(0, len(chosen) - 1, max(1, keep)).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(at))]
    if seeded:
        refs = ([{"chapter": "exemplar", "pin": True, "seed": True,
                  "feat": [round(float(v), 6) for v in f]} for f in SF]
                + [_ref(ixs[samp[best[c]][0]], samp[best[c]][1]) for c in chosen])
    else:
        refs = [_ref(ixs[samp[j][0]], samp[j][1]) for j in [s] + [best[c] for c in chosen]]
    seed = dict(refs[0])
    if pins is not None and len(pins) and not seeded:
        if ccip_diff(pins, _feats(refs)).min(axis=1).max() >= SAME_CHAR:
            reasons.append("an owner exemplar matches none of the automatic references")
        refs += [{"chapter": "registry", "pin": True,
                  "feat": [round(float(v), 6) for v in f]} for f in _unit(pins)]
    R = _feats(refs)
    matched, chapters_with = 0, 0
    for ix in ixs:
        m = int((ccip_diff(ix["feats"], R).min(axis=1) < CUT).sum())
        matched += m
        chapters_with += m > 0
    spread = chapters_with / len(every)
    rest = np.where(ccip_diff(S, R).min(axis=1) >= SEED_RADIUS)[0]
    second = int((D[np.ix_(rest, rest)] < SEED_RADIUS).sum(axis=1).max()) if len(rest) else 0
    dominance = float(dens[s]) / max(1, second)
    if len(every) < MIN_CHAPTERS:
        reasons.append(f"chapters {len(every)} < {MIN_CHAPTERS}")
    if matched < MIN_MATCHED:
        reasons.append(f"matched heads {matched} < {MIN_MATCHED}")
    if spread < MIN_SPREAD:
        reasons.append(f"spread {spread:.2f} < {MIN_SPREAD}")
    if dominance < MIN_DOMINANCE and not seeded:     # a picked lead needs no majority
        reasons.append(f"dominance {dominance:.2f} < {MIN_DOMINANCE}")
    prof = {"status": "provisional" if reasons else "active", "reasons": reasons, "alarms": [],
            "version": 1, "model": MODEL_TAG, "cut": CUT, "chapters_indexed": len(every),
            "seed": seed, "refs": refs, "seeded": seeded,
            "stats": {"matched": matched, "spread": round(spread, 3),
                      "dominance": round(dominance, 2), "density": int(dens[s]),
                      "heads": int(sum(len(ix["heads"]) for ix in ixs)), "sampled": len(samp)}}
    if not prior or prior.get("model") != MODEL_TAG:
        return prof
    prof["version"] = int(prior.get("version", 0)) + 1
    if prior.get("status") != "active" or seeded:   # a picked lead overrides continuity
        return prof
    # continuity: an active profile is only replaced by the same character,
    # judged at SAME_CHAR (a look-alike lead of the same archetype is within 0.178)
    old = live_refs(prior, every)
    if not old:
        prof["alarms"].append("refs_reset: every old reference was re-indexed")
        return prof
    gap = float(ccip_diff(_feats([seed]), _feats(old)).min())
    why = (f"profile_flip: new seed {seed['chapter']}/{seed['panel']} is {gap:.3f} from the old refs"
           if gap >= SAME_CHAR else
           f"rebuild_provisional: {'; '.join(reasons)}" if reasons else "")
    if why:
        kept = dict(prior)
        kept.pop("_meta", None)
        kept["alarms"] = [why]
        kept["chapters_indexed"] = len(every)
        return kept
    return prof


def profile_due(profile: Optional[Dict[str, Any]], n_indexed: int) -> bool:
    if not profile:
        return True
    last = int(profile.get("chapters_indexed") or 0)
    if n_indexed < REFRESH_EVERY or profile.get("status") != "active":
        return n_indexed != last
    return n_indexed - last >= REFRESH_EVERY


def save_profile(series_dir, profile: Dict[str, Any]) -> None:
    write_manifest(_cache_dir(series_dir) / "profile.json", profile, tool="panel_identity_ccip")


def load_profile(series_dir) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((_cache_dir(series_dir) / "profile.json").read_text())
    except (OSError, ValueError):
        return None


@contextmanager
def profile_lock(series_dir):
    """The sweep and the worker may both refresh one series' profile."""
    cache = _cache_dir(series_dir)
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "profile.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def pin_features(series_cast, *, heads_fn: Optional[Callable] = None,
                 embed_fn: Optional[Callable] = None) -> Optional[np.ndarray]:
    """The owner's exemplar panels for the protagonist (cast/<slug>.json), one
    head each; a panel with several heads is skipped (whose head is unknown)."""
    try:
        reg = json.loads(Path(series_cast).read_text())
    except (OSError, ValueError, TypeError):
        return None
    members = reg.get("cast") if isinstance(reg, dict) else reg
    ex = next((m.get("exemplars") or [] for m in members or []
               if isinstance(m, dict) and m.get("is_protagonist")), [])
    heads_fn = heads_fn or _real_heads
    embed_fn = embed_fn or _real_embed
    items = []
    for p in ex:
        path = str(p if os.path.isabs(p) else REPO_ROOT / p)
        if not os.path.exists(path):
            continue
        heads = heads_fn(path)
        if len(heads) == 1:
            items.append((path, tuple(heads[0][0])))
        else:
            print(f"[identity] exemplar {p}: {len(heads)} heads, skipped as a pin")
    return _unit(np.asarray(embed_fn(items), dtype=np.float32)) if items else None


# ---------------------------------------------------------------- identify
def _protagonist(cast_path: Path) -> Optional[str]:
    try:
        cast = json.loads(cast_path.read_text())
    except (OSError, ValueError):
        return None
    members = cast.get("cast") if isinstance(cast, dict) else cast
    for m in members or []:
        if isinstance(m, dict) and m.get("is_protagonist") and m.get("canonical_name"):
            return str(m["canonical_name"])
    return None


def identify_chapter(ep_dir, profile: Optional[Dict[str, Any]],
                     index: Dict[str, Any], write: bool = True) -> Optional[Dict[str, Any]]:
    """manifest.identity.json for one chapter, or None (nothing written) while
    the series profile is provisional: the keyword identity then stands.
    write=False computes it in memory (the sweep's census).

    The picture only PROPOSES (owner decision 2026-10-10): a panel is a
    candidate (`cand`) when exactly one head scores under CUT. Nobody is named
    here — `names` stays [] and `mc` false until panel_identity.py --verify-ccip
    has gemma confirm the candidate against the series' exemplar panels. Alone
    the picture was ~70-85% right on ORV; agreeing with gemma, ~95%."""
    if not profile or profile.get("status") != "active":
        return None
    ep = Path(ep_dir)
    understood = ep / "manifest.panels.understood.json"
    cast_path = ep / "manifest.cast.json"
    panels = people_panels(json.loads(understood.read_text()).get("panels") or [])
    mc_name = _protagonist(cast_path)
    cut = CUT                              # a profile's stored cut is history
    rows = defaultdict(list)
    for i, h in enumerate(index["heads"]):
        rows[h["panel"]].append(i)
    d_all = (ccip_diff(index["feats"], _feats(profile["refs"])).min(axis=1)
             if len(index["heads"]) else np.zeros(0))
    out: Dict[str, Any] = {}
    for fn, p in panels:
        r = rows.get(fn)
        if not r:
            continue                       # no head found: all-unknown, as before
        d = d_all[r]
        cand = int((d < cut).sum()) == 1   # two matches = a look-alike beside him
        persons = sum(subject_person_count(str(s)) for s in p.get("subjects") or [])
        out[fn] = {"names": [], "others": len(r) + max(0, persons - len(r)),
                   "mc": False, "cand": cand, "heads": len(r),
                   "diff": round(float(d.min()), 3)}
    obj: Dict[str, Any] = {"panels": out}
    if not write:
        return obj
    write_manifest(ep / "manifest.identity.json", obj, inputs=[understood, cast_path],
                   tool="panel_identity_ccip",
                   extra_meta={"backend": "ccip", "profile_version": profile.get("version"),
                               "cut": cut, "protagonist": mc_name, "verified_by": None,
                               "refs": [f"{r['chapter']}/{r.get('panel', '')}"
                                        for r in profile["refs"]]})
    return obj


def _seed_exemplars(path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """An exemplars file marked seed: exemplars, and a key of its lead's faces
    (a new pick rebuilds the profile); (None, None) otherwise."""
    try:
        ex = json.loads(Path(path).read_text()) if path else None
    except (OSError, ValueError):
        ex = None
    if not isinstance(ex, dict) or ex.get("seed") != "exemplars":
        return None, None
    faces = (ex.get("heads") or {}).get("protagonist") or ex.get("protagonist")
    return ex, hashlib.sha1(json.dumps(faces, sort_keys=True).encode()).hexdigest()[:12]


def run(ep_dir, series_cast=None, *, exemplars=None, heads_fn=None,
        embed_fn=None, identify: bool = True) -> Optional[Dict[str, Any]]:
    """index this chapter -> refresh the series profile when due -> identify.
    *exemplars* marked seed: exemplars (confirmed on the Series page) grow the
    profile from the picked lead instead of the most-drawn face. identify=False
    stops after the profile: a chapter READ while the series waits for its
    protagonist (worker _read_first) names nobody yet."""
    ep = Path(ep_dir)
    index = index_chapter(ep, heads_fn=heads_fn, embed_fn=embed_fn)
    ex, ex_key = _seed_exemplars(exemplars)
    with profile_lock(ep.parent):
        prof = load_profile(ep.parent)
        indexes = load_indexes(ep.parent)
        if profile_due(prof, len(indexes)) or (prof or {}).get("exemplars_key") != ex_key:
            if ex is not None:
                sf = exemplar_feats(indexes, ex)
                prof = (build_profile(indexes, prior=prof, seed_feats=sf) if sf is not None
                        else {**build_profile([], prior=prof), "chapters_indexed": len(indexes),
                              "reasons": ["the exemplar faces are not in the index"]})
            else:
                pins = (pin_features(series_cast, heads_fn=heads_fn, embed_fn=embed_fn)
                        if series_cast else None)
                prof = build_profile(indexes, pins=pins, prior=prof)
            prof["exemplars_key"] = ex_key
            save_profile(ep.parent, prof)
            print(f"[identity] profile v{prof['version']} {prof['status']} "
                  f"{prof.get('reasons') or ''} {prof.get('alarms') or ''} {prof.get('stats')}")
    if not identify:
        return None
    got = identify_chapter(ep, prof, index)
    if got is None and (ep / "manifest.identity.json").exists():
        # provisional = the keyword identity stands; a leftover file from an
        # earlier profile or another backend would keep naming from it
        (ep / "manifest.identity.json").unlink()
        print("[identity] removed a leftover manifest.identity.json (profile provisional)")
    return got


# ---------------------------------------------------------------- grading
def contact_sheet(series_dir, out_path, n: int = 40, seed: int = 0) -> int:
    """n random heads the profile names, from ONE-head panels (the panels the
    identity gate acts on), labelled chapter/panel/diff — for grading by eye."""
    from PIL import Image, ImageDraw
    series = Path(series_dir)
    prof = load_profile(series)
    if not prof or not prof.get("refs"):
        return 0
    R = _feats(prof["refs"])
    picks = []
    for ix in load_indexes(series):
        count = defaultdict(int)
        for h in ix["heads"]:
            count[h["panel"]] += 1
        d = ccip_diff(ix["feats"], R).min(axis=1)
        picks += [(ix["chapter"], h, float(d[i])) for i, h in enumerate(ix["heads"])
                  if d[i] < prof["cut"] and count[h["panel"]] == 1]
    rng = np.random.default_rng(seed)
    picks = [picks[i] for i in sorted(rng.choice(len(picks), min(n, len(picks)), replace=False))]
    size, per_row = 150, 10
    rows = (len(picks) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (size + 6) + 6, 30 + rows * (size + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), f"{series.name}: {len(picks)} random heads named the protagonist "
              f"(one-head panels, cut {prof['cut']}) - grade each", fill="blue")
    for k, (ch, h, d) in enumerate(picks):
        x, y = 5 + (k % per_row) * (size + 6), 30 + (k // per_row) * (size + 22)
        c = _crop(_open(str(series / ch / "scenes" / h["panel"])), h["box"])
        c.thumbnail((size, size))
        sheet.paste(c, (x, y))
        draw.text((x, y + size + 3), f"{ch[-8:]} {h['panel'][:7]} {d:.2f}", fill="black")
    sheet.save(out_path, quality=85)
    return len(picks)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--series-cast", default="",
                    help="cast/<slug>.json; protagonist exemplars become pinned references")
    ap.add_argument("--exemplars", default="",
                    help="exemplars file; seed: exemplars grows the profile from its lead")
    ap.add_argument("--sheet", default="", help="also write a grading contact sheet here")
    ap.add_argument("--index-only", action="store_true",
                    help="only record the chapter's faces (for the automatic protagonist pick)")
    args = ap.parse_args()
    if args.index_only and args.exemplars:
        run(args.episode_dir, exemplars=args.exemplars, identify=False)
        print("[identity] faces recorded; the locked protagonist's profile refreshed")
        return 0
    if args.index_only:
        ix = index_chapter(args.episode_dir)
        print(f"[identity] {len(ix['heads'])} faces recorded for the automatic protagonist pick")
        return 0
    got = run(args.episode_dir, args.series_cast or None, exemplars=args.exemplars or None)
    if got is None:
        print("[identity] ccip profile provisional -> no manifest.identity.json "
              "(keyword identity stands)")
    else:
        P = got["panels"]
        print(f"[ok] identity (ccip): {len(P)} panels with heads, "
              f"{sum(1 for v in P.values() if v.get('cand'))} picture candidates "
              "(named only after gemma confirms them)")
    if args.sheet:
        print(f"[identity] sheet: {contact_sheet(Path(args.episode_dir).parent, args.sheet)} heads "
              f"-> {args.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
