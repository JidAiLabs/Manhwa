#!/usr/bin/env python3
"""
local_tts_from_manifest.py

Free, LOCAL text-to-speech for the `voiced` stage — a drop-in alternative to
elevenlabs_tts_from_manifest.py that costs nothing per chapter. Produces the
SAME output contract so the timeline stays aligned:

  - one clip per paragraph named ``{segment_id}.wav`` under ``<out-dir>/clips/``
  - ``<out-dir>/tts_index.json`` with clips[] (segment_id, group_id, ...,
    audio_file, duration_sec) — exactly what timeline_planner consumes.

Backends (selected by --backend):
  - chatterbox : Resemble AI Chatterbox (MIT). Expressive; maps the script's
    mood tags -> emotion exaggeration. Optional voice cloning via --voice-ref.
  - kokoro     : Kokoro-82M (Apache). Fast, CPU-friendly, flatter delivery.

The heavy model deps are imported LAZILY (only when actually synthesizing), so
this module imports — and its pure logic tests — run without them installed.

Install (one-time, into the project venv):
  chatterbox:  pip install chatterbox-tts torchaudio
  kokoro:      pip install kokoro soundfile   (+ `espeak-ng` system package)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pure helpers (mood tags, item extraction, duration) — all unit-tested
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\[([^\]]+)\]")

# Mood/intensity -> Chatterbox `exaggeration` (0..1, ~0.5 neutral). The script's
# tts_paragraphs_v3 lead with an ElevenLabs-v3 style tag (e.g. "[tense]"); we map
# its sentiment to expressiveness so local TTS still tracks scene emotion.
_EMOTION_BY_KEYWORD: List[Tuple[str, float]] = [
    ("whisper", 0.25), ("calm", 0.30), ("somber", 0.30), ("sad", 0.35),
    ("serious", 0.45), ("neutral", 0.45), ("curious", 0.50),
    ("tense", 0.62), ("nervous", 0.62), ("excited", 0.70), ("intense", 0.72),
    ("dramatic", 0.78), ("angry", 0.85), ("shout", 0.90), ("explosive", 0.92),
    ("scream", 0.95),
]
_DEFAULT_EXAGGERATION = 0.5


def leading_tag(text: str) -> Optional[str]:
    """Return the lowercased first ``[tag]`` of *text*, or None."""
    m = _TAG_RE.match(text.strip())
    return m.group(1).strip().lower() if m else None


def strip_bracket_tags(text: str) -> str:
    """Remove all ``[tag]`` markers and collapse whitespace (what TTS speaks)."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def mood_to_exaggeration(tag: Optional[str]) -> float:
    """Map a mood tag (or intensity word) to a Chatterbox exaggeration value."""
    if not tag:
        return _DEFAULT_EXAGGERATION
    t = tag.lower()
    for kw, val in _EMOTION_BY_KEYWORD:
        if kw in t:
            return val
    return _DEFAULT_EXAGGERATION


def wav_duration_sec(path: str) -> float:
    """Duration in seconds of a WAV file.

    Tries the stdlib ``wave`` (PCM, zero extra deps); falls back to ``soundfile``
    for float/IEEE WAVs that ``wave`` can't parse (Chatterbox/torchaudio output).
    """
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate() or 1
            return w.getnframes() / float(rate)
    except Exception:
        pass
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return 0.0


def extract_items_from_manifest(script_obj: Dict[str, Any], text_source: str = "tts_v3") -> List[Dict[str, Any]]:
    """Stable per-paragraph items keyed by canonical segment_id (g####_p##).

    Mirrors elevenlabs_tts_from_manifest.extract_items_from_manifest so the two
    backends are interchangeable.
    """
    out: List[Dict[str, Any]] = []
    for sec in script_obj.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        sec_idx = int(sec.get("section_index") or 0)
        shots = sec.get("shots") or []
        if not isinstance(shots, list):
            continue
        if text_source == "tts_v3":
            paras = sec.get("tts_paragraphs_v3") or []
        elif text_source == "script":
            paras = sec.get("script_paragraphs") or []
        else:
            paras = sec.get("tts_paragraphs_ssml") or []
        if not isinstance(paras, list):
            paras = []

        n = min(len(shots), len(paras))
        for i in range(n):
            shot = shots[i] if isinstance(shots[i], dict) else {}
            gid = int(shot.get("group_id") or 0)
            if gid <= 0:
                continue
            beat_id = int(shot.get("beat_id") or gid)
            segment_id = f"g{gid:04d}_p{i:02d}"
            p = paras[i]
            text = str(p.get("text") if isinstance(p, dict) else p or "").strip()
            if not text:
                continue
            out.append({
                "segment_id": segment_id,
                "group_id": gid,
                "section_index": sec_idx,
                "beat_id": beat_id,
                "paragraph_index": i,
                "text": text,
            })
    out.sort(key=lambda x: (x["group_id"], x["section_index"], x["paragraph_index"]))
    return out


# ---------------------------------------------------------------------------
# Orchestrator (synth + duration injected -> fully testable without a model)
# ---------------------------------------------------------------------------

# SynthFn(text, out_path, exaggeration) -> None  (writes a wav to out_path)
SynthFn = Callable[[str, str, float], None]
DurationFn = Callable[[str], float]


def synthesize_manifest(
    script_obj: Dict[str, Any],
    out_dir: str,
    *,
    backend: str,
    synth_fn: SynthFn,
    duration_fn: DurationFn = wav_duration_sec,
    text_source: str = "tts_v3",
    overwrite: bool = False,
    voice_ref: str = "",
) -> Dict[str, Any]:
    """Drive TTS over every paragraph and build the tts_index.json dict.

    *synth_fn* does the actual audio synthesis (injected so tests need no model);
    it receives the tag-stripped text, the target wav path, and the per-paragraph
    exaggeration derived from the mood tag.
    """
    items = extract_items_from_manifest(script_obj, text_source)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    index: Dict[str, Any] = {
        "source_script": os.path.abspath(script_obj.get("_path", "")) if script_obj.get("_path") else "",
        "backend": backend,
        "voice_ref": voice_ref,
        "text_source": text_source,
        "clips": [],
        "total_duration_sec": 0.0,
    }
    total = 0.0
    for it in items:
        seg_id = it["segment_id"]
        source_text = str(it["text"])
        tag = leading_tag(source_text)
        sent_text = strip_bracket_tags(source_text)
        if not sent_text:
            continue
        exaggeration = mood_to_exaggeration(tag)
        audio_path = os.path.join(clips_dir, f"{seg_id}.wav")

        cached = os.path.exists(audio_path) and not overwrite
        if not cached:
            synth_fn(sent_text, audio_path, exaggeration)
        dur = duration_fn(audio_path)

        index["clips"].append({
            "segment_id": seg_id,
            "group_id": int(it["group_id"]),
            "section_index": int(it["section_index"]),
            "beat_id": int(it["beat_id"]),
            "paragraph_index": int(it["paragraph_index"]),
            "source_text": source_text,
            "sent_text": sent_text,
            "mood_tag": tag or "",
            "exaggeration": exaggeration,
            "audio_file": os.path.relpath(audio_path, out_dir),
            "duration_sec": round(dur, 4),
            "cached": cached,
        })
        total += dur
        print(f"[{'cache' if cached else 'ok'}] {seg_id} dur={dur:.2f}s mood={tag or '-'}")

    index["total_duration_sec"] = round(total, 4)
    return index


# ---------------------------------------------------------------------------
# Real backends (lazy-loaded)
# ---------------------------------------------------------------------------

def _make_chatterbox_synth(voice_ref: str) -> SynthFn:
    import torch
    import torchaudio as ta

    # Chatterbox's real Perth watermarker needs pkg_resources (setuptools); when
    # that's absent it imports as None and model init crashes. Narration doesn't
    # need watermarking, so fall back to perth's no-op DummyWatermarker.
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    from chatterbox.tts import ChatterboxTTS

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"[chatterbox] loaded on {device}")

    def synth(text: str, out_path: str, exaggeration: float) -> None:
        kwargs: Dict[str, Any] = {"exaggeration": float(exaggeration), "cfg_weight": 0.5}
        if voice_ref and os.path.exists(voice_ref):
            kwargs["audio_prompt_path"] = voice_ref
        wav = model.generate(text, **kwargs)
        # Save standard 16-bit PCM (stdlib-wave readable + what Blender VSE wants),
        # not the float WAV torchaudio writes by default for float tensors.
        ta.save(out_path, wav, model.sr, encoding="PCM_S", bits_per_sample=16)

    return synth


def _make_kokoro_synth(voice: str = "af_heart") -> SynthFn:
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    print("[kokoro] loaded")

    def synth(text: str, out_path: str, exaggeration: float) -> None:
        # Kokoro has no emotion control; exaggeration is ignored.
        audio = None
        for _, _, audio in pipe(text, voice=voice):
            break
        sf.write(out_path, audio, 24000)

    return synth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="manifest.script.json")
    ap.add_argument("--out-dir", required=True, help="creates clips/ + tts_index.json")
    ap.add_argument("--backend", choices=["chatterbox", "kokoro"], default="chatterbox")
    ap.add_argument("--voice-ref", default="", help="chatterbox: 5-10s reference wav to clone")
    ap.add_argument("--kokoro-voice", default="af_heart")
    ap.add_argument("--text-source", choices=["tts_v3", "script", "tts_ssml"], default="tts_v3")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.script, "r", encoding="utf-8") as f:
        script_obj = json.load(f)
    script_obj["_path"] = os.path.abspath(args.script)

    if args.backend == "chatterbox":
        synth_fn = _make_chatterbox_synth(args.voice_ref)
    else:
        synth_fn = _make_kokoro_synth(args.kokoro_voice)

    index = synthesize_manifest(
        script_obj, os.path.abspath(args.out_dir),
        backend=args.backend, synth_fn=synth_fn,
        text_source=args.text_source, overwrite=bool(args.overwrite),
        voice_ref=args.voice_ref,
    )

    out_index = os.path.join(os.path.abspath(args.out_dir), "tts_index.json")
    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote={out_index} clips={len(index['clips'])} total={index['total_duration_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
