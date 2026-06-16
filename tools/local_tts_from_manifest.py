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
import sys
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from narration_consistency import narration_sha  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (mood tags, item extraction, duration) — all unit-tested
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\[([^\]]+)\]")
# canonical per-paragraph clip filename: g####_p##.wav (prune targets ONLY these)
_SEGMENT_WAV_RE = re.compile(r"^g\d{4}_p\d{2}\.wav$")

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


def exaggeration_to_instruction(exaggeration: float) -> str:
    """Map the per-clip intensity (0..1) to a natural-language emotion instruction
    for instruction-driven backends like Qwen3-TTS. Keeps the adapter's synth_fn
    interface unchanged (intensity already encodes the mood)."""
    e = float(exaggeration)
    if e < 0.35:
        return "Speak in a calm, somber, restrained tone."
    if e < 0.55:
        return "Speak in a serious, measured cinematic narrator voice."
    if e < 0.70:
        return "Speak with rising tension and unease."
    if e < 0.85:
        return "Speak with intense, dramatic energy."
    return "Speak forcefully and urgently, with explosive intensity."


def exaggeration_to_speed(exaggeration: float) -> float:
    """Map per-clip intensity (0..1) to a Kokoro speaking rate. Kokoro has no
    emotion control, so pace is the main expressive lever: slow + weighty for
    calm/somber beats, brisk + urgent for intense/explosive ones."""
    e = float(exaggeration)
    if e < 0.35:
        return 0.90    # calm/somber — slow, contemplative
    if e < 0.55:
        return 0.96    # serious narrator
    if e < 0.70:
        return 1.00    # tense
    if e < 0.85:
        return 1.06    # intense
    return 1.12        # explosive — fast, urgent


# ---------------------------------------------------------------------------
# Clip conditioning — fixes the "first word is swallowed" defect measured on
# the Modal ch1 run: some takes open at 10-22% of body loudness for 300ms+
# (perceptually silent), and lead/tail dead air varies per clip. Conditioning
# trims to uniform pads and lifts a soft attack with a bounded gain.
# ---------------------------------------------------------------------------
PAD_LEAD_SEC = 0.12      # uniform lead-in kept before the first word
PAD_TRAIL_SEC = 0.20     # natural breath kept after the last word
ATTACK_WINDOW_SEC = 0.4  # window judged (and lifted) at the clip head
ATTACK_MIN_RATIO = 0.5   # head RMS below this fraction of body RMS = soft
ATTACK_MAX_GAIN = 4.0    # +12 dB ceiling so noise is never blasted
_SILENCE_AMP = 0.01      # ~-40 dBFS


def condition_wav(x, sr: int):
    """Condition one mono float32 waveform; returns (y, info).

    info: lead_trim_sec / trail_trim_sec (dead air removed beyond the pads),
    soft_attack (head needed lifting), attack_gain (bounded make-up gain,
    constant over the head then ramped to 1.0 so there is no audible step).
    Pure numpy — unit-testable without any TTS model.
    """
    import numpy as np

    info = {"lead_trim_sec": 0.0, "trail_trim_sec": 0.0,
            "soft_attack": False, "attack_gain": 1.0}
    x = np.asarray(x, dtype=np.float32)
    loud = np.abs(x) > _SILENCE_AMP
    if not loud.any():
        return x, info

    first = int(np.argmax(loud))
    last = int(len(x) - np.argmax(loud[::-1]) - 1)
    start = max(0, first - int(PAD_LEAD_SEC * sr))
    end = min(len(x), last + 1 + int(PAD_TRAIL_SEC * sr))
    info["lead_trim_sec"] = round(start / sr, 3)
    info["trail_trim_sec"] = round((len(x) - end) / sr, 3)
    y = x[start:end].copy()

    aw = int(ATTACK_WINDOW_SEC * sr)
    if len(y) > 2 * aw:
        def _rms(seg) -> float:
            return float(np.sqrt((seg.astype(np.float64) ** 2).mean()))

        head = _rms(y[:aw])
        win = max(1, int(0.1 * sr))
        vals = [_rms(y[i:i + win]) for i in range(aw, len(y) - win, win)]
        vals = [v for v in vals if v > _SILENCE_AMP]
        body = float(np.median(vals)) if vals else 0.0

        if body > 0.0 and 0.0 < head < ATTACK_MIN_RATIO * body:
            g = min(ATTACK_MAX_GAIN, (0.8 * body) / head)
            if g > 1.0:
                fade = max(1, int(0.1 * sr))
                gain = np.full(aw, g, dtype=np.float32)
                gain[aw - fade:] = np.linspace(g, 1.0, fade, dtype=np.float32)
                y[:aw] *= gain
                np.clip(y, -1.0, 1.0, out=y)
                info["soft_attack"] = True
                info["attack_gain"] = round(float(g), 2)
    return y, info


def condition_wav_file(path: str) -> dict:
    """Condition a mono wav in place (PCM16 out); returns the info dict.

    Fail-soft: conditioning is an enhancement — an unreadable/corrupt clip is
    left untouched and the skip is recorded as ``condition_error`` in the
    returned dict (which lands in tts_index.json), never raised.
    """
    import numpy as np
    try:
        try:
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                x = x.astype(np.float32) / 32768.0
        except Exception:
            import soundfile as sf
            x, sr = sf.read(path, dtype="float32")
            if getattr(x, "ndim", 1) > 1:
                x = x[:, 0]
        y, info = condition_wav(x, int(sr))
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm.tobytes())
        return info
    except Exception as exc:
        print(f"[warn] conditioning skipped for {os.path.basename(path)}: {exc}")
        return {"condition_error": str(exc)[:120]}


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

    # Text-aware incremental cache: a clip is reused only when its narration is
    # UNCHANGED (same text_sha). When the beats/script are regenerated, only the
    # changed segments are re-voiced — the deterministic audio↔narration gate.
    # Caching on file existence alone (the old rule) is what let stale audio ship.
    prior_sha: Dict[str, str] = {}
    prior_index_path = os.path.join(out_dir, "tts_index.json")
    if os.path.exists(prior_index_path) and not overwrite:
        try:
            prev = json.load(open(prior_index_path))
            for c in prev.get("clips") or []:
                sid = c.get("segment_id")
                sha = c.get("text_sha") or (
                    narration_sha(c["sent_text"]) if c.get("sent_text") is not None else None)
                if sid and sha:
                    prior_sha[str(sid)] = str(sha)
        except Exception:
            prior_sha = {}

    index: Dict[str, Any] = {
        "source_script": os.path.abspath(script_obj.get("_path", "")) if script_obj.get("_path") else "",
        "backend": backend,
        "voice_ref": voice_ref,
        "text_source": text_source,
        "clips": [],
        "total_duration_sec": 0.0,
    }
    total = 0.0
    kept_ids: set = set()
    for it in items:
        seg_id = it["segment_id"]
        source_text = str(it["text"])
        tag = leading_tag(source_text)
        sent_text = strip_bracket_tags(source_text)
        if not sent_text:
            continue
        text_sha = narration_sha(source_text)
        exaggeration = mood_to_exaggeration(tag)
        audio_path = os.path.join(clips_dir, f"{seg_id}.wav")
        kept_ids.add(seg_id)

        # reuse only when the audio exists AND the narration is unchanged
        cached = (os.path.exists(audio_path) and not overwrite
                  and prior_sha.get(seg_id) == text_sha)
        cond: Dict[str, Any] = {}
        if not cached:
            synth_fn(sent_text, audio_path, exaggeration)
            # uniform lead/tail pads + soft-attack lift (first word audible)
            cond = condition_wav_file(audio_path)
        dur = duration_fn(audio_path)

        index["clips"].append({
            "segment_id": seg_id,
            "group_id": int(it["group_id"]),
            "section_index": int(it["section_index"]),
            "beat_id": int(it["beat_id"]),
            "paragraph_index": int(it["paragraph_index"]),
            "source_text": source_text,
            "sent_text": sent_text,
            "text_sha": text_sha,
            "mood_tag": tag or "",
            "exaggeration": exaggeration,
            "audio_file": os.path.relpath(audio_path, out_dir),
            "duration_sec": round(dur, 4),
            "cached": cached,
            **cond,
        })
        total += dur
        flag = " SOFT-ATTACK-LIFTED" if cond.get("soft_attack") else ""
        print(f"[{'cache' if cached else 'ok'}] {seg_id} dur={dur:.2f}s mood={tag or '-'}{flag}")

    # prune orphan SEGMENT clips (g####_p## no longer in the script) so a stale
    # wav never leaks into the voice-preview concat or a later render. Only
    # canonical segment names are touched — never branding or other sidecar wavs.
    for fn in os.listdir(clips_dir):
        if _SEGMENT_WAV_RE.match(fn) and fn[:-4] not in kept_ids:
            try:
                os.remove(os.path.join(clips_dir, fn))
            except OSError as e:
                print(f"[warn] could not prune orphan clip {fn}: {e}")

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


def _make_chatterbox_turbo_synth(voice_ref: str) -> SynthFn:
    """Chatterbox TURBO — much faster than standard Chatterbox, but it IGNORES
    emotion exaggeration (flatter delivery). Still supports voice cloning."""
    import torch
    import torchaudio as ta
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print(f"[chatterbox-turbo] loaded on {device}")

    def synth(text: str, out_path: str, exaggeration: float) -> None:
        kwargs: Dict[str, Any] = {}
        if voice_ref and os.path.exists(voice_ref):
            kwargs["audio_prompt_path"] = voice_ref
        wav = model.generate(text, **kwargs)   # exaggeration ignored by Turbo
        ta.save(out_path, wav, model.sr, encoding="PCM_S", bits_per_sample=16)

    return synth


# Voice persona prepended to every Qwen instruction (VoiceDesign builds the voice
# from the instruction, so this is how we get a consistent MALE narrator).
QWEN_VOICE_PERSONA = "A deep, resonant male narrator voice, clear and dramatic."

# VoiceDesign re-DESIGNS a voice on every call (same persona → audibly different
# narrators), so the locked production narrator is a CLONE of the user-picked
# clip (g0021_p02 → assets/voice/narrator_ref.wav). Cloning needs the Base
# checkpoint; VoiceDesign checkpoints refuse generate_voice_clone.
QWEN_MODEL_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Clone generation is stochastic. UNSEEDED it drifts clip-to-clip (timbre wobble)
# and occasionally emits a robotic/buzzy take — measured on Nano ch1: spectral
# flatness ~0.33 across clips, but g0014_p01 came out at 0.42 (audibly robotic).
# So: seed per-text (reproducible), and if a take reads robotic, re-roll on the
# next seed and KEEP the least-buzzy. Most clips pass on the first try, so cost
# is unchanged except for the rare bad take.
QWEN_CLONE_MAX_TRIES = 3
QWEN_ROBOTIC_FLATNESS = 0.40   # narration sits ~0.33; >0.40 reads buzzy/robotic


def spectral_flatness(wav: Any) -> float:
    """Geometric-mean / arithmetic-mean of the magnitude spectrum (0..1).
    High = noise-like/buzzy (a robotic TTS take); clean speech sits low."""
    import numpy as np
    a = np.asarray(wav, dtype=float)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.size < 256:
        return 0.0
    sp = np.abs(np.fft.rfft(a * np.hanning(a.size))) + 1e-12
    return float(np.exp(np.mean(np.log(sp))) / np.mean(sp))


def ref_text_for(voice_ref: str) -> str:
    """Transcript of a voice-clone reference wav, from its `.txt` sidecar
    (same path, .txt extension). Empty string when absent — the clone then
    falls back to x-vector-only mode instead of ICL."""
    base, _ = os.path.splitext(voice_ref or "")
    sidecar = base + ".txt"
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _make_qwen_synth(voice_ref: str, language: str = "English",
                     persona: str = QWEN_VOICE_PERSONA) -> SynthFn:
    """Qwen3-TTS (Apache-2.0). Two modes:

    - voice_ref given  → CLONE the locked narrator (Base checkpoint,
      `generate_voice_clone` with the ref wav + its .txt transcript). Clone
      mode takes no emotion instruction — delivery follows the reference
      prosody + the text's own punctuation.
    - no voice_ref     → VoiceDesign persona + per-clip emotion instruction
      (voice varies run-to-run; audition/exploration only).

    Apple-Silicon adapted: device=mps, fp16 (~1.7x faster than fp32 on MPS), and
    `sdpa` attention (flash-attention_2 is CUDA-only; sdpa has an MPS path and
    beats `eager`). On CUDA it uses flash-attention + bf16 and is far faster.
    """
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        device, dtype, attn = "cuda", torch.bfloat16, "flash_attention_2"
    elif torch.backends.mps.is_available():
        device, dtype, attn = "mps", torch.float16, "sdpa"
    else:
        device, dtype, attn = "cpu", torch.float32, "sdpa"

    clone = bool(voice_ref) and os.path.exists(voice_ref)
    model_id = QWEN_MODEL_BASE if clone else QWEN_MODEL_VOICE_DESIGN
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map=device, dtype=dtype, attn_implementation=attn,
    )
    print(f"[qwen3-tts] {model_id} on {device} ({attn}, {dtype})"
          + (f" — cloning {voice_ref}" if clone else ""))

    if clone:
        rtext = ref_text_for(voice_ref)
        # ICL mode (with transcript) clones prosody best; x-vector-only is the
        # fallback when no transcript sidecar exists.
        prompt = model.create_voice_clone_prompt(
            ref_audio=voice_ref,
            ref_text=(rtext or None),
            x_vector_only_mode=not bool(rtext),
        )

        def synth(text: str, out_path: str, exaggeration: float) -> None:
            import hashlib
            # per-text seed -> the clip is reproducible (re-runs identical, so the
            # audio<->narration gate is exact); if a take reads robotic, re-roll on
            # the next seed and keep the least-buzzy one.
            base = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            best_wav = None
            best_sr = None
            best_flat = 1.0
            for attempt in range(QWEN_CLONE_MAX_TRIES):
                seed = (base + attempt) & 0x7FFFFFFF
                torch.manual_seed(seed)
                if device == "mps" and hasattr(torch, "mps"):
                    torch.mps.manual_seed(seed)
                elif device == "cuda":
                    torch.cuda.manual_seed_all(seed)
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=prompt)
                flat = spectral_flatness(wavs[0])
                if flat < best_flat:
                    best_flat, best_wav, best_sr = flat, wavs[0], sr
                if flat <= QWEN_ROBOTIC_FLATNESS:
                    break
            sf.write(out_path, best_wav, best_sr, subtype="PCM_16")

        return synth

    def synth(text: str, out_path: str, exaggeration: float) -> None:
        instruct = f"{persona} {exaggeration_to_instruction(exaggeration)}".strip()
        wavs, sr = model.generate_voice_design(text=text, language=language, instruct=instruct)
        sf.write(out_path, wavs[0], sr, subtype="PCM_16")

    return synth


def _make_kokoro_synth(voice: str = "af_heart") -> SynthFn:
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    print("[kokoro] loaded")

    def synth(text: str, out_path: str, exaggeration: float) -> None:
        # No emotion conditioning in Kokoro; use speaking-rate as the expressive
        # lever (intensity -> pace), its only real control besides punctuation.
        spd = exaggeration_to_speed(exaggeration)
        audio = None
        for _, _, audio in pipe(text, voice=voice, speed=spd):
            break
        sf.write(out_path, audio, 24000)

    return synth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="manifest.script.json")
    ap.add_argument("--out-dir", required=True, help="creates clips/ + tts_index.json")
    ap.add_argument("--backend", choices=["chatterbox", "chatterbox-turbo", "qwen", "kokoro"], default="chatterbox")
    ap.add_argument("--voice-ref", default="",
                    help="reference wav to clone: chatterbox (5-10s sample) or qwen "
                         "(locked narrator, e.g. assets/voice/narrator_ref.wav + .txt transcript)")
    ap.add_argument("--kokoro-voice", default="af_heart")
    ap.add_argument("--text-source", choices=["tts_v3", "script", "tts_ssml"], default="tts_v3")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.script, "r", encoding="utf-8") as f:
        script_obj = json.load(f)
    script_obj["_path"] = os.path.abspath(args.script)

    if args.backend == "chatterbox":
        synth_fn = _make_chatterbox_synth(args.voice_ref)
    elif args.backend == "chatterbox-turbo":
        synth_fn = _make_chatterbox_turbo_synth(args.voice_ref)
    elif args.backend == "qwen":
        synth_fn = _make_qwen_synth(args.voice_ref)
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
