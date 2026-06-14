#!/usr/bin/env python3
"""
elevenlabs_tts_from_manifest.py (v4 - segment_id aligned)

Key fixes:
- Audio clips are named by paragraph index: g####_p##.mp3
- segment_id written to tts_index.json matches timeline_planner expectation
- Uses manifest.script.json:
    - sections[].tts_paragraphs_v3 (default)
    - sections[].tts_meta[].voice_settings (optional per-paragraph)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from narration_consistency import narration_sha  # noqa: E402

ELEVEN_BASE = "https://api.elevenlabs.io"
_TAG_RE = re.compile(r"\[[^\[\]]+\]")

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ffprobe_duration_sec(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr.strip()}")
    return float(p.stdout.strip())

def ensure_api_key() -> str:
    k = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not k:
        raise SystemExit("Missing ELEVENLABS_API_KEY env var.")
    return k

def output_format_to_ext(output_format: str) -> str:
    codec = (output_format.split("_", 1)[0] if output_format else "mp3").strip().lower()
    if codec in ("mp3", "wav", "pcm", "ulaw", "mulaw"):
        return "wav" if codec == "mulaw" else codec
    return codec or "mp3"

def strip_bracket_tags(text: str) -> str:
    s = re.sub(_TAG_RE, "", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_v3_model(model_id: str) -> bool:
    return "v3" in (model_id or "").lower().strip()

def _quantize_v3_stability(x: float) -> float:
    x = float(x)
    if x < 0.25:
        return 0.0
    if x < 0.75:
        return 0.5
    return 1.0

def _coerce_voice_settings(vs: Dict[str, Any], v3: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "stability" in vs:
        st = float(vs["stability"])
        out["stability"] = _quantize_v3_stability(st) if v3 else st
    if "similarity_boost" in vs:
        out["similarity_boost"] = float(vs["similarity_boost"])
    if "style" in vs:
        out["style"] = float(vs["style"])
    if "use_speaker_boost" in vs:
        out["use_speaker_boost"] = bool(vs["use_speaker_boost"])
    if "speed" in vs:
        out["speed"] = float(vs["speed"])
    return out

def eleven_convert_audio(
    *,
    api_key: str,
    voice_id: str,
    text: str,
    model_id: str,
    output_format: str,
    optimize_streaming_latency: Optional[int],
    voice_settings: Optional[Dict[str, Any]],
) -> Tuple[bytes, Dict[str, str]]:
    url = f"{ELEVEN_BASE}/v1/text-to-speech/{voice_id}"
    params: Dict[str, str] = {"output_format": output_format}
    if optimize_streaming_latency is not None:
        params["optimize_streaming_latency"] = str(int(optimize_streaming_latency))

    payload: Dict[str, Any] = {"text": text, "model_id": model_id}
    if voice_settings:
        payload["voice_settings"] = voice_settings

    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}

    r = requests.post(url, params=params, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS failed {r.status_code}: {r.text[:800]}")

    keep: Dict[str, str] = {}
    for hk in ("x-character-count", "request-id"):
        if hk in r.headers:
            keep[hk] = r.headers.get(hk, "")
    return r.content, keep

def extract_items_from_manifest(script_obj: Dict[str, Any], text_source: str) -> List[Dict[str, Any]]:
    """
    Returns stable list of:
    {segment_id, group_id, section_index, beat_id, paragraph_index, text, voice_settings}
    """
    out: List[Dict[str, Any]] = []
    sections = script_obj.get("sections") or []
    if not isinstance(sections, list):
        return out

    for sec in sections:
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

        meta = sec.get("tts_meta") or []
        if not isinstance(meta, list):
            meta = []

        # Build a quick map segment_id -> voice_settings from tts_meta
        vs_by_seg: Dict[str, Dict[str, Any]] = {}
        for m in meta:
            if isinstance(m, dict):
                sid = str(m.get("segment_id") or "").strip()
                vs = m.get("voice_settings")
                if sid and isinstance(vs, dict):
                    vs_by_seg[sid] = vs

        n = min(len(shots), len(paras))
        for i in range(n):
            shot = shots[i] if isinstance(shots[i], dict) else {}
            gid = int(shot.get("group_id") or 0)
            if gid <= 0:
                continue
            beat_id = int(shot.get("beat_id") or gid)

            segment_id = str(shot.get("segment_id") or f"g{gid:04d}_p{i:02d}").strip()
            # force canonical
            paragraph_index = i
            segment_id = f"g{gid:04d}_p{paragraph_index:02d}"

            p = paras[i]
            if isinstance(p, dict):
                text = str(p.get("text") or "").strip()
            else:
                text = str(p or "").strip()
            if not text:
                continue

            out.append({
                "segment_id": segment_id,
                "group_id": gid,
                "section_index": sec_idx,
                "beat_id": beat_id,
                "paragraph_index": paragraph_index,
                "text": text,
                "voice_settings": vs_by_seg.get(segment_id),
            })

    out.sort(key=lambda x: (x["group_id"], x["section_index"], x["paragraph_index"]))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="manifest.script.json")
    ap.add_argument("--out-dir", required=True, help="Output directory (creates clips/ and tts_index.json)")
    ap.add_argument("--voice-id", required=True)
    ap.add_argument("--model-id", default="eleven_v3")
    ap.add_argument("--output-format", default="mp3_44100_128")
    ap.add_argument("--optimize-streaming-latency", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--text-source", choices=["tts_v3", "script", "tts_ssml"], default="tts_v3")
    ap.add_argument("--strip-tags", action="store_true", help="If set, removes [tags] before sending to ElevenLabs")

    # Defaults if no per-paragraph tts_meta.voice_settings exists
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity-boost", type=float, default=0.78)
    ap.add_argument("--style", type=float, default=0.35)
    ap.add_argument("--speaker-boost", action="store_true")
    ap.add_argument("--speed", type=float, default=1.08)

    args = ap.parse_args()

    api_key = ensure_api_key()
    script_obj = load_json(args.script)

    items = extract_items_from_manifest(script_obj, text_source=args.text_source)
    if not items:
        raise SystemExit("No items extracted from manifest.script.json (check sections/shots/tts_paragraphs_v3).")

    out_dir = os.path.abspath(args.out_dir)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    ext = output_format_to_ext(args.output_format)

    v3 = is_v3_model(args.model_id)
    cli_stability = _quantize_v3_stability(args.stability) if v3 else float(args.stability)

    default_vs = {
        "stability": cli_stability,
        "similarity_boost": float(args.similarity_boost),
        "style": float(args.style),
        "use_speaker_boost": bool(args.speaker_boost),
        "speed": float(args.speed),
    }

    index: Dict[str, Any] = {
        "source_script": os.path.abspath(args.script),
        "voice_id": args.voice_id,
        "model_id": args.model_id,
        "output_format": args.output_format,
        "text_source": args.text_source,
        "strip_tags": bool(args.strip_tags),
        "default_voice_settings": default_vs,
        "clips": [],
        "total_duration_sec": 0.0,
    }

    total = 0.0

    # text-aware incremental cache (mirrors local_tts): reuse a clip only when
    # its narration is UNCHANGED, so a beats/script regen re-voices ONLY the
    # changed segments. Existence-only caching shipped stale audio.
    prior_sha: Dict[str, str] = {}
    prior_index_path = os.path.join(out_dir, "tts_index.json")
    if os.path.exists(prior_index_path) and not args.overwrite:
        try:
            for c in (json.load(open(prior_index_path)).get("clips") or []):
                if c.get("segment_id") and c.get("text_sha"):
                    prior_sha[str(c["segment_id"])] = str(c["text_sha"])
        except Exception:
            prior_sha = {}

    for it in items:
        seg_id = it["segment_id"]
        gid = int(it["group_id"])
        sec_idx = int(it["section_index"])
        beat_id = int(it["beat_id"])
        para_idx = int(it["paragraph_index"])

        source_text = str(it["text"])
        sent_text = strip_bracket_tags(source_text) if args.strip_tags else source_text
        if not sent_text:
            continue
        text_sha = narration_sha(source_text)

        # Canonical filename tied to segment_id (this is the critical fix)
        audio_name = f"{seg_id}.{ext}"
        audio_path = os.path.join(clips_dir, audio_name)

        vs = it.get("voice_settings")
        voice_settings = _coerce_voice_settings(vs, v3=v3) if isinstance(vs, dict) else dict(default_vs)

        if (os.path.exists(audio_path) and not args.overwrite
                and prior_sha.get(seg_id) == text_sha):
            dur = ffprobe_duration_sec(audio_path)
            index["clips"].append({
                "segment_id": seg_id,
                "group_id": gid,
                "section_index": sec_idx,
                "beat_id": beat_id,
                "paragraph_index": para_idx,
                "source_text": source_text,
                "sent_text": sent_text,
                "text_sha": text_sha,
                "voice_settings": voice_settings,
                "audio_file": os.path.relpath(audio_path, out_dir),
                "duration_sec": round(dur, 4),
                "cached": True,
            })
            total += dur
            print(f"[cache] {seg_id} dur={dur:.2f}s file={audio_name}")
            continue

        audio_bytes, hdr = eleven_convert_audio(
            api_key=api_key,
            voice_id=args.voice_id,
            text=sent_text,
            model_id=args.model_id,
            output_format=args.output_format,
            optimize_streaming_latency=args.optimize_streaming_latency,
            voice_settings=voice_settings,
        )

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        dur = ffprobe_duration_sec(audio_path)

        index["clips"].append({
            "segment_id": seg_id,
            "group_id": gid,
            "section_index": sec_idx,
            "beat_id": beat_id,
            "paragraph_index": para_idx,
            "source_text": source_text,
            "sent_text": sent_text,
            "text_sha": text_sha,
            "voice_settings": voice_settings,
            "audio_file": os.path.relpath(audio_path, out_dir),
            "duration_sec": round(dur, 4),
            "cached": False,
            "request_id": hdr.get("request-id", ""),
            "char_cost": hdr.get("x-character-count", ""),
        })
        total += dur
        print(f"[ok] {seg_id} dur={dur:.2f}s file={audio_name}")

    index["total_duration_sec"] = round(total, 4)
    out_index = os.path.join(out_dir, "tts_index.json")
    dump_json(out_index, index)
    print(f"[ok] wrote={out_index} clips={len(index['clips'])} total_sec={index['total_duration_sec']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
