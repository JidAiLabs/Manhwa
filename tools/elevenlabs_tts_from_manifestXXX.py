#!/usr/bin/env python3
"""
elevenlabs_tts_from_manifest.py (v3-tag aware + per-tag presets + v3 stability fix)

Fixes:
- Eleven v3 requires stability to be one of: [0.0, 0.5, 1.0]
  (0.0=Creative, 0.5=Natural, 1.0=Robust).
  We quantize stability when model_id is v3.
- Keeps bracket tags for v3 by default.
- Maps your leading tags ([tense], [urgent], ...) to v3-friendly audio tags.
- Optional per-tag voice_settings presets (quantized stability for v3).
- Sends speed inside voice_settings.
"""

import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests

ELEVEN_BASE = "https://api.elevenlabs.io"
_TAG_RE = re.compile(r"\[[^\[\]]+\]")
_LEADING_TAG_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*")


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
    try:
        return float(p.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe output not float for {path}: {p.stdout!r}") from e


def ensure_api_key() -> str:
    k = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not k:
        raise SystemExit("Missing ELEVENLABS_API_KEY env var. Add to ~/.zshrc and restart shell.")
    return k


def output_format_to_ext(output_format: str) -> str:
    if not output_format:
        return "mp3"
    codec = output_format.split("_", 1)[0].strip().lower()
    if codec in ("mp3", "wav", "pcm", "ulaw", "mulaw"):
        return "wav" if codec == "mulaw" else codec
    return codec or "mp3"


def strip_bracket_tags(text: str) -> str:
    s = re.sub(_TAG_RE, "", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_v3_model(model_id: str) -> bool:
    s = (model_id or "").lower().strip()
    return "v3" in s  # e.g., "eleven_v3"


# ---- tag mapping (your internal -> v3-friendly) ----
_TAG_MAP_V3: Dict[str, str] = {
    "calm": "calm",
    "tense": "nervously",
    "urgent": "panicked",
    "excited": "excited",
    "awe": "amazed",
    "sad": "sad",
    "whisper": "whispers",
    "angry": "angry",
}

# For v3, stability must be 0.0/0.5/1.0. We keep presets aligned to that.
_PRESET_BY_TAG_V3: Dict[str, Dict[str, Any]] = {
    "calm":    {"stability": 0.5, "style": 0.25, "speed": 1.02},
    "tense":   {"stability": 0.5, "style": 0.45, "speed": 1.05},
    "urgent":  {"stability": 0.0, "style": 0.60, "speed": 1.12},
    "excited": {"stability": 0.0, "style": 0.65, "speed": 1.10},
    "awe":     {"stability": 0.5, "style": 0.45, "speed": 1.00},
    "sad":     {"stability": 1.0, "style": 0.30, "speed": 0.95},
    "whisper": {"stability": 1.0, "style": 0.25, "speed": 0.98},
    "angry":   {"stability": 0.0, "style": 0.70, "speed": 1.06},
}

# Non-v3 presets (your original idea; kept)
_PRESET_BY_TAG: Dict[str, Dict[str, Any]] = {
    "calm":    {"stability": 0.55, "style": 0.25, "speed": 1.00},
    "tense":   {"stability": 0.45, "style": 0.45, "speed": 1.00},
    "urgent":  {"stability": 0.40, "style": 0.55, "speed": 1.08},
    "excited": {"stability": 0.40, "style": 0.60, "speed": 1.08},
    "awe":     {"stability": 0.50, "style": 0.45, "speed": 0.98},
    "sad":     {"stability": 0.55, "style": 0.35, "speed": 0.92},
    "whisper": {"stability": 0.55, "style": 0.30, "speed": 0.95},
    "angry":   {"stability": 0.45, "style": 0.60, "speed": 1.02},
}


def extract_leading_tag(text: str) -> Tuple[Optional[str], str]:
    """Return (tag_without_brackets, remainder_text). Only removes ONE leading tag if present."""
    s = str(text or "").strip()
    if not s:
        return None, ""
    m = _LEADING_TAG_RE.match(s)
    if not m:
        return None, s
    tag = (m.group(1) or "").strip().lower()
    rest = s[m.end():].strip()
    return tag, rest


def apply_v3_tag_mapping(text: str) -> Tuple[str, Optional[str]]:
    """
    If your text starts with [tense]/[urgent]/..., replace with a v3-friendly audio tag.
    Returns (new_text, internal_tag_detected).
    """
    internal_tag, rest = extract_leading_tag(text)
    if not internal_tag:
        return text, None

    mapped = _TAG_MAP_V3.get(internal_tag, None)
    if not mapped:
        return text, internal_tag

    return f"[{mapped}] {rest}".strip(), internal_tag


def pick_paragraph_list(section: Dict[str, Any], text_source: str) -> List[str]:
    if text_source == "script":
        v = section.get("script_paragraphs") or []
        return v if isinstance(v, list) else []
    if text_source == "tts_v3":
        v = section.get("tts_paragraphs_v3") or []
        return v if isinstance(v, list) else []
    if text_source == "tts_ssml":
        v = section.get("tts_paragraphs_ssml") or []
        return v if isinstance(v, list) else []
    return []


def pick_tts_meta(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = section.get("tts_meta") or []
    return v if isinstance(v, list) else []


def script_items_from_manifest(script_obj: Dict[str, Any], text_source: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    sections = script_obj.get("sections") or []
    if not isinstance(sections, list):
        return items

    for sec in sections:
        if not isinstance(sec, dict):
            continue

        sec_idx = int(sec.get("section_index") or 0)
        paras = pick_paragraph_list(sec, text_source=text_source)
        shots = sec.get("shots") or []
        meta = pick_tts_meta(sec)

        if not isinstance(paras, list) or not isinstance(shots, list):
            continue

        n = min(len(paras), len(shots))
        for pi in range(n):
            s = shots[pi]
            if not isinstance(s, dict):
                continue

            gid = int(s.get("group_id") or 0)
            if gid <= 0:
                continue

            beat_id = int(s.get("beat_id") or gid)
            p = paras[pi]
            text = str(p or "").strip()
            if not text:
                continue

            voice_settings = None
            if pi < len(meta) and isinstance(meta[pi], dict):
                vs = meta[pi].get("voice_settings")
                if isinstance(vs, dict):
                    voice_settings = vs

            items.append(
                {
                    "group_id": gid,
                    "section_index": sec_idx,
                    "beat_id": beat_id,
                    "paragraph_index": pi,
                    "text": text,
                    "voice_settings": voice_settings,
                }
            )

    items.sort(key=lambda x: (int(x["group_id"]), int(x["section_index"]), int(x["beat_id"]), int(x["paragraph_index"])))
    return items


def _quantize_v3_stability(x: float) -> float:
    """
    Eleven v3 stability must be one of [0.0, 0.5, 1.0].
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="manifest.script.json")
    ap.add_argument("--out-dir", required=True, help="Output directory (creates clips/ and tts_index.json)")
    ap.add_argument("--voice-id", required=True, help="ElevenLabs voice_id")
    ap.add_argument("--model-id", default="eleven_multilingual_v2")
    ap.add_argument("--output-format", default="mp3_44100_128")
    ap.add_argument("--optimize-streaming-latency", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--text-source", choices=["script", "tts_v3", "tts_ssml"], default="tts_v3")

    # Tag behavior:
    ap.add_argument("--no-strip-tags", action="store_true")
    ap.add_argument("--disable-v3-tag-mapping", action="store_true")
    ap.add_argument("--disable-tag-presets", action="store_true")

    # Defaults
    ap.add_argument("--stability", type=float, default=0.5)  # safer default for v3
    ap.add_argument("--similarity-boost", type=float, default=0.78)
    ap.add_argument("--style", type=float, default=0.35)
    ap.add_argument("--speaker-boost", action="store_true")
    ap.add_argument("--speed", type=float, default=1.12)  # faster default (you wanted quicker narration)

    args = ap.parse_args()

    api_key = ensure_api_key()
    script_obj = load_json(args.script)
    items = script_items_from_manifest(script_obj, text_source=args.text_source)
    if not items:
        raise SystemExit("No (group_id, text) items extracted from manifest.script.json")

    out_dir = os.path.abspath(args.out_dir)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    ext = output_format_to_ext(args.output_format)

    v3 = is_v3_model(args.model_id)

    # Tag stripping default
    force_keep = bool(args.no_strip_tags)
    strip_tags = (not force_keep) and (not v3)  # strip only for non-v3 by default

    # Quantize CLI stability if v3
    cli_stability = _quantize_v3_stability(args.stability) if v3 else float(args.stability)

    index: Dict[str, Any] = {
        "source_script": os.path.abspath(args.script),
        "voice_id": args.voice_id,
        "model_id": args.model_id,
        "output_format": args.output_format,
        "text_source": args.text_source,
        "strip_tags": strip_tags,
        "v3_tag_mapping": (v3 and (not args.disable_v3_tag_mapping)),
        "tag_presets": (not args.disable_tag_presets),
        "default_voice_settings": {
            "stability": cli_stability,
            "similarity_boost": float(args.similarity_boost),
            "style": float(args.style),
            "use_speaker_boost": bool(args.speaker_boost),
            "speed": float(args.speed),
        },
        "clips": [],
        "total_duration_sec": 0.0,
    }

    gid_counts: Dict[int, int] = {}
    total = 0.0

    for it in items:
        gid = int(it["group_id"])
        sec_idx = int(it["section_index"])
        beat_id = int(it["beat_id"])
        para_idx = int(it["paragraph_index"])

        source_text = str(it["text"])
        sent_text = source_text if not strip_tags else strip_bracket_tags(source_text)
        if not sent_text:
            continue

        internal_tag = None
        if v3 and (not args.disable_v3_tag_mapping):
            sent_text, internal_tag = apply_v3_tag_mapping(sent_text)

        gid_counts[gid] = gid_counts.get(gid, 0) + 1
        suffix = "" if gid_counts[gid] == 1 else f"_{gid_counts[gid]:02d}"
        audio_name = f"g_{gid:04d}{suffix}.{ext}"
        audio_path = os.path.join(clips_dir, audio_name)

        # Per-clip voice settings override (from manifest) OR defaults
        vs = it.get("voice_settings")
        if isinstance(vs, dict):
            voice_settings = _coerce_voice_settings(vs, v3=v3)
        else:
            voice_settings = {
                "stability": cli_stability,
                "similarity_boost": float(args.similarity_boost),
                "style": float(args.style),
                "use_speaker_boost": bool(args.speaker_boost),
                "speed": float(args.speed),
            }

        # Optional: reinforce your internal tag via presets (only when manifest didn't override)
        if (not args.disable_tag_presets) and (not isinstance(vs, dict)) and internal_tag:
            preset = (_PRESET_BY_TAG_V3 if v3 else _PRESET_BY_TAG).get(internal_tag)
            if preset:
                voice_settings["stability"] = preset.get("stability", voice_settings.get("stability"))
                if v3:
                    voice_settings["stability"] = _quantize_v3_stability(float(voice_settings["stability"]))
                voice_settings["style"] = float(preset.get("style", voice_settings.get("style", args.style)))
                voice_settings["speed"] = float(preset.get("speed", voice_settings.get("speed", args.speed)))

        if os.path.exists(audio_path) and not args.overwrite:
            dur = ffprobe_duration_sec(audio_path)
            meta = {
                "group_id": gid,
                "group_id_occurrence": gid_counts[gid],
                "section_index": sec_idx,
                "beat_id": beat_id,
                "paragraph_index": para_idx,
                "source_text": source_text,
                "sent_text": sent_text,
                "internal_tag": internal_tag,
                "voice_settings": voice_settings,
                "audio_file": os.path.relpath(audio_path, out_dir),
                "duration_sec": round(dur, 4),
                "cached": True,
            }
            index["clips"].append(meta)
            total += dur
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

        meta = {
            "group_id": gid,
            "group_id_occurrence": gid_counts[gid],
            "section_index": sec_idx,
            "beat_id": beat_id,
            "paragraph_index": para_idx,
            "source_text": source_text,
            "sent_text": sent_text,
            "internal_tag": internal_tag,
            "voice_settings": voice_settings,
            "audio_file": os.path.relpath(audio_path, out_dir),
            "duration_sec": round(dur, 4),
            "cached": False,
            "request_id": hdr.get("request-id", ""),
            "char_cost": hdr.get("x-character-count", ""),
        }
        index["clips"].append(meta)
        total += dur

        print(f"[ok] group_id={gid}#{gid_counts[gid]} dur={dur:.2f}s file={audio_name}")

    index["total_duration_sec"] = round(total, 4)
    out_index = os.path.join(out_dir, "tts_index.json")
    dump_json(out_index, index)
    print(f"[ok] wrote={out_index} clips={len(index['clips'])} total_sec={index['total_duration_sec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
