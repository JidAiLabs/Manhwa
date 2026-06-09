"""
tools/modal_tts.py — expressive TTS (Qwen3-TTS) on a Modal serverless GPU.

Same output contract as local_tts_from_manifest.py (clips/{segment_id}.wav +
tts_index.json), but the model runs on an NVIDIA GPU where it's FAST
(~3-5 min/chapter vs hours on Mac MPS). The local machine only orchestrates —
no torch/model needed locally.

ONE-TIME SETUP
  .eval_venv/bin/pip install modal
  .eval_venv/bin/modal token new          # interactive browser login (do this yourself)

RUN (voices a chapter on the GPU, writes clips locally)
  .eval_venv/bin/modal run tools/modal_tts.py \
      --script ongoing/nano-machine/Chapter_1/manifest.script.json \
      --out-dir ongoing/nano-machine/Chapter_1/tts

Then rebuild the QA report as usual; it picks up tts/tts_index.json.

Cost: A10G ≈ $1.10/hr, a chapter ≈ 3-5 min ⇒ ~$0.05-0.10/chapter of GPU time.
"""

import contextlib
import io
import json
import os
import wave

import modal

app = modal.App("manhwa-tts")

# CUDA image with Qwen3-TTS. sdpa attention (fast on CUDA, no flash-attn build
# hassle); bump to flash_attention_2 later for more speed if desired.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "sox", "ffmpeg")
    .pip_install("qwen-tts", "soundfile", "torch")
)


@app.function(gpu="A10G", image=image, timeout=1800)
def synth_qwen(items: list, persona: str) -> list:
    """Generate every paragraph on the GPU (model loads once). items: list of
    (segment_id, sent_text, emotion_instruction). Returns (segment_id, wav_bytes, sr)."""
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    out = []
    for seg_id, sent, instruct in items:
        full = f"{persona} {instruct}".strip()
        wavs, sr = model.generate_voice_design(text=sent, language="English", instruct=full)
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
        out.append((seg_id, buf.getvalue(), int(sr)))
        print(f"[gpu] {seg_id} ok")
    return out


def _load_adapter(tools_dir: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lt", os.path.join(tools_dir, "local_tts_from_manifest.py"))
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    return lt


@app.local_entrypoint()
def main(script: str, out_dir: str):
    """Read the script manifest locally, voice it on the GPU, write clips + index."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    lt = _load_adapter(tools_dir)

    obj = json.load(open(script))
    raw = lt.extract_items_from_manifest(obj, "tts_v3")
    items = []
    for it in raw:
        tag = lt.leading_tag(it["text"])
        sent = lt.strip_bracket_tags(it["text"])
        instruct = lt.exaggeration_to_instruction(lt.mood_to_exaggeration(tag))
        items.append((it["segment_id"], sent, instruct))

    print(f"[modal] voicing {len(items)} paragraphs on GPU …")
    results = synth_qwen.remote(items, lt.QWEN_VOICE_PERSONA)

    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    by_id = {it["segment_id"]: it for it in raw}
    index = {"backend": "qwen-modal", "voice": "qwen3-tts male persona",
             "clips": [], "total_duration_sec": 0.0}
    total = 0.0
    for seg_id, wav_bytes, _sr in results:
        p = os.path.join(clips_dir, f"{seg_id}.wav")
        with open(p, "wb") as f:
            f.write(wav_bytes)
        with contextlib.closing(wave.open(p, "rb")) as w:
            dur = w.getnframes() / float(w.getframerate() or 1)
        it = by_id.get(seg_id, {})
        index["clips"].append({
            "segment_id": seg_id, "group_id": it.get("group_id"),
            "section_index": it.get("section_index"), "beat_id": it.get("beat_id"),
            "paragraph_index": it.get("paragraph_index"),
            "audio_file": f"clips/{seg_id}.wav", "duration_sec": round(dur, 4),
        })
        total += dur
    index["total_duration_sec"] = round(total, 4)
    with open(os.path.join(out_dir, "tts_index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {len(index['clips'])} clips to {out_dir} (total {total/60:.1f} min)")
