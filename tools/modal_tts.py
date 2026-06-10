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

Cost: A10G ≈ $1.10/hr. Measured (clone mode, sdpa): ~20 min/chapter ≈ $0.37.
With flash-attn (now in the image): expected ~2-3× faster ⇒ ~$0.12-0.18/chapter
— verify on the next run (first run also rebuilds the image once).
"""

import contextlib
import io
import json
import os
import wave

import modal

app = modal.App("manhwa-tts")

# Persistent HF cache: the 3.4 GB Qwen3-TTS weights download ONCE into this
# Volume, then every later run mounts them (kills the cold-start tax that
# dominated the ~$0.19/chapter cost). Modal auto-commits the Volume on exit.
HF_CACHE_PATH = "/root/.cache/huggingface"
hf_cache_vol = modal.Volume.from_name("manhwa-hf-cache", create_if_missing=True)

# CUDA image with Qwen3-TTS + flash-attn. Measured on A10G with sdpa: 36 clips
# / 8.9 min audio in ~20 min ≈ $0.37/chapter (generation-dominated); flash-attn
# is the speed/cost lever. It compiles from source, so the base image must be
# the CUDA *devel* registry image (nvcc) and torch must be installed first
# (--no-build-isolation). One-time build, cached as an image layer.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # silence the verbose NVIDIA banner
    .apt_install("espeak-ng", "sox", "ffmpeg")
    .pip_install("torch", "ninja", "packaging", "wheel")
    .pip_install("qwen-tts", "soundfile")
    .env({"MAX_JOBS": "8"})  # bound flash-attn's parallel compile memory
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env({"HF_HOME": HF_CACHE_PATH})
)


@app.function(gpu="A10G", image=image, timeout=1800, volumes={HF_CACHE_PATH: hf_cache_vol})
def synth_qwen(items: list, persona: str, ref_wav: bytes = b"", ref_text: str = "") -> list:
    """Generate every paragraph on the GPU (model loads once). items: list of
    (segment_id, sent_text, emotion_instruction). Returns (segment_id, wav_bytes, sr).

    With *ref_wav* (the locked narrator, e.g. assets/voice/narrator_ref.wav):
    CLONES that exact voice via the Base checkpoint — clone mode ignores the
    per-clip emotion instruction (delivery = reference prosody + punctuation).
    Without it: VoiceDesign persona + instruction (voice varies run-to-run).
    """
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    model_id = ("Qwen/Qwen3-TTS-12Hz-1.7B-Base" if ref_wav
                else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"[gpu] {model_id} loaded" + (" — voice-clone mode" if ref_wav else ""))

    prompt = None
    if ref_wav:
        ref_path = "/tmp/narrator_ref.wav"
        with open(ref_path, "wb") as f:
            f.write(ref_wav)
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text=(ref_text or None),
            x_vector_only_mode=not bool(ref_text),
        )

    out = []
    for seg_id, sent, instruct in items:
        if prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=sent, language="English", voice_clone_prompt=prompt)
        else:
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
def main(script: str, out_dir: str, voice_ref: str = ""):
    """Read the script manifest locally, voice it on the GPU, write clips + index.

    voice_ref: wav of the locked narrator to clone (defaults to
    assets/voice/narrator_ref.wav when present; pass --voice-ref "" stays
    cloning, --voice-ref none forces VoiceDesign exploration mode).
    """
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    lt = _load_adapter(tools_dir)

    repo_root = os.path.dirname(tools_dir)
    if not voice_ref:
        default_ref = os.path.join(repo_root, "assets", "voice", "narrator_ref.wav")
        if os.path.exists(default_ref):
            voice_ref = default_ref
    if voice_ref.lower() == "none":
        voice_ref = ""

    ref_wav, ref_text = b"", ""
    if voice_ref:
        with open(voice_ref, "rb") as f:
            ref_wav = f.read()
        ref_text = lt.ref_text_for(voice_ref)
        print(f"[modal] cloning locked narrator: {voice_ref}"
              + (" (ICL w/ transcript)" if ref_text else " (x-vector only)"))

    obj = json.load(open(script))
    raw = lt.extract_items_from_manifest(obj, "tts_v3")
    items = []
    for it in raw:
        tag = lt.leading_tag(it["text"])
        sent = lt.strip_bracket_tags(it["text"])
        instruct = lt.exaggeration_to_instruction(lt.mood_to_exaggeration(tag))
        items.append((it["segment_id"], sent, instruct))

    print(f"[modal] voicing {len(items)} paragraphs on GPU …")
    results = synth_qwen.remote(items, lt.QWEN_VOICE_PERSONA, ref_wav, ref_text)

    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    by_id = {it["segment_id"]: it for it in raw}
    index = {"backend": "qwen-modal",
             "voice": (f"clone:{os.path.basename(voice_ref)}" if voice_ref
                       else "qwen3-tts male persona (voice-design)"),
             "clips": [], "total_duration_sec": 0.0}
    total = 0.0
    for seg_id, wav_bytes, _sr in results:
        p = os.path.join(clips_dir, f"{seg_id}.wav")
        with open(p, "wb") as f:
            f.write(wav_bytes)
        # uniform lead/tail pads + soft-attack lift (first word audible)
        cond = lt.condition_wav_file(p)
        with contextlib.closing(wave.open(p, "rb")) as w:
            dur = w.getnframes() / float(w.getframerate() or 1)
        it = by_id.get(seg_id, {})
        index["clips"].append({
            "segment_id": seg_id, "group_id": it.get("group_id"),
            "section_index": it.get("section_index"), "beat_id": it.get("beat_id"),
            "paragraph_index": it.get("paragraph_index"),
            "audio_file": f"clips/{seg_id}.wav", "duration_sec": round(dur, 4),
            **cond,
        })
        if cond.get("soft_attack"):
            print(f"[fix] {seg_id}: soft attack lifted x{cond['attack_gain']}")
        total += dur
    index["total_duration_sec"] = round(total, 4)
    with open(os.path.join(out_dir, "tts_index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {len(index['clips'])} clips to {out_dir} (total {total/60:.1f} min)")
