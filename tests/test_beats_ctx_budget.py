"""The narration writer's context budget (2026-09-14).

A production writer call ran num_ctx=8192 with prompt_eval_count=8025 and got
167 tokens of answer: done_reason="length", prompt + eval == num_ctx. The prompt
FIT, so ollama raised nothing and the only safeguard (_bumped_num_ctx, which
reacts to a prompt that does NOT fit) never ran. The text-only JSON formatter
then turned the cut fragment into VALID JSON with the content cut mid-sentence,
and the beat went to pads. These pin that a cut is SEEN, retried once at a
window that fits the real prompt, and never laundered.
"""
import json
import sys

sys.path.insert(0, "tools")
import gemini_narrative_pass as gnp  # noqa: E402
import ollama_compat  # noqa: E402
import usage_cost  # noqa: E402

GOOD = json.dumps({"beat_title": "T", "what_happens": "W",
                   "narration": "A line."})


def _resp(content, p, e, done):
    r = {"message": {"content": content}, "prompt_eval_count": p,
         "eval_count": e}
    if done is not None:
        r["done_reason"] = done
    return r


def _call(monkeypatch, responses, fit_ctx=True, num_predict=2400):
    seen = []

    def fake_chat(**kw):
        seen.append(dict(kw["options"]))
        return responses[min(len(seen) - 1, len(responses) - 1)]

    monkeypatch.setattr(ollama_compat, "chat", fake_chat)
    kw = dict(model="gemma4:26b", system_instruction="s", user_payload={},
              image_paths=[], response_schema={"type": "object"},
              max_output_tokens=num_predict, temperature=0.2)
    if fit_ctx:
        kw["fit_ctx"] = True
    obj, raw, usage = gnp._call_model(**kw)
    return obj, raw, usage, seen


def _default_window(monkeypatch):
    monkeypatch.delenv("STUDIO_BEATS_NUM_CTX", raising=False)
    monkeypatch.delenv("STUDIO_BEATS_NUM_CTX_MAX", raising=False)
    # the fitted retry is exercised through the 8192 escape hatch
    # (STUDIO_BEATS_WINDOW=8192): at the production window a cut is at the cap
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", 8192, raising=False)


# ---- classification ---------------------------------------------------------

def test_cut_kind_tells_a_full_window_from_an_exhausted_budget():
    # the production case: prompt + answer filled the window
    assert gnp._cut_kind(_resp("", 8025, 167, "length"), 8192, 2400) == "window"
    # the answer used its whole num_predict: a loop or verbosity, not the window
    assert gnp._cut_kind(_resp("", 3000, 2400, "length"), 8192, 2400) == "predict"
    # both bounds hit at once: the output budget is the binding one
    assert gnp._cut_kind(_resp("", 5792, 2400, "length"), 8192, 2400) == "predict"


def test_cut_kind_is_none_for_a_clean_or_unexplained_stop():
    assert gnp._cut_kind(_resp("", 8025, 167, "stop"), 8192, 2400) is None
    # backends and older fakes send no done_reason at all: that is a stop
    assert gnp._cut_kind(_resp("", 8025, 167, None), 8192, 2400) is None
    # "length" with neither bound reached: nothing a bigger window would fix
    assert gnp._cut_kind(_resp("", 3000, 500, "length"), 8192, 2400) is None


def test_fit_num_ctx_is_the_bump_arithmetic():
    assert gnp._fit_num_ctx(8025, 2400, 8192, 16384) == 12288
    assert gnp._fit_num_ctx(9358, 2048, 8192, 16384) == 13312
    assert gnp._fit_num_ctx(9358, 2048, 16384, 16384) is None   # nowhere bigger
    # the error-string bump still delegates to the same arithmetic
    err = ('{"error":{"message":"request (9358 tokens) exceeds the available '
           'context size (8192 tokens)","n_prompt_tokens":9358}}')
    assert gnp._bumped_num_ctx(err, 8192, 2048) == 13312


# ---- _call_model under fit_ctx ----------------------------------------------

def test_a_window_cut_is_retried_once_at_a_window_that_fits(monkeypatch,
                                                           capsys):
    _default_window(monkeypatch)
    cut = _resp('{"beat_title": "T", "narration": "He watches the', 8025, 167,
                "length")
    ok = _resp(GOOD, 8025, 300, "stop")
    obj, _, usage, seen = _call(monkeypatch, [cut, ok])
    assert [o["num_ctx"] for o in seen] == [8192, 12288]
    assert obj["narration"] == "A line."          # the RETRY is what is parsed
    assert usage["calls"] == 2
    assert usage["input"] == 8025 * 2 and usage["output"] == 167 + 300
    assert usage["cut"] is None
    err = capsys.readouterr().err
    assert ("[beats] answer cut at num_ctx=8192 prompt=8025 out=167 "
            "-> retry at num_ctx=12288") in err


def test_a_cut_at_the_cap_is_logged_and_not_retried(monkeypatch, capsys):
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "16384")
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX_MAX", "16384")
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", 16384, raising=False)
    obj, _, usage, seen = _call(monkeypatch,
                                [_resp('{"a', 16000, 384, "length")])
    assert len(seen) == 1
    assert obj is None and usage["cut"] == "window"
    assert "answer cut at the cap num_ctx=16384" in capsys.readouterr().err


def test_an_exhausted_output_budget_is_not_retried_as_a_window_problem(
        monkeypatch, capsys):
    _default_window(monkeypatch)
    loop = _resp('["p1.jpg", "p1.jpg", "p1.jpg"', 3000, 2400, "length")
    obj, _, usage, seen = _call(monkeypatch, [loop])
    assert len(seen) == 1 and obj is None and usage["cut"] == "predict"
    err = capsys.readouterr().err
    assert "output budget exhausted num_predict=2400" in err
    assert "answer cut" not in err


def test_a_clean_stop_is_never_retried_even_when_the_window_is_full(
        monkeypatch):
    _default_window(monkeypatch)
    obj, _, usage, seen = _call(monkeypatch, [_resp(GOOD, 8025, 167, "stop")])
    assert len(seen) == 1 and obj["narration"] == "A line."
    assert usage["cut"] is None and usage["calls"] == 1


def test_every_budgeted_call_logs_what_it_spent(monkeypatch, capsys):
    _default_window(monkeypatch)
    _call(monkeypatch, [_resp(GOOD, 10, 5, "stop")])
    assert ("[beats] call num_ctx=8192 prompt=10 out=5 done=stop"
            in capsys.readouterr().err)


def test_callers_without_fit_ctx_see_no_change_on_the_wire(monkeypatch,
                                                          capsys):
    # understanding, grouping, accept_better, sanitize and the teaser share
    # _call_model: none of them may pick up the writer's budget behaviour
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    obj, _, usage, seen = _call(monkeypatch,
                                [_resp('{"a', 8025, 167, "length")],
                                fit_ctx=False, num_predict=10)
    assert seen == [{"temperature": 0.2, "num_predict": 10, "num_ctx": 8192}]
    assert usage == {"input": 8025, "output": 167, "cached": 0}
    assert capsys.readouterr().err == ""


def test_usage_accumulator_counts_a_retried_call():
    acc = usage_cost.UsageAccumulator("gemma4:26b")
    acc.add(input_tokens=1, output_tokens=1)
    acc.add(input_tokens=1, output_tokens=1, calls=2)
    assert acc.calls == 3                  # [cost] calls= stays the exact count


# ---- _generate_beat_for_group: a cut is regenerated, never laundered -------

def _gen(monkeypatch, answers, retries=1, usage=None):
    calls = []

    def stub(**kw):
        calls.append(kw)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(gnp, "_call_model_with_backoff", stub)
    out = gnp._generate_beat_for_group(
        model="m", system_instruction="s", payload={"scene_files": ["p1.jpg"]},
        image_paths=[], beat_schema={}, gid=5, retries=retries,
        max_output_tokens=2400, backoff_max=1.0, usage=usage)
    return out, calls


def test_a_cut_answer_is_never_laundered_by_the_json_formatter(monkeypatch):
    cut = (None, '{"beat_title": "T", "narration": "cut mid',
           {"input": 1, "output": 1, "cached": 0, "cut": "window", "calls": 1})
    out, calls = _gen(monkeypatch, [cut])
    assert out is None
    assert len(calls) == 2                            # two FULL attempts
    assert all("strict JSON formatter" not in c["system_instruction"]
               for c in calls)
    assert all(c.get("fit_ctx") is True for c in calls)


def test_a_complete_but_malformed_answer_still_gets_the_formatter(monkeypatch):
    malformed = (None, "not json at all", {"input": 1, "output": 1, "cached": 0})
    fixed = ({"beat_title": "T", "what_happens": "W", "narration": "A line."},
             "raw", {"input": 1, "output": 1, "cached": 0})
    out, calls = _gen(monkeypatch, [malformed, fixed])
    assert out["narration"] == "A line."
    assert "strict JSON formatter" in calls[1]["system_instruction"]
    assert all(c.get("fit_ctx") is True for c in calls)


def test_generate_beat_counts_a_retried_call_in_usage(monkeypatch):
    acc = usage_cost.UsageAccumulator("m")
    beat = ({"beat_title": "T", "what_happens": "W", "narration": "A line."},
            "raw", {"input": 5, "output": 5, "cached": 0, "cut": None,
                    "calls": 2})
    _gen(monkeypatch, [beat], usage=acc)
    assert acc.calls == 2


# ---- A2: ONE window for every writer call + an over-cap shrink --------------
# ollama reloads the runner whenever num_ctx changes. Over 42 h on the Mini,
# 2,189 of 2,214 consecutive gemma reloads (99%) were window changes — mostly
# the writer flip-flopping 8192 -> 12288 (error-driven bump) -> 8192 (next
# group), a reload and a cold prompt cache per call. Every writer call now asks
# for one window; an input too long for it is shrunk BEFORE the call.

def test_every_budgeted_call_asks_for_the_one_writer_window(monkeypatch):
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", 16384, raising=False)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")   # the shared default
    _, _, _, seen = _call(monkeypatch, [_resp(GOOD, 7000, 800, "stop")])
    assert seen[0]["num_ctx"] == 16384


def test_non_writer_callers_keep_their_own_window(monkeypatch):
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", 16384, raising=False)
    monkeypatch.setenv("STUDIO_BEATS_NUM_CTX", "8192")
    _, _, _, seen = _call(monkeypatch, [_resp(GOOD, 10, 5, "stop")],
                          fit_ctx=False)
    assert seen[0]["num_ctx"] == 8192


def _exact_estimates(monkeypatch, window):
    # 1 char = 1 token, 100 tokens per image, no margin/reserve: the shrink's
    # arithmetic becomes the payload's serialized length, so thresholds are exact
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", window, raising=False)
    monkeypatch.setattr(gnp, "_BEATS_CHARS_PER_TOKEN", 1.0, raising=False)
    monkeypatch.setattr(gnp, "_BEATS_IMAGE_TOKENS", 100, raising=False)
    monkeypatch.setattr(gnp, "_BEATS_CTX_MARGIN", 0, raising=False)
    monkeypatch.setattr(gnp, "_REASK_RESERVE_TOKENS", 0, raising=False)


def _payload(n_panels, words_each, prev=True):
    files = [f"p{i}.jpg" for i in range(n_panels)]
    text = " ".join(["word"] * words_each)
    p = {"scene_files": list(files), "taggable_panels": list(files),
         "facts": {"dead_by_now": ["the old man"]},
         "scenes_signals": [{"scene_file": f, "ocr_clean": text,
                             "description": text, "dialogue": text,
                             "action": text, "subjects": ["the prince"]}
                            for f in files]}
    if prev:
        p["previous_narration"] = [" ".join(["x"] * 300),
                                   " ".join(["y"] * 300)]
    return p


def _input_chars(payload):
    return len("INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False))


def test_an_input_that_fits_is_left_exactly_as_it_is(monkeypatch, capsys):
    p, imgs = _payload(2, 10), ["a.jpg", "b.jpg"]
    _exact_estimates(monkeypatch, _input_chars(p) + 200 + 1000)
    out_p, out_i = gnp._fit_beats_input("", p, imgs, 0, 7)
    assert out_p is p and out_i is imgs
    assert capsys.readouterr().err == ""


def test_the_shrink_stops_at_the_first_step_that_fits(monkeypatch, capsys):
    p = _payload(2, 10)
    no_prev = {k: v for k, v in p.items() if k != "previous_narration"}
    _exact_estimates(monkeypatch, _input_chars(no_prev))     # fits once dropped
    out_p, out_i = gnp._fit_beats_input("", p, [], 0, 7)
    steps = [l for l in capsys.readouterr().err.splitlines()
             if "prompt over cap" in l]
    assert len(steps) == 1 and steps[0].endswith("drop previous_narration")
    assert "previous_narration" not in out_p
    assert out_p["scenes_signals"] == p["scenes_signals"]    # not gisted
    assert "previous_narration" in p                         # input untouched


def test_an_input_nothing_can_fit_is_shrunk_in_order_and_never_silent(
        monkeypatch, capsys):
    p, imgs = _payload(6, 200), ["a.jpg", "b.jpg", "c.jpg"]
    _exact_estimates(monkeypatch, 10)                        # nothing fits
    out_p, out_i = gnp._fit_beats_input("", p, imgs, 0, 7)
    err = capsys.readouterr().err
    steps = [l.rsplit(" — ", 1)[1] for l in err.splitlines()
             if "prompt over cap" in l]
    assert steps == ["drop previous_narration",
                     "gist panel text 300/160/160/100",
                     "gist panel text 160/100/100/60",
                     "images -> 2", "images -> 1"]
    assert "STILL over cap" in err
    assert out_i == ["a.jpg"]
    # grounding the writer needs survives every step
    for key in ("scene_files", "taggable_panels", "facts"):
        assert out_p[key] == p[key]
    assert [s["subjects"] for s in out_p["scenes_signals"]] == \
        [s["subjects"] for s in p["scenes_signals"]]
    assert len(out_p["scenes_signals"][0]["ocr_clean"]) < \
        len(p["scenes_signals"][0]["ocr_clean"])


def test_a_prompt_counted_far_below_its_estimate_is_flagged(monkeypatch,
                                                           capsys):
    # ollama context-shifts an over-long prompt SILENTLY (probe: ~14k tokens at
    # 13312 -> prompt_eval_count 6,659, head and system prompt dropped)
    monkeypatch.setattr(gnp, "_BEATS_WINDOW", 16384, raising=False)
    monkeypatch.setattr(gnp, "_BEATS_CHARS_PER_TOKEN", 1.0, raising=False)
    _call(monkeypatch, [_resp(GOOD, 2, 5, "stop")])
    assert "context-shifted" in capsys.readouterr().err
