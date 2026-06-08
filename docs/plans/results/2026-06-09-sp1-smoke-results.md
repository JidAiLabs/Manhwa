# SP1 Live Smoke Results — 2026-06-09

**Sub-project:** Acquisition + Catalog Spine (SP1)
**Test type:** Manual live validation (network required)
**Environment:** macOS, `.eval_venv`, `studio` installed via `pip install -e .`

---

## Summary

| Source   | Series                     | add-series        | fetch ch1              | Result |
|----------|----------------------------|-------------------|------------------------|--------|
| webtoon  | Omniscient Reader          | 309 chapters      | 64 images, downloaded  | PASSED |
| asura    | Nano Machine               | not yet live-run  | —                      | —      |
| elftoon  | Infinite Evolution From Zero | not yet live-run | —                      | —      |

---

## Webtoon — Omniscient Reader (PASSED)

**Command run:**

```bash
studio add-series webtoon \
  "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"
# Output: series_id=1 chapters=309
```

```bash
studio fetch 1 --chapters 1
# Output: Fetched 1 chapter(s).
```

**Verified state (post-run):**

- Series row: `source=webtoon`, `slug=omniscient-reader`, `title=Omniscient Reader`
- Chapter 1 `status=downloaded`
- `ep_dir=/Users/anka/repos/Manhwa/ongoing/omniscient-reader/Episode_1`
- Image count: **64 files** (`001.jpg` – `064.jpg`)
- Image validity: PIL opens `001.jpg` as JPEG 800×1000 without error

---

## Bug caught and fixed during live run

**Symptom:** `add-series` raised `RuntimeError: gallery-dl -j failed (exit 127)`.

**Root cause:** `webtoon.py` originally invoked `gallery-dl` as a bare
subprocess command (`["gallery-dl", "-j", url]`).  `gallery-dl` is installed
into the `.eval_venv` virtualenv but not onto the system `PATH`, so the
subprocess call received exit code 127 (command not found).

**Fix (commit `648f238`):** Changed the invocation to use `sys.executable -m
gallery_dl` — running gallery-dl as a Python module through the same
interpreter that is executing `studio`.  This is PATH-independent and works
regardless of whether the virtualenv is activated.

```python
# Before (broken)
result = subprocess.run(["gallery-dl", "-j", url], ...)

# After (fixed)
result = subprocess.run([sys.executable, "-m", "gallery_dl", "-j", url], ...)
```

---

## Asura / Elftoon

Not yet live-run.  Adapters are implemented and pass offline conformance
tests using recorded HTML fixtures.  Live validation is deferred to a future
smoke run.

---

## Test artifact

`tests/test_live_smoke.py` — `@pytest.mark.live` test that reproduces the
webtoon add-series + fetch sequence against a temp DB and temp `ongoing/`
directory, asserting `status=downloaded` and PIL-valid `001.jpg`.

Run with:

```bash
.eval_venv/bin/python -m pytest -m live tests/test_live_smoke.py -v
```
