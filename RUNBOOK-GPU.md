# GPU runbook — exactly what to run on the rented GPU, in order

This file exists so a rented-GPU session spends its credits on *measurements*, not on
scaffold debugging. The model-facing code paths below were CPU-verified against
`transformers==5.13.0` with a random-weight Qwen2.5-VL (see
`tests/test_probe_contract_cpu.py`, `tests/test_train_qlora_lora_targets.py`) — that
verification caught and fixed three run-blocking defects (`model.visual` →
`model.model.visual`, merged tokens living in `pooler_output` not `last_hidden_state`,
and a PEFT suffix-matching leak that silently LoRA-tuned the vision blocks). What has
*never* run anywhere is the same code against **real weights** — that is what this GPU
session is for.

Hardware assumptions: one CUDA GPU. Step 0 (the probe) fits comfortably on 24 GB with the
3B model in bf16 (~8 GB weights) and works on smaller cards at `--dtype float16`. The
QLoRA stage (step 4) wants ≥ 24 GB for the 7B model in 4-bit.

## 0. Setup (~5 min)

```bash
git clone <this repo> && cd heliogram
pip install -e ".[gpu]"            # or: pip install -e . -r requirements-gpu.txt
pytest -q                          # CPU suite must be green before you trust anything below
pytest -q tests/test_probe_contract_cpu.py tests/test_train_qlora_lora_targets.py
                                   # these now run FOR REAL (torch present) — they gate the
                                   # model-interface contracts against YOUR installed versions
```

If the two contract test files fail here, **stop**: the installed transformers/peft resolved
the Qwen2.5-VL module layout differently than 5.13.0 did, and running the scripts anyway
will measure garbage or crash mid-run. Fix the contract first (the tests say exactly which
seam broke).

## 1. Measure the text-encoding baselines (~1 min, CPU, needs HF Hub access)

The environment this branch was prepared in could not reach HuggingFace Hub, so the
multi-encoding baseline (is base64 even the right bar, or do ascii85/base85 beat it on
Qwen's tokenizer?) is **unmeasured**. This is a 1-minute CPU command and it materially
moves the project's economic bar:

```bash
python -m heliogram.baselines --measure
git add heliogram/data/base64_baseline.json heliogram/data/text_baselines.json
```

Then re-run `python -m heliogram.harness` (CPU, ~15 min) — RESULTS.md's Bar A qualifier
flips from "UNMEASURED CAVEAT" to a measured statement either way, and every verdict is
recomputed against whatever the strongest encoding turned out to be.

## 2. Step 0 — the frozen-encoder linear probe (the decisive experiment, ~minutes)

**Run this before any training spend.** It decides whether the LM-token accounting branch
— the entire surviving economic case — is alive:

```bash
python scripts/run_probe.py \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --palettes 16,128,256 \
    --corruptions clean,jpeg_q85,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report.md --json probe_report.json
```

Decision rules (also printed per-cell by the script):

| Observation | Meaning | Next action |
|---|---|---|
| `palette=16 / clean` at or below the RS budget (~6.3%) | the tower preserves 4 symbols/merged token; the LM only has to learn to read them | proceed to step 3/4 |
| `palette=16 / clean` **at chance** | either the token-order assumption is wrong (check first — it's the documented loud failure mode) or the tower discards the signal | try `--model-id Qwen/Qwen2.5-VL-7B-Instruct` once; if still chance, **the LM-token branch is dead — stop, write it up** |
| `palette=128 / jpeg_q70` below budget | the `bayes_bound` headroom survives INTO the embeddings, not just in pixel statistics | the corruption-axis fine-tune (stage 3/4 of the curriculum) has real support |
| clean passes but every corruption cell fails | perception of the lattice survives, robustness doesn't | fine-tune is still justified, but expect the corruption stages to carry the load |

Paste `probe_report.md` back into the repo (commit it) — it is the Phase-2 Step-0 artifact
the README's roadmap points at.

## 2.5 Localization follow-ups (run these — the 2026-07 session measured the merged probe at/near chance)

**Status from the first GPU session (committed on `gpu-results`):** the merged-stage probe on
the 3B tower came back at/near chance on every cell *including clean* (P=16: 73.6% error vs
6.3% budget, with 23% error even on its own training set — genuine linear non-separability,
not an underpowered probe; P=128/256: near-total, with heavy train/test overfit gaps). Weak
above-chance signal exists everywhere (4–24× chance accuracy), so the token-order assumption
holds; the signal is simply ~10–15× too weak. The measured text baseline also moved the bar
up: ascii85 = 8.374 bits/token > base64's 8.096.

Per the decision table above, that is a FAIL pending three cheap follow-ups (~$1–2 total).
Run all three before deciding anything:

```bash
# (a) escalate the tower: does 7B preserve more?
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --palettes 16,256 --corruptions clean \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report_7b.md --json probe_report_7b.json

# (b) easy mode: if even BINARY color (chance=50%) is unreadable, the result is airtight
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --palettes 2,4 --corruptions clean \
    --n-train-images 12 --n-test-images 4 \
    --out probe_report_easy.md --json probe_report_easy.json

# (c) LOCALIZATION: probe the merger's INPUT (per-patch states, 1 symbol per row).
#     The merged-stage fail localizes the loss to at-or-before the merger OUTPUT;
#     this run splits that ambiguity.
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 16,256 --corruptions clean,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report_premerger.md --json probe_report_premerger.json
```

Decision rules for (c), the one that decides step 7:

| pre_merger result (clean) | Meaning | Step 7 (QLoRA)? |
|---|---|---|
| at/below RS budget (~6.3%) | the vision blocks DO carry per-patch color; the **merger MLP** is what destroys it | **Justified, targeted**: the default LoRA config already tunes the merger (`visual.merger.mlp.0/.2`) — that layer now has a concrete, measured job. Consider raising `--lora-rank` since the merger is carrying the fix. |
| well above budget but far below chance | partial signal reaches the merger input; the tower attenuates it progressively | Long shot. If you run step 7 anyway, use `--include-vision-blocks` and treat the first curriculum stage's held-out accuracy as a hard kill gate. |
| at/near chance | the vision **blocks** already discarded flat-color identity — nothing downstream (merger, LM, LoRA on either) can recover it | **No. Stop.** Write up the negative result — a designed-for-the-channel code, a measured channel, and a tower that provably discards it before the LM boundary is a complete, publishable answer. |

If (a) shows the 7B tower passing where 3B failed, prefer switching the whole Phase-2 target
to 7B over any amount of 3B fine-tuning.

### Session-2 verdict (2026-07, reports committed at repo root)

All three follow-ups ran. Measured outcomes against the table above:

- **(a) 7B escalation: FAILED.** Merged-stage clean error 65.5% (P=16) / 89.8% (P=256) —
  marginally better than 3B, nowhere near the 6.3% budget. Scale does not rescue the
  merged-token branch; the escalation clause is exhausted.
- **(b) Easy mode: pipeline validated, economics not.** P=2 merged reads at 18.3% error
  (81.7% binary accuracy) — the probe/labels/ordering provably work end to end, and the
  tower does carry coarse color. But even binary is ~3× over budget, and readability
  collapses with palette size (P=2: 18% → P=4: 33% → P=16: 66% → P=256: 90%).
- **(c) Pre-merger localization: SPLIT verdict, and it decides step 7.**
  - `P=16 / clean`: **13.4% error** (train 0.2%) vs 73.6% post-merger — the vision blocks
    largely PRESERVE 16-color identity and **the merger MLP is what destroys it** (a ~5×
    error blow-up across one layer). `jpeg_q70`: 19.0% — corruption costs ~6 points
    pre-merger, consistent with the bayes_bound pixel-level headroom finding.
  - `P=256 / clean`: **80.8% error even pre-merger** — the vision BLOCKS already discard
    fine color identity. Per the decision table: for the large palettes, nothing
    downstream (merger, LM, LoRA on either) can recover what never arrives.

**Step-7 decision: the ORIGINAL step 7 (large-palette curriculum, P∈{64,128,256}) is
DEAD — do not run it.** Its entire premise (a learned reader recovering large-palette
color through corruption) requires information the tower's own blocks are measured to
discard before the merger. The one live, narrower option is a P=16-RETARGETED experiment
(merger-focused fine-tune; prize ≈ 16 bits/LM-token ≈ 1.9× the measured ascii85 text bar)
— gated on the two cheap scans below coming back favorable:

```bash
# (d) is P=16's 13.4% data-limited? (train error was 0.2% -- classic small-sample gap)
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 16 --corruptions clean,jpeg_q70 \
    --n-train-images 24 --n-test-images 8 \
    --out probe_report_premerger_p16_big.md --json probe_report_premerger_p16_big.json

# (e) where is the pre-merger palette cliff? (16 -> 13%; 256 -> 81%; map the middle)
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 32,64 --corruptions clean \
    --n-train-images 12 --n-test-images 4 \
    --out probe_report_premerger_cliff.md --json probe_report_premerger_cliff.json
```

Read (d) as: error trending toward the 6.3% budget with more data → a merger-targeted
fine-tune has a credible shot and a retargeted P=16 curriculum is worth its ~$15–40;
stuck at ~13% → even the narrow case rests on a nonlinear readout the probe can't see —
possible, but bet accordingly. (e) sets the max economical palette if (d) passes.

### If (d)/(e) pass: the P=16 merger experiment

`scripts/train_qlora.py` has a second curriculum builder,
`build_p16_merger_curriculum()`, retargeted to exactly this narrow case — palette=16,
`subpatch=1` throughout, never the dead `DEFAULT_PALETTES={64,128,256}` regime. Select it
with `--curriculum p16_merger` (default stays `large_palette`, i.e. the original, now-dead
`build_curriculum()`, kept only for back-compat):

```bash
python scripts/train_qlora.py --curriculum p16_merger \
    --output-dir checkpoints/qwen25vl-heliogram-p16-merger-lora
```

- **VRAM/time expectation:** same as section 4 below — one modern GPU with ≥24GB VRAM,
  tens of GPU-hours total across all stages (see `build_p16_merger_curriculum`'s own
  docstring and the module-level docstring's hardware paragraph; this curriculum has
  fewer, palette=16-only stages than the original, so expect the low end of that range).
- The merger MLP (`visual.merger.mlp.0`/`.2`) is **already** in the default
  `--target_modules` regardless of `--curriculum` — see `LORA_MERGER_TARGET_MODULES` and
  `_build_lora_target_modules` — because tuning exactly that layer is the entire point of
  this experiment: it is the layer the pre-merger/post-merger probe pair (13.4% vs.
  65.5–73.6% symbol error, `docs/FINDINGS.md` section 3) measures to destroy the P=16
  signal the vision blocks upstream still carry.
- **Pre-committed decision rule:** per-stage held-out teacher-forced symbol accuracy
  (printed by `_evaluate_stage_per_symbol_accuracy`, same mechanism as section 4) must
  **beat 86.6%** (= 100% − the 13.4% pre-merger linear-probe symbol error from
  `probe_report_premerger.md`) to show the fine-tune added value beyond what the frozen
  tower already had pre-merger. A run that lands at or below 86.6% means the LoRA-tuned
  merger + LM did no better than the frozen pre-merger embeddings already measured via a
  cheap linear probe — i.e. the expensive fine-tune bought nothing a linear readout on
  frozen weights didn't already show, and the answer is a negative result, not a partial
  win to keep tuning around.
- **The prize, restated, so it isn't oversold:** 4 palette=16 symbols per 2×2-merged LM
  token = 16 bits/token, ≈1.9× the measured ascii85 text bar (8.374 bits/token,
  `docs/FINDINGS.md` section 2) — modest, not transformative, and **unverified**: the
  13.4% number is a linear-probe result on frozen weights, not evidence a LoRA-tuned
  merger can actually get there.
- This trains a machine-dense reader. Per the README's "Phase-2 safety release gate", any
  adapter this run produces is governed by that gate before release — see section 5 below
  (`measure_behavioral_capacity` + `foreign_tile.guard` TPR/FPR against the *tuned* model,
  published together, before anything is released).

## 3. Optional: stock-model zero-shot floor (~30 min)

Before fine-tuning, measure what the *unmodified* model does, so the fine-tune has a real
before/after:

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from heliogram.vlm import zero_shot_symbol_error

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=28*28, max_pixels=16_000_000)

results = zero_shot_symbol_error(model, processor,
    configs=[{"palette": p, "payload_size": 1024} for p in (16, 64, 128, 256)])
for r in results:
    print(r.palette, r.symbol_error_rate, r.decode_success_rate)
```

Expected: near-total failure (the model has never seen this format). That number is the
floor the fine-tune gets compared against — commit it.

Note the processor pixel bounds above: **always** pass `min_pixels=28*28,
max_pixels=16_000_000` (or per-image identity bounds) when feeding heliogram grids, or the
stock processor's ~1MP default budget silently downscales the larger grids — this is now a
measured corruption row (`qwen_smart_resize_1mp`) in RESULTS.md, not a hypothesis.

## 3.5 Typography readability (Option 2 — does the tower actually OCR dense text?)

Separate from the color-codec branch above (steps 0–4 are about `decode_pixels`/`QwenVLDecoder`
reading solid-color patches): `heliogram/typography.py` measured a second, independent pivot —
render the payload as **dense typeset ascii85 text** instead of color blocks, and rely on the
tower's pretrained OCR competence rather than a learned color classifier. That module's own gate
is model-free and already passed: RS-framed ascii85 text clears the color codec's 6.996
bits/patch ceiling at every swept font size, and clears the harder measured ascii85 text-token
bar (8.374 bits/token, `heliogram/data/text_baselines.json`) from 12px font size down. But that
number assumes **perfect legibility** — it says nothing about whether a real, un-fine-tuned
Qwen2.5-VL can actually read text that small. This step answers that, cheaply, before any
typography-focused fine-tune is considered:

```bash
python scripts/run_typography_ocr.py \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --font-sizes 14,12,10,8 \
    --payload-size 256 --n-trials 5 \
    --out typography_ocr_report.md --json typography_ocr_report.json
```

**What it measures:** for each font size, renders `--n-trials` random payloads (both raw ascii85
and RS-framed ascii85, `heliogram.ocr_eval.render_ocr_example` — reusing
`heliogram.typography`'s renderer, not a separate one) and asks the STOCK model to transcribe
each one. Reports, per font size: character error rate (CER), exact-match rate, and —
the metric that actually matters — `decode_success_rate`: the fraction of transcriptions that,
fed through `recover_payload_from_transcription`, recover the exact original payload bytes.
Cross-references all of this against the geometric bars from `heliogram.typography` (reused via
`sweep_typography`/`load_reference_bars`, not recomputed) so readability and density economics
show up side by side in one table, not two documents a reader has to reconcile by hand.

**Three-way verdict** (printed by the script, and in `typography_ocr_report.md`):

| Observation | Meaning |
|---|---|
| some font size both beats 8.374 bits/token geometrically **and** reads at a CER low enough to RS-decode reliably (`decode_success_rate` clears the script's 50% threshold) | **REAL** — the pivot's density economics are backed by actual stock-model readability, not just a geometric assumption. A fine-tune only has to improve on a working zero-shot floor. |
| readability only holds at font sizes too big to beat the bar (the tower can read the text, just not small/dense enough for the economics to work) | **NEEDS FINE-TUNING** — a targeted fine-tune aimed at exactly the gap between "readable" and "dense enough" is the next bounded experiment. |
| even the largest swept font size (14px, deliberately kept as an "should read trivially" control) fails to transcribe reliably | **DEAD** — the tower cannot OCR this alphabet/layout at all; no fine-tune rescues a channel the tower cannot perceive in the first place. |

**Cost and status:** this is a **zero-shot** run (no training, no adapter) — cheap (~$1–2 of GPU
time for the default sweep) and decisive: a clean fail here kills the typography pivot before any
GPU-hours are spent on a typography-focused QLoRA curriculum, exactly the same "cheap experiment
before the expensive one" role the frozen-encoder linear probe (section 2 above) plays for the
color-codec branch. Commit `typography_ocr_report.md` + `.json` verbatim, same convention as the
probe reports in section 2.

## 4. QLoRA fine-tune (only if step 2 passed; tens of GPU-hours)

**This section describes the DEFAULT curriculum (`--curriculum large_palette`, i.e.
`build_curriculum()` over `DEFAULT_PALETTES={64,128,256}`) — the original Slice C bet, which
the session-2 verdict above measured DEAD (the vision blocks discard that much color depth
before the merger ever runs). It is kept as the CLI default only for back-compatibility; do
not run it expecting it to work. If you're here because (d)/(e) in section 2.5 passed, use
the "If (d)/(e) pass" subsection above (`--curriculum p16_merger`) instead.**

```bash
python scripts/train_qlora.py --output-dir checkpoints/qwen25vl-heliogram-lora
```

- The curriculum's cheapest decisive stage runs first; each stage prints held-out
  teacher-forced per-symbol accuracy so you can kill a failing run early.
- `--include-vision-blocks` opts into LoRA on the full vision tower (bigger, riskier);
  the default tunes the LM decoder + the 2×2 merger MLP only — and, as of this branch,
  *verifiably* only that (the PEFT suffix-matching leak is fixed and pinned by test).
- After training, evaluate through the same decode path Phase 1 used:

```python
from heliogram import decode
from heliogram.vlm import QwenVLDecoder
decoder = QwenVLDecoder(model=tuned_model, processor=processor, palette=256, subpatch=1)
payload = decode(img, palette=256, subpatch=1, decoder=decoder)
```

## 5. Before releasing anything trained

The README's "Phase-2 safety release gate" section is binding here: run
`heliogram.instruments.injection_bench.measure_behavioral_capacity` against the tuned
model, measure `foreign_tile.guard`'s TPR/FPR against tiles the *tuned* model reads, and
publish both alongside any adapter — the decision rule is written down in the README so it
can't be softened after the fact.

## What to bring back into the repo

1. `heliogram/data/base64_baseline.json` + `heliogram/data/text_baselines.json` (step 1)
2. regenerated `RESULTS.md` / `results.csv` (step 1)
3. `probe_report.md` + `probe_report.json` (step 2) — commit these verbatim
4. the zero-shot floor numbers (step 3), in whatever file the write-up lands in
5. if step 4 ran: per-stage accuracy logs and the final adapter's measured
   symbol-error/decode-success table — numbers, not the checkpoint's existence, are the
   result
