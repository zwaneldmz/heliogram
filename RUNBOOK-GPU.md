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

> **Superseded by Session 3 (2026-07-12): do not run this curriculum.** The cheaper,
> more direct test of the same premise — section 7's Design A/B merger-adapter go/no-go —
> has since been run on real weights and came back **NO-GO** (trained merger plateaus at
> ~0.38 symbol error, ~6× over budget and above the frozen 0.1344 pre-merger linear
> readout this curriculum's own 86.6% decision rule is anchored to). See the Session-3
> verdict in section 7. The (d)/(e) scans below and the curriculum they gate are retained
> for the record only.

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

## 6. GPU repro cost per row (estimate, derived from the committed probe reports)

**None of the committed `probe_report*.json` files contain wall-clock or dollar-cost metadata**
— each JSON cell records `n_train_positions`/`n_test_positions`/`n_classes`/`epochs`/`seed` only
(see e.g. `probe_report.json`'s per-cell `fit` block). Everything below is therefore an
**ESTIMATE derived from the reports' own configuration fields (model, palettes, corruptions,
image counts) plus the runtime assumptions this repo already states elsewhere** — not a
measured benchmark. It is flagged as an estimate throughout; do not treat any number in this
section as a committed result the way `probe_report*.md`'s symbol-error numbers are.

### 6.1 What actually costs GPU time here

Per `heliogram/probe.py` and `scripts/run_probe.py`: the frozen-tower probe does **one forward
pass per image** through the (frozen, no-gradient) vision tower to extract embeddings; the
linear-probe fit itself (`fit_linear_probe`, up to 60 epochs) is plain numpy on the CPU and is
negligible next to a real vision-tower forward pass. So GPU time per report is proportional to
**total images processed** = sum over cells of `(n_train_images + n_test_images)`, since each
`(palette, corruption)` cell in `scripts/run_probe.py`'s `main()` generates and forwards its own
fresh set of images (`_cell_arrays`, called independently per cell — confirmed by reading
`scripts/run_probe.py`'s `main`, not assumed).

### 6.2 Per-row image counts, read directly from each committed report's config

| Report | Model | Cells (palette × corruption) | Images/cell (train+test) | Total images |
|---|---|---:|---:|---:|
| `probe_report.md` | 3B | 3 palettes × 3 corruptions = 9 | 6+3 = 9 | 81 |
| `probe_report_7b.md` | 7B | 2 palettes × 1 corruption = 2 | 6+3 = 9 | 18 |
| `probe_report_premerger.md` | 3B, pre-merger tap | 2 palettes × 2 corruptions = 4 | 6+3 = 9 | 36 |
| `probe_report_easy.md` | 3B | 2 palettes × 1 corruption = 2 | 12+4 = 16 | 32 |

(Cell/image counts read from each `.md`/`.json`'s rows and from the commands in section 2/2.5
above that produced them — `--n-train-images`/`--n-test-images` per invocation — not
re-derived from any timing data, since none exists in the committed files.)

### 6.3 Rough per-row and per-report time/cost estimate (FLAGGED: estimate, not measured)

Anchoring assumptions (stated so they can be checked/replaced against a real GPU session):

- This file's own section 2 header already characterizes the whole multi-cell `probe_report.md`
  run as **"~minutes"** on one GPU — the only runtime claim this repo makes anywhere for this
  step. Spread over 81 images, that is roughly **2–5 seconds of GPU time per image** for the 3B
  tower in bf16 (forward pass + processor overhead), which is the per-image rate used below.
- `scripts/train_merger_adapter.py`'s own cost-estimate module (`estimate_runtime_seconds`)
  independently assumes **0.5 s/image** for a single frozen forward pass with no backprop
  (its `DESIGN_A_SECONDS_PER_IMAGE` constant) — a lower bound consistent with, and cheaper than,
  the 2–5 s/image range above (that script's number excludes processor/dataset generation
  overhead this estimate includes, so a difference in the same direction is expected, not a
  contradiction).
- The 7B tower is assumed **~1.5–2×** the 3B per-image cost (larger vision tower forward pass);
  no 7B timing is available anywhere in this repo to check that multiplier against, so treat it
  as the least-certain part of this estimate.

| Report | Images | Assumed s/image | Estimated GPU time | Estimated cost @ $2/GPU-hr |
|---|---:|---:|---:|---:|
| `probe_report.md` (3B) | 81 | 2–5 | ~3–7 min | ~$0.10–$0.25 |
| `probe_report_7b.md` (7B) | 18 | 3–10 | ~1–3 min | ~$0.05–$0.10 |
| `probe_report_premerger.md` (3B, pre-merger) | 36 | 2–5 | ~1–3 min | ~$0.05–$0.10 |
| `probe_report_easy.md` (3B) | 32 | 2–5 | ~1–3 min | ~$0.05–$0.10 |
| **All four, one session** | **167** | — | **~10–20 min** (plus ~1–2 model-load events) | **~$0.30–$0.60** |

`$2/GPU-hour` is the same rented-GPU rate assumption `scripts/train_merger_adapter.py` uses
(`GPU_HOURLY_RATE_USD_DEFAULT`), kept here for consistency rather than re-derived — adjust
linearly for your actual rented rate. **Bottom line, consistent with every prior claim in this
file: the frozen-encoder probe is cheap — single-digit minutes and well under $1 per report,
tens of minutes and roughly $1 for a full from-scratch reproduction of every committed
`probe_report*` file** — small enough that re-running all of section 2/2.5 to sanity-check a
new environment (e.g. to satisfy `scripts/train_merger_adapter.py`'s alignment assert, section 7
below) is not a meaningful cost concern relative to any of the training stages further down
this file.

## 7. Merger-adapter go/no-go (`scripts/train_merger_adapter.py`, Design A then B)

This is the staged next test after the pre/post-merger localization in section 2.5 — it asks
whether *training* the merger (not just reading it linearly, frozen, the way Step 0's probe
does) can recover the palette=16 signal the pre-merger probe already showed the vision blocks
still carry (`probe_report_premerger.md`: 13.4% clean / 19.0% jpeg_q70) even though it is
destroyed post-merger (`probe_report.md`/`probe_report_7b.md`: 65.5–73.6%). **This has not been
run against real weights anywhere in this repo — there is no GPU here.** It is a designed,
CPU-contract-tested (`tests/test_merger_adapter_contract_cpu.py`), REFUSING scaffold: its CLI
entry point never loads a model itself and always raises before touching torch (see the
script's own module docstring, "REAL INVOCATION").

**Easiest path — the shipped driver + one-shot script** (no snippet to hand-write):

```bash
# One-shot: alignment gate + post-merger probe + Design A, in order, with the decision rule
# printed at the end. GPU required.
bash scripts/gpu_gonogo.sh Qwen/Qwen2.5-VL-3B-Instruct

# Or drive a single design directly (scripts/drive_merger_adapter.py loads the tower via
# run_probe._load_tower, enforces the budget cap, then calls run_design_a/b with the real model):
python scripts/drive_merger_adapter.py --design a --palette 16 --corruptions clean,jpeg_q70 --out design_a.json
python scripts/drive_merger_adapter.py --design b --lora-variant B1 --palette 16 \
    --corruptions clean,jpeg_q70 --epochs 20 --lora-rank 8 --out design_b.json
```

Under the hood the driver is exactly the scaffold's own "REAL INVOCATION" recipe:

```python
from scripts.run_probe import _load_tower       # the exact loader run_probe.py's main() uses
model, processor, dtype = _load_tower("Qwen/Qwen2.5-VL-3B-Instruct", "cuda", "bfloat16")
import scripts.train_merger_adapter as tma
report_a = tma.run_design_a(model, processor, dtype=dtype, device="cuda")            # cheap first
report_b = tma.run_design_b(model, processor, dtype=dtype, device="cuda", variant="B1")  # the gate
```

(Related: the frozen-embedding nonlinear question Design A answers at the merger *input* can also
be asked at any probe tap directly with `scripts/run_probe.py --probe-head mlp`, now implemented —
`heliogram.probe.fit_mlp_probe`, a one-hidden-layer readout — though it, too, needs a real tower.)

**Alignment sanity assert — run this first, every time.** Both `run_design_a` and
`run_design_b` re-run the exact (palette=16, clean, pre_merger) probe cell
`probe_report_premerger.md` was generated from, before trusting any new number, and raise
`RuntimeError` unless the result reproduces the committed **13.4%** clean symbol error
(`PREMERGER_CLEAN_SYMBOL_ERROR = 0.1344`) within a fixed tolerance (`ALIGNMENT_TOLERANCE =
0.02`, i.e. must land in roughly 11.4–15.4%). This is not optional or skippable from the CLI —
it is the first thing either design's real-run path does. If it fails, a GPU session's
transformers/peft version, model id, or window-shuffle handling has drifted since
`probe_report_premerger.md` was generated (`transformers==5.13.0`) — **do not trust any further
number from that session until this reproduces.** Both designs are hard-pinned to
`palette=16` only; any other palette raises immediately (`probe_report_premerger.md` already
measured `palette=256` at/near chance even pre-merger, 80.8% clean, so no merger-side adapter
could recover it regardless of training).

**Per-run cost estimate — tens of dollars, by the script's own design.** The script's own
`estimate_cost_usd`/`check_budget` machinery (pure Python, no torch, unit-tested in
`tests/test_merger_adapter_contract_cpu.py`) is the authoritative estimate here, not a
number recomputed independently in this runbook:

The estimator counts the one-time palette=16/clean alignment sanity cell **and** multiplies the
per-corruption work by `len(corruptions)` — both designs re-extract/re-train once per requested
corruption (`for cname in corruptions:`), so `estimate_cost_usd`/`RunConfig.__post_init__` pass
`n_corruptions=len(corruptions)`.

- **Design A** (`DESIGN_A_SECONDS_PER_IMAGE = 0.5`s): one frozen forward pass per image, no
  backprop — at the CLI's default `--n-train-images 6 --n-test-images 3` and the default two
  corruptions (`clean,jpeg_q70`), that is the alignment cell (`(6+3)·0.5 = 4.5`s) plus one
  per-corruption extraction block per corruption (`2 · 4.5 = 9`s) ≈ **13.5s** of GPU time —
  cents, not dollars, at the default `$2`/GPU-hour rate. Cheap enough that it should always be
  run before Design B.
- **Design B** (`DESIGN_B_SECONDS_PER_TRAIN_STEP = 2.5`s per training image per epoch, plus
  `DESIGN_B_SECONDS_PER_EVAL_IMAGE = 0.5`s per eval image): real backprop through the merger
  LoRA/adapter parameters across `--epochs` passes. At the CLI's shared `--epochs` default (60),
  `--n-train-images 6 --n-test-images 3`, and the default two corruptions,
  `estimate_cost_usd("b", 6, 3, 60, n_corruptions=2)` ≈ the alignment cell (`4.5`s) plus
  `2·(6·60·2.5 + 3·0.5) = 2·901.5 ≈ 1803`s ≈ 0.5 GPU-hours ≈ **~$1 at $2/GPU-hour** for the two
  default corruptions (the per-corruption multiplier is now included in the printed estimate, not
  a gap to correct for by hand). Reaching the
  **tens-of-dollars** range this section's title refers to requires deliberately scaling up
  `--n-train-images`/`--epochs` beyond the tiny diagnostic defaults (e.g. dozens of images and
  hundreds of epochs) for a more thorough gate run — the script's own module docstring is
  explicit that its per-cell diagnostic run "should cost a small fraction of" the tens-of-dollars
  ceiling by design, not consume the whole thing.
- `check_budget` **aborts before any GPU spend** if the projected cost exceeds
  `--budget-cap-usd` (default `$40` — the same tens-of-dollars ceiling `RUNBOOK-GPU.md` section
  2.5 already uses for the full p16 merger curriculum, `BUDGET_CAP_USD_DEFAULT`'s own comment
  cites it directly) — this cap is a ceiling not to exceed, not a target cost; raise it
  explicitly only after reviewing the printed projected cost (which now already includes the
  per-corruption multiplier and the alignment cell), never silently.

**Reading the result:** per the script's own `HONEST_CAVEAT_TEXT` — a trained readout head is
not the language model. Design-B success (error at/below the RS budget, or clearly below the
stock frozen-merger baseline of 65.5–73.6%) means the palette=16 signal *can* be made
recoverable at the LM boundary by a cheaply-trained merger, which reopens the fine-tune question
(see section 2.5's "If (d)/(e) pass" subsection) — it does **not** prove a zero-shot LM already
uses those symbols, or that the economics ultimately win. If Design B stays at or above the
frozen-merger baseline despite training, the negative result is **stronger** than Step 0's own
merged-probe fail: even a cheaply-trained, code-aware merger cannot carry the signal, scoped to
Qwen2.5-VL.

### Session-3 verdict (2026-07-12, RunPod RTX 4090, CUDA/bf16, 3B — raw artifacts in `gpu_gonogo_out/`)

**This section HAS now been run against real weights. Verdict: NO-GO — the merger-adapter
branch closes as a measured negative.** Every run below independently re-measured the
alignment gate and reproduced the committed pre-merger number **exactly** (0.13442 vs
target 0.1344) — also the first GPU/bf16 reproduction of `probe_report_premerger.md`'s
CPU/fp32 headline, so version/dtype drift is ruled out for the whole session. The
post-merger negative reproduced too (clean 0.7358, within the committed 0.66–0.74 band).

The measurement ladder (palette=16, test symbol error; RS budget 0.0627, chance 0.9375):

| # | measurement | config | clean | jpeg_q70 |
|---|---|---|---|---|
| 1 | pre-merger linear (info in the quad, frozen) | reference | 0.1344 | 0.1902 |
| 2 | Design A: nonlinear readout of merger INPUT | 6/3 imgs | 0.4916 (train 0.000) | 0.4871 |
| 3 | Design A, more data | 24/8 imgs | **0.2464** (train ~0.000) | 0.2680 |
| 4 | Design B1: LoRA r8 merger + joint head | 20 ep, 24/8 | 0.5547 | 0.8849 |
| 5 | Design B1, escalated | r32/α64, 80 ep, 48/12 | **0.3919** | **0.4100** |
| 6 | Design B2: parallel adapter around frozen merger | 80 ep, 48/12 | **0.3752** | 0.9075 |
| — | stock frozen merger, linear (post-merger) | reference | 0.6551–0.7358 | — |

How the pre-committed rules read against this:

- **Design A cleared its gate** — the merger input carries the symbols (rows 2→3: the
  default-config 0.49 was pure small-sample overfit; 4× the data halved it to 0.2464 with
  train error still ~0, i.e. 0.2464 is itself a data-limited floor-not-a-ceiling). So the
  "info already gone at the merger input → stop" branch did NOT fire; Design B was run.
- **Design B beat the stock baseline but nothing else.** Two independent mechanisms — a
  low-rank delta inside the merger MLP (B1) and fresh full-rank parallel capacity around
  it (B2) — plateau at **0.375–0.392 clean**, agreeing with each other. That is: (a) **6×
  above the 0.0627 RS budget** (no decodable channel — Reed–Solomon at nsym=32 corrects
  nothing at this error rate); (b) **worse than Design A's own 0.2464 frozen-input floor**
  (the trainable merger never even matched a plain MLP reading its frozen input); (c)
  **worse than the 0.1344 frozen pre-merger LINEAR readout** — so per section 2.5's
  pre-committed "must beat 86.6% accuracy" rule, the trained merger added nothing a cheap
  linear probe on frozen weights didn't already show.
- **Honest rule-letter note:** section 7's weaker success clause ("clearly below the stock
  frozen-merger baseline") technically fired — rows 5/6 sit well under 0.6551, so training
  the merger demonstrably recovers *some* coarse signal. Recorded, not overturned: the
  clause exists as a proxy for "the signal can be made recoverable at the LM boundary",
  and the budget/floor comparisons above measure directly that it cannot — a ~0.38
  symbol-error channel carries no byte-exact payload regardless of downstream cleverness.
- **Run-quality caveats, so the negative is not overstated:** row 4 (r8/20ep) was
  under-optimized (jpeg near chance, worse than stock) and is superseded by row 5; row 6's
  jpeg cell (0.9075) is an optimization collapse of the B2 variant, not channel evidence —
  B1's stable 0.4100 is the trustworthy corruption number. Design A's floor is
  conservative (still data-limited), so "info present at the input, unextractable by a
  cheap trained merger" is if anything understated.

**Consequences:** the P=16-retargeted curriculum (section 2.5, `--curriculum p16_merger`)
is NOT justified — its premise was that a trainable merger could approach the pre-merger
signal, and the direct cheap test measured it stuck ~3× above even the frozen linear
number. The scoped-to-Qwen2.5-VL negative chain is now complete and measured end to end:
density bar (model-free) → frozen tower discards structure (Step 0) → scale doesn't rescue
it (7B) → the loss localizes to the merger (2.5c) → **and a cheaply-trained, code-aware
merger cannot recover it either (this section)**. Remaining unmeasured escalations (full
merger retrain / vision-block fine-tune / learned encoders) are outside this project's
cheap-gate scope and are documented as such in `docs/FINDINGS.md` §5.

## What to bring back into the repo

1. `heliogram/data/base64_baseline.json` + `heliogram/data/text_baselines.json` (step 1)
2. regenerated `RESULTS.md` / `results.csv` (step 1)
3. `probe_report.md` + `probe_report.json` (step 2) — commit these verbatim
4. the zero-shot floor numbers (step 3), in whatever file the write-up lands in
5. if step 4 ran: per-stage accuracy logs and the final adapter's measured
   symbol-error/decode-success table — numbers, not the checkpoint's existence, are the
   result
6. if section 7 (merger-adapter go/no-go) ran: both designs' full report dicts (config,
   alignment-check result, per-corruption symbol errors, and the printed cost estimate) —
   commit these the same way `probe_report*.md`/`.json` are committed, verbatim, whichever way
   the result comes out.
