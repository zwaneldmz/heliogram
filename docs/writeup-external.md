# heliogram: an external-replication writeup (optical byte codecs, a negative result, and a reusable localization probe)

> **STATUS: UNPOSTED DRAFT.** This document has not been published anywhere outside this
> repository. It is written for a reader outside the project who wants to independently
> replicate the measurements below — it assumes no prior context beyond what is stated here.
> Every number cited is read from a file already committed to this repository
> (`probe_report*.md`/`.json`, `RESULTS.md`, `docs/FINDINGS.md`); nothing is invented, and no
> GPU probe was re-run to produce this document. If you find a discrepancy between a number
> here and the cited file, trust the file and treat this document as stale.

## Why this document exists

heliogram is primarily a **defensive-security research artifact**: it exists to answer "can a
self-hosted vision-language model be fed a machine-dense, human-opaque image payload, and if
so, what would catch that at ingest time?" Answering that required first building the payload
this document's byte-economics question also happens to bound — so the same repository
contains (1) a structural pre-ingest detector and a measured threat model, and (2) a fully
honest, negative economic result about whether such payloads are even a *cheap* way to feed a
model data. This writeup is about (2), but it is offered to external readers because the
**method** — a reusable way to localize, inside any patch-merging vision-language model, where
a structured signal a covert channel would need gets destroyed — is useful independent of
which way the economics come out, including to someone building or evaluating the defensive
side.

## Abstract

**The question:** for a self-hosted, open-weight VLM (target: Qwen2.5-VL) where the operator
controls image preprocessing end to end, is a patch-aligned optical codec — solid-color 14px
blocks, one symbol per ViT patch, Reed–Solomon error correction — a cheaper way to hand the
model high-entropy, must-be-bit-exact bytes (a key, a hash, a binary blob) than sending the
same bytes as base64/ascii85 text?

**The answer, on two independent measurement channels, is no.** First, a model-free channel
sweep shows the codec's own error-correction-adjusted physical ceiling (6.996 bits/patch) sits
below the measured text-tokenizer baseline (ascii85 at 8.374 bits/token) — the codec cannot win
on density even before asking whether a real model can read it. Second, a frozen-tower linear
probe pushed real image grids through the real, frozen Qwen2.5-VL vision tower and found that
the merged, LM-visible token embeddings do not linearly carry the per-patch symbols at any
tested palette — post-merger probe error sits at/near chance everywhere. A follow-up probe
tapping the tower *before* the 2×2 patch-merger step localizes the loss: at the coarsest
practical palette (16 colors, 4 bits/symbol) the pre-merger vision blocks *do* preserve a real,
partial signal (13.4% symbol error, still above the RS-decodable budget but far below chance);
the merger step itself is what destroys most of it. At a finer palette (256 colors, 8
bits/symbol) even the pre-merger blocks have already discarded the signal.

**The reusable part:** independent of which way this particular codec's economics land, the
probe method — tapping a patch-merging VLM's vision tower on both sides of its merger boundary,
recovering the tower's internal window-shuffle permutation from its own outputs (no private
library internals required), and scoring a linear readout's error against an explicit,
error-correction-derived decision budget — is a general way to answer "does this tower's own
perception preserve structured, non-natural-image signal, and if not, at which layer is it
lost?" That question matters equally for someone chasing the economic angle and for someone
building a detector against it.

## Scope, and what this document is not

- Every number below is specific to **Qwen2.5-VL** (3B and 7B variants), measured against
  `transformers==5.13.0`. Nothing here transfers to other model families, other tower
  architectures, or closed/API-only models (whose preprocessing is opaque and out of scope by
  construction).
- All data used in every measurement is **synthetic and seed-deterministic** — no real secrets,
  no scraped or third-party content, no natural images. Every payload is generated
  programmatically from a fixed seed.
- The probe measures **linear** separability only. A pass means a linear readout recovers the
  symbols; a fail means a linear readout does not — it does not prove no computation whatsoever
  downstream could recover the signal. This scope limitation is stated explicitly in every
  probe report and is not new information invented for this writeup (see "Honest scope of the
  linear probe" below).
- This is a measurement and defensive-security writeup of **already-completed** work. It does
  not propose, design, or improve any offensive capability, and no instrument or result file is
  modified by writing this document.

## The measurement method, in enough detail to replicate

Three independent, composable measurements make up the full picture. All three are runnable
from a clean checkout; the first two need no GPU at all.

### 1. Model-free channel sweep (`heliogram.harness`)

`python -m heliogram.harness` sweeps palette size (2–256 colors), `subpatch` (1 or 2 sub-cells
per ViT patch), payload size (48B–16KB), and a corruption suite (bilinear resize, JPEG at
q70/q85/q95, crop/pad, and the target model's own mandatory `smart_resize` preprocessing). For
each configuration it encodes a synthetic payload into a heliogram grid, applies the
corruption, decodes with a deliberately dumb reference decoder (`decode_pixels`, which samples
only patch centers — this measures channel capacity, not decoder cleverness), and reports
**effective bits/patch**: payload bits divided by total grid patches (calibration row, RS
parity, and framing overhead all counted against it), computed only on successful,
Reed–Solomon-verified decodes. Output lands in `RESULTS.md` and `results.csv`. This step
requires no model weights and no GPU; it establishes the codec's own physical ceiling
independent of any VLM.

### 2. Text-tokenizer baseline (`heliogram.baselines`)

`python -m heliogram.baselines --measure` runs the **real** Qwen2.5-VL tokenizer
(`transformers==5.13.0`, needs Hugging Face Hub access to fetch the tokenizer) against
base64/ascii85/base85/hex encodings of synthetic byte samples (multiple sizes, multiple seeds)
and measures the actual achieved bits-per-token rate for each — not an analytic estimate. The
result is persisted to `heliogram/data/text_baselines.json`. This is the bar the optical codec
has to beat on a per-token/per-patch accounting to be worth using at all.

### 3. Frozen-tower linear probe (`heliogram/probe.py`, `scripts/run_probe.py`)

This is the step that needs a GPU and the actual model weights, and it is the one this document
is really about. The question: after a heliogram grid passes through the real, **frozen**
(never fine-tuned) Qwen2.5-VL vision tower, do the resulting token embeddings still linearly
separate the ground-truth per-patch symbols?

- **Labeling.** `heliogram.probe.merged_token_labels` lays ground-truth symbols out in raster
  order over the merged token grid, matching Qwen2.5-VL's documented 2×2-merger output order.
  This ordering assumption is load-bearing and self-checking: if a real tower emitted tokens in
  a different order, the probe would score at chance *even on clean images* — a loud,
  unambiguous failure signature rather than a silently wrong number. Every probe report states
  this explicitly and none of the committed reports hit it.
- **The probe itself.** `heliogram.probe.fit_linear_probe` is plain multi-position softmax
  regression (no torch — pure numpy, CPU-testable in isolation), fit with deterministic
  minibatch SGD (fixed seed, no schedule tuning, no early stopping against the test set). One
  weight matrix maps embedding → `K × P` logits, where `K` is the number of original patch
  positions folded into one merged token (`K=4` for Qwen2.5-VL's 2×2 merger) and `P` is the
  palette size. Standardization statistics are computed on the training split only.
- **Two tap points.** The tower is probed at two points: **post-merger** (the actual
  LM-visible, merged-token embeddings — this is the direct go/no-go for whether an LM-token
  accounting scheme could ever work) and **pre-merger** (per-patch hidden states at the vision
  blocks' output, before the 2×2 merge — this localizes *where* in the pipeline a post-merger
  failure originates: in the vision blocks themselves, or in the merger step).
- **Window-shuffle-permutation recovery by exact row-matching.** Qwen2.5-VL's vision tower
  internally reorders patches into a window-attention shuffle before running its blocks, and
  undoes that shuffle after the merger. To probe the *pre*-merger state in the tower's true
  raster order, the pre-merger extraction code must know and invert this permutation — without
  importing any private `transformers` internals to do it. The technique (`_match_reverse_indices`
  in `scripts/run_probe.py`): the tower's public `pooler_output` is exactly the merger's raw
  output rows, permuted (`pooler_output = merger_out[reverse_indices]`). Matching each pooled
  row to its byte-identical row in the raw merger output — by literal `.tobytes()` equality,
  not by index arithmetic — recovers `reverse_indices` exactly. This is version-robust (it works
  for any tower where the returned merged rows are a permutation of the merger's raw output
  rows, including the identity permutation) and fails loudly: it raises on any duplicate,
  unmatched, or non-permutation row rather than silently returning a plausible-but-wrong
  ordering. Duplicate merged rows are effectively impossible for real inputs because rotary
  position embeddings make every token position-distinct, so hitting that guard means something
  is deeply wrong with the run, not a benign edge case.
- **Decision budget.** A probe's symbol error is compared against two reference lines:
  `chance_error = 1 - 1/P` (the probe learned nothing) and the Reed–Solomon symbol-error budget
  (`rs_symbol_error_budget`, ≈6.27% for the default 32 parity bytes — the sustained per-symbol
  error rate below which end-to-end decode succeeds). A probe at or below the RS budget means
  the information survives, linearly readable, to that tap point. A probe at/near chance on
  *clean* images means no linearly-decodable per-patch signal survives that far — and, given the
  ordering self-check above, that failure is attributable to the tower's own representation, not
  a labeling bug.
- **Model-interface contract, CPU-verified.** The exact attribute path to the vision tower
  (`model.model.visual`, not `model.visual`, in `transformers==5.13.0`), which output field
  actually carries the merged tokens (`pooler_output`, not `last_hidden_state`, which is
  pre-merger and window-shuffled), and the full pre-merger row-ordering chain are all verified
  against a real (tiny, random-weight) Qwen2.5-VL instantiated from a local config in
  `tests/test_probe_contract_cpu.py` — this CPU-only test caught the `model.visual` and
  `pooler_output` mistakes before any GPU time was spent, and it is the test to run first on any
  new environment before trusting a real run's numbers.

Replication command (needs a CUDA GPU and Hugging Face Hub access to the Qwen2.5-VL weights):

```bash
pip install -e ".[gpu]"          # or: pip install -e . -r requirements-gpu.txt
pytest -q tests/test_probe_contract_cpu.py   # verify the model-interface contract first

# post-merger (the LM-boundary go/no-go)
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --palettes 16,128,256 --corruptions clean,jpeg_q85,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report.md --json probe_report.json

# pre-merger localization
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 16,256 --corruptions clean,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report_premerger.md --json probe_report_premerger.json
```

## The key committed results

All numbers below are read directly from committed files; each row cites its source.

**Economic ceiling vs. the measured text bar** (`README.md` "Capacity sweep"; `RESULTS.md`
"Headline"): the codec's architectural ceiling for the only VLM-meaningful regime
(`subpatch=1`) is `log2(palette)` bits/symbol, reaching 8 bits only at `palette=256`;
Reed–Solomon and calibration-row overhead cap the achievable net ceiling at
`log2(256) × 223/255 ≈ 6.996 bits/patch` (measured max observed: 6.827, at palette=256/16KB).
The measured text baseline (`heliogram/data/text_baselines.json`): **ascii85 at 8.374
bits/token** (base85 8.178, base64 8.096, hex 4.534). Since 6.996 < 8.096 < 8.374, no
`subpatch=1` configuration can beat any measured text baseline on a per-patch/per-token density
accounting.

**Post-merger linear probe (`probe_report.md`, `probe_report_7b.md`):**

| Report | Tower | Palette | Corruption | Probe symbol error | RS budget | Chance |
|---|---|---:|---|---:|---:|---:|
| `probe_report.md` | Qwen2.5-VL-3B | 16 | clean | 0.7358 | 0.0627 | 0.9375 |
| `probe_report.md` | Qwen2.5-VL-3B | 128 | clean | 0.9245 | 0.0627 | 0.9922 |
| `probe_report.md` | Qwen2.5-VL-3B | 256 | clean | 0.9070 | 0.0627 | 0.9961 |
| `probe_report_7b.md` | Qwen2.5-VL-7B | 16 | clean | 0.6551 | 0.0627 | 0.9375 |
| `probe_report_7b.md` | Qwen2.5-VL-7B | 256 | clean | 0.8979 | 0.0627 | 0.9961 |

Every post-merger cell, on both model sizes, is at/near chance. `palette=16` gets closest
(65.5–73.6% error against 93.75% chance) but remains nowhere near the 6.27% RS decode budget.
Scaling 3B→7B does not rescue it.

**Pre-merger localization (`probe_report_premerger.md`), the split that matters:**

| Palette | Corruption | Probe symbol error | RS budget | Chance | Reading |
|---:|---|---:|---:|---:|---|
| 16 | clean | **0.1344** | 0.0627 | 0.9375 | above budget, far below chance — real partial signal |
| 16 | jpeg_q70 | 0.1902 | 0.0627 | 0.9375 | above budget, far below chance — degrades modestly |
| 256 | clean | 0.8081 | 0.0627 | 0.9961 | at/near chance — already lost pre-merger |
| 256 | jpeg_q70 | 0.8193 | 0.0627 | 0.9961 | at/near chance — already lost pre-merger |

Comparing tap points at `palette=16` isolates exactly where the signal dies: 13.4% error
pre-merger versus 65.5–73.6% error post-merger on the same code, same model family — the 2×2
merger MLP is what destroys most of the linearly-recoverable signal the vision blocks upstream
still carried. At the finer `palette=256`, the vision blocks themselves have already discarded
the signal before the merger ever runs, so no merger-side intervention could help there
regardless.

**Supporting easy-mode check (`probe_report_easy.md`):** at the coarsest possible codes
(`palette=2`, chance 50%; `palette=4`, chance 75%), post-merger probe error is 18.3% and 33.3%
respectively — still above the RS budget, but confirming the pipeline, labels, and ordering
work end to end, and that coarser codes survive further into the pipeline than finer ones,
consistent with the palette=16-vs-256 pre-merger split above.

## Honest scope of the linear probe

Every number above characterizes **linear** separability of a **frozen** tower's embeddings for
**one** hand-designed code (solid 14px-block symbols, a fixed deterministic palette, one
calibration row). This is a real, informative measurement, but it has a specific and stated
limit: **a nonlinear probe could recover more signal than this linear one does, at the same tap
point.** A clean-image FAIL under a linear probe is strong negative evidence — no
linearly-decodable per-patch signal survives to that point for a linear readout to exploit — but
it is not proof that no computation whatsoever, downstream, could ever extract the symbols.
Every committed probe report states this scope limit in its own header; it is not new
information added for this document.

The concrete next step this project staged to test that boundary — but **has not yet run** —
is `scripts/train_merger_adapter.py`, a "merger-adapter go/no-go." It is fully designed,
CPU-contract-tested, and documented, but requires a GPU that this environment does not have, so
it remains a designed-but-unexecuted scaffold, not a result. Concretely, it offers two designs:

- **Design A** (`--design a`): a cheap, GPU-light diagnostic. It re-uses the same frozen
  pre-merger embeddings the linear probe already extracted, groups them into the merger's own
  2×2 quads, and fits a small **nonlinear** (one hidden ReLU layer) readout head in plain numpy
  — no gradients through the tower at all. This tests whether a slightly higher-capacity readout
  recovers more than the linear probe already measured, at essentially the linear probe's cost.
- **Design B** (`--design b`): the actual gate. It freezes the vision blocks and the language
  model, and trains *only* the merger — either via LoRA on the merger MLP's two Linear layers
  (`mlp.0`/`mlp.2`, variant B1, preferred) or via a small parallel residual adapter around the
  frozen merger (variant B2) — jointly with a trained readout head, with gradients flowing back
  into the merger's own trainable parameters. This is the one experiment design that could
  actually test whether *training* the merger (not just reading it linearly, frozen) recovers
  the palette=16 signal the pre-merger probe shows the vision blocks still carry.

Both designs are scoped to `palette=16` only — `probe_report_premerger.md` already measured
`palette=256` at/near chance even pre-merger (80.8% clean), so no merger-side adapter, however
trained, could recover a palette that coarse a probe already showed the vision blocks discard
before the merger runs.

Before either design trusts a single new number, both first re-run the *exact* (palette=16,
clean, pre-merger) probe cell `probe_report_premerger.md` was generated from and check the
result against the committed 0.1344 within a fixed tolerance (`check_alignment_sanity`) — a
fresh GPU session whose transformers/peft version, model id, or window-shuffle handling has
drifted since that report was generated fails loudly here instead of silently reporting an
incomparable number.

This script is explicit, in its own module docstring, that a trained readout head is not the
language model: Design-B success would mean the information *can* be made recoverable at the LM
boundary by a cheaply-trained merger — reopening the fine-tune question — not that a zero-shot
LM actually uses those symbols, or that the economics ultimately win. It is necessary evidence,
not sufficient. Conversely, if Design B stays well above budget, the negative result becomes
*stronger* than the frozen linear probe's own fail: even a cheaply-trained, code-aware merger
could not carry the signal, not just a frozen, generically pretrained one.

## Defensive framing (why this matters beyond the economics)

Whatever the economic answer turns out to be, the same measurement infrastructure answers a
defensive question: does a self-hosted VLM's own perception preserve machine-dense,
human-opaque image structure well enough that it could carry a covert instruction past a human
reviewer and into the language model? The negative economic result here is, at the same time, a
favorable defensive finding for this architecture and this hand-designed code: the representation
a language model actually consumes does not linearly preserve the fine symbolic structure a
covert channel of this kind would need. Separately and independently of this economic question,
the project also ships a model-free, structural pre-ingest detector
(`heliogram.instruments.foreign_tile`) and a behavioral-capacity benchmark harness
(`heliogram.instruments.injection_bench`) built *before* any capability-facing fine-tuning work
was attempted, plus a written release gate conditioning any future capability work on measuring
both a behavioral-influence number and a detector TPR/FPR number together before anything is
published. None of that defensive tooling is modified, extended, or re-measured by this
document; see `README.md` ("Defensive contribution and safety gate") and `docs/FINDINGS.md`
(Section 4) for the full, separately-maintained account of that side of the project.

## Reproducing this document's claims yourself

```bash
git clone <this repo> && cd heliogram
pip install -e .
pytest -q                                  # full CPU test suite

# CPU-only, no GPU or model weights needed:
python -m heliogram.harness                # codec/corruption sweep -> RESULTS.md, results.csv
python -m heliogram.baselines --measure    # real-tokenizer text baselines -> heliogram/data/*.json

# GPU-dependent (CUDA GPU + HF Hub access to Qwen2.5-VL weights):
pip install -e ".[gpu]"
pytest -q tests/test_probe_contract_cpu.py             # verify the model-interface contract first
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --palettes 16,128,256 --corruptions clean,jpeg_q85,jpeg_q70 \
    --out probe_report.md --json probe_report.json
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 16,256 --corruptions clean,jpeg_q70 \
    --out probe_report_premerger.md --json probe_report_premerger.json
```

See `RUNBOOK-GPU.md` for the full ordered GPU procedure, including per-step decision rules,
the merger-adapter go/no-go's cost/time budget, and which artifacts to commit back after each
run.
