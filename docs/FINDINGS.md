# heliogram: findings

**For the full-length, paper-style version of this document — background and
related work, methods, and discussion sections — see
[`docs/writeup.md`](writeup.md).**

This document summarizes what heliogram actually measured, in one place, for
readers who want the verdict without the sweep tables. It invents no numbers:
every quantitative claim below cites the file it comes from. It is a defensive-
security and measurement writeup of completed work — it does not propose,
design, or improve any offensive capability, and none of the instruments it
describes are modified here.

## 1. Summary

heliogram's primary contribution is defensive-security tooling: a structural,
model-free pre-ingest detector (`heliogram.instruments.foreign_tile`) for
heliogram-like payloads, a written threat model (`docs/THREAT-MODEL.md`) for
the covert-channel risk a capability-facing optical codec would pose, and a
reusable pre/post-merger information-localization probe method — all shipped
and measured independent of, and before, any capability-facing (fine-tuning)
work (§4). The secondary question that motivated building those instruments
in the first place was economic, and it is reported in full below, honestly,
as a negative result: it does not become less true for being secondary.

heliogram tested one economic hypothesis: that a patch-aligned optical codec —
solid-color 14px blocks, one symbol per ViT patch, Reed–Solomon error
correction — could pack more payload bits per vision token than base64/ascii85
text costs per text token, making encoded images a cheaper context medium for
self-hosted VLMs (target: Qwen2.5-VL). The hypothesis was measured, not
assumed, on two independent tracks: a model-free channel/harness sweep (Phase
1, this repo, `RESULTS.md`), and a frozen-vision-tower linear probe run on real
Qwen2.5-VL weights (Phase-2 Step 0, `probe_report*.md`). **Both measurements
say no.** The codec's own physical ceiling for the only VLM-meaningful regime
(one symbol per patch) sits below the measured text baseline, and — more
fundamentally — the probes show the model's own vision pipeline actively
discards this kind of structure before an LM ever sees it. The project also
shipped two defensive instruments (`heliogram.instruments.foreign_tile`, a
structural pre-ingest detector, and `heliogram.instruments.injection_bench`, a
behavioral-payload benchmark harness) before doing any capability-facing work,
and wrote a release gate in `README.md` conditioning any future fine-tuned
reader on measuring both a behavioral-capacity number and a detector TPR/FPR
number together. The behavioral-attack side of that gate has not been measured
here — no tuned reader exists in this repo — so what follows is a threat model
and a detector, honestly labeled as such, not a demonstrated exploit.

## 2. The economic result (negative)

The harness (`python -m heliogram.harness`, output in `RESULTS.md` and
`results.csv`) sweeps palette (2–256 colors), `subpatch` (1 or 2), payload size
(48B–16KB), and a realistic corruption suite (resize, JPEG q70/85/95,
crop/pad, and the target model's own `smart_resize` preprocessing) and reports
**effective bits/patch** — payload bits divided by total grid patches, on
successful Reed–Solomon-verified decode only.

The only VLM-meaningful regime is `subpatch=1` (one symbol per one nominal
vision token); `subpatch>1` is a pixel-decoder-only geometric ceiling with no
evidence a real ViT can resolve sub-patch structure (`README.md`, "Capacity
sweep"). For `subpatch=1`, the hard architectural ceiling is `log2(palette)`,
capped at exactly 8 bits at `palette=256`; Reed–Solomon and calibration-row
overhead push the achievable **net** ceiling, as payload size grows without
bound, to `log2(256) × 223/255 ≈ 6.996 bits/patch` (`README.md`, "Capacity
sweep"; `RESULTS.md`, Headline table — the measured max observed in the sweep
is 6.827 bits/patch at palette=256, 16KB). That ceiling is a mathematical
consequence of the code's own framing/ECC overhead, not a fitting artifact.

The text-context bar it needs to beat was measured, not assumed:
`heliogram.baselines.measure_text_encoding_baselines` ran the real Qwen2.5-VL
tokenizer (`transformers==5.13.0`) against base64/ascii85/base85/hex samples
and persisted the result to `heliogram/data/text_baselines.json`. The
strongest text encoding measured is **ascii85 at 8.374 bits/token**
(base64 measures 8.096 bits/token; base85 8.178; hex 4.534 —
`heliogram/data/text_baselines.json`). Since 6.996 < 8.096 < 8.374, no
`subpatch=1` configuration can beat the text bar on a per-patch density
accounting, against either baseline — and `RESULTS.md` confirms 0 of 64 swept
`subpatch=1` configs beat even the weaker base64 bar clean, and none can by
construction at any payload size (`RESULTS.md`, "Headline"; `README.md`,
"Capacity sweep" and "Roadmap / Phase-2 boundary"). An exact byte-granular
token-count scan up to 64KB likewise found no `subpatch=1` palette crossing
below base64 token count on per-patch accounting anywhere in range — the best
ratio observed is 1.16, i.e. 16% *more* tokens than base64 (`RESULTS.md`,
"Token crossover: the actual measured benefit"). The only route under which
heliogram's token accounting crosses below base64 at all requires assuming a
model can read four patch-symbols out of one 2×2-merged LM token embedding —
an explicitly **unverified** assumption the README flags as "the same
epistemic class as the sub-patch caveat" and "the entire load-bearing wall of
the project's economic case" (`README.md`, headline paragraph; `RESULTS.md`,
"LM-token accounting" section) — and that is exactly the question Section 3
below reports a direct measurement against.

**The `smart_resize` robustness finding.** Because the operator-controlled
in-scope deployment cannot opt out of the target model's own image
preprocessing, the harness added Qwen2.5-VL's `smart_resize` (which snaps
input resolution to 28px multiples before the ViT ever sees a pixel) as a
measured corruption row. Under the *default* encoding, several palettes that
had cleared the density gate in an earlier revision (`palette=8`/`subpatch=2`)
were newly measured to fail this corruption at every payload size, because
their default grids land on an odd patch dimension and the 28px snap resamples
data rows off the symbol lattice — decode success drops to 0.00 for exactly
those rows (`README.md`, "Capacity sweep", Bar B verdict). The fix,
`encode(..., align=2)`, rounds the grid to even patch dimensions before
layout, making the `smart_resize` snap a no-op with zero wire-format change;
this is implemented and pinned by test (`tests/test_smart_resize.py`,
`spec/format-v0.1.md` §6), and re-encoding the affected configs with
`align=2` was measured to survive the full corruption suite at 1KB (9.752
clean bits/patch) and 4KB (10.089), clearing the 8.0-bit Gate #1 bar again,
while the 16KB grid separately exceeds the stock processor's ~1-megapixel
budget and needs the operator to widen `max_pixels` (`README.md`, "Capacity
sweep", Bar B verdict). This is a real, measured robustness finding about the
codec's interaction with the target model's mandatory preprocessing — not
merely a caveat, since it changed which configs clear the gate and shipped
with a working fix.

**Cost asymmetry and a conditional bit-cost.** Two further honest, model-free
notes (`heliogram.benefit`, `python -m heliogram.benefit`). First, equal
*token count* is not equal *compute cost*: an image token's merged embedding
only exists after a full vision-tower forward pass and its own activation
footprint, cost a same-length text prompt never pays, so the token-count
comparisons above answer "how many tokens," never "how much compute"
(`cost_asymmetry_points`, a structural argument with no invented FLOP or
latency figure). Second, an assumption-gated *effective cost per recovered
bit*: **if** a post-merger reader achieved the RS correction budget's own
error rate (an optimistic anchor, never observed on any real reader), a
6000-byte binary payload at `palette=256` costs 0.149 tokens per recovered
bit (measured just now via `python -m heliogram.benefit`); at the
chance-level error rate the probe (§3) actually measured post-merger, the
same arithmetic returns an undefined (infinite) cost — zero bits recovered.
This is a conditional projection under an assumption the stock tower does
not realize, not a second economic result; §3's probe remains the real,
measured verdict on the stock tower.

## 3. The mechanism, from the probes

The harness above only measures a model-free reference pixel decoder — it
cannot say whether a real Qwen2.5-VL vision tower even carries this
information as far as the language model. Phase-2 Step 0
(`scripts/run_probe.py`, `heliogram/probe.py`) closes that gap directly: it
pushes heliogram grids through the stock, frozen Qwen2.5-VL vision tower and
trains a linear probe from token embeddings to the ground-truth patch symbols.
A probe at/below the Reed–Solomon symbol-error budget means the information
survives to the point the probe reads from; a probe at chance means no
linearly-decodable signal survives to that point — not that no fine-tune of
the layers downstream could recover it: a nonlinear or higher-capacity probe
could differ (module docstring intent, echoed in every probe report's
header; see §5's existing nonlinear-probe caveat, which this is consistent
with, and `scripts/train_merger_adapter.py`, a scaffolded, not-yet-run test
of exactly this question at the merger).

Four probe runs are in the repo, at two tap points in the model (post the 2×2
merger — the default report; and, in `probe_report_premerger.md`, before the
merger, directly on ViT block output) and two model sizes:

| Report | Tower | Tap point | Palette | Corruption | Probe symbol error | RS budget | Chance | Verdict |
|---|---|---|---:|---|---:|---:|---:|---|
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 16 | clean | 0.7358 | 0.0627 | 0.9375 | at/near chance |
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 128 | clean | 0.9245 | 0.0627 | 0.9922 | at/near chance |
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 256 | clean | 0.9070 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_7b.md` | Qwen2.5-VL-7B | post-merger | 16 | clean | 0.6551 | 0.0627 | 0.9375 | at/near chance |
| `probe_report_7b.md` | Qwen2.5-VL-7B | post-merger | 256 | clean | 0.8979 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_easy.md` | Qwen2.5-VL-3B | post-merger | 2 | clean | 0.1831 | 0.0627 | 0.5000 | above RS budget, far below chance — partial signal |
| `probe_report_easy.md` | Qwen2.5-VL-3B | post-merger | 4 | clean | 0.3329 | 0.0627 | 0.7500 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | **pre-merger** | 16 | clean | **0.1344** | 0.0627 | 0.9375 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 16 | jpeg_q70 | 0.1902 | 0.0627 | 0.9375 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 256 | clean | 0.8081 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 256 | jpeg_q70 | 0.8193 | 0.0627 | 0.9961 | at/near chance |

Reading this table with the two tap points side by side is the whole finding:

- **Post-merger, every palette on both 3B and 7B is at/near chance** —
  `palette=16` gets closest (0.6551–0.7358 symbol error against 0.9375
  chance), but is still nowhere near the 0.0627 RS budget a real decode would
  need. The merged-token embedding a language model actually consumes does not
  linearly carry these symbols, at either model size tested.
- **Pre-merger tells a sharper story: it depends on palette.** At `palette=16`
  (4 bits/symbol), the pre-merger ViT-block output gives a probe symbol error
  of **13.4%** clean — still above the 6.27% RS budget (so not directly
  decodable end-to-end via a linear readout) but "far below chance," i.e. a
  real partial signal the vision blocks themselves preserve, that survives
  degrading only modestly under `jpeg_q70` (19.0%). At `palette=256` (8
  bits/symbol), pre-merger is already at/near chance (80.8% clean, 81.9%
  jpeg_q70) — the linear probe finds no linearly-decodable trace of this much
  finer color structure at the vision blocks themselves, before the merger
  ever runs (linear-probe scope, per §5; a nonlinear readout at this tap point
  is untested).
- **The 2×2 merger MLP is what finishes off the palette=16 signal.** The same
  16-color code that survives the vision blocks at 13.4% error is back at
  65.5–73.6% error (still at/near chance) once read post-merger on 3B/7B. The
  merger's job is to fold 4 ViT-patch embeddings into 1 language-visible
  token; that fold is measured here to erase most of the linearly-recoverable,
  fine-grained color-symbol information the blocks upstream still held.

> **Session addendum (single CPU/fp32 run, unreplicated — pending GPU confirmation).**
> The committed **13.4%** pre-merger `palette=16` clean number was measured at
> `scripts/run_probe.py`'s default probe config (6 train / 3 test images). A
> higher-probe-data follow-up this session (Qwen2.5-VL-3B, **CPU, float32**, 48
> train / 12 test images) measured the *same* pre-merger clean cell at **0.0700**
> (train 0.0322) — essentially **at** the 0.0627 RS budget — with everything but
> the probe's training-set size held fixed (the same box measured 0.1322 at 6
> images). This indicates the committed 0.1344 was **undertrained**, not a
> tap-point ceiling: the vision blocks preserve the per-patch `palette=16` signal
> *nearly to the decode threshold*, which sharpens (does not overturn) the
> localization above — the loss is concentrated even more squarely at the 2×2
> merger (≈0.07 pre-merger → 0.66–0.74 post-merger). This is **one CPU/fp32 run**
> (`gpu_gonogo_out/probe_pre_big.json`); it does not yet revise the headline
> `probe_report*.md` numbers, which stand until an independent GPU run reproduces
> the ~0.07. (The *reference-config* 0.1344 itself IS now GPU-confirmed: the
> 2026-07-12 RTX 4090 CUDA/bf16 session reproduced it exactly — 0.13442 — five
> independent times as the merger-adapter runs' alignment gate, see
> `gpu_gonogo_out/`; the 48-image ~0.07 estimate remains CPU-only/unreplicated.) The same run's Design-A nonlinear quad readout landed at 0.157
> (train 0.000) — still data-limited on the harder joint 4-symbol task, i.e. a
> floor-not-a-ceiling, consistent with the 0.07 per-patch estimate.
- **`palette=2`/`4` (the easiest, lowest-bit-depth codes) leave a comparable
  partial-but-undecodable signal even post-merger** (`probe_report_easy.md`:
  18.3%/33.3% error against 50%/75% chance) — consistent with the same
  mechanism: coarser symbol codes survive further into the pipeline, finer
  ones don't, and none of the tested codes reach the RS budget at the point a
  language model actually reads from.

The unifying read: a VLM's own perception is a strong learned prior over
natural images, and it visibly discards machine-dense, out-of-distribution
color structure at multiple stages — first partially in the vision blocks
(more so as color depth rises), then decisively in the patch-merger MLP. This
single mechanism is simultaneously **why the compression scheme fails**
(there is no bit-exact channel left for a linear reader to exploit once the LM
actually sees the tokens) **and a measured, favorable fact for anyone worried
about human-opaque payloads slipping through image inputs**: on this
architecture and this hand-designed code, the representation a downstream
language model consumes does not linearly preserve the fine symbolic
structure a covert channel would need.

## 4. Defensive contribution

Independent of the economic result, the project shipped two defensive
instruments deliberately *before* any capability-facing (fine-tuning) work —
"detector before capability" (`README.md`, "Phase-2 safety release gate").

- **`heliogram.instruments.foreign_tile`** is a lightweight, model-free
  pre-ingest guard: `guard(img, allowlist)` flags an image carrying a
  heliogram-like payload that is not on a trusted allow-list, before that
  image ever reaches a decoder or model. Its design rests on a structural
  signature (module docstring): a genuine heliogram tile tiles evenly into
  patch-size cells that are each near-solid color, which natural photos
  essentially never exhibit by accident; `patch_structure_score` is the cheap
  first filter, and only images that pass it face the decisive second check —
  whether they decode under any of the caller's trusted `(palette,
  patch_size, subpatch)` configs. Anything patch-structured that decodes under
  none of them is flagged as foreign. `evaluate_detector` reports true-positive
  rate at a bounded false-positive rate, using "learned-alphabet" tiles (data
  cells repainted through a permuted palette, specifically constructed to
  defeat the reference decoder's calibration-from-row-0 recovery) as the hard
  positive case — the module is explicit that this is a structural/decode-
  based classifier, not a learned one, and contains no model import at any
  scope (`heliogram/instruments/foreign_tile.py` module docstring). Measured
  against its standard hard-positive set: TPR=1.000 (8/8) at FPR=0.000
  (0/16) (`python -m heliogram.instruments.foreign_tile`). **This detector's
  own blind spot is now measured, not merely predicted:** a texture-
  camouflage adversary — an otherwise-foreign tile dusted with seeded
  per-pixel noise so its cells still average to the right color but clear
  the cheap first filter's variance threshold — collapses that filter
  completely, measured at TPR=0.000 (0/4) against the same FPR=0.000 (0/16)
  negatives (`python -m heliogram.instruments.foreign_tile`, "texture-
  camouflage family eval"). Every texture-camouflaged positive is, by
  construction, still a heliogram-shaped payload nobody approved — it is
  simply waved through by the cheap structural filter before ever reaching
  the decisive decode-attempt stage. This is an honest, measured limitation
  of this particular defensive instrument, not a hypothetical one; see
  `docs/THREAT-MODEL.md` for the fuller adversarial framing this result
  informs.
- **`heliogram.instruments.injection_bench`** is the harness pointed at
  *behavioral* payloads (persona hijack, forced output schema, triggered tool
  call), split explicitly into two halves per the project's data-honesty
  rule: `measure_behavioral_capacity` requires a real model+processor+judge
  and raises immediately without one, rather than ever fabricating an
  influence number — it is not run anywhere in this repo (no GPU here); and
  `evaluate_defense`/`build_detector_eval_set`, the detector-evaluation mode,
  is pure CPU image-counting that scores a candidate defense (demonstrated
  against `foreign_tile.guard`) and is fully buildable and tested here. The
  module is explicit about the resulting coverage gap: `guard` is blind to
  decoded payload *content* by design, so it cannot by itself catch a
  behavioral payload encoded under a trusted, allow-listed config — a real
  limitation of that particular defense, not a bug (`heliogram/instruments/
  injection_bench.py` module docstring). All `BEHAVIORAL_PAYLOADS` fixtures
  are labeled synthetic test strings — fictional personas, invented tool
  names, JSON actions no real system implements — used only to probe channel
  capacity, not working exploits.
- **The written release gate** (`README.md`, "Phase-2 safety release gate")
  commits, in advance of any tuned reader existing, to publishing three things
  together before releasing any Phase-2 adapter or fine-tuned weights: (1)
  measured behavioral capacity against the actual tuned model via
  `measure_behavioral_capacity`; (2) `foreign_tile.guard`'s TPR at a bounded
  FPR measured against tiles the *tuned* model decodes, not only tiles the
  reference pixel decoder decodes; and (3) a decision rule, stated before
  there is any incentive to soften it, that no adapter is released if
  behavioral payloads survive corruption at rates the guard cannot catch at
  bounded FPR. The README is explicit that this is "a commitment about what
  gets published and what release is conditioned on, not a claim that we have
  already run it" — nothing in this repo has a GPU, so neither number has
  been measured yet. See `docs/THREAT-MODEL.md` for the fuller threat model
  (attacker capabilities, assets, and where each instrument sits in the
  pipeline) this release gate operationalizes.

**Honestly:** the behavioral-attack capacity itself has **not** been
measured — there is no tuned reader in this repo, and
`measure_behavioral_capacity` is designed to refuse to run without one. What
exists today is a threat model, a structural detector with a measured
detection methodology, and a behavioral-benchmark harness ready to run against
a future model — not a demonstrated exploit.

## 5. Honest limitations

- The probe measures the **linear** readability of embeddings for **one**
  hand-designed code (solid 14px-block symbols, a fixed deterministic
  palette, one calibration row). It is a de-risking signal, not a proven
  capacity bound: a nonlinear probe, or a probe with more capacity or more
  training data, could recover more than a linear readout does at the same
  tap point (probe report headers: "a PASS here means the frozen embeddings
  linearly carry the symbols — it de-risks, but does not replace, the
  fine-tune"). That nonlinear readout is now implemented and unit-tested —
  `scripts/run_probe.py --probe-head mlp` (a one-hidden-layer MLP head,
  `heliogram.probe.fit_mlp_probe`) — but has **not** been run against real
  weights here (no GPU); a probe, linear or not, is still not the LM using
  the symbols.
- A **fine-tuned** adversary — one that trains the vision tower or merger
  itself to preserve this structure, rather than reading frozen embeddings —
  is a different, unmeasured question; the probes only characterize what the
  stock, frozen weights already preserve. The cheapest version of that
  question — can training only the merger (ViT and LM left frozen) recover
  the `palette=16` signal the merger is measured to erase — is now DESIGNED
  and staged as a refuse-without-model scaffold (`scripts/train_merger_adapter.py`:
  Design A, a frozen-feature nonlinear readout diagnostic; Design B, an
  actual trainable-merger LoRA/adapter gate), reusing the probe's exact
  window-shuffle alignment code and re-checking against the committed
  `probe_report_premerger.md` numbers before trusting anything new. **This has
  now been run against real weights** (2026-07-12, RTX 4090, CUDA/bf16, 3B;
  raw artifacts in `gpu_gonogo_out/`, full reading in `RUNBOOK-GPU.md`
  "Session-3 verdict") **and the answer is no**: Design A confirms the merger
  input still carries the symbols (nonlinear quad readout 0.2464 clean at
  24 train images, train error ~0, i.e. a data-limited floor), but the
  trainable merger itself — both a rank-32 LoRA delta (B1) and a parallel
  full-rank adapter (B2) — plateaus at **0.375–0.392 clean symbol error**:
  below the stock frozen-merger baseline (0.6551–0.7358, so training does
  recover *coarse* signal), but ~6× above the 0.0627 RS budget, above Design
  A's own frozen-input floor, and above even the 0.1344 frozen pre-merger
  *linear* readout. A cheaply-trained, code-aware merger cannot carry the
  `palette=16` signal to the LM boundary, scoped to Qwen2.5-VL. The
  still-unmeasured escalations shrink to: a full (non-cheap) merger retrain,
  or fine-tuning the vision blocks themselves.
- A **different code** (different palette design, different patch geometry,
  a learned encoding rather than a hand-designed one) could interact with the
  vision blocks and merger differently; nothing here bounds codes not tested.
- Every number is specific to **Qwen2.5-VL** (3B and 7B variants tested,
  `transformers==5.13.0`). Nothing here transfers to other model families or
  to closed API models, which the project explicitly places out of scope
  because their preprocessing is opaque and changeable (`README.md`,
  "Scope").

## 5b. The typography alternative, also measured dead (for a different reason)

The natural pivot from the failed color codec is to render the payload as dense
typeset text — the channel VLM towers demonstrably *do* read (DeepSeek-OCR,
Glyph). `heliogram.typography` first confirmed the geometric prerequisite:
RS-framed ascii85 text on the 14px patch grid clears the 8.374 bits/token bar
from a 12px font down (assuming perfect legibility). `heliogram.ocr_eval` +
`scripts/run_typography_ocr.py` then measured the readability that geometry
assumes away, running a stock Qwen2.5-VL zero-shot over the rendered images
(28px-aligned so the tower's own `smart_resize` is the identity — an earlier
run that skipped this alignment measured a resampled image and was discarded).

Measured (7B, zero-shot, RS-framed, 256B payload, mean character error rate):

| font | CER | reads? | geom bits/patch | beats 8.374? |
|---:|---:|---|---:|---|
| 28px | 0.041 | yes (~96%) | 1.94 | no |
| 24px | 0.074 | yes | 2.56 | no |
| 20px | 0.152 | mostly | 3.71 | no |
| 16px | 0.370 | poorly | 5.84 | no |
| 14px | 0.467 | badly | 7.42 | no |
| 12px | 0.595 | no | 9.75 | **yes** |

The tower **can** OCR rendered ascii85 — 96% character accuracy at 28px — so
this is not a perception failure like the color codec. It is a geometric one:
**readability and density are inversely coupled through font size and cross
below the economic bar.** The sizes the tower reads (≥24px) sit at ~2 bits/patch,
roughly 4× *below* the bar; the only size that beats the bar (12px) is
illegible (60% CER). Even a perfect-decoding encoding at a readable size would
cost more tokens than sending the text itself. No encoding cleverness moves the
geometry — only a model that reads far smaller text would, and this run
measures that the stock tower does not. This is the same reason optical
compression pays off for redundant *prose* (where the LM prior fills OCR gaps)
but not for the high-entropy arbitrary bytes heliogram targets, where every
character must be exact. (`decode_success` was 0% at every size, but that
number is dominated by ascii85's positional brittleness — one OCR
insertion/deletion defeats Reed–Solomon — and is *not* the reason for the
verdict; the readability/density crossing is, and it holds regardless of
encoding.)

## 6. What would make this a stronger result

The single measurement that would resolve the open question on both sides —
the economic one (can a fine-tuned reader realize the LM-token accounting
route despite the frozen-tower probe's negative signal?) and the safety one
(is the detector still adequate once there is something real to detect?) — is
one measured behavioral run against a tuned reader, executed exactly as
`README.md`'s Phase-2 safety release gate already specifies: run
`measure_behavioral_capacity` against the actual fine-tuned model and publish
whatever reliable behavioral influence survives the corruption suite,
alongside `foreign_tile.guard`'s TPR/FPR measured against tiles that same
tuned model decodes (not only tiles the reference pixel decoder decodes), and
apply the pre-committed decision rule to the result. That run requires a GPU
this environment does not have and a fine-tuned reader that does not exist
yet; it is out of scope for this document and is not performed here.

A cheaper, intermediate step already existed to gate whether that full run is
even worth its cost: `scripts/train_merger_adapter.py`'s Design A/Design B
merger-only go/no-go (§5, above), settling whether a trainable merger can
recover the `palette=16` signal the frozen merger is measured to erase,
before spending the tens-of-GPU-hours a full behavioral fine-tune curriculum
would cost. **That gate has now been run (2026-07-12, ~$1–2 of GPU time; §5
above, `gpu_gonogo_out/`, `RUNBOOK-GPU.md` "Session-3 verdict") and came back
NO-GO** — the trainable merger plateaus ~6× above the RS budget — so the full
behavioral fine-tune run this section describes is not merely unfunded but
now measured to be unsupported: the cheap gate that was designed to spare its
cost did exactly that.
