# Optical context codecs for arbitrary bytes: a negative result on two channels

*heliogram — full technical report. Companion to the one-page [`docs/FINDINGS.md`](FINDINGS.md),
which this document elaborates without repeating verbatim. Every quantitative claim below cites
the in-repo file it comes from; no number is invented or extrapolated beyond what that file
states. This is a measurement and defensive-security writeup of already-completed work — it does
not propose, design, or improve any offensive capability, and no instrument, result file, or probe
report is modified here.*

## Abstract

We tested whether a patch-aligned optical codec — solid-color 14px blocks, one symbol per ViT
patch, Reed–Solomon error correction — could make encoded images a cheaper context medium than
text (base64/ascii85) for self-hosted vision-language models, for arbitrary high-entropy byte
payloads rather than prose. The hypothesis treats one ViT patch as roughly one vision token when
the operator controls preprocessing, making bits/patch and bits/token directly comparable. Two
independent channels were measured, and both came back negative. The color codec's own
error-correction-adjusted ceiling, 6.996 bits/patch, sits below the measured text baseline
(ascii85, 8.374 bits/token); frozen-tower linear probes on real Qwen2.5-VL weights (3B, 7B) show
the model's patch-merger step leaves no linearly-decodable trace of the fine symbolic structure
the codec needs, even where the vision blocks upstream partially preserve it linearly (a
nonlinear probe was not tested and could differ — Section 7). A typography pivot — rendering the
payload as dense typeset ascii85 text, the kind of input VLM towers demonstrably do read (DeepSeek-OCR,
Glyph) — passes its geometric gate but fails a zero-shot OCR readability measurement: font sizes
the tower reads accurately sit roughly 4x below the economic bar in density, and sizes dense
enough to beat the bar are illegible. Both failures trace to one mechanism: optical compression
works for redundant prose, where a language model's prior fills OCR gaps, but not for high-entropy
arbitrary bytes, where every character must be exact. The project also shipped a defensive
contribution — a structural pre-ingest detector and a behavioral injection benchmark — before any
capability-facing work, honestly labeled as a threat model, not a demonstrated exploit.

## 1. Introduction

Self-hosted, open-weight VLMs are increasingly used as general-purpose text processors. An
operator who controls the full serving pipeline can, in principle, hand a model arbitrary encoded
data as "context," provided the vision tower carries it faithfully to the language model. This
raises an economic question: is it cheaper, in the units a serving system pays for, to send data
as image pixels or as text?

Each side has a well-defined unit of account. Text sent as base64/ascii85 costs some number of BPE
tokens, at an exact, measurable bits-per-token rate for a given encoding and tokenizer. A
self-hosted deployment where the operator controls preprocessing can be treated, to first
approximation, as costing roughly one vision token per ViT patch (14px square for Qwen2.5-VL,
before any patch-merging step folds several patches into one LM-visible token). If an encoding can
pack more bits into one patch than a text token costs, the image route uses fewer total tokens for
the same payload.

This report is about arbitrary bytes, not prose — a distinction that matters because the existing
"optical context compression" literature (Section 2) targets natural language, where a model's
own prior can paper over an imperfect optical read (a smudged character is often inferable from
context). An arbitrary binary blob, key, or hash has no such redundancy: every symbol must be
recovered exactly. heliogram's thesis was that a code *designed for the channel* — solid color
blocks aligned to the patch grid, with explicit ECC rather than implicit language-model denoising
— might reach a different operating point than typeset prose can for this harder payload class.

Two channels were tested:

1. **The color codec** — solid-color 14x14px blocks, one symbol per patch, a deterministic
   palette, Reed–Solomon ECC (Section 4.1–4.2).
2. **Typography** — rendering the same arbitrary-byte payload (after the same ECC framing) as
   small, dense typeset text, betting on the tower's pretrained OCR competence instead of a
   learned color classifier (Section 4.3).

Both were measured end to end — a model-free channel harness for the physical ceiling, and either
a frozen-tower linear probe (color codec) or a zero-shot OCR readability run (typography) for
whether a real model's own perception carries the signal that far.

## 2. Background and related work

heliogram's prior-art survey (`NOTES.md`) groups related work into three lines, plus a fourth
category — patch-aligned machine codes — where no direct prior work was found.

**Optical/visual-text compression.** DeepSeek-OCR ("Contexts Optical Compression," Oct 2025,
arXiv:2510.18234) is the empirical anchor: ~97% OCR precision at under 10x text-token-to-
vision-token compression of *rendered prose*, degrading to ~60% at 20x. Glyph (arXiv:2510.17800)
continual-pretrains and RL-post-trains a VLM to read visually-compressed text, reporting 3–4x
compression at accuracy comparable to a text baseline, via a genetic search over rendering
configuration — but only within the space of typeset human text. Related benchmarks (VTCBench,
arXiv:2512.15649; VTC-R1, arXiv:2601.22069) document compression ratios from 3x to 576x claimed
across methods, and where they break down (`NOTES.md` §1).

**Critical distinction:** DeepSeek-OCR's figure is semantic-adjacent OCR of prose, not exact byte
recovery. heliogram's numbers, targeting bit-exact recovery of arbitrary payloads, are not
comparable and must never be marketed as "beating" it (`NOTES.md`, "Honest caveats").

**Capacity measurements of vision tokens.** "How Much Information Can a Vision Token Hold? A
Scaling Law for Recognition Limits in VLMs" (arXiv:2602.02539) is the closest prior work: it
stress-tests VLMs with increasing character count per image and finds a three-regime phase
transition (stable/instability/collapse), attributing instability to spatial-alignment
sensitivity of ViT patch partitioning. Its payload is still rendered text (confounding capacity
with OCR ability and font geometry), it characterizes an existing distribution rather than
designing a code, and uses no ECC, no patch-aligned symbol layout, no model-free reference
decoder. Its phase-transition finding predicts that per-patch capacity claims must be validated
under misalignment — exactly what heliogram's corruption suite covers.

**Pixel-native language models.** PIXEL (arXiv:2207.06991) and CLIPPO (arXiv:2212.08045)
establish ViT patches over rendered text as a viable language-modeling substrate — again with
the alphabet being human glyphs, not an arbitrary machine code.

**The gap heliogram fills.** Every system above uses language rendered as glyphs as the payload
and learned OCR as the decoder, conflating the channel (pixels → patches → tokens), the code
(fonts/layout evolved for human eyes), and the decoder. No prior work combines (`NOTES.md` §4): a
patch-aligned *symbolic* code designed for the channel (solid symbol per 14x14 patch,
deterministic palette, calibration row, RS ECC); a model-free channel measurement reporting
effective bits/patch after ECC overhead, isolating channel robustness from model ability; and the
economic framing for arbitrary binary payloads rather than prose. Classical 2D barcodes (QR,
DataMatrix, Aztec) are the nearest structural neighbor but target camera optics and arbitrary
orientation, aren't aligned to a ViT patch grid, and have no published evidence of native VLM
decodability.

## 3. Methods

**The codec** (`spec/format-v0.1.md`, `heliogram/codec.py`). Payload bytes are framed (version
byte + 4-byte length), Reed–Solomon coded (default `nsym=32`), split into `log2(P)`-bit symbols
for a palette `P ∈ {2,4,8,16,32,64,128,256}`, and painted one symbol per 14x14px patch on a
roughly square grid. Row 0 is a calibration row cycling the palette so the reference decoder can
recover post-corruption RGB values and nearest-neighbor classify data patches. `decode_pixels`
samples patch centers only, by design, so it measures channel capacity, not decoder cleverness.
`encode(..., align=2)` rounds the grid to even patch dimensions so Qwen2.5-VL's mandatory
`smart_resize` (which snaps resolution to 28px multiples) becomes a no-op, with zero
wire-format change (`tests/test_smart_resize.py`, `spec/format-v0.1.md` §6).

**The harness and corruption suite** (`heliogram/harness.py`, `python -m heliogram.harness` →
`RESULTS.md`/`results.csv`). Sweeps palette, `subpatch` (1 or 2), payload size (48B–16KB), and a
corruption suite: bilinear resize (±3–5%), JPEG q70/q85/q95, crop/pad, and the target model's own
`smart_resize`, measured directly rather than assumed away. Effective bits/patch = payload bits /
total grid patches (calibration row, RS parity, framing, padding all counted against it), on
successful RS-verified decode only — an earlier formula credited padding as free capacity and
overstated density up to 3x; this was corrected before any headline number was reported.

**Measured text baselines** (`heliogram/baselines.py`, `heliogram/data/text_baselines.json`).
`measure_text_encoding_baselines` ran the real Qwen2.5-VL tokenizer (`transformers==5.13.0`)
against base64/ascii85/base85/hex (3 sizes × 3 seeds each) rather than assuming an analytic rate;
the earlier analytic estimate (~6.0 bits/token) undermeasured base64 substantially once real BPE
merges were accounted for.

**The frozen-tower linear probe** (`heliogram/probe.py`, `scripts/run_probe.py`). heliogram grids
pass through the stock, frozen Qwen2.5-VL vision tower; a linear softmax probe is trained from
token embeddings to ground-truth patch symbols (oracle labels, train-statistics-only
standardization). Two tap points: **post-merger** (LM-visible merged-token embeddings — the
LM-token route's go/no-go) and **pre-merger** (per-patch ViT-block output before the 2×2 merger —
localizes whether a post-merger failure originates in the vision blocks or the merger MLP). A
methodological note: Qwen2.5-VL's tower applies a window-attention reordering to patches before
its blocks run and undoes it after the merger; the pre-merger probe recovers this permutation
directly from the tower's own outputs by exact row-matching (no private `transformers` internals),
failing loudly if the recovered index set is not a valid permutation (`scripts/run_probe.py`). A
probe at or below the RS symbol-error budget (~6.27% for `nsym=32`) means the information survives
to that tap point; a probe at chance means no linearly-decodable signal survives there — not that
no fine-tune of downstream layers could recover it (`heliogram/probe.py`) — the reason a
clean-image FAIL is treated as close to conclusive for the linear-readout question specifically.
Whether a *trained* merger could do better is a different, open question; `scripts/train_merger_adapter.py`
scaffolds exactly that test (Design A: frozen-feature nonlinear readout; Design B: a trainable
merger LoRA/adapter), reusing this probe's own alignment code, but has not been run against real
weights (Section 7).

**Typography geometric gate and zero-shot OCR readability** (`heliogram/typography.py`,
`heliogram/ocr_eval.py`, `scripts/run_typography_ocr.py`). `heliogram.typography` first checks,
with no GPU, whether RS-framed ascii85 text at the codec's 14px patch grid clears the measured
8.374 bits/token bar *assuming perfect legibility* — a geometric upper bound only.
`heliogram.ocr_eval` then measures the readability that assumption defers: a stock, un-fine-tuned
Qwen2.5-VL transcribes the rendered images zero-shot, reporting character error rate (CER),
exact-match rate, and `decode_success_rate` (whether the transcription, run through ascii85-decode
and the same RS/framing contract `decode_pixels` uses, recovers the exact payload). Runs were
28px-aligned so `smart_resize` is the identity on the rendered image (an earlier, unaligned run
was discarded rather than reported). This separates whether the tower can *read* the glyphs (CER)
from whether a fragile encoding can be exactly *decoded* end to end (`decode_success_rate`) —
ascii85's positional brittleness means one OCR insertion/deletion defeats Reed–Solomon regardless
of legibility.

Throughout, any function requiring a real model raises immediately without one rather than
fabricating a number (mirrored across `heliogram/vlm.py`, `heliogram/ocr_eval.py`,
`heliogram/instruments/injection_bench.py`).

## 4. Results

### 4.1 Economic ceiling vs. text bar

The architectural ceiling for the only VLM-meaningful regime (`subpatch=1`, one symbol per one
nominal vision token) is `log2(P)` bits/symbol, reaching 8 bits only at `P=256`. RS and
calibration-row overhead cap the achievable *net* ceiling, as payload size grows without bound,
at `log2(256) × 223/255 ≈ 6.996` bits/patch (`README.md`, "Capacity sweep"; measured max observed
is 6.827 at palette=256, 16KB — `RESULTS.md`, "Headline"). This is a consequence of the code's own
framing/ECC overhead, not a fitting artifact.

The measured bar (`heliogram/data/text_baselines.json`): ascii85 at **8.374 bits/token** is
strongest (base85 8.178, base64 8.096, hex 4.534). Since 6.996 < 8.096 < 8.374, no `subpatch=1`
config can beat either baseline on density; `RESULTS.md` ("Headline") confirms 0 of 64 swept
`subpatch=1` configs beat even base64 clean. A byte-granular token-count scan up to 64KB found no
`subpatch=1` crossing below base64 token count anywhere; best ratio observed is 1.16, i.e. 16%
*more* tokens than base64 (`RESULTS.md`, "Token crossover").

A separate robustness finding: because an in-scope operator cannot opt out of Qwen2.5-VL's
`smart_resize`, palettes that had cleared the density gate in an earlier revision
(`palette=8`/`subpatch=2`) were newly measured to fail it at every payload size — their default
grids land on an odd patch dimension, and the 28px snap resamples data rows off the symbol
lattice, dropping decode to 0.00 (`README.md`, "Capacity sweep," Bar B verdict). The `align=2` fix
restores them: re-encoding survives the full suite at 1KB (9.752 clean bits/patch) and 4KB
(10.089), while 16KB separately exceeds the stock processor's ~1MP budget and needs the operator
to widen `max_pixels` (`README.md`, same section).

The only route under which token accounting crosses below base64 requires assuming a model can
read four patch-symbols out of one 2×2-merged token — an explicitly unverified assumption the
README calls "the entire load-bearing wall of the project's economic case." Section 4.2 reports
the direct measurement against it.

Two further honest, model-free notes bound even the token-count claim above (`heliogram.benefit`,
`python -m heliogram.benefit`). First, equal token *count* is not equal compute *cost*: reaching a
merged image token at all requires routing pixels through the frozen vision tower's own forward
pass and activation footprint, a cost a same-length text prompt never pays — a structural argument,
no FLOP or latency figure invented (`cost_asymmetry_points`). Second, an assumption-gated
*effective cost per recovered bit*: assuming, optimistically, a post-merger reader that hit exactly
the RS correction budget's own error rate, a 6000-byte binary payload at `palette=256` would cost
0.149 tokens per recovered bit (measured live via `python -m heliogram.benefit`); at the
chance-level error rate Section 4.2's probe actually measured post-merger, the same arithmetic
returns an undefined (infinite) cost. This is a conditional projection under an assumption the
stock tower does not realize, not a second measured result — Section 4.2 remains the real verdict.

### 4.2 Probe mechanism: post- and pre-merger localization

Four probe runs, two tap points, two model sizes, committed at the repo root:

| Report | Tower | Tap point | Palette | Corruption | Probe symbol error | RS budget | Chance | Verdict |
|---|---|---|---:|---|---:|---:|---:|---|
| `probe_report.md` | 3B | post-merger | 16 | clean | 0.7358 | 0.0627 | 0.9375 | at/near chance |
| `probe_report.md` | 3B | post-merger | 128 | clean | 0.9245 | 0.0627 | 0.9922 | at/near chance |
| `probe_report.md` | 3B | post-merger | 256 | clean | 0.9070 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_7b.md` | 7B | post-merger | 16 | clean | 0.6551 | 0.0627 | 0.9375 | at/near chance |
| `probe_report_7b.md` | 7B | post-merger | 256 | clean | 0.8979 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_easy.md` | 3B | post-merger | 2 | clean | 0.1831 | 0.0627 | 0.5000 | partial signal, above budget |
| `probe_report_easy.md` | 3B | post-merger | 4 | clean | 0.3329 | 0.0627 | 0.7500 | partial signal, above budget |
| `probe_report_premerger.md` | 3B | **pre-merger** | 16 | clean | **0.1344** | 0.0627 | 0.9375 | partial signal, above budget |
| `probe_report_premerger.md` | 3B | pre-merger | 16 | jpeg_q70 | 0.1902 | 0.0627 | 0.9375 | partial signal, above budget |
| `probe_report_premerger.md` | 3B | pre-merger | 256 | clean | 0.8081 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_premerger.md` | 3B | pre-merger | 256 | jpeg_q70 | 0.8193 | 0.0627 | 0.9961 | at/near chance |

Every post-merger cell, on both sizes, is at/near chance: `palette=16` gets closest (65.5–73.6%
error vs. 93.75% chance) but is nowhere near the 6.27% RS budget. Going 3B→7B does not rescue it.

Pre-merger splits by palette. At `palette=16` (4 bits/symbol), pre-merger error is 13.4% clean —
above the RS budget, so not directly decodable via a linear readout, but "far below chance": a
real partial signal the vision blocks preserve, degrading only modestly under `jpeg_q70` (19.0%).
At `palette=256` (8 bits/symbol), pre-merger is already at/near chance (80.8% clean, 81.9%
`jpeg_q70`) — the linear probe finds no linearly-decodable trace of this finer color structure at
the vision blocks themselves, before the merger runs (linear-probe scope; see Section 7).
Comparing tap points at `palette=16` isolates the loss: the same code at 13.4% error
pre-merger is back at 65.5–73.6% post-merger — the 2×2 merger MLP erases most of the
linearly-recoverable signal the blocks upstream still held. The lowest-bit-depth codes
(`palette=2`/`4`) leave a comparable partial-but-undecodable signal even post-merger
(18.3%/33.3% error vs. 50%/75% chance, `probe_report_easy.md`) — coarser codes survive further
into the pipeline, finer ones don't, and none reach the RS budget where a language model actually
reads from.

### 4.3 Typography readability/density crossing

The geometric gate passes: RS-framed ascii85 text clears the 8.374-bit bar from a 12px font down,
assuming perfect legibility. Zero-shot readability against a stock 7B model (RS-framed, 256B
payload, mean CER; table from `docs/FINDINGS.md` §5b — no separate committed report file for this
run):

| font | CER | reads? | geometric bits/patch | beats 8.374? |
|---:|---:|---|---:|---|
| 28px | 0.041 | yes (~96%) | 1.94 | no |
| 24px | 0.074 | yes | 2.56 | no |
| 20px | 0.152 | mostly | 3.71 | no |
| 16px | 0.370 | poorly | 5.84 | no |
| 14px | 0.467 | badly | 7.42 | no |
| 12px | 0.595 | no | 9.75 | **yes** |

The tower *can* OCR rendered ascii85 (96% accuracy at 28px), so this is not a perception failure
like the color codec — it is geometric. Readability and density cross below the bar rather than
above it: sizes the tower reads (≥24px) sit around 2 bits/patch, roughly 4x below the bar; the
only size that beats it (12px) is illegible (60% CER). Even a perfect-decoding encoding at a
readable size would cost more tokens than sending text directly. `decode_success_rate` was 0% at
every size, but that is dominated by ascii85's positional brittleness, not the readability/density
crossing itself, which holds regardless of the encoding chosen.

## 5. Discussion

Both channels fail for versions of the same reason. A VLM's perception is a strong learned prior
over natural images and typeset text, not a general-purpose lossless channel for machine-designed
symbolic structure. The color codec's pre/post-merger split shows this directly: the vision
blocks partially tolerate coarse, out-of-distribution color structure (13.4% error at
palette=16) but progressively discard finer color depth (80.8% at palette=256, same tap point),
and whatever linearly-decodable trace survives the blocks is then, per the linear probe, mostly
gone once read post the 2×2 merger MLP — whose job is to compress four patch embeddings into
one, with no particular reason to preserve fine-grained per-patch identity. Whether a nonlinear
readout or a trained merger would do better is untested here (Section 7). Typography fails
differently: the tower's OCR competence extends
to rendered text, but only at font sizes large enough to be comfortably legible, and legibility
and packing density move in opposite directions as font size shrinks.

Scale did not rescue either failure. 3B→7B moved post-merger error from 73.6% to 65.5% at
palette=16 — real but modest, nowhere near the 6.27% budget (`probe_report_7b.md`) — suggesting
an architectural bottleneck (what the merger structurally discards) rather than a capacity limit
more parameters straightforwardly relieve, though only two sizes from one family were tested.

The unifying frame: redundant natural language carries enough statistical structure that a
language model's prior can fill small perceptual gaps left by an imperfect optical read; arbitrary
high-entropy bytes have no such redundancy, and every symbol must be recovered exactly with no
downstream prior to lean on. This bounds, generally, what patch-aligned or typographic
optical-context schemes can gain specifically for *exact byte recovery* on a model whose
perception was trained on natural images and prose: the training signal that makes a VLM good at
reading real-world content is also what discards dense, non-prose structure such a scheme needs.

## 6. Defensive contribution

Independent of the economic result, and shipped *before* any capability-facing work — "detector
before capability" — the project built two defensive instruments and wrote a release gate
conditioning any future capability work on measuring both sides of the resulting risk together
(`README.md`, "Phase-2 safety release gate").

`heliogram.instruments.foreign_tile` is a model-free pre-ingest guard: `guard(img, allowlist)`
flags an image carrying a heliogram-like payload not on a trusted allow-list, before it reaches a
decoder or model. It rests on a structural signature: a genuine tile tiles evenly into
patch-size cells that are each near-solid color, a regularity natural photos essentially never
exhibit by accident. `patch_structure_score` is the cheap first filter; only images clearing it
face the decisive second check — does the image decode under a trusted `(palette, patch_size,
subpatch)` config. `evaluate_detector` reports TPR at a bounded FPR, using "learned-alphabet"
tiles (data cells repainted through a permuted palette, built to defeat calibration-from-row-0
recovery) as hard positives. It contains no model import at any scope — a structural/decode-based
classifier, not a learned one, by design. Measured against that standard hard-positive set:
TPR=1.000 (8/8) at FPR=0.000 (0/16) (`python -m heliogram.instruments.foreign_tile`).

That measurement also exposes this detector's own blind spot, directly rather than by prediction: a
texture-camouflage adversary — an otherwise-foreign tile dusted with seeded per-pixel noise so its
cells still average to the correct color but clear the cheap first filter's variance threshold —
collapses that filter completely, measured at TPR=0.000 (0/4) against the same FPR=0.000 (0/16)
negatives (`python -m heliogram.instruments.foreign_tile`, "texture-camouflage family eval"). Every
texture-camouflaged positive is, by construction, still a heliogram-shaped payload nobody approved
— it is simply waved through by the cheap structural filter before ever reaching the decisive
decode-attempt stage. This is a measured limitation of this particular defensive instrument, not a
hypothetical one; see `docs/THREAT-MODEL.md` for the fuller adversarial framing this result feeds.

`heliogram.instruments.injection_bench` addresses a harder question: if a payload decodes cleanly
under a *trusted* config, can its bytes still carry an instruction that changes downstream model
behavior, and how much survives corruption? Split per the project's data-honesty rule:
`measure_behavioral_capacity` requires a real model, processor, and judge and raises without them
rather than fabricating an influence number — not run anywhere in this repo (no GPU here).
`evaluate_defense`/`build_detector_eval_set` is pure CPU image-counting, fully tested, demonstrated
scoring `foreign_tile.guard` as one candidate defense. The module is explicit about the resulting
gap: `guard` is blind to decoded payload content by design, so it cannot alone catch a behavioral
payload under a trusted, allow-listed config — a real limitation, not a bug. All
`BEHAVIORAL_PAYLOADS` fixtures are labeled synthetic test strings (fictional personas, invented
tool names, actions no real system implements), used only to probe channel capacity.

The written release gate commits, in advance of any tuned reader existing, to publishing together:
measured behavioral capacity against the actual tuned model; the detector's TPR at bounded FPR
measured against tiles the *tuned* model decodes (not only the reference decoder); and a decision
rule, stated before there is incentive to soften it, that no adapter releases if behavioral
payloads survive corruption at rates the guard cannot catch at bounded FPR.

**Stated honestly:** behavioral-attack capacity has not been measured. There is no tuned reader in
this repo, and `measure_behavioral_capacity` refuses to run without one. What exists is a threat
model (`docs/THREAT-MODEL.md`), a structural detector with a measured methodology against its
defined hard-positive case (and a measured blind spot against a texture-camouflage adversary, above),
and a benchmark harness ready for a future model — not a demonstrated exploit.

## 7. Limitations

- The probe measures **linear** readability of embeddings for **one** hand-designed code (solid
  14px-block symbols, fixed palette, one calibration row). It de-risks but does not bound
  capacity: a nonlinear probe, or one with more capacity or data, could recover more at the same
  tap point.
- A **fine-tuned** adversary that trains the tower or merger itself to preserve this structure is
  unmeasured; the probes only characterize what stock, frozen weights preserve. A cheaper,
  intermediate go/no-go for exactly this question — can training only the merger (tower and LM
  left frozen) recover the `palette=16` signal the frozen merger is measured to erase —
  is scaffolded in `scripts/train_merger_adapter.py`: Design A, a frozen-feature nonlinear readout
  diagnostic (no gradients into the tower at all); Design B, the actual gate, LoRA/adapter-training
  the merger jointly with a readout head. Both reuse the probe's own window-shuffle alignment code,
  refuse to run without a real model, and re-check against the committed `probe_report_premerger.md`
  numbers before trusting anything new — neither has been run against real weights (no GPU here).
  The larger, one gated, live experiment that would test the full fine-tune question — a
  merger-retargeted fine-tune at `palette=16` (`build_p16_merger_curriculum` in
  `scripts/train_qlora.py`, `--curriculum p16_merger`) — is designed and CPU-contract-tested but
  not run against real weights; it is gated on two cheap pre-scans (data-limitedness of the 13.4%
  pre-merger error, and mapping the pre-merger palette cliff between 16 and 256) that are also
  unrun (`RUNBOOK-GPU.md` §2.5). In practice, the cheaper `train_merger_adapter.py` gate is meant
  to run before that larger curriculum is attempted at all.
- A **different code** (palette design, patch geometry, a learned encoding) could interact with
  the vision blocks and merger differently; nothing here bounds untested codes.
- Every claim is specific to **Qwen2.5-VL** (3B/7B, `transformers==5.13.0`); nothing transfers to
  other model families or closed API models, out of scope because their preprocessing is opaque.
- The typography measurement is **zero-shot** against a **stock** model. A fine-tune could
  plausibly read smaller text than the stock tower does; this measures the stock floor, not a
  fine-tune's ceiling, and does not claim the channel is dead for a tuned reader — only as
  measured here.

## 8. Conclusion

heliogram set out to measure whether a patch-aligned optical codec could make images a cheaper
context medium than text for arbitrary, high-entropy byte payloads on self-hosted VLMs. On two
independent channels, against a measured text baseline and a real frozen tower at two model
sizes, the answer is no. This is a genuine contribution: it bounds what optical-context schemes
can gain, for exact byte recovery, given how a VLM's own perception treats machine-dense,
out-of-distribution structure — discarding it partially in the vision blocks and more decisively
in the patch-merger step, and inversely coupling readability against density once the payload is
rendered as text instead.

Two durable byproducts survive the negative result. First, the exactness niche: Reed–Solomon
gives detection *and* correction on every successful decode, which raw VLM transcription of
rendered text does not provide without bolting on a checksum/ECC layer — at which point it has
reinvented what heliogram already has natively (`README.md`, "Roadmap"). This holds even where
rendered-text OCR density matches the color codec's, since exactness and density are different
axes. Second, the pre/post-merger information-localization probe technique — recovering a
tower's window-shuffle permutation from its own outputs, tapping both sides of the merger
boundary, and comparing linear-probe error against an explicit RS-derived decision budget — is a
reusable method for locating where in any patch-merging VLM's pipeline a given structured signal
is lost, independent of whether the answer turns out favorable.

## Appendix: Reproducibility

**CPU-only (no GPU or model weights required):**

```bash
pip install -e .
python -m heliogram.harness       # codec/corruption sweep -> RESULTS.md, results.csv
python -m heliogram.baselines --measure   # text-encoding baselines -> heliogram/data/*.json
                                            # (needs HF Hub access; falls back to the committed
                                            # heliogram/data/text_baselines.json otherwise)
python -m heliogram.typography    # geometric gate for the typography pivot (model-free)
python -m heliogram.benefit       # token-savings + exactness-argument demo, no model
pytest -q                         # full CPU test suite
```

**GPU-dependent (requires a CUDA GPU and, for the probe/OCR steps, HF Hub access to the
Qwen2.5-VL weights):**

```bash
pip install -e ".[gpu]"                         # or: pip install -r requirements-gpu.txt

# Phase-2 Step 0: frozen-tower linear probe (post-merger, the default)
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --palettes 16,128,256 --corruptions clean,jpeg_q85,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report.md --json probe_report.json

# Pre-merger localization run
python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --probe-stage pre_merger --palettes 16,256 --corruptions clean,jpeg_q70 \
    --n-train-images 6 --n-test-images 3 \
    --out probe_report_premerger.md --json probe_report_premerger.json

# Typography zero-shot OCR readability
python scripts/run_typography_ocr.py --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --font-sizes 14,12,10,8 --payload-size 256 --n-trials 5 \
    --out typography_ocr_report.md --json typography_ocr_report.json
```

See [`RUNBOOK-GPU.md`](../RUNBOOK-GPU.md) for the full ordered sequence (setup, contract tests,
baselines, Step 0, localization follow-ups, the gated P=16 merger fine-tune, the typography
readability step, and the QLoRA stage), including per-step decision rules and which artifacts to
commit back after each run.
