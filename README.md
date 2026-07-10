# heliogram

**If one ViT patch (~1 vision token) can reliably carry more than one text token's worth of data (~6 bits for base64), then encoded images are a cheaper context medium for self-hosted VLMs — heliogram measures whether that's true.**

heliogram is a patch-aligned optical codec plus an evaluation harness. It encodes
arbitrary bytes as a grid of solid-color 14x14 blocks (one symbol per ViT patch),
protects them with Reed–Solomon ECC, runs them through the corruptions a real
serving pipeline applies (resize, JPEG, crop/pad), and reports **effective
bits/patch** — the channel capacity of the patch grid, measured with a model-free
reference decoder. Format details: [`spec/format-v0.1.md`](spec/format-v0.1.md).

Apache-2.0. All data is synthetic and seed-deterministic; same inputs produce
byte-identical PNGs.

## Scope

- **In scope:** self-hosted / open-weight VLMs where the operator controls image
  preprocessing end to end (resolution, patching, no server-side re-encoding they
  can't inspect).
- **Out of scope:** closed API models. Their preprocessing (resizing, tiling,
  re-compression, token accounting) is opaque and changeable, so no capacity claim
  made here transfers to them. We do not test against them and won't.

This is a measurement project. If the numbers come out below the base64 line, that
is a result, not a failure — it bounds what optical-context schemes can gain from
denser-than-text symbol coding, and we'll report it as such.

## Results

> Measured by the eval harness (`python -m heliogram.harness`), which writes
> `RESULTS.md` and `results.csv`. Full breakdown, self-consistency checks, the
> capacity/amortization/Gate sweep, the token-crossover benefit, and a
> diagnostic stress test beyond this envelope are in
> [`RESULTS.md`](RESULTS.md).

**The honest headline, from the actual sweep:** at `palette=256`,
`subpatch=1` (one symbol per patch — the only VLM-meaningful regime),
heliogram beats base64 on both density (6.611 bits/patch clean at a 4KB
payload, versus base64's ~6.0 bits/token) *and* on raw token count
(encoding a payload this way costs fewer total patches than base64-ing it
into text tokens, from ~3KB payloads onward — see "Token crossover"
below), and is bit-exact on a successful decode (Reed–Solomon-verified).
**The open question is purely whether a fine-tuned VLM can read a
256-colour palette under real corruption — and on the model-free reference
decoder measured here, it cannot:** `palette=256` (and `palette=128`)
clean-decode exactly but are MEASURED to fail decode under JPEG q70 at
*every* payload size this sweep tested, and, at larger payloads, even the
milder JPEG q85. That failure is not a footnote — it is the entire reason
this benefit is a Phase-2 bet and not a working result. See "Capacity
sweep" and "Token crossover" below for the numbers.

Effective bits/patch = bits_per_symbol x (data patches / total patches) x
(payload bytes / ECC-coded bytes), counted only on successful RS decode (0 on
a failed decode). "Corrupted" is the mean bits/patch over every non-clean
corruption tested: bilinear resize ±3–5%, JPEG q70–95, a slight crop/pad, and
their composition ("combined"). Reference pixel decoder (`decode_pixels`), no
model. Table below: 48-byte synthetic payloads, subpatch=1 (one symbol per
patch — see "Capacity sweep" below for the fuller grid), nsym=32,
patch_size=14px, 3 trials/cell.

| Palette | bits/symbol | Clean bits/patch | Corrupted bits/patch |
|--------:|------------:|-----------------:|----------------------:|
| 2       | 1           | 0.544            | 0.544                 |
| 4       | 2           | 1.070            | 1.070                 |
| 8       | 3           | 1.588            | 1.588                 |
| 16      | 4           | 2.071            | 2.071                 |
| 32      | 5           | 2.353            | 2.353                 |
| 64      | 6           | 2.259            | 2.259                 |
| 128     | 7           | 1.976            | 1.412                 |
| 256     | 8           | 2.259            | 1.506                 |

The break-even line to beat is **~6 bits/patch** (the base64 baseline, see
Baselines); at this small (48-byte) payload every palette here is below it,
including the two largest — this is a real result, not a failure to report:
fixed overhead (32-byte RS parity + a calibration row that widens with the
palette) is the bottleneck at 48 bytes, not palette size. For `palette` in
`{2,4,8,16,32,64}`, clean and corrupted are identical at this payload size
(as in the original release) because `decode_success_rate` is 1.00 across
the whole realistic-serving-pipeline envelope tested here (resize ±3–5%,
JPEG q70–95, slight crop/pad) — Reed–Solomon (nsym=32) fully absorbs the
symbol errors that envelope introduces for those six palettes at this size.
**That stops being true for `palette=128/256`:** both already fail JPEG q70
(and, for 256, the fully-composed `combined`) even at this smallest tested
payload, which is why their corrupted column (1.976→1.412, 2.259→1.506) is
measurably below their own clean column, unlike every smaller palette here —
see `RESULTS.md` for the exact per-corruption rates. A diagnostic stress
test well outside the realistic envelope (50% resize, JPEG q10, 6px
crop/pad, composed) finds the expected fall-off at larger palettes — decode
success drops to 0.00 at palette≥4 under the composed stress case,
including 128/256 — confirming the channel has a real breaking point, just
further out than a typical serving pipeline pushes it for the smaller
palettes and well *inside* it for the two largest. See `RESULTS.md` for
that table. bits/patch (and whether corruption is survived at all) both
change with payload size — see the capacity sweep below, which measures
that directly instead of via a single one-off comparison.

### Capacity sweep: sub-patch and payload size (three bars, not one)

> Full sweep (64 configs), self-consistency checks, the token-crossover
> table, and the complete 512-row per-corruption breakdown are in the
> "Headline" and "Token crossover" sections of [`RESULTS.md`](RESULTS.md).

Beyond the single-payload table above, the harness sweeps `subpatch` (k ∈
{1, 2}: k×k solid-color sub-cells per data patch instead of 1 — a purely
geometric density knob, `spec/format-v0.1.md` §6a), `payload_size` (48B –
16KB, to amortize the fixed 5-byte header + RS parity + calibration-row
overhead), and now all 8 palettes in `VALID_PALETTES` (2–256) against the
same realistic corruption suite, at 3 trials/cell (reduced from 5 to bound
wall-clock at the 16KB tier — 64 configs × 8 corruptions = 512 cells total).
This project tracks three separate bars over that sweep, on purpose — see
`RESULTS.md`'s Headline section for why conflating them is exactly the
overclaiming this file exists to avoid:

- **Bar A — beat base64 density, clean (6.0 bits/patch).** The real economic
  break-even for density alone.
- **Bar B — Gate #1 comfort margin (8.0 bits/patch, clean *and* worst-tested
  corruption).** Deliberately padded above Bar A (see "Decision Gate" below)
  — **not** the real economic bar.
- **Bar C — token crossover.** Does encoding a payload cost fewer total
  patches than base64-ing it into text tokens? An accounting comparison of
  token *count*, not bits/patch density — see below.

**Bar B verdict, from the actual run: 3 of 64 configs clear the gate —
palette=8, subpatch=2, at payload_size 1024B / 4096B / 16384B (9.978 /
10.255 / 10.389 bits/patch clean, identical under every one of the 7 tested
corruptions, including `combined`). None of the 3 are `subpatch=1`
(VLM-meaningful).** At the smallest payload (48B), the same
palette=8/subpatch=2 config still fails `combined` (0.00 decode success) — it
only clears once payload size is large enough to amortize the fixed overhead.

**Bar A verdict: 30 of 64 configs beat base64 density clean — including,
for the first time, `subpatch=1` configs:** `palette=128` at 16KB (6.073
bits/patch) and `palette=256` at 4KB and 16KB (6.611 / 6.895 bits/patch).
This is the real economic bar, and `subpatch=1`/`palette=256` clears it —
but see the mandatory corruption caveat below before reading that as usable.

**Bar C verdict — the actual currently-measured benefit claim: at
`subpatch=1`, `palette=256` crosses below base64 token count at ~3KB
payloads (3055B), and `palette=128` crosses (barely) at ~12.9KB (13159B);
`palette≤64` never
crosses anywhere in the swept range (up to 16KB).** Concretely, `palette=256`
needs 5,120 total patches at a 4KB payload versus base64's ~5,464 estimated
tokens for the same bytes (0.94×), improving to 19,200 vs. ~21,848 (0.88×) at
16KB — fewer tokens *and* denser *and* bit-exact on a successful decode.

**Honesty caveat (mandatory, same as `RESULTS.md`, and the whole reason Bar
C is a bet and not a result): `palette=128`/`256` clean-decode exactly on
this pixel decoder but are MEASURED to fail decode under JPEG q70 at every
payload size in this sweep** (0.00 decode success at 48B/1KB/4KB/16KB, both
palettes), and, at larger payloads, even JPEG q85. **The token-count and
density benefit above is therefore a clean-channel-only number — it is not
usable end to end on this reference decoder.** Whether a fine-tuned VLM
reader can survive JPEG q70 at a 256-colour palette where `decode_pixels`
cannot is exactly the Phase-2 question this project exists to answer, not
something claimed here.

**A second, independent honesty caveat, this one about `subpatch>1`: every
Bar B (Gate #1)-clearing config, and most Bar A/C-clearing configs at
`subpatch=2`, use `subpatch=2`, which is a PIXEL-DECODER GEOMETRIC CEILING,
not a VLM capability claim.** `decode_pixels` reads k×k sub-cells trivially
— it samples known, exact pixel coordinates off a grid it's told the size of
in advance — but there is no evidence a real ViT/VLM image encoder can
resolve structure below its own patch. `subpatch=1` is the only
VLM-meaningful regime (one symbol per patch = one nominal vision token). The
hard architectural ceiling for `subpatch=1` is `log2(P)`, which reaches
exactly 8 at `P=256` — but Reed–Solomon/calibration overhead caps the
*net* ceiling at `log2(P) × 223/255 ≈ 6.996` bits/patch as payload size
grows without bound, strictly below 8 for every finite payload; the measured
max in this sweep is 6.895 (palette=256, 16KB payload). **No `subpatch=1`
config clears Bar B (Gate #1) anywhere in this sweep, or ever can.**

Two more things this sweep found, both measured (not cherry-picked):

- **Higher palette ≠ better once corruption is in the loop, and it gets
  monotonically worse.** At `subpatch=2`/16KB, each palette from 8 upward has
  a *higher clean* ceiling than the last (8: 10.389, 16: 13.833, 32: 17.271,
  64: 20.702, 128: 23.889, 256: 26.554 bits/patch) but fails against a
  strictly larger set of the 7 tested corruptions: palette=16 fails only the
  fully-composed `combined`; 32 and 64 add `JPEG q70`; 128 adds `JPEG q85`
  too; and 256 fails `JPEG q95` as well — the *mildest* JPEG setting this
  harness tests. Of palette 8 and up, only palette=8/subpatch=2 survives
  every tested corruption (palette=2/4 also do, but their clean ceiling is
  too low for the comparison to be interesting here).
- **Amortization raises the floor but not the ceiling.** For `subpatch=1`,
  bits/patch rises with payload size (palette=64: 2.259 → 4.969 → 5.154 →
  5.208; palette=256: 2.259 → 5.742 → 6.611 → 6.895 bits/patch, at
  48B/1KB/4KB/16KB) but asymptotically approaches `log2(palette) × 223/255`
  — it pays down the fixed per-message overhead, it cannot raise the
  per-symbol ceiling itself. `palette=256` is the first palette whose
  amortization curve crosses the base64 baseline (between 1KB and 4KB).

Every number above is `decode_pixels` (pixel decoder, CPU, no VLM) — see the
Scope note at the top of `RESULTS.md`.

## Baselines

- **base64 in text context: ~6 bits/token.** base64 encodes 6 payload bits per
  character, and common tokenizers emit roughly one token per base64 character
  (often slightly better via multi-char merges). `heliogram.baselines.
  base64_bits_per_token()` returns the 6.0 analytic figure by default and accepts
  a HuggingFace tokenizer to measure the real ratio for your model.
- **Rendered text (honesty guardrail).** The obvious competitor is just
  typesetting the payload small and letting the VLM read it — that's what
  DeepSeek-OCR and Glyph exploit. `heliogram.baselines` typesets the payload onto
  the same patch grid and reports chars/patch as a geometric, model-free estimate.
  The *true* rendered-text capacity depends on the un-fine-tuned VLM's OCR and is
  a Phase-2 measurement. If rendered text matches the codec's bits/patch, the
  codec's only remaining advantages are exactness (ECC-verified bytes) and
  robustness — and we'll say so.
- **Token crossover (the actual benefit claim — see "Capacity sweep" above
  and `RESULTS.md`).** Beating bits/patch density is necessary but not
  sufficient for beating base64 end to end: base64 pays a fixed 4/3-per-byte
  expansion with no per-message overhead, while heliogram pays a calibration
  row + Reed–Solomon parity once per *image*, amortized differently as
  payload grows. `heliogram.harness._token_crossover` compares
  `total_patches` (the grid's width×height, ~1 token/patch for a self-hosted
  VLM) directly against `base64_token_est` (`ceil(payload/3)*4`, ~1
  token/char) for the SAME payload. Measured: only `palette` 128/256 at
  `subpatch=1` currently cross that line, from ~3–13KB payloads onward.

## Quickstart

```bash
pip install -e .
python demo.py            # encode a small JSON doc, decode it back,
                          # print patch count vs. base64 token count
python -m heliogram.harness   # full palette x subpatch x payload x corruption sweep -> RESULTS.md, CSV
python -m heliogram.benefit   # token-savings demo (P=256, a real ~6KB JSON doc) vs. base64/hex
                               # tokens, PLUS the exactness-vs-OCR argument -- CPU-only, no model
pytest                    # roundtrip, determinism, calibration-recovery tests
```

Requires Python with `pillow`, `numpy`, `reedsolo` (installed by the command above).

## How it works (v0.1, one paragraph)

Payload bytes are framed (version byte + 4-byte length), Reed–Solomon coded
(default 32 parity bytes), split into log2(P)-bit symbols for a palette of
P ∈ {2, 4, 8, 16, 32, 64, 128, 256} deterministic, maximally separable colors, and
painted one symbol per 14x14 patch (or, with `subpatch`>1, a k×k grid of
sub-cells per patch — see the capacity sweep above) on a square-ish grid.
Row 0 is a calibration row cycling
through the palette so the decoder can recover the palette's post-corruption RGB
values and nearest-neighbor classify every data patch. The reference decoder
(`decode_pixels`) samples patch centers only — it is deliberately dumb, so that
what it measures is the channel, not decoder cleverness. Full spec:
[`spec/format-v0.1.md`](spec/format-v0.1.md).

## Roadmap / Phase-2 boundary

Phase 1 (this repo) is entirely model-free: codec, corruption suite, harness,
baselines. It answers one question — *how many bits does a patch carry through
realistic preprocessing?*

**Decision Gate #1:** Phase 2 starts only if the corrupted (worst-tested-case,
not mean) bits/patch number clears a working bar of **~8 bits/patch** — a
deliberate margin above the ~6 bit/token base64 break-even (Baselines above),
not just any number over it. **This bar is a conservative comfort margin,
not the real economic bar** — Bar A (beat base64 density, clean) and Bar C
(token crossover) are the two bars this project's actual benefit claim rests
on (see "Capacity sweep" above), and are reported alongside Gate #1 for
exactly that reason, not as a replacement for it.

The capacity/amortization/Gate sweep (see "Capacity sweep" above; full detail
in `RESULTS.md`) found that **3 of the 64 tested (palette, subpatch,
payload_size) configs clear the Gate #1 (8-bit) bar, and all 3 use
`subpatch=2`** — the unverified, pixel-decoder-only geometric regime. **No
`subpatch=1` (the actually VLM-meaningful regime) config clears Gate #1, and
none can by construction at any payload size:** the hard per-symbol ceiling
for `subpatch=1` is `log2(P)`, which reaches exactly 8 only at the largest
palette, `P=256` — but Reed–Solomon/calibration overhead caps the achievable
*net* ceiling at `log2(P) × 223/255 ≈ 6.996` bits/patch as payload size grows
without bound, strictly below 8 for any finite payload (measured max in this
sweep: 6.895, palette=256, 16KB).

**That result does not, by itself, open or close this gate — but the sweep
found something else that does motivate starting Phase 2 anyway: at
`subpatch=1`, `palette=256` clears Bar A (beats base64 density clean, from
4KB payloads: 6.611 bits/patch) and Bar C (crosses below base64 token count,
from ~3KB payloads), and `palette=128` clears both too, later (~13KB for Bar
C).** This is the project's actual, currently-measured benefit signal —
cheaper-than-text context for large binary payloads, bit-exact — **and it is
explicitly conditional on a learned reader recovering it: `palette=128/256`
are measured, on this same model-free decoder, to fail decode under JPEG q70
at every payload size tested here.** Phase 2 is GPU work and is **not** in
this repo yet:

- **Fine-tune an open VLM to decode heliogram images natively, retargeted at exactly the
  measured gap above:** `palette` in `{64, 128, 256}` at `subpatch=1`, under corruption
  augmentation concentrated on the specific corruptions (`jpeg_q70`/`jpeg_q85`/`combined`) this
  same palette range is measured to fail — i.e. learned classification of a big color palette
  through realistic corruption, *not* sub-patch geometry (`subpatch>1` stays a documented,
  secondary, pixel-decoder-only geometric ceiling, unchanged from the discussion above). See
  `heliogram/dataset.py`'s retargeted defaults (`DEFAULT_PALETTES`) and
  `scripts/train_qlora.py`'s retargeted curriculum (`build_curriculum()`) for where this plays
  out; `heliogram.vlm.QwenVLDecoder` (see "Phase 2 (GPU)" below) is the decoder plug point, and
  the `VLMDecoder` stub in `heliogram/codec.py` remains the minimal hand-rolled alternative (it
  currently raises `NotImplementedError`).
- Measure the un-fine-tuned VLM's OCR on the rendered-text baseline for a fair comparison — not
  implemented anywhere in this repo; needs a real model run.
- **Report the exactness niche regardless of how that comparison lands** (see
  `heliogram/benefit.py`): Reed–Solomon gives detection *and* correction on every decode; a raw
  VLM transcription of rendered text gives neither, unless a checksum/ECC layer is added on top
  of it — at which point it has reinvented what heliogram already has natively. This holds even
  if a future measurement shows rendered-text OCR matching heliogram's bits/patch density; no
  OCR error rate is invented to make this argument, and the real number is itself Phase-2 work.
- Semantic use of optically-injected context (retrieval, tool outputs, logs).

No model-based numbers appear anywhere in this repo until Phase 2 produces them.

## Phase 2 (GPU) — how to run when you have a GPU

Everything above this section is Phase 1: model-free. This repo also ships a **Phase-2
scaffold** — code to generate training data, fine-tune a VLM, and plug it into the same
`decode()` call Phase 1 uses — that only *runs* once you have a GPU. Nothing in it has been
executed here: there is no GPU in this environment, so every number anywhere else in this
README and in `RESULTS.md` still comes from `decode_pixels` (see the Scope note at the top of
`RESULTS.md`). Importing `heliogram` never requires `torch`/`transformers`/`peft`/
`bitsandbytes` — those are imported lazily, only inside the functions that actually need them
(`heliogram/vlm.py`, `scripts/train_qlora.py`); `python -c "import heliogram"` succeeds with
only `pillow`/`numpy`/`reedsolo` installed, same as Phase 1. One exception to "nothing in this
section has been executed": `heliogram/benefit.py` (step 6 below) is model-free like Phase 1
and its numbers ARE real, run on demand on CPU — it is scaffold in the sense of anticipating
Phase 2's outcome (the exactness argument, the token-count accounting), not in the sense of
being unrun.

1. **Install GPU extras** (on a machine with a GPU):

   ```bash
   pip install -e ".[gpu]"          # or: pip install -r requirements-gpu.txt
   ```

2. **Generate a synthetic training set** — `heliogram/dataset.py`, CPU-only, needs no GPU and
   runs fine in this same environment. It reuses `encode`/`extract_symbols` from
   `heliogram.codec`, so every training target is the *exact* ground-truth symbol sequence the
   codec wrote for that image — no hand labeling, no drift between what was encoded and what
   the model is asked to transcribe. Its defaults are retargeted at the large-palette bet above:
   `--palettes` defaults to `{64, 128, 256}` and `--corruption-prob` defaults to `0.5` (see
   `heliogram/dataset.py`'s module docstring), so a bare invocation already generates data for
   exactly the gap this project needs closed:

   ```bash
   python scripts/gen_dataset.py --out data/phase2_train --n 2000 --seed 0
   python scripts/gen_dataset.py --out data/phase2_val   --n 200  --seed 1
   ```

   This writes `data/phase2_train/images/*.png` plus a `manifest.jsonl` with one
   `{"image_path", "palette", "subpatch", "target", ...}` record per line — `target` is the
   symbol sequence rendered as a compact string (see `heliogram.dataset.symbols_to_target`).

3. **(Optional) Measure the stock model's zero-shot symbol error first.**
   `heliogram.vlm.zero_shot_symbol_error(model, processor, configs)` runs an *unmodified*
   Qwen2.5-VL (or whatever you load) directly over encoded images and reports real
   symbol-error/decode-success numbers against ground truth. It never returns a number without
   a real model behind it — pass `model=None` and it raises immediately instead of guessing.

4. **QLoRA fine-tune** — `scripts/train_qlora.py` loads Qwen2.5-VL-7B-Instruct in 4-bit
   (bitsandbytes NF4) via Transformers, attaches LoRA adapters (PEFT) on the attention/
   projection layers, and trains across a curriculum: a cheap small-palette warm-up, then three
   stages that concentrate specifically on `palette` in `{64, 128, 256}` at `subpatch=1` —
   clean first, then corrupted, then a final stage that up-weights the specific corruptions
   (`jpeg_q70`/`jpeg_q85`/`combined`) this palette range is measured to fail under — see
   `build_curriculum()`:

   ```bash
   python scripts/train_qlora.py --output-dir checkpoints/qwen25vl-heliogram-lora
   ```

   This script is a documented **starting point, not a verified recipe** — see its top-of-file
   docstring for VRAM expectations and its explicit "GPU required, untested here" caveat.

5. **Plug the fine-tuned model into `decode()`**, exactly like the model-free reference decoder:

   ```python
   from heliogram import decode
   from heliogram.vlm import QwenVLDecoder

   decoder = QwenVLDecoder(model=my_loaded_model, processor=my_loaded_processor,
                            palette=256, subpatch=1)   # the large-palette bet -- see above
   payload = decode(img, palette=256, subpatch=1, decoder=decoder)
   ```

   `QwenVLDecoder` prompts the model to transcribe the symbol grid, parses the response back
   into symbols, and feeds them through the *same* Reed–Solomon/framing layer `decode_pixels`
   uses — there is no separate RS implementation for the model path.

6. **See the token savings and the exactness niche, right now, without a GPU** —
   `python -m heliogram.benefit` encodes a real ~6KB synthetic JSON payload at `palette=256`,
   reports patches vs. base64 tokens vs. a raw-byte (hex) text-encoding baseline, and then
   *demonstrates* the mandatory caveat live: it decodes the clean image (passes, bit-exact) and
   then a real JPEG q70 re-encode of that same image (fails, exactly as `codec.py`/`RESULTS.md`
   measure) — so the "this needs the Phase-2 reader" caveat is a fresh measurement on every run,
   not a citation. The same module also prints the exactness argument: Reed–Solomon gives
   detection *and* correction on every decode; free-form OCR gives neither — a durable niche
   even if a future measurement shows rendered-text OCR matching heliogram's bits/patch density
   (see "Roadmap / Phase-2 boundary" above and `heliogram/benefit.py`'s module docstring).

**The boundary this doesn't cross.** `heliogram/dataset.py`, `scripts/gen_dataset.py`, and
`heliogram/benefit.py` are all CPU-only and *have* been exercised in this repo (the first two
via `tests/`; `heliogram/benefit.py` by direct runs of `python -m heliogram.benefit`, not yet
its own pytest file). Everything past that — loading a real VLM, `zero_shot_symbol_error` with
a real model, `scripts/train_qlora.py`, `QwenVLDecoder.__call__` actually reaching
`.generate()` — requires a GPU this environment doesn't have and is untested here. **Decision Gate #1 (see "Roadmap / Phase-2 boundary" above)
is decided by measured numbers from an actual fine-tuned model, run through this same
`decode()` path, on held-out data — not by this scaffold's existence.** Until that run happens
and its numbers are written up the same way `RESULTS.md` was, treat everything in this section
as "code that should let someone with a GPU go measure this," not as a result.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 heliogram contributors.
