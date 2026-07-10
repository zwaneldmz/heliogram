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
> capacity/amortization/Gate sweep, and a diagnostic stress test beyond this
> envelope are in [`RESULTS.md`](RESULTS.md).

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

The break-even line to beat is **~6 bits/patch** (the base64 baseline, see
Baselines); the codec is currently below it at every palette here — this is a
real result, not a failure to report. Clean and corrupted are identical at
every palette, including the two (32, 64) added since the original v0.1
4-palette release, because `decode_success_rate` is 1.00 across the whole
realistic-serving-pipeline corruption envelope tested here (resize ±3–5%, JPEG
q70–95, slight crop/pad): Reed–Solomon (nsym=32) fully absorbs the symbol
errors that envelope introduces at this payload size (largest observed symbol
error rate: 0.0417, at palette=32 / JPEG q70 — still fully corrected). A
diagnostic stress test well outside that envelope (50% resize, JPEG q10, 6px
crop/pad, composed) does find the expected fall-off at larger palettes —
decode success drops to 0.00 at palette≥4 under the composed stress case —
confirming the channel has a real breaking point, just further out than a
typical serving pipeline pushes it. See `RESULTS.md` for that table. The
bottleneck at this small (48B) payload size is fixed overhead (32-byte RS
parity + calibration row), not corruption; bits/patch grows with payload
size — see the capacity sweep below, which measures that directly instead of
via a single one-off comparison.

### Capacity sweep: sub-patch and payload size (does it clear the Gate #1 bar?)

> Full sweep (48 configs), self-consistency checks, and the complete
> 384-row per-corruption breakdown are in the "Headline" section of
> [`RESULTS.md`](RESULTS.md).

Beyond the single-payload table above, the harness sweeps `subpatch` (k ∈
{1, 2}: k×k solid-color sub-cells per data patch instead of 1 — a purely
geometric density knob, `spec/format-v0.1.md` §6a) and `payload_size` (48B –
16KB, to amortize the fixed 5-byte header + RS parity + calibration-row
overhead) against all 6 palettes and the same realistic corruption suite, at
3 trials/cell (reduced from 5 to bound wall-clock at the 16KB tier — 384
cells total). **Gate #1 bar: 8.0 bits/patch**, cleared only if a config is at
or above it both clean *and* under its single worst tested corruption (not
just on average).

**Verdict, from the actual run: 3 of 48 configs clear the gate — palette=8,
subpatch=2, at payload_size 1024B / 4096B / 16384B (9.978 / 10.255 / 10.389
bits/patch clean, identical under every one of the 7 tested corruptions,
including `combined`).** At the smallest payload (48B), the same
palette=8/subpatch=2 config still fails `combined` (0.00 decode success) — it
only clears once payload size is large enough to amortize the fixed overhead,
which is exactly the mechanism this sweep exists to measure.

**Honesty caveat (mandatory, same as `RESULTS.md`): every clearing config has
`subpatch=2`, which is a PIXEL-DECODER GEOMETRIC CEILING, not a VLM capability
claim.** `decode_pixels` reads k×k sub-cells trivially — it samples known,
exact pixel coordinates off a grid it's told the size of in advance — but
there is no evidence a real ViT/VLM image encoder can resolve structure below
its own patch. `subpatch=1` is the only VLM-meaningful regime (one symbol per
patch = one nominal vision token), and **no `subpatch=1` config clears the
gate anywhere in this sweep**: the measured ceiling there tops out at 5.208
bits/patch (palette=64, 16KB payload), and the hard architectural ceiling
(`log2(64) = 6 < 8`) rules a `subpatch=1` clear out entirely, for any palette
in `VALID_PALETTES` and any amount of payload-size amortization.

Two more things this sweep found, both measured (not cherry-picked):

- **Higher palette ≠ better, once corruption is in the loop.** At
  `subpatch=2`/16KB, palette 16/32/64 have a *higher clean* ceiling (13.833 /
  17.271 / 20.702 bits/patch) than palette=8 (10.389) — but palette=16 fails
  only the fully-composed `combined` corruption, while palette=32 and 64
  already fail on `JPEG q70` alone (0.00 decode success). That lines up with
  `spec/format-v0.1.md` §2a's note that palettes above 16 add an untested
  value-tiering axis for JPEG-chroma robustness — this sweep is the first
  empirical test of that choice, and at `subpatch=2` it stops paying off past
  palette=8. Only palette=8/subpatch=2 clears both bars.
- **Amortization raises the floor but not the ceiling.** For `subpatch=1`,
  bits/patch rises with payload size (palette=64: 2.259 → 4.969 → 5.154 →
  5.208 bits/patch at 48B/1KB/4KB/16KB) but asymptotically approaches
  `log2(palette)` — it pays down the fixed per-message overhead, it cannot
  raise the per-symbol ceiling itself.

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

## Quickstart

```bash
pip install -e .
python demo.py            # encode a small JSON doc, decode it back,
                          # print patch count vs. base64 token count
python -m heliogram.harness   # full palette x subpatch x payload x corruption sweep -> RESULTS.md, CSV
pytest                    # roundtrip, determinism, calibration-recovery tests
```

Requires Python with `pillow`, `numpy`, `reedsolo` (installed by the command above).

## How it works (v0.1, one paragraph)

Payload bytes are framed (version byte + 4-byte length), Reed–Solomon coded
(default 32 parity bytes), split into log2(P)-bit symbols for a palette of
P ∈ {2, 4, 8, 16, 32, 64} deterministic, maximally separable colors, and
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
not just any number over it. The capacity/amortization/Gate sweep (see
"Capacity sweep" above; full detail in `RESULTS.md`) found that **3 of the 48
tested (palette, subpatch, payload_size) configs clear that bar, and all 3
use `subpatch=2`** — the unverified, pixel-decoder-only geometric regime.
**No `subpatch=1` (the actually VLM-meaningful regime) config clears it, and
none can by construction** (`log2(64) = 6 < 8`, the hard ceiling for one
symbol per patch, regardless of palette or payload-size amortization). That
result does not, by itself, open or close this gate: it says the checked
geometric ceiling can clear the bar on this model-free decoder, not that any
model has been shown to read it. Phase 2 is GPU work and is **not** in this
repo yet:

- Fine-tune an open VLM to decode heliogram images natively (the
  `VLMDecoder` stub in `heliogram/codec.py` is the plug point; it currently
  raises `NotImplementedError`).
- Measure the un-fine-tuned VLM's OCR on the rendered-text baseline for a fair
  comparison.
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
only `pillow`/`numpy`/`reedsolo` installed, same as Phase 1.

1. **Install GPU extras** (on a machine with a GPU):

   ```bash
   pip install -e ".[gpu]"          # or: pip install -r requirements-gpu.txt
   ```

2. **Generate a synthetic training set** — `heliogram/dataset.py`, CPU-only, needs no GPU and
   runs fine in this same environment. It reuses `encode`/`extract_symbols` from
   `heliogram.codec`, so every training target is the *exact* ground-truth symbol sequence the
   codec wrote for that image — no hand labeling, no drift between what was encoded and what
   the model is asked to transcribe:

   ```bash
   python scripts/gen_dataset.py --out data/phase2_train --n 2000 --seed 0
   python scripts/gen_dataset.py --out data/phase2_val   --n 200  --seed 1 --corruption-prob 0.5
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
   projection layers, and trains across a curriculum (low-density/clean first, widening the
   palette and turning on corruption augmentation in later stages — see `build_curriculum()`):

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
                            palette=8, subpatch=1)
   payload = decode(img, palette=8, subpatch=1, decoder=decoder)
   ```

   `QwenVLDecoder` prompts the model to transcribe the symbol grid, parses the response back
   into symbols, and feeds them through the *same* Reed–Solomon/framing layer `decode_pixels`
   uses — there is no separate RS implementation for the model path.

**The boundary this doesn't cross.** `heliogram/dataset.py` and `scripts/gen_dataset.py` are
CPU-only and *have* been exercised in this repo (see `tests/`). Everything past that — loading
a real VLM, `zero_shot_symbol_error` with a real model, `scripts/train_qlora.py`,
`QwenVLDecoder.__call__` actually reaching `.generate()` — requires a GPU this environment
doesn't have and is untested here. **Decision Gate #1 (see "Roadmap / Phase-2 boundary" above)
is decided by measured numbers from an actual fine-tuned model, run through this same
`decode()` path, on held-out data — not by this scaffold's existence.** Until that run happens
and its numbers are written up the same way `RESULTS.md` was, treat everything in this section
as "code that should let someone with a GPU go measure this," not as a result.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 heliogram contributors.
