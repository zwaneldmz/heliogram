# Threat model: the heliogram pre-ingest detector

This document scopes `heliogram.instruments.foreign_tile`'s `guard`/`is_foreign`
— the model-free, pre-ingest detector this repository ships — against the
questions a reader will actually have: what does it catch, what does it
measurably fail to catch, and what image-security concerns are simply out of
scope for it. It exists because the earlier "detector before capability"
framing in [`README.md`](../README.md#defensive-contribution-and-safety-gate)
needs one place that separates these clearly instead of leaving them implied
across module docstrings.

**Scope, throughout this document:** everything here is specific to
**Qwen2.5-VL** (3B and 7B, `transformers==5.13.0`) and to *this codec family* —
the v0.1 wire format in [`spec/format-v0.1.md`](../spec/format-v0.1.md): solid-color
`PATCH_SIZE`-px (14px default, `heliogram/codec.py`) blocks, one symbol per
patch at `subpatch=1`, a row-0 calibration row, Reed–Solomon framing. Nothing
here transfers to other model families, other patch sizes, or other optical
encodings — see README's "Scope" section for why closed API models in
particular are excluded (opaque, changeable preprocessing).

## What this detector catches

`heliogram.instruments.foreign_tile.is_foreign` (wrapped by `guard`) is a
two-stage, model-free classifier over pixels only — no torch/transformers
import at any scope (see the module's DATA HONESTY note):

1. **Structural signature.** `patch_structure_score` measures mean within-cell
   pixel variance over the whole patch tiling. A genuine heliogram tile of any
   palette/subpatch is exactly solid per `patch_size`×`patch_size` cell (score
   0.0); a natural photo has real texture and scores far above
   `DEFAULT_VAR_THRESHOLD=250.0`. Anything that does not clear this cheap first
   filter is treated as "looks like a natural image" and passed through
   without ever attempting a decode.
2. **Decode-under-trusted-config check.** Anything that is patch-structured is
   then run through `heliogram.codec.decode_pixels` under every
   `AllowListEntry` (palette, patch_size, subpatch) the caller trusts. A tile
   that decodes under some allow-listed entry is trusted traffic, not flagged.
   A tile that is patch-structured but decodes under **none** of them is
   flagged as foreign: a heliogram-shaped payload nobody approved.

This second check is what catches the hard case the detector was specifically
built for: a **learned-alphabet** tile (`_shuffled_alphabet_tile`) — genuinely
patch-aligned and solid-per-cell, so it clears step 1 exactly like a trusted
tile, but with its data cells repainted through a seeded permutation of the
palette so it is internally inconsistent with its own row-0 calibration and
decodes under no allow-listed config. This is intended to stand in for "a model
learned a non-standard color/symbol mapping for the same palette."

**Measured result** (`build_eval_set` + `evaluate_detector`, reproduce with
`python -m heliogram.instruments.foreign_tile`):

```
=== Standard eval: evaluate_detector(*build_eval_set(seed)) ===
TPR=1.000 (8/8 positives correctly flagged foreign, including the
learned-alphabet-like hard positives), FPR=0.000 (0/16 negatives —
natural images + allow-listed trusted tiles — incorrectly flagged).
```

At this eval set's seed and construction, the detector catches every
off-allowlist and learned-alphabet positive, with zero false positives against
natural images and the detector's own trusted traffic.

## What it does NOT catch (measured / by construction)

**Texture-camouflaged tiles — measured, not just predicted.** The `# ponytail:`
note in `is_foreign` (`heliogram/instruments/foreign_tile.py`) predicts that a
single global variance threshold cannot be adversarially robust: a payload
whose cells still average to the right symbol color but carry added per-pixel
noise can clear `DEFAULT_VAR_THRESHOLD` and sail through step 1 as "natural,"
never reaching the decisive decode check at all. `_texture_camouflaged_tile` /
`build_texture_camouflage_positives` build exactly this attack (zero-mean,
seeded per-pixel noise at amplitude 50, on top of both an off-allowlist tile
and a learned-alphabet tile) and this is now a direct measurement:

```
=== Texture-camouflage family eval: build_texture_camouflage_positives vs. the SAME negatives ===
TPR=0.000 (0/4 positives correctly flagged foreign), FPR=0.000 (0/16
negatives incorrectly flagged).
```

Against the *same* negative set the standard eval used (so FPR is directly
comparable), TPR collapses from 1.000 to **0.000**: every texture-camouflaged
positive — still, by construction, a heliogram-shaped payload nobody approved —
is waved through by the cheap variance filter before the decode check ever
runs. Reproduce both numbers together with `python -m heliogram.instruments.foreign_tile`.
**Do not treat `DEFAULT_VAR_THRESHOLD`, or any single global variance cutoff,
as adversarially robust.** It is a cheap ordinary-image filter, not a security
boundary against an adversary who can shape pixel-level texture.

**Anything shaped to look like natural-image statistics, more generally.** The
texture-camouflage attack above is one concrete instance of a broader class:
the detector's structural signature relies on genuine heliogram tiles having
near-zero within-cell variance, which is a strong but narrow prior. Any
payload construction that mimics natural-image texture statistics closely
enough to clear the variance threshold — whether by adding noise (measured
above) or by other means not tested here — is not caught by step 1, and if it
also happens to decode under no allow-listed config it is still only caught if
it fails to clear step 1 in the first place. No claim is made here about the
full space of such constructions; only the one measured attack above is
characterized.

**Payload content under a trusted, allow-listed config — by construction.**
`is_foreign`/`guard` is a *channel* detector: it asks "was this encoded under a
config nobody approved," never "what does this decode to." A payload that
decodes cleanly under a trusted allow-listed config is, by this detector's own
design, never flagged — regardless of what bytes it contains. Whether those
bytes carry a behavioral attack (a persona hijack, a forced output schema, a
triggered tool call) once decoded and read by a real model is a different
question this detector makes no attempt to answer; see
`heliogram.instruments.injection_bench`'s module docstring, which names this
gap explicitly: "`guard` is a structural/channel detector, blind to decoded
payload CONTENT by design, so it cannot by itself catch a behavioral payload
encoded under a trusted, allow-listed config." `measure_behavioral_capacity`
(the half of that module that would measure how much behavioral influence
survives) requires a real model and is not run anywhere in this repo (no GPU
here) — see README's "Defensive contribution and safety gate" section.

## What exists more broadly (out of scope for this detector)

Image-borne prompt injection is a general problem; this is a narrow structural
detector for one specific codec family, and catching heliogram-family payloads
is **not** the same as catching image-borne injection in general. In
particular, this detector makes no claim to catch:

- **Natural-image adversarial perturbations** — small, imperceptible
  pixel-space perturbations on an otherwise ordinary photo that steer a real
  VLM's behavior. These do not produce heliogram's structural signature
  (even patch tiling, near-solid cells) at all, so `patch_structure_score`
  correctly treats them as ordinary images and `is_foreign` never flags them.
  Nothing in this repository measures or defends against this class.
- **Typographic / visual-text injection** — instructions rendered as ordinary
  typeset text inside an image, read by a VLM's OCR-like perception rather than
  by any patch-color code. This repository's typography work
  (`heliogram.typography`, README's "typography pivot") only measures whether
  *dense, high-entropy* rendered text is readable/dense enough to compete with
  heliogram's own codec — it is not a defense against typeset instructions and
  says nothing about detecting them.
- **Steganographic channels** — payloads hidden in the low-order bits or subtle
  statistics of an otherwise natural-looking image, designed specifically to
  avoid any structural signature. This detector's entire method depends on a
  gap between "genuine heliogram tile" and "natural photo" that a
  steganographic scheme is, by design, built to erase; it is untested here and
  there is no reason to expect `is_foreign` catches it.

None of this is a gap in `heliogram.instruments.foreign_tile` specifically — it
is a scope boundary. The detector was built to catch one well-defined thing (a
heliogram-like payload of this codec family, outside a trusted allow-list) and
does that, with the measured exception above; it was never intended as, and
must not be read as, general-purpose defense against image-borne prompt
injection.

## Summary table

| Threat | Status | Evidence |
|---|---|---|
| Off-allowlist heliogram tile (different palette/config) | Caught | TPR=1.000 (`build_eval_set` "plain" positives) |
| Learned-alphabet tile (defeats row-0 calibration) | Caught | TPR=1.000 (`build_eval_set` hard positives, same eval run) |
| Texture-camouflaged tile (noise dusted on a foreign tile) | **Not caught** | Measured TPR=0.000 at the same FPR=0.000 |
| Payload content under a trusted config | Not in scope, by construction | `injection_bench` module docstring |
| Natural-image adversarial perturbation | Not in scope | No structural signature to detect |
| Typographic / visual-text injection | Not in scope | `heliogram.typography` measures density/OCR only, not detection |
| Steganographic channel | Not in scope, untested | No claim made |

Reproduce the two measured rows together:

```bash
python -m heliogram.instruments.foreign_tile
```
