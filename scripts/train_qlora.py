#!/usr/bin/env python3
"""scripts/train_qlora.py -- QLoRA fine-tune Qwen2.5-VL-7B-Instruct to read heliogram grids.

*** GPU REQUIRED. UNTESTED IN THIS REPO. *** There is no GPU, torch, transformers, peft, or
bitsandbytes in the environment this script was written in -- it has never been run or imported
end to end. Treat it as a documented, reasonable-defaults STARTING POINT for a training recipe,
not a verified one. Every function that needs torch/transformers/peft/bitsandbytes/datasets
imports those packages locally, inside itself -- nothing above `if __name__ == "__main__":`
requires them, so `python scripts/train_qlora.py --help` works without a GPU (it just prints
usage), but actually running training does not and cannot work here.

Expected hardware: one modern GPU with >= 24GB VRAM to hold Qwen2.5-VL-7B-Instruct in 4-bit NF4
(bitsandbytes) plus LoRA adapters and activations at bf16 compute dtype and batch size 1 (the
default below). Multi-GPU or more VRAM lets you raise --batch-size/--grad-accum-steps; this
script relies entirely on `transformers.Trainer` + `accelerate` for any parallelism, it does not
implement its own.

What this does (the curriculum from the Phase-2 handover, retargeted -- Slice C):
  1. Generate synthetic heliogram images + target symbol strings via heliogram.dataset (see
     scripts/gen_dataset.py) for each curriculum stage -- CPU-only, no GPU needed for this part,
     and every target is exact ground truth from the codec itself (no hand labeling anywhere).
     Every generated image is also padded to an EVEN patch-grid width/height
     (heliogram.dataset.pad_to_even_patch_grid) so the Qwen processor's own `smart_resize` step
     is the identity transform on it -- see `_assert_processor_alignment`/`_identity_pixel_bounds`
     below and heliogram/dataset.py's "PROCESSOR RESIZE HAZARD" module-docstring note (D4 of the
     Phase-2 scaffold review).
  2. Load Qwen2.5-VL-7B-Instruct in 4-bit (bitsandbytes NF4) via transformers, attach LoRA
     adapters (peft) on the attention projections (q_proj/k_proj/v_proj/o_proj) and MLP
     projections (gate_proj/up_proj/down_proj) of the language-model decoder stack, PLUS -- BY
     DEFAULT, not opt-in -- the vision tower's 2x2 patch-merger MLP (`LORA_MERGER_TARGET_MODULES`)
     and, optionally (`--include-vision-blocks`), the full vision-block attention/MLP stack
     (`LORA_VISION_BLOCK_TARGET_MODULES`). See `LORA_MERGER_TARGET_MODULES`'s docstring (D3 of the
     Phase-2 scaffold review) for why the merger is included by default: this task is a
     PERCEPTION shift onto flat color grids, and the merger MLP is exactly the layer whose job is
     to combine 2x2 raw-patch color information into the single feature the LM ever sees.
  3. Fine-tune across a CURRICULUM of increasing difficulty (see build_curriculum() below):
     stage 1 is a cheap, low-density warm-up (small palette, subpatch=1, no corruption); every
     stage after that concentrates specifically on `palette` in `{64, 128, 256}` at
     `subpatch=1` -- the exact (palette, subpatch) regime `heliogram.codec`/RESULTS.md MEASURE
     `decode_pixels` to clean-decode exactly but FAIL under JPEG q70 (and, at larger payloads,
     JPEG q85) -- with corruption augmentation (heliogram.dataset's DEFAULT_CORRUPTIONS,
     mirroring heliogram.harness's realistic envelope) turned on and increasingly concentrated
     on those same failure-inducing corruptions in the final stage. Each stage runs a normal
     Trainer.train() call (loss computed ONLY over assistant-response tokens -- see
     `_mask_prompt_tokens`, D1 of the Phase-2 scaffold review -- with a padding
     `HeliogramVLCollator`, D2, and `gradient_checkpointing=True`) and continues from the previous
     stage's adapter weights. Each stage ALSO generates a small held-out evaluation split
     (disjoint seed space, see `EVAL_SEED_OFFSET`) and, after training, reports its teacher-forced
     per-symbol accuracy (`heliogram.vlm.teacher_forced_symbol_accuracy`, D5(c)) so a curriculum
     run is not blind for tens of GPU-hours -- see `_evaluate_stage_per_symbol_accuracy`.
  4. Save the resulting LoRA adapter (not a full model merge -- get_peft_model()'s
     save_pretrained() saves adapter weights only) to --output-dir, once per stage and once
     more as "final".

THE BET this curriculum trains toward (see heliogram/dataset.py's module docstring for the full
argument): learned classification of a BIG COLOR PALETTE through realistic corruption, NOT
sub-patch geometry. `subpatch` is pinned to 1 in every stage below (see build_curriculum()'s own
docstring) -- `subpatch>1` remains a separate, documented, PIXEL-DECODER-ONLY geometric ceiling
(heliogram/codec.py's DATA HONESTY note, spec/format-v0.1.md section 6a) with no evidence a real
ViT patch embedding can resolve structure smaller than its own patch. A patch's dominant color,
by contrast, is a coarse whole-patch feature a patch embedding could plausibly encode even
through JPEG/resize blur -- untested, which is exactly what this training run and the
before/after measurement it enables (heliogram.vlm.zero_shot_symbol_error on the stock model,
then the same metric on the fine-tuned checkpoint) are for. Succeeding at this is what would
make the README's Bar C (token crossover: fewer total patches than base64 tokens, from ~3-13KB
payloads at palette=128/256) usable end to end instead of a clean-channel-only accounting fact
-- see RESULTS.md's "Token crossover" section for the exact numbers this curriculum is trying to
make real. Nothing in this file measures whether that succeeds; only a real GPU run can.

This is a MINIMAL starting recipe: hyperparameters below (LoRA rank/alpha/dropout, learning
rate, batch size) are standard QLoRA defaults from the literature, not the result of any sweep
-- there is no GPU here to sweep on. Expect to retune once this actually runs against real
hardware and real validation numbers.

DATA HONESTY: this script produces a checkpoint, not a measured result. After training,
evaluate the checkpoint with heliogram.vlm.QwenVLDecoder plugged into
heliogram.codec.decode(..., decoder=...) (or heliogram.vlm.zero_shot_symbol_error for a
before/after comparison against the stock model) and report THOSE numbers. This script's mere
existence, or having "finished successfully", is not a capability claim.

Usage (once you have a GPU and have installed the `gpu` extra, or requirements-gpu.txt):
    python scripts/train_qlora.py --output-dir checkpoints/qwen25vl-heliogram-lora
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Allow running as `python scripts/train_qlora.py` from anywhere without an editable install --
# see scripts/gen_dataset.py's identical comment for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heliogram.codec import PATCH_SIZE  # noqa: E402 -- no heavy deps
from heliogram.dataset import (  # noqa: E402 -- no heavy deps
    DEFAULT_CORRUPTIONS,
    DEFAULT_PALETTES,
    RECOMMENDED_TRAINING_CORRUPTION_PROB,
    build_prompt,
    format_output_text,
    iter_manifest,
    write_dataset,
)
from heliogram.vlm import teacher_forced_symbol_accuracy  # noqa: E402 -- no heavy deps (lazy torch)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Standard QLoRA target-module set for Qwen2-VL/Qwen2.5-VL's language-model decoder stack:
# attention projections + MLP projections ("attn/proj" from the handover). Module names are
# UNVERIFIED here (no GPU to load the model and introspect `model.named_modules()`) -- they
# match the module-naming convention transformers' Qwen2VL/Qwen2_5_VL implementations use as of
# writing, but double-check against your installed transformers version before a real run.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Vision merger MLP target modules (D3 of the Phase-2 scaffold review) -- INCLUDED BY DEFAULT in
# LoRA fine-tuning, not opt-in. Qwen2-VL/Qwen2.5-VL's vision tower ends in a 2x2 spatial
# patch-merger (`Qwen2VLPatchMerger`/`Qwen2_5_VLPatchMerger` in transformers' implementation as of
# writing -- module path `visual.merger`) that projects each 2x2 block of raw ViT-patch embeddings
# down to the single merged feature the language model actually sees; its own submodule names are
# `mlp.0`/`mlp.2` (two nn.Linear layers around a GELU) -- UNVERIFIED here (no GPU to load the
# model and introspect `model.named_modules()`), same caveat as LORA_TARGET_MODULES above.
#
# WHY DEFAULT, NOT OPT-IN: this task is a PERCEPTION shift onto flat color grids surviving JPEG
# corruption, not a language/reasoning shift -- fine-tuning ONLY the LM decoder (the pre-Slice-C
# target_modules list) trains the LM to better GUESS from whatever vision features it is handed,
# without ever letting the model relearn WHICH vision features to compute in the first place. The
# merger MLP is the cheapest (lowest additional trainable-parameter count -- two small Linear
# layers, not an entire vision block) place in the vision path to test whether that matters: it is
# the layer whose entire job is to combine 2x2 raw-patch color information into one feature, i.e.
# exactly the step at which "which of P colors is this patch" would need to be preserved (or
# actively relearned) through JPEG chroma-subsampling blur, per heliogram/dataset.py's "THE BET"
# paragraph. Leaving it out of target_modules by default would silently bias this experiment
# toward "can the LM decoder work around whatever the frozen, ImageNet/web-photo-trained vision
# tower already computes" rather than the actual question ("can this model learn to perceive a
# flat color-grid lattice at all").
LORA_MERGER_TARGET_MODULES = ["mlp.0", "mlp.2"]

# Full vision-BLOCK attention/MLP target modules -- opt-in only (`--include-vision-blocks`), NOT
# part of the default target_modules. Qwen2.5-VL's vision transformer blocks
# (`Qwen2_5_VLVisionBlock` as of writing) use `attn.qkv`/`attn.proj` for attention and a SwiGLU
# MLP (`mlp.gate_proj`/`mlp.up_proj`/`mlp.down_proj`) per the model's technical report -- AGAIN
# UNVERIFIED here (no GPU to introspect a real checkpoint's `model.named_modules()`), and a
# meaningfully bigger, more speculative bet than the merger MLP above: fine-tuning the FULL vision
# tower risks catastrophically forgetting whatever general-purpose visual features the tower
# learned from its (almost certainly natural-photo-dominated) pretraining corpus, for a workload
# (flat solid-color patches) that looks nothing like that corpus. Opt-in via `--include-vision-
# blocks` lets a real GPU run A/B this against the merger-only default once actual validation
# numbers (see `_evaluate_stage_per_symbol_accuracy`) are available to judge whether the extra
# trainable-parameter budget (and forgetting risk) is worth it -- this script does not claim it is.
LORA_VISION_BLOCK_TARGET_MODULES = [
    "attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Seed offset for each curriculum stage's held-out EVALUATION split (D2 of the Phase-2 scaffold
# review), so the eval split's random payloads never collide with the TRAINING split's for the
# same stage/--seed: `heliogram.dataset.random_payload(seed, size)` is a pure function of `seed`
# alone, so training uses `args.seed` and eval uses `args.seed + EVAL_SEED_OFFSET`. Not
# cryptographically guaranteed disjoint against an adversarially chosen --seed -- just large
# enough (10 million) to never collide in practice for the small integer seeds (0, 1, 2, ...) this
# project actually uses.
EVAL_SEED_OFFSET = 10_000_000


def _build_lora_target_modules(include_vision_blocks: bool) -> List[str]:
    """Assemble the full `target_modules` list for `LoraConfig` (D3 of the Phase-2 scaffold
    review): the LM decoder stack (`LORA_TARGET_MODULES`) plus the vision merger MLP
    (`LORA_MERGER_TARGET_MODULES`, always included), plus -- only when `include_vision_blocks` is
    True (`--include-vision-blocks`) -- the full vision-block attention/MLP stack
    (`LORA_VISION_BLOCK_TARGET_MODULES`). See those three constants' own docstrings/comments for
    what each covers and why the merger is default-on while the full vision blocks are opt-in."""
    target_modules = list(LORA_TARGET_MODULES) + list(LORA_MERGER_TARGET_MODULES)
    if include_vision_blocks:
        target_modules += list(LORA_VISION_BLOCK_TARGET_MODULES)
    return target_modules


@dataclass
class CurriculumStage:
    """One stage of the training curriculum: a (palette, subpatch, payload_sizes,
    corruption_prob) regime to generate fresh examples for and train on, continuing from
    whatever adapter weights the previous stage produced. Ordered low-density/clean -> higher-
    density/corrupted, per the Phase-2 handover's curriculum guidance, retargeted (Slice C) to
    concentrate later stages on the large-palette-under-corruption bet (see module docstring).

    `corruptions` (optional) overrides the corruption suite `write_dataset` draws from for this
    stage only (default `None` -- falls back to `heliogram.dataset.DEFAULT_CORRUPTIONS`, the
    full realistic-envelope suite). A stage that wants to concentrate specifically on the
    corruptions RESULTS.md pins as this palette range's actual measured failure points (rather
    than spreading its corruption budget uniformly across the whole realistic-envelope suite)
    passes a narrower dict here -- see stage4 in build_curriculum() below. Any custom dict here
    must still include a "clean" entry whenever this stage's `corruption_prob < 1.0` (see
    heliogram.dataset.generate_examples' contract: "clean" is looked up unconditionally for the
    "no corruption fired" branch).
    """

    name: str
    n_examples: int
    palettes: Sequence[int]
    subpatches: Sequence[int]
    payload_sizes: Sequence[int]
    corruption_prob: float
    epochs: int
    corruptions: Optional[Dict[str, Callable]] = field(default=None)


# The specific corruptions RESULTS.md pins as this palette range's actual measured failure
# points (see heliogram/codec.py's DATA HONESTY note and RESULTS.md's Headline/Token-crossover
# sections): palette=128/256 FAIL jpeg_q70 at every tested payload size, and jpeg_q85 too at
# larger payloads; "combined" (resize + JPEG q70 + crop/pad composed) is the harness's own
# worst-tested corruption. "clean" stays included -- see CurriculumStage's docstring: a custom
# `corruptions` dict is looked up for the "no corruption fired" case too whenever
# corruption_prob < 1.0, so even a "hard focus" stage that is mostly (not always) corrupted
# needs a "clean" entry.
_HARD_CORRUPTION_NAMES = {"clean", "jpeg_q70", "jpeg_q85", "combined"}


def build_curriculum(n_examples_per_stage: int = 2000) -> List[CurriculumStage]:
    """Default curriculum (Slice C retarget): a cheap warm-up, then three stages that
    concentrate on `DEFAULT_PALETTES` (`{64, 128, 256}` -- see heliogram/dataset.py's module
    docstring for why) at `subpatch=1`, clean first and then increasingly corrupted, finishing
    with a stage that up-weights the SPECIFIC corruptions (`jpeg_q70`/`jpeg_q85`/`combined`)
    RESULTS.md measures this palette range to actually fail under -- rather than spreading a
    generic corruption budget uniformly across the whole realistic-envelope suite for the whole
    run. See the module docstring's "THE BET" paragraph for why this is the retargeted goal.

    `subpatch` is pinned to 1 in every stage below: subpatch>1 is a pixel-decoder-only
    geometric ceiling with no evidence a real ViT encoder can resolve it (see
    heliogram/codec.py's DATA HONESTY note and spec/format-v0.1.md section 6a) -- training a
    VLM to emit subpatch>1 targets is speculative and left to a manually constructed curriculum
    (this function's return value is a plain list; edit it or write your own), not this default.

    Palette/payload/corruption-prob/epoch numbers below are a reasonable-starting-point design,
    same caveat as everywhere else in this file: not the result of a sweep (no GPU here to sweep
    on), and not a claim about what actually works -- only a real training + evaluation run can
    show that.
    """
    return [
        CurriculumStage(
            name="stage1_warmup_small_palette_clean",
            n_examples=n_examples_per_stage,
            palettes=[2, 4],
            subpatches=[1],
            payload_sizes=[16, 48],
            corruption_prob=0.0,
            epochs=1,
        ),
        CurriculumStage(
            name="stage2_large_palette_clean",
            n_examples=n_examples_per_stage,
            palettes=list(DEFAULT_PALETTES),
            subpatches=[1],
            payload_sizes=[128, 1024, 4096],
            corruption_prob=0.0,
            epochs=1,
        ),
        CurriculumStage(
            name="stage3_large_palette_corrupted",
            n_examples=n_examples_per_stage,
            palettes=list(DEFAULT_PALETTES),
            subpatches=[1],
            payload_sizes=[128, 1024, 4096],
            corruption_prob=RECOMMENDED_TRAINING_CORRUPTION_PROB,
            epochs=2,
        ),
        CurriculumStage(
            name="stage4_hard_corruption_focus",
            n_examples=n_examples_per_stage,
            palettes=list(DEFAULT_PALETTES),
            subpatches=[1],
            payload_sizes=[128, 1024, 4096],
            corruption_prob=0.9,
            epochs=2,
            corruptions={
                name: fn
                for name, fn in DEFAULT_CORRUPTIONS.items()
                if name in _HARD_CORRUPTION_NAMES
            },
        ),
    ]


def _assert_processor_alignment(image: Any, patch_size: int) -> None:
    """Guard against Qwen's processor `smart_resize` silently resampling a heliogram grid OFF its
    symbol lattice (D4 of the Phase-2 scaffold review -- see heliogram/dataset.py's "PROCESSOR
    RESIZE HAZARD" module-docstring note). Qwen2-VL/Qwen2.5-VL's image processor snaps input
    pixel dimensions to a multiple of `patch_size * 2` (2x2 spatial merge) BEFORE the vision
    tower ever sees them. `heliogram.dataset.write_dataset` (via
    `heliogram.dataset.pad_to_even_patch_grid`) already guarantees every image it writes has
    even patch-grid dimensions, i.e. pixel dimensions already at that alignment -- this assertion
    is the second half of that guarantee: it fires HERE, at the point an image actually enters
    the processor, so a caller who fed this script some other image source gets a loud,
    immediate, specific error instead of silent color corruption several layers away inside the
    processor's own resampling code."""
    merge_px = patch_size * 2
    if image.width % merge_px != 0 or image.height % merge_px != 0:
        raise ValueError(
            f"image size {image.size} is not aligned to the Qwen processor's {merge_px}px "
            f"(patch_size={patch_size} * 2x2 spatial merge) grid -- feeding this image to "
            "processor(...) would let smart_resize silently resample it OFF the heliogram symbol "
            "lattice before the model ever sees it. This should be unreachable for images "
            "written by heliogram.dataset.write_dataset (pad_to_even_patch_grid guarantees "
            "alignment) -- if you hit this, some other image source bypassed that padding step."
        )


def _identity_pixel_bounds(image: Any) -> Dict[str, int]:
    """`min_pixels = max_pixels = image.width * image.height` forces Qwen's `smart_resize` to
    treat the image's CURRENT pixel count as both its lower and upper bound, so -- PROVIDED the
    image's dimensions are already `patch_size * 2`-aligned (see `_assert_processor_alignment`,
    called immediately before every use of this function below) -- `smart_resize`'s search for
    "the closest aligned (h, w) within [min_pixels, max_pixels]" has exactly one feasible point:
    the image's own (already-aligned) dimensions, i.e. the identity transform. Without this, the
    processor's own DEFAULT `min_pixels`/`max_pixels` bounds (tuned for natural photos, not
    heliogram grids that can be far smaller or, at large payloads, far larger) could still force
    a resize even on an already-aligned image."""
    return {"min_pixels": image.width * image.height, "max_pixels": image.width * image.height}


def _mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> List[int]:
    """Build a `labels` sequence that masks everything except the assistant-response tokens to
    `ignore_index` (D1 of the Phase-2 scaffold review). Given the FULL (prompt+response)
    tokenized sequence's `input_ids` and the length of the PROMPT-ONLY tokenization (`prompt_len`
    -- see `_build_hf_dataset`'s `_load` for how that is computed), returns a same-length list:
    the first `prompt_len` positions replaced with `ignore_index`, the rest equal to `input_ids`.

    WHY THIS MATTERS: `labels = input_ids.clone()` (the previous behavior) computed loss over the
    ENTIRE sequence, including the prompt text AND the huge image-placeholder token block (a
    single heliogram image at typical training resolution expands to hundreds of vision tokens)
    -- neither of which the model is being trained to PRODUCE. That dilutes the actual target-
    token (the transcribed grid) loss by roughly 50-90% depending on image/grid size, since most
    of the sequence's tokens are prompt/image tokens the model never needs to generate.

    WHY LENGTH-BASED, NOT STRING SEARCH: locating "where does the assistant response start" via
    tokenizing prompt-only and full sequences and using the prompt-only length as the split point
    is robust to tokenizer/chat-template internals (special tokens, image-placeholder expansion,
    role markers) that a fragile string search over already-tokenized IDs would have to
    reimplement or guess at. See `teacher_forced_symbol_accuracy` in heliogram/vlm.py for the
    same prompt-length-via-separate-tokenization technique used again for eval.

    Pure Python/list based (no torch) SPECIFICALLY so this masking arithmetic is unit-testable in
    this repo's CPU/no-torch environment (see tests/test_phase2_scaffold.py) -- `_build_hf_dataset`
    (the real Trainer-facing path) is a thin `torch.tensor(...)` wrapper around this exact
    function's output, not a separate reimplementation.
    """
    if prompt_len < 0 or prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len ({prompt_len}) out of range for input_ids of length {len(input_ids)}"
        )
    return [ignore_index] * prompt_len + list(input_ids[prompt_len:])


def _pad_sequences(
    sequences: Sequence[Sequence[int]], pad_value: int, max_len: Optional[int] = None
) -> List[List[int]]:
    """Right-pad a batch of integer sequences (plain Python lists, e.g. already `.tolist()`'d
    tensors) to a common length with `pad_value`. Pure Python -- this is the actual padding
    arithmetic `HeliogramVLCollator` (D2 of the Phase-2 scaffold review) delegates to, factored
    out so it is unit-testable without torch installed (see tests/test_phase2_scaffold.py).
    `max_len` defaults to the longest sequence in the batch; passing it explicitly lets a caller
    pad several related batches (e.g. input_ids/attention_mask/labels for the SAME examples) to
    one consistent length."""
    if max_len is None:
        max_len = max((len(s) for s in sequences), default=0)
    return [list(s) + [pad_value] * (max_len - len(s)) for s in sequences]


@dataclass
class HeliogramVLCollator:
    """Padding data collator for variable-length Qwen2-VL/Qwen2.5-VL training examples (D2 of the
    Phase-2 scaffold review). Without this, `transformers.Trainer`'s default collation either
    crashes (it cannot stack ragged `input_ids`/`labels` tensors of different lengths into one
    batch tensor) or -- if `--batch-size 1` happens to mask the crash -- silently never exercises
    real batching at all. Right-pads `input_ids` (with `pad_token_id`), `attention_mask` (with 0),
    and `labels` (with `label_pad_token_id`, -100, so padding never contributes to the loss); for
    Qwen2-VL-style vision inputs, `pixel_values` is CONCATENATED (not stacked -- Qwen2-VL flattens
    all images' patches into one leading dimension, with `image_grid_thw` recording each image's
    own (t, h, w) grid to un-flatten downstream) and `image_grid_thw` is concatenated to match.

    `pad_token_id` has no library-wide default (it is a property of the loaded tokenizer, see
    `_pad_token_id`) -- pass it explicitly at construction.

    ALL torch use is local to `__call__` -- constructing this dataclass (e.g. to introspect it in
    a CPU-only test) never requires torch; only actually collating a batch of real torch tensors
    does. The padding LENGTH/VALUE arithmetic itself is delegated to `_pad_sequences` (pure
    Python, see that function's docstring) precisely so it is unit-testable without torch.

    UNTESTED against a real Trainer run (see module docstring): the `pixel_values`/
    `image_grid_thw` concatenation shape follows Qwen2-VL's documented flattened-patches
    convention, but has never been exercised against real processor output in this environment.
    """

    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        input_ids = [f["input_ids"].tolist() for f in features]
        attention_mask = [f["attention_mask"].tolist() for f in features]
        labels = [f["labels"].tolist() for f in features]
        max_len = max((len(x) for x in input_ids), default=0)

        batch: Dict[str, Any] = {
            "input_ids": torch.tensor(
                _pad_sequences(input_ids, self.pad_token_id, max_len),
                dtype=features[0]["input_ids"].dtype,
            ),
            "attention_mask": torch.tensor(
                _pad_sequences(attention_mask, 0, max_len),
                dtype=features[0]["attention_mask"].dtype,
            ),
            "labels": torch.tensor(
                _pad_sequences(labels, self.label_pad_token_id, max_len),
                dtype=features[0]["labels"].dtype,
            ),
        }
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"].reshape(-1, f["image_grid_thw"].shape[-1]) for f in features],
                dim=0,
            )
        return batch


def _pad_token_id(processor: Any) -> int:
    """Read a usable pad token id off `processor` (its own `pad_token_id`, or its wrapped
    `tokenizer.pad_token_id`, falling back to `eos_token_id` -- many causal-LM tokenizers, Qwen's
    included, have no dedicated pad token and conventionally reuse EOS for padding). Raises
    ValueError (never silently picks an arbitrary id like 0, which could collide with a real
    vocabulary token) if neither is set."""
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError(
            "processor's tokenizer has neither pad_token_id nor eos_token_id set -- cannot build "
            "a padding HeliogramVLCollator without a valid pad id"
        )
    return int(pad_id)


def _load_model_and_processor(
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
):
    """Load the base model in 4-bit and attach LoRA adapters. ALL heavy imports are local to
    this function -- torch/transformers/peft/bitsandbytes are only required once this is
    actually called (i.e. only from main(), only on a real GPU box).

    `target_modules` (D3 of the Phase-2 scaffold review -- see `_build_lora_target_modules`) now
    includes the vision merger MLP by default, and optionally the full vision-block
    attention/MLP stack, alongside the LM decoder projections `LORA_TARGET_MODULES` alone used to
    cover.

    UNTESTED (see module docstring): this exact call sequence -- BitsAndBytesConfig +
    from_pretrained(quantization_config=..., device_map="auto") + get_peft_model -- follows the
    standard documented QLoRA recipe, but has never been run here. The
    Qwen2_5_VLForConditionalGeneration class name in particular is version-sensitive; the
    fallback below is a hedge, not a verified alternative.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:  # pragma: no cover -- depends on installed transformers version
        from transformers import AutoModelForVision2Seq as ModelClass  # best-effort fallback

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = ModelClass.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(base_model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def _build_hf_dataset(manifest_path: Path, processor):
    """Wrap a heliogram.dataset manifest.jsonl into a `datasets.Dataset` of Trainer-ready
    examples (processed image + tokenized target string, chat-template formatted, with
    prompt-masked `labels`). Local import of `datasets` (the Hugging Face library, a `gpu`
    extra) -- see module docstring.

    Uses the CANONICAL prompt/output-contract functions from heliogram.dataset
    (`build_prompt`/`format_output_text`, D5(b) of the Phase-2 scaffold review) -- the exact same
    functions `heliogram.vlm.QwenVLDecoder` uses at inference time -- rather than an
    independently-written prompt string.

    LABEL MASKING (D1 of the Phase-2 scaffold review, see `_mask_prompt_tokens`'s docstring for
    the full "why"): tokenizes the PROMPT-ONLY chat text (with the same image, so image-
    placeholder token expansion matches) separately from the FULL (prompt+response) chat text,
    and uses the prompt-only tokenization's length as the split point for masking `labels` --
    everything before that point (prompt text AND the image-placeholder token block) becomes
    -100 (ignored by the loss), everything from that point on (the assistant's transcribed grid)
    keeps its real token id as its own label.

    PROCESSOR ALIGNMENT (D4): calls `_assert_processor_alignment` before every `processor(...)`
    call, and passes `_identity_pixel_bounds(image)` so `smart_resize` cannot resize an
    already-aligned image just because it falls outside the processor's own default pixel bounds.

    UNTESTED (see module docstring): the exact processor(...) call shape for Qwen2.5-VL's
    "assistant turn is the target" training setup, and the prompt/full-tokenization prefix-
    consistency assumption label-masking relies on (see `_mask_prompt_tokens`'s docstring), are
    both version-sensitive / unverified without a real GPU + tokenizer.
    """
    import datasets
    import torch
    from PIL import Image

    records = list(iter_manifest(manifest_path))

    def _load(record):
        image = Image.open(record["image_path"]).convert("RGB")
        patch_size = int(record["patch_size"])
        subpatch = int(record["subpatch"])
        palette = int(record["palette"])
        width = image.width // patch_size
        height = image.height // patch_size

        _assert_processor_alignment(image, patch_size)
        pixel_bounds = _identity_pixel_bounds(image)

        prompt = build_prompt(palette, width, height, subpatch)
        response_text = format_output_text(record["target"], width, subpatch)

        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            },
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
        prompt_chat_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_chat_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_inputs = processor(
            text=[prompt_chat_text], images=[image], return_tensors="pt", **pixel_bounds
        )
        full_inputs = processor(
            text=[full_chat_text], images=[image], return_tensors="pt", **pixel_bounds
        )

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full_input_ids = full_inputs["input_ids"][0]
        labels_list = _mask_prompt_tokens(full_input_ids.tolist(), prompt_len)

        model_inputs = {k: v.squeeze(0) for k, v in full_inputs.items()}
        model_inputs["labels"] = torch.tensor(labels_list, dtype=full_input_ids.dtype)
        return model_inputs

    ds = datasets.Dataset.from_list(records)
    return ds.map(_load, remove_columns=ds.column_names)


def _evaluate_stage_per_symbol_accuracy(
    model, processor, eval_manifest_path: Path, n_examples: int
) -> float:
    """Cheap, per-stage held-out validation metric (D2 of the Phase-2 scaffold review): average
    TEACHER-FORCED per-symbol accuracy (`heliogram.vlm.teacher_forced_symbol_accuracy`, D5(c))
    over the first `n_examples` records of `eval_manifest_path` -- deliberately a HANDFUL of
    examples (not the full held-out split), since this runs once per curriculum stage on top of
    an already-expensive training run and only needs to be cheap enough to catch an obviously
    broken stage before the NEXT one burns more GPU-hours on top of it, not to be a
    publication-grade evaluation. See `teacher_forced_symbol_accuracy`'s own docstring for why
    teacher-forced (not free-running generate-and-compare) is the metric used here: it isolates
    PERCEPTION from sequence-drift/RS-adjacent confounds, which is exactly what this project's
    actual bet (heliogram/dataset.py's "THE BET") is about. Returns 0.0 if the eval manifest has
    no records (should be unreachable given `--eval-examples-per-stage`'s `max(1, ...)` floor)."""
    from PIL import Image

    records = list(iter_manifest(eval_manifest_path))[:n_examples]
    if not records:
        return 0.0
    accuracies = []
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        accuracies.append(
            teacher_forced_symbol_accuracy(
                model,
                processor,
                image,
                record["target"],
                palette=int(record["palette"]),
                patch_size=int(record["patch_size"]),
                subpatch=int(record["subpatch"]),
            )
        )
    return sum(accuracies) / len(accuracies)


def _train_stage(
    model, processor, stage: CurriculumStage, args: argparse.Namespace, stage_dir: Path
):
    """Generate this stage's training AND held-out evaluation datasets, run one Trainer.train()
    call (padded batches via `HeliogramVLCollator`, D2; masked labels via `_build_hf_dataset`,
    D1; gradient checkpointing on), then report the stage's held-out per-symbol accuracy (D2/
    D5(c)) so a curriculum run is not blind for tens of GPU-hours. Local import of
    `transformers.Trainer`/`TrainingArguments` -- see module docstring."""
    from transformers import Trainer, TrainingArguments

    manifest_path = write_dataset(
        stage_dir / "train",
        stage.n_examples,
        palettes=stage.palettes,
        subpatches=stage.subpatches,
        payload_sizes=stage.payload_sizes,
        patch_size=args.patch_size,
        nsym=args.nsym,
        seed=args.seed,
        corruption_prob=stage.corruption_prob,
        corruptions=stage.corruptions,
    )
    dataset = _build_hf_dataset(manifest_path, processor)

    # Held-out eval split: disjoint seed space (EVAL_SEED_OFFSET) from the training split above,
    # same (palette/subpatch/payload_size/corruption) regime as this stage so the eval number is
    # actually representative of what this stage trained on.
    n_eval = max(1, min(args.eval_examples_per_stage, stage.n_examples))
    eval_manifest_path = write_dataset(
        stage_dir / "eval",
        n_eval,
        palettes=stage.palettes,
        subpatches=stage.subpatches,
        payload_sizes=stage.payload_sizes,
        patch_size=args.patch_size,
        nsym=args.nsym,
        seed=args.seed + EVAL_SEED_OFFSET,
        corruption_prob=stage.corruption_prob,
        corruptions=stage.corruptions,
    )
    eval_dataset = _build_hf_dataset(eval_manifest_path, processor)

    collator = HeliogramVLCollator(pad_token_id=_pad_token_id(processor))

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / stage.name / "trainer_state"),
        num_train_epochs=stage.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    per_symbol_accuracy = _evaluate_stage_per_symbol_accuracy(
        model, processor, eval_manifest_path, args.eval_symbol_accuracy_examples
    )
    print(
        f"=== {stage.name}: held-out teacher-forced per-symbol accuracy "
        f"(n={args.eval_symbol_accuracy_examples}): {per_symbol_accuracy:.4f} ==="
    )
    trainer.log({"stage_held_out_per_symbol_accuracy": per_symbol_accuracy})

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HF model id (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/qwen25vl-heliogram-lora"),
        help="where to save LoRA adapter checkpoints, one subdir per stage plus 'final'",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/phase2_curriculum"),
        help="working directory for per-stage generated datasets (images + manifest)",
    )
    parser.add_argument(
        "--n-examples-per-stage",
        type=int,
        default=2000,
        help="examples generated per curriculum stage (default: 2000)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"patch size in px, must match what the deployed decoder will see "
        f"(default: {PATCH_SIZE})",
    )
    parser.add_argument(
        "--nsym",
        type=int,
        default=32,
        help="Reed-Solomon parity bytes for generated images (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=0, help="dataset generation seed (default: 0)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--include-vision-blocks",
        action="store_true",
        help="also LoRA-tune the full vision-tower attention/MLP stack "
        "(LORA_VISION_BLOCK_TARGET_MODULES), not just the LM decoder + merger MLP -- see that "
        "constant's docstring for the (bigger, more speculative) tradeoff this opts into "
        "(default: off; the merger MLP is always included regardless of this flag)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="per-device train batch size (default: 1)"
    )
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--eval-examples-per-stage",
        type=int,
        default=50,
        help="held-out evaluation examples generated per curriculum stage, disjoint seed space "
        "from that stage's training data (default: 50; see EVAL_SEED_OFFSET)",
    )
    parser.add_argument(
        "--eval-symbol-accuracy-examples",
        type=int,
        default=8,
        help="number of held-out examples to run heliogram.vlm.teacher_forced_symbol_accuracy "
        "over after each stage -- deliberately a cheap handful, not the full eval split "
        "(default: 8)",
    )
    return parser


def main(argv: list = None) -> int:
    args = build_parser().parse_args(argv)

    print(__doc__)
    print(
        "\n*** This script has not been run in the environment that generated it (no GPU "
        "available there). Review every default above against your actual hardware and "
        "installed package versions before trusting it to work unmodified. ***\n"
    )

    target_modules = _build_lora_target_modules(args.include_vision_blocks)
    model, processor = _load_model_and_processor(
        args.base_model, args.lora_rank, args.lora_alpha, args.lora_dropout, target_modules
    )

    for stage in build_curriculum(args.n_examples_per_stage):
        print(f"=== curriculum stage: {stage.name} ===")
        stage_dataset_dir = Path(args.dataset_dir) / stage.name
        model = _train_stage(model, processor, stage, args, stage_dataset_dir)
        stage_output = Path(args.output_dir) / stage.name
        model.save_pretrained(str(stage_output))
        print(f"saved adapter for {stage.name} -> {stage_output}")

    final_output = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_output))
    processor.save_pretrained(str(final_output))
    print(f"\ndone. Final adapter + processor saved to {final_output}")
    print(
        "Next: load this checkpoint, pass it to heliogram.vlm.QwenVLDecoder, plug that into "
        "heliogram.codec.decode(..., decoder=...), and measure REAL symbol-error/decode-"
        "success numbers (heliogram.vlm.zero_shot_symbol_error gives a before/after baseline "
        "against the stock model). Do not report this checkpoint's existence as a capability "
        "result on its own -- see the README's 'Phase 2 (GPU)' section."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
