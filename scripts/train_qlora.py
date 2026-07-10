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

What this does (the curriculum from the Phase-2 handover):
  1. Generate synthetic heliogram images + target symbol strings via heliogram.dataset (see
     scripts/gen_dataset.py) for each curriculum stage -- CPU-only, no GPU needed for this part,
     and every target is exact ground truth from the codec itself (no hand labeling anywhere).
  2. Load Qwen2.5-VL-7B-Instruct in 4-bit (bitsandbytes NF4) via transformers, attach LoRA
     adapters (peft) on the attention projections (q_proj/k_proj/v_proj/o_proj) and MLP
     projections (gate_proj/up_proj/down_proj) of the language-model decoder stack.
  3. Fine-tune across a CURRICULUM of increasing difficulty (see build_curriculum() below):
     stage 1 is the lowest-density, cleanest regime (small palette, subpatch=1 -- one symbol
     per patch, no corruption augmentation); later stages widen the palette and turn on
     corruption augmentation (heliogram.dataset's DEFAULT_CORRUPTIONS, mirroring
     heliogram.harness's realistic envelope). Each stage runs a normal Trainer.train() call and
     continues from the previous stage's adapter weights.
  4. Save the resulting LoRA adapter (not a full model merge -- get_peft_model()'s
     save_pretrained() saves adapter weights only) to --output-dir, once per stage and once
     more as "final".

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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

# Allow running as `python scripts/train_qlora.py` from anywhere without an editable install --
# see scripts/gen_dataset.py's identical comment for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heliogram.codec import PATCH_SIZE, VALID_PALETTES  # noqa: E402 -- no heavy deps
from heliogram.dataset import iter_manifest, write_dataset  # noqa: E402 -- no heavy deps

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


@dataclass
class CurriculumStage:
    """One stage of the training curriculum: a (palette, subpatch, payload_sizes,
    corruption_prob) regime to generate fresh examples for and train on, continuing from
    whatever adapter weights the previous stage produced. Ordered low-density/clean -> higher-
    density/corrupted, per the Phase-2 handover's curriculum guidance."""

    name: str
    n_examples: int
    palettes: Sequence[int]
    subpatches: Sequence[int]
    payload_sizes: Sequence[int]
    corruption_prob: float
    epochs: int


def build_curriculum(n_examples_per_stage: int = 2000) -> List[CurriculumStage]:
    """Default curriculum: low density (small palette, subpatch=1 -- one symbol/patch, no
    corruption) first, then widen the palette, then turn on corruption augmentation.

    `subpatch` is pinned to 1 in every stage below: subpatch>1 is a pixel-decoder-only
    geometric ceiling with no evidence a real ViT encoder can resolve it (see
    heliogram/codec.py's DATA HONESTY note and spec/format-v0.1.md section 6a) -- training a
    VLM to emit subpatch>1 targets is speculative and left to a manually constructed curriculum
    (this function's return value is a plain list; edit it or write your own), not this default.
    """
    return [
        CurriculumStage(
            name="stage1_low_density_clean",
            n_examples=n_examples_per_stage,
            palettes=[2, 4],
            subpatches=[1],
            payload_sizes=[16, 48],
            corruption_prob=0.0,
            epochs=1,
        ),
        CurriculumStage(
            name="stage2_full_palette_clean",
            n_examples=n_examples_per_stage,
            palettes=list(VALID_PALETTES),
            subpatches=[1],
            payload_sizes=[16, 48, 128],
            corruption_prob=0.0,
            epochs=1,
        ),
        CurriculumStage(
            name="stage3_full_palette_corrupted",
            n_examples=n_examples_per_stage,
            palettes=list(VALID_PALETTES),
            subpatches=[1],
            payload_sizes=[16, 48, 128],
            corruption_prob=0.5,
            epochs=1,
        ),
    ]


def _load_model_and_processor(
    base_model: str, lora_rank: int, lora_alpha: int, lora_dropout: float
):
    """Load the base model in 4-bit and attach LoRA adapters. ALL heavy imports are local to
    this function -- torch/transformers/peft/bitsandbytes are only required once this is
    actually called (i.e. only from main(), only on a real GPU box).

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
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def _build_hf_dataset(manifest_path: Path, processor):
    """Wrap a heliogram.dataset manifest.jsonl into a `datasets.Dataset` of Trainer-ready
    examples (processed image + tokenized target string, chat-template formatted). Local import
    of `datasets` (the Hugging Face library, a `gpu` extra) -- see module docstring.

    UNTESTED (see module docstring): the exact processor(...) call shape for Qwen2.5-VL's
    "assistant turn is the target" training setup is version-sensitive; treat the labels
    handling here (predict every token, including the prompt) as a simple starting point --
    many recipes mask the prompt tokens out of the loss instead, which this does not do.
    """
    import datasets
    from PIL import Image

    records = list(iter_manifest(manifest_path))

    def _load(record):
        image = Image.open(record["image_path"]).convert("RGB")
        prompt = (
            "Transcribe this heliogram-encoded data grid: row 0 is the calibration row, every "
            f"cell below it is DATA from a {record['palette']}-color palette. Output the data "
            "cells' color-index string, one character per cell, row-major, wrapped in a single "
            "fenced code block, nothing else."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"```\n{record['target']}\n```"}],
            },
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False)
        model_inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    ds = datasets.Dataset.from_list(records)
    return ds.map(_load, remove_columns=ds.column_names)


def _train_stage(
    model, processor, stage: CurriculumStage, args: argparse.Namespace, stage_dir: Path
):
    """Generate this stage's dataset and run one Trainer.train() call. Local import of
    `transformers.Trainer`/`TrainingArguments` -- see module docstring."""
    from transformers import Trainer, TrainingArguments

    manifest_path = write_dataset(
        stage_dir,
        stage.n_examples,
        palettes=stage.palettes,
        subpatches=stage.subpatches,
        payload_sizes=stage.payload_sizes,
        patch_size=args.patch_size,
        nsym=args.nsym,
        seed=args.seed,
        corruption_prob=stage.corruption_prob,
    )
    dataset = _build_hf_dataset(manifest_path, processor)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / stage.name / "trainer_state"),
        num_train_epochs=stage.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
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
        "--batch-size", type=int, default=1, help="per-device train batch size (default: 1)"
    )
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser


def main(argv: list = None) -> int:
    args = build_parser().parse_args(argv)

    print(__doc__)
    print(
        "\n*** This script has not been run in the environment that generated it (no GPU "
        "available there). Review every default above against your actual hardware and "
        "installed package versions before trusting it to work unmodified. ***\n"
    )

    model, processor = _load_model_and_processor(
        args.base_model, args.lora_rank, args.lora_alpha, args.lora_dropout
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
