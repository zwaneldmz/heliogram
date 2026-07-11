"""Pins scripts/train_qlora.py's LoRA target_modules regex against ACTUAL peft wrapping of a
random-weight Qwen2.5-VL (local config, no Hub download, CPU). Skipped when torch/transformers/
peft aren't installed -- they are GPU-path extras. When they are installed (e.g. on the GPU box
right before a training run), this is the test that proves the LoRA config touches exactly the
modules the script's docstring says it touches:

  default:  every language_model q/k/v/o/gate/up/down projection + the two merger Linears
            (visual.merger.mlp.0/.2), and ZERO vision-block modules;
  opt-in:   additionally exactly the vision blocks' attn.qkv/attn.proj + SwiGLU projections.

This caught a real bug: a plain suffix list ("gate_proj", ...) peft-matches the vision blocks'
SwiGLU MLPs too (suffix matching runs against the FULL dotted path), silently LoRA-tuning the
vision tower that --include-vision-blocks was supposed to gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_qlora():
    spec = importlib.util.spec_from_file_location(
        "train_qlora_mod", _REPO_ROOT / "scripts" / "train_qlora.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_qlora_mod", mod)
    spec.loader.exec_module(mod)
    return mod


def _tiny_model():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    vision = dict(
        depth=2, hidden_size=32, num_heads=2, intermediate_size=48, out_hidden_size=24,
        patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
        fullatt_block_indexes=[1], in_channels=3,
    )
    text = dict(
        hidden_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=48, vocab_size=512, max_position_embeddings=128,
    )
    return Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(vision_config=vision, text_config=text)
    )


def _lora_wrapped_module_names(pattern):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        _tiny_model(),
        LoraConfig(r=2, target_modules=pattern, task_type="CAUSAL_LM"),
    )
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and not name.rsplit(".", 1)[-1].startswith("lora")
    ]


def test_default_regex_wraps_lm_and_merger_only():
    tq = _load_train_qlora()
    wrapped = _lora_wrapped_module_names(
        tq._build_lora_target_modules(include_vision_blocks=False)
    )
    assert wrapped, "LoRA wrapped nothing -- the regex is broken"
    vision_block_hits = [n for n in wrapped if ".visual.blocks." in n]
    assert vision_block_hits == [], (
        "the default LoRA config wrapped vision-block modules -- the suffix-matching leak "
        f"is back: {vision_block_hits[:4]}"
    )
    assert sum(".visual.merger.mlp." in n for n in wrapped) == 2  # exactly mlp.0 and mlp.2
    # tiny model: 2 LM layers x 7 projections
    assert sum("language_model" in n for n in wrapped) == 14


def test_opt_in_regex_adds_exactly_the_vision_block_modules():
    tq = _load_train_qlora()
    wrapped = _lora_wrapped_module_names(
        tq._build_lora_target_modules(include_vision_blocks=True)
    )
    vision_block_hits = sorted(n for n in wrapped if ".visual.blocks." in n)
    # tiny model: 2 vision blocks x (attn.qkv, attn.proj, mlp.gate/up/down) = 10
    assert len(vision_block_hits) == 10
    assert sum(".visual.merger.mlp." in n for n in wrapped) == 2
    assert sum("language_model" in n for n in wrapped) == 14
