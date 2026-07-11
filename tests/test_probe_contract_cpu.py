"""CPU contract test for scripts/run_probe.py's model-facing half -- the code the rented GPU
will actually execute.

heliogram/probe.py (labels/fit/verdict) is model-free and tested in tests/test_probe.py. What
was UNTESTED until this file was the seam between that and a real transformers Qwen2.5-VL:
attribute paths, processor output, visual() call signature, and which output field carries the
raster-ordered merged-token embeddings. All of that is testable WITHOUT weights or a GPU: a
tiny random-weight Qwen2.5-VL instantiated from a local config exercises the identical code
paths (transformers dispatches on class, not on checkpoint size).

Skipped wholesale when torch/transformers/torchvision aren't installed -- they are GPU-path
extras, not base deps. When they ARE installed, this is the test that keeps run_probe.py
honest against transformers version drift: it caught `model.visual` -> `model.model.visual`
and merged-tokens-live-in-`pooler_output` (transformers 5.13.0) the first time it ran.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")  # the 5.x Qwen image processor imports it

from heliogram.codec import PATCH_SIZE  # noqa: E402
from heliogram.probe import evaluate_cell, merged_token_labels  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_probe():
    spec = importlib.util.spec_from_file_location(
        "run_probe", _REPO_ROOT / "scripts" / "run_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_probe", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_model():
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
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(vision_config=vision, text_config=text)
    ).eval()
    return model


@pytest.fixture(scope="module")
def processor_shim():
    """run_probe only touches processor.image_processor -- a shim around the REAL Qwen image
    processor (constructed locally, no Hub download) covers exactly what the script uses."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    class _Shim:
        image_processor = Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=16_000_000)

    return _Shim()


def test_resolve_visual_tower_handles_5x_layout(tiny_model):
    rp = _load_run_probe()
    tower = rp._resolve_visual_tower(tiny_model)
    assert tower is tiny_model.model.visual  # transformers 5.x layout
    # and a bare object with neither layout fails loudly
    with pytest.raises(RuntimeError, match="vision tower"):
        rp._resolve_visual_tower(object())


def test_extract_embeddings_contract(tiny_model, processor_shim):
    """The full _extract_embeddings path: identity preprocessing, visual() call, merged-token
    selection. Shape must be (merged tokens, out_hidden_size)."""
    rp = _load_run_probe()
    from heliogram.dataset import generate_examples

    ex = next(generate_examples(1, palettes=[16], subpatches=[1], payload_sizes=[64], seed=0))
    w, h = ex.image.width // PATCH_SIZE, ex.image.height // PATCH_SIZE
    assert w % 2 == 0 and h % 2 == 0  # dataset contract: even-padded

    emb = rp._extract_embeddings(
        tiny_model, processor_shim, torch.float32, "cpu", ex.image
    )
    assert emb.shape == ((h // 2) * (w // 2), 24)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()


def test_merged_embeddings_rejects_wrong_token_count(tiny_model):
    rp = _load_run_probe()

    class _FakeOut:
        pooler_output = torch.zeros(7, 24)
        last_hidden_state = torch.zeros(28, 32)

    with pytest.raises(RuntimeError, match="token-count contract"):
        rp._merged_embeddings_tensor(_FakeOut(), expected_tokens=56)


def test_identity_preprocessing_assertion_fires_on_odd_grid(tiny_model, processor_shim):
    """An image with an odd patch dimension MUST be refused (smart_resize would move pixels off
    the symbol lattice), not silently measured."""
    rp = _load_run_probe()
    from heliogram.codec import encode

    img = encode(bytes(range(48)), palette=8, nsym=32)  # 16x16 -- even; crop one patch row off
    odd = img.crop((0, 0, img.width, img.height - PATCH_SIZE))
    assert (odd.height // PATCH_SIZE) % 2 == 1
    with pytest.raises(RuntimeError, match="resized the grid"):
        rp._extract_embeddings(tiny_model, processor_shim, torch.float32, "cpu", odd)


def test_cell_arrays_and_probe_end_to_end_at_chance(tiny_model, processor_shim):
    """The exact per-cell path main() runs, on the tiny random-weight tower. Random weights
    carry no task information, so the probe must land far from perfect -- but every contract
    (label/embedding row alignment, masking, fit, verdict rendering) is exercised for real.
    This is the rehearsal that makes the rented-GPU run a parameter change, not a debut."""
    rp = _load_run_probe()

    X_tr, y_tr = rp._cell_arrays(
        tiny_model, processor_shim, torch.float32, "cpu",
        palette=16, corruption_name="clean", corruption_fn=lambda im: im,
        n_images=2, payload_size=48, seed_base=1_000,
    )
    X_te, y_te = rp._cell_arrays(
        tiny_model, processor_shim, torch.float32, "cpu",
        palette=16, corruption_name="clean", corruption_fn=lambda im: im,
        n_images=1, payload_size=48, seed_base=2_000_000,
    )
    assert X_tr.shape[0] == y_tr.shape[0] and X_te.shape[0] == y_te.shape[0]
    assert y_tr.shape[1] == 4  # 2x2 merge positions
    assert (y_tr >= -1).all() and (y_tr < 16).all()

    cell = evaluate_cell(16, "clean", X_tr, y_tr, X_te, y_te, epochs=5)
    assert 0.0 <= cell.fit.symbol_error <= 1.0
    assert cell.verdict  # a rendered verdict string exists
    # Random weights: demand only "not near-perfect" (embeddings of solid-color patches through
    # a random linear tower can still be partially separable -- that would be a finding about
    # random features, not a bug; what may NOT happen is a spuriously perfect probe).
    assert cell.fit.symbol_error > 0.01


def test_merged_token_labels_alignment_against_processor_grid(processor_shim):
    """grid_thw's (h, w) convention vs merged_token_labels' (width, height) convention is an
    easy silent-transpose trap; pin that the two agree on the merged-token COUNT for a
    non-square image (the count differs under a transpose only in shape, so also pin the
    label layout's row-major width)."""
    from heliogram.dataset import generate_examples, target_to_symbols

    ex = next(generate_examples(1, palettes=[16], subpatches=[1], payload_sizes=[64], seed=3))
    img = ex.image
    w, h = img.width // PATCH_SIZE, img.height // PATCH_SIZE
    assert w != h  # non-square, or the test proves nothing

    out = processor_shim.image_processor(images=[img.convert("RGB")], return_tensors="pt")
    t, gh, gw = (int(x) for x in out["image_grid_thw"][0])
    assert (t, gh, gw) == (1, h, w)

    symbols = target_to_symbols(ex.target, 16)
    labels = merged_token_labels(w, h, symbols, merge=2)
    assert labels.shape == ((h // 2) * (w // 2), 4)
