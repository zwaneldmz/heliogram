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


def test_every_name_main_references_is_defined():
    """Regression guard for a real bug: an edit once swallowed `def _load_tower(...)`, leaving
    its body as unreachable-but-syntactically-valid code inside the previous function -- the
    file imported fine and every shim-based contract test passed, then main() crashed with
    NameError on the GPU box. Walk main()'s (and the module's other functions') global-name
    references and assert each resolves, so a dangling reference fails HERE, not on a pod."""
    import builtins
    import dis

    rp = _load_run_probe()
    for fn_name in ("main", "_parse_args", "_load_tower", "_extract_embeddings",
                    "_extract_pre_merger_embeddings", "_cell_arrays",
                    "_merged_embeddings_tensor", "_match_reverse_indices",
                    "_resolve_visual_tower"):
        fn = getattr(rp, fn_name, None)
        assert callable(fn), f"run_probe.{fn_name} is missing or not callable"
        for instr in dis.get_instructions(fn):
            if instr.opname == "LOAD_GLOBAL":
                name = instr.argval
                assert hasattr(rp, name) or hasattr(builtins, name), (
                    f"run_probe.{fn_name} references global {name!r}, which does not exist in "
                    "the module -- a dangling reference that would NameError at runtime"
                )


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


def _color_image(colors_by_patch, patch=14):
    """Build an image from a 2D list of per-patch RGB colors (rows of patches)."""
    import numpy as np
    from PIL import Image

    rows = len(colors_by_patch)
    cols = len(colors_by_patch[0])
    arr = np.zeros((rows * patch, cols * patch, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            arr[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = colors_by_patch[r][c]
    return Image.fromarray(arr)


def test_pixel_values_row_order_is_unit_raster_then_within_unit_raster(processor_shim):
    """THE ordering assumption pre-merger probing rests on, verified at the pixel level: the
    processor's pixel_values rows are grouped 4-per-merge-unit, units in raster order over the
    merged grid, TL/TR/BL/BR within each unit. Verified by decoding patch COLORS back out of
    pixel_values: build reference signatures from single-color images (layout-agnostic), then
    check an 8-distinct-color 4x2-patch image (2 merge units side by side) row by row."""
    import numpy as np

    ip = processor_shim.image_processor

    def rows_of(img):
        return ip(images=[img.convert("RGB")], return_tensors="pt")["pixel_values"].numpy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 255), (30, 30, 30),
    ]
    # signature per color: one row from a 2x2-patch single-color image (all 4 rows identical)
    signatures = {}
    for color in colors:
        r = rows_of(_color_image([[color, color], [color, color]]))
        assert np.allclose(r, r[0])  # sanity: uniform image -> identical patch rows
        signatures[color] = r[0]

    # 4x2-patch image = two 2x2 merge units. Left unit: colors[0..3] as TL,TR,BL,BR;
    # right unit: colors[4..7] likewise.
    img = _color_image([
        [colors[0], colors[1], colors[4], colors[5]],
        [colors[2], colors[3], colors[6], colors[7]],
    ])
    rows = rows_of(img)
    assert rows.shape[0] == 8
    expected = [colors[0], colors[1], colors[2], colors[3],
                colors[4], colors[5], colors[6], colors[7]]
    for i, exp_color in enumerate(expected):
        dists = {c: float(np.abs(rows[i] - sig).mean()) for c, sig in signatures.items()}
        nearest = min(dists, key=dists.get)
        assert nearest == exp_color, (
            f"pixel_values row {i} decodes to color {nearest}, expected {exp_color} -- the "
            "unit-raster/TL-TR-BL-BR ordering assumption does not hold for this processor"
        )


def test_pre_merger_unshuffle_reproduces_pooler_output_exactly(tiny_model, processor_shim):
    """End-to-end permutation correctness: re-running the tower's OWN merger over the
    unshuffled pre-merger states must reproduce pooler_output (raster order) -- if the
    recovered permutation were wrong anywhere, rows would disagree."""
    rp = _load_run_probe()
    from heliogram.dataset import generate_examples

    ex = next(generate_examples(1, palettes=[16], subpatches=[1], payload_sizes=[64], seed=0))
    out = processor_shim.image_processor(images=[ex.image.convert("RGB")], return_tensors="pt")
    w, h = ex.image.width // 14, ex.image.height // 14
    n_units = (h // 2) * (w // 2)
    visual = tiny_model.model.visual

    unshuffled, pooled, reverse = rp._extract_pre_merger_embeddings(
        visual, out["pixel_values"].float(), out["image_grid_thw"], n_units
    )
    assert unshuffled.shape == (n_units * 4, 32)  # vision hidden_size
    assert sorted(reverse.tolist()) == list(range(n_units))
    with torch.no_grad():
        remerged = visual.merger(unshuffled)
    torch.testing.assert_close(remerged, pooled, rtol=0, atol=0)


def test_match_reverse_indices_recovers_permutation_and_fails_loud():
    rp = _load_run_probe()
    g = torch.Generator().manual_seed(0)
    merger_out = torch.randn(10, 6, generator=g)
    perm = torch.randperm(10, generator=g)
    pooled = merger_out[perm]
    recovered = rp._match_reverse_indices(pooled, merger_out)
    assert torch.equal(recovered, perm)

    # duplicate rows -> refuse rather than guess
    dup = merger_out.clone()
    dup[3] = dup[7]
    with pytest.raises(RuntimeError, match="byte-identical"):
        rp._match_reverse_indices(dup[perm], dup)

    # unmatched rows -> refuse
    with pytest.raises(RuntimeError, match="no byte-identical row"):
        rp._match_reverse_indices(pooled + 1.0, merger_out)


def test_pre_merger_cell_arrays_align_labels_and_probe_runs(tiny_model, processor_shim):
    """The full pre-merger cell path: per-patch embeddings, per-patch labels (calibration row
    masked), probe fit + verdict -- the rehearsal for the localization run."""
    rp = _load_run_probe()

    X_tr, y_tr = rp._cell_arrays(
        tiny_model, processor_shim, torch.float32, "cpu",
        palette=16, corruption_name="clean", corruption_fn=lambda im: im,
        n_images=2, payload_size=48, seed_base=1_000, stage="pre_merger",
    )
    X_te, y_te = rp._cell_arrays(
        tiny_model, processor_shim, torch.float32, "cpu",
        palette=16, corruption_name="clean", corruption_fn=lambda im: im,
        n_images=1, payload_size=48, seed_base=2_000_000, stage="pre_merger",
    )
    assert y_tr.shape[1] == 1  # one symbol per patch row
    assert X_tr.shape[1] == 32  # vision hidden, not out_hidden (24)
    # 4x the sample count of the merged stage for the same images
    X_tr_m, _ = rp._cell_arrays(
        tiny_model, processor_shim, torch.float32, "cpu",
        palette=16, corruption_name="clean", corruption_fn=lambda im: im,
        n_images=2, payload_size=48, seed_base=1_000, stage="merged",
    )
    assert X_tr.shape[0] == 4 * X_tr_m.shape[0]
    # calibration-row patches masked, data patches labeled in range
    assert (y_tr == -1).sum() > 0
    assert y_tr.max() < 16

    cell = evaluate_cell(16, "clean", X_tr, y_tr, X_te, y_te, epochs=5)
    assert 0.0 <= cell.fit.symbol_error <= 1.0 and cell.verdict


def test_pre_merger_labels_match_patch_colors_through_pixel_values(processor_shim):
    """Label/row alignment sanity at full strength: for a real heliogram image, the color
    decoded from pixel_values row i must be the palette color of the symbol that
    merged_token_labels.reshape(-1) says lives at row i (calibration rows excluded). This ties
    the LABELS (probe ground truth) to the PIXELS (what the tower actually ingests) with no
    model in between -- if this holds and the unshuffle/merger test above holds, the
    pre-merger probe's features and labels are aligned end to end."""
    import numpy as np
    from heliogram.codec import get_palette
    from heliogram.dataset import generate_examples, target_to_symbols

    ex = next(generate_examples(1, palettes=[16], subpatches=[1], payload_sizes=[48], seed=5))
    img = ex.image
    w, h = img.width // PATCH_SIZE, img.height // PATCH_SIZE
    symbols = target_to_symbols(ex.target, 16)
    labels = merged_token_labels(w, h, symbols, merge=2).reshape(-1)

    ip = processor_shim.image_processor
    rows = ip(images=[img.convert("RGB")], return_tensors="pt")["pixel_values"].numpy()
    assert rows.shape[0] == labels.shape[0]

    palette_colors = get_palette(16)
    signatures = {}
    for idx, color in enumerate(palette_colors):
        r = ip(images=[_color_image([[color, color], [color, color]]).convert("RGB")],
               return_tensors="pt")["pixel_values"].numpy()
        signatures[idx] = r[0]

    checked = 0
    for i, lab in enumerate(labels.tolist()):
        if lab < 0:
            continue  # calibration-row patch
        dists = {s: float(np.abs(rows[i] - sig).mean()) for s, sig in signatures.items()}
        assert min(dists, key=dists.get) == lab, f"row {i}: pixel color != label {lab}"
        checked += 1
    assert checked > 100  # the assertion actually ran over real data patches


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
