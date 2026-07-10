"""Pytest suite for the Phase-2 (GPU) scaffold: heliogram.dataset and heliogram.vlm.

Everything in this file runs on the same CPU-only, no-torch environment as
tests/test_roundtrip.py -- it is a hard requirement (see heliogram/vlm.py's module docstring)
that neither module needs torch/transformers/peft/bitsandbytes merely to be imported or to have
its non-model code paths exercised. Accordingly, this file never loads a real VLM: everywhere
heliogram.vlm would need one (QwenVLDecoder.__call__ reaching _generate,
zero_shot_symbol_error's forward pass), the tests below only exercise the guard rails that fire
BEFORE a model is touched (model=None -> clear error) plus the pure-Python
parsing/framing/prompt code that does not need one. That boundary is the point of this file, not
an oversight: it is exactly what "no GPU in this environment" means in practice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import heliogram
from heliogram.codec import (
    VALID_PALETTES,
    VALID_SUBPATCHES,
    HeliogramDecodeError,
    decode_pixels,
    encode,
    extract_symbols,
)
from heliogram.dataset import (
    SYMBOL_ALPHABET,
    DEFAULT_CORRUPTIONS,
    Example,
    generate_examples,
    iter_manifest,
    random_payload,
    symbols_to_target,
    target_to_symbols,
    write_dataset,
)
from heliogram.vlm import (
    QwenVLDecoder,
    ZeroShotResult,
    _extract_symbol_string,
    _payload_from_symbols,
    zero_shot_symbol_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- import-time / package boundary -----------------------------------------------------------


def test_import_heliogram_does_not_pull_in_torch():
    """The central Phase-2 scaffold invariant: `import heliogram` (and, transitively,
    heliogram.dataset / heliogram.vlm) must never require torch/transformers/peft/bitsandbytes.
    This test doesn't just check the import succeeded (that's implicit in collecting this file
    at all) -- it asserts torch was never actually loaded as a side effect."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_heliogram_reexports_phase2_names():
    assert heliogram.QwenVLDecoder is QwenVLDecoder
    assert heliogram.zero_shot_symbol_error is zero_shot_symbol_error
    assert heliogram.write_dataset is write_dataset
    assert heliogram.generate_examples is generate_examples


# --- target string encoding -------------------------------------------------------------------


def test_symbol_alphabet_covers_every_valid_palette():
    assert len(SYMBOL_ALPHABET) == 64
    assert len(set(SYMBOL_ALPHABET)) == 64  # all distinct
    assert max(VALID_PALETTES) <= len(SYMBOL_ALPHABET)


def test_symbols_to_target_roundtrip():
    for palette in VALID_PALETTES:
        symbols = list(range(palette)) * 3  # exercise every symbol value at least 3x
        target = symbols_to_target(symbols, palette)
        assert len(target) == len(symbols)
        assert target_to_symbols(target, palette) == symbols


def test_symbols_to_target_empty():
    assert symbols_to_target([], 8) == ""
    assert target_to_symbols("", 8) == []


def test_symbols_to_target_rejects_out_of_range_symbol():
    with pytest.raises(ValueError):
        symbols_to_target([0, 8], palette=8)  # 8 is out of range for palette=8 (0..7)


def test_target_to_symbols_rejects_unknown_character():
    with pytest.raises(ValueError):
        target_to_symbols("!!!", palette=8)


def test_target_to_symbols_rejects_value_too_large_for_palette():
    # SYMBOL_ALPHABET[8] ('I', since it starts 'ABCDEFGHI...') is a valid character but out of
    # range for a palette that only spans indices 0..7.
    ch = SYMBOL_ALPHABET[8]
    with pytest.raises(ValueError):
        target_to_symbols(ch, palette=8)


# --- generate_examples: determinism + ground-truth correctness -------------------------------


def test_generate_examples_deterministic():
    kwargs = dict(palettes=(8, 16), subpatches=(1,), payload_sizes=(16, 48), seed=7)
    run1 = list(generate_examples(5, **kwargs))
    run2 = list(generate_examples(5, **kwargs))
    assert len(run1) == len(run2) == 5
    for a, b in zip(run1, run2):
        assert a.target == b.target
        assert a.payload == b.payload
        assert a.palette == b.palette
        assert a.subpatch == b.subpatch
        assert a.corruption == b.corruption
        assert list(a.image.getdata()) == list(b.image.getdata())


def test_generate_examples_ground_truth_matches_codec_clean_roundtrip():
    """The core correctness property: target is exactly what extract_symbols reads off a fresh
    clean re-encode of the same payload -- and, when corruption_prob=0.0 (the default), decoding
    the returned image with decode_pixels recovers the exact original payload."""
    for ex in generate_examples(
        6, palettes=VALID_PALETTES, subpatches=(1,), payload_sizes=(16, 48), seed=1
    ):
        assert isinstance(ex, Example)
        clean_img = encode(
            ex.payload,
            palette=ex.palette,
            patch_size=ex.patch_size,
            nsym=ex.nsym,
            seed=0,
            subpatch=ex.subpatch,
        )
        _, _, truth_symbols = extract_symbols(
            clean_img, palette=ex.palette, patch_size=ex.patch_size, subpatch=ex.subpatch
        )
        assert target_to_symbols(ex.target, ex.palette) == truth_symbols

        assert ex.corruption == "clean"  # corruption_prob defaults to 0.0
        recovered = decode_pixels(
            ex.image,
            palette=ex.palette,
            patch_size=ex.patch_size,
            nsym=ex.nsym,
            subpatch=ex.subpatch,
        )
        assert recovered == ex.payload


def test_generate_examples_corruption_prob_zero_is_always_clean():
    examples = list(generate_examples(10, payload_sizes=(16,), seed=3, corruption_prob=0.0))
    assert all(ex.corruption == "clean" for ex in examples)


def test_generate_examples_corruption_prob_one_is_never_clean():
    examples = list(generate_examples(10, payload_sizes=(16,), seed=3, corruption_prob=1.0))
    assert all(ex.corruption != "clean" for ex in examples)
    assert all(ex.corruption in DEFAULT_CORRUPTIONS for ex in examples)


def test_generate_examples_target_survives_corruption():
    """Even when the returned image IS corrupted, `target` still matches the clean encode of
    the same payload -- ground truth is never re-derived from the (possibly corrupted) image."""
    for ex in generate_examples(6, payload_sizes=(48,), seed=5, corruption_prob=1.0):
        clean_img = encode(
            ex.payload,
            palette=ex.palette,
            patch_size=ex.patch_size,
            nsym=ex.nsym,
            seed=0,
            subpatch=ex.subpatch,
        )
        _, _, truth_symbols = extract_symbols(
            clean_img, palette=ex.palette, patch_size=ex.patch_size, subpatch=ex.subpatch
        )
        assert target_to_symbols(ex.target, ex.palette) == truth_symbols


def test_generate_examples_rejects_bad_corruption_prob():
    with pytest.raises(ValueError):
        list(generate_examples(1, corruption_prob=1.5))
    with pytest.raises(ValueError):
        list(generate_examples(1, corruption_prob=-0.1))


def test_generate_examples_rejects_empty_ranges():
    with pytest.raises(ValueError):
        list(generate_examples(1, palettes=()))


def test_generate_examples_rejects_invalid_palette_or_subpatch():
    with pytest.raises(ValueError, match="palettes"):
        list(generate_examples(1, palettes=(3, 8)))  # 3 is not in VALID_PALETTES
    with pytest.raises(ValueError, match="subpatches"):
        list(generate_examples(1, subpatches=(1, 5)))  # 5 is not in VALID_SUBPATCHES


# --- write_dataset / iter_manifest -------------------------------------------------------------


def test_write_dataset_manifest_contract(tmp_path):
    out_dir = tmp_path / "ds"
    manifest_path = write_dataset(
        out_dir, 4, palettes=(8, 16), payload_sizes=(16,), seed=2, corruption_prob=0.5
    )
    assert manifest_path == out_dir / "manifest.jsonl"
    assert manifest_path.exists()

    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) == 4

    records = [json.loads(line) for line in lines]
    for record in records:
        for key in ("image_path", "palette", "subpatch", "target"):
            assert key in record
        assert record["palette"] in (8, 16)
        img_path = out_dir / record["image_path"]
        assert img_path.exists()
        img = Image.open(img_path)
        img.load()  # force a real decode, not just header parsing


def test_write_dataset_matches_generate_examples(tmp_path):
    """write_dataset must not silently transform what generate_examples would have produced."""
    kwargs = dict(palettes=(8,), payload_sizes=(16, 48), seed=9, corruption_prob=0.0)
    direct = list(generate_examples(3, **kwargs))
    manifest_path = write_dataset(tmp_path / "ds2", 3, **kwargs)
    records = list(iter_manifest(manifest_path))
    assert len(records) == len(direct) == 3
    for ex, record in zip(direct, records):
        assert record["target"] == ex.target
        assert record["palette"] == ex.palette
        assert record["subpatch"] == ex.subpatch
        assert Path(record["image_path"]).exists()


def test_iter_manifest_resolves_paths_relative_to_manifest_dir(tmp_path):
    manifest_path = write_dataset(tmp_path / "ds3", 2, payload_sizes=(16,), seed=4)
    for record in iter_manifest(manifest_path):
        assert Path(record["image_path"]).is_absolute()
        assert Path(record["image_path"]).exists()


# --- random_payload (shared with heliogram.harness's private helper, by construction) --------


def test_random_payload_deterministic():
    assert random_payload(0, 16) == random_payload(0, 16)
    assert random_payload(0, 16) != random_payload(1, 16)
    assert len(random_payload(0, 32)) == 32


# --- heliogram.vlm: pure parsing / framing, no model required ---------------------------------


def test_extract_symbol_string_prefers_fenced_code_block():
    text = "Sure, here you go:\n```\nABCD\n```\nHope that helps!"
    assert _extract_symbol_string(text) == "ABCD"


def test_extract_symbol_string_falls_back_to_whole_text():
    assert _extract_symbol_string("  ABCD  \n") == "ABCD"


def test_payload_from_symbols_matches_decode_pixels():
    """The whole point of _payload_from_symbols: given the SAME symbols decode_pixels would
    have classified off an image, it must produce the exact same payload -- proving the VLM
    decoder path and the pixel decoder path share the identical RS/framing layer."""
    for palette in VALID_PALETTES:
        payload = b"phase-2 scaffold RS/framing reuse check"
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        _, _, symbols = extract_symbols(img, palette=palette, patch_size=14)
        assert _payload_from_symbols(symbols, palette=palette, nsym=32) == payload
        # and it's consistent with decode_pixels on the same image/args
        assert decode_pixels(img, palette=palette, patch_size=14, nsym=32) == payload


def test_payload_from_symbols_raises_heliogram_decode_error_on_garbage():
    with pytest.raises(HeliogramDecodeError):
        _payload_from_symbols([0, 1, 2], palette=8, nsym=32)  # far too short to be a real frame


def test_qwen_vl_decoder_full_pipeline_without_a_model():
    """End-to-end proof that QwenVLDecoder's non-model plumbing (prompt -> [pretend the model
    echoed the target perfectly] -> parse -> RS/framing) reconstructs the original payload,
    without ever touching _generate/torch."""
    palette = 8
    payload = b"end-to-end parse+framing check, no model involved"
    img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=14)
    target = symbols_to_target(symbols, palette)

    decoder = QwenVLDecoder(model=object(), processor=object(), palette=palette, subpatch=1)
    # Simulate a perfect model response (wrapped in a fenced code block, as the prompt asks):
    fake_response = f"```\n{target}\n```"
    parsed_symbols = decoder._parse_symbols(fake_response)
    assert parsed_symbols == symbols
    assert _payload_from_symbols(parsed_symbols, palette=palette, nsym=32) == payload


# --- heliogram.vlm: guard rails that must fire BEFORE any model is touched --------------------


def test_qwen_vl_decoder_requires_a_model():
    decoder = QwenVLDecoder(model=None, processor=None, palette=8, subpatch=1)
    img = encode(b"x", palette=8)
    with pytest.raises(RuntimeError, match="scripts/train_qlora.py"):
        decoder(img, palette=8, subpatch=1)


def test_decode_plug_point_actually_reaches_qwen_vl_decoder():
    """Proves the binding-decision claim that `decode(img, decoder=QwenVLDecoder(...))` "works"
    -- i.e. heliogram.codec.decode()'s unconditional `subpatch=subpatch` forwarding to whatever
    `decoder=` callable it's given reaches QwenVLDecoder.__call__ without a TypeError on an
    unexpected keyword (which is exactly the documented pre-existing failure mode for the OLD
    VLMDecoder stub in heliogram/codec.py, whose __call__ does not accept `subpatch`). It should
    reach QwenVLDecoder's own model=None guard (RuntimeError), not a TypeError from decode()."""
    img = encode(b"decode() plug point check", palette=8, subpatch=1)
    decoder = QwenVLDecoder(model=None, processor=None, palette=8, subpatch=1)
    with pytest.raises(RuntimeError, match="scripts/train_qlora.py"):
        heliogram.decode(img, palette=8, subpatch=1, decoder=decoder)


def test_qwen_vl_decoder_requires_processor_too():
    decoder = QwenVLDecoder(model=object(), processor=None, palette=8, subpatch=1)
    img = encode(b"x", palette=8)
    with pytest.raises(RuntimeError, match="scripts/train_qlora.py"):
        decoder(img, palette=8, subpatch=1)


def test_qwen_vl_decoder_rejects_config_mismatch():
    decoder = QwenVLDecoder(model=object(), processor=object(), palette=8, subpatch=1)
    img = encode(b"x", palette=16)
    with pytest.raises(ValueError):
        decoder(img, palette=16, subpatch=1)  # constructed for palette=8, called with 16


def test_qwen_vl_decoder_invalid_palette_or_subpatch_raise_value_error():
    with pytest.raises(ValueError):
        QwenVLDecoder(palette=3)  # not in VALID_PALETTES
    with pytest.raises(ValueError):
        QwenVLDecoder(palette=8, subpatch=3)  # not in VALID_SUBPATCHES


def test_zero_shot_symbol_error_requires_a_model():
    with pytest.raises(ValueError, match="model"):
        zero_shot_symbol_error(model=None, processor=None, configs=[{"palette": 8}])


def test_zero_shot_result_is_a_plain_dataclass_with_expected_fields():
    result = ZeroShotResult(
        palette=8,
        subpatch=1,
        patch_size=14,
        payload_size=48,
        trials=3,
        symbol_error_rate=0.0,
        decode_success_rate=0.0,
    )
    assert result.raw_responses == []  # default_factory, never a shared mutable default


# --- CLI smoke tests (subprocess, no GPU needed for either script's --help or gen_dataset) ----


def test_gen_dataset_cli_help_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_dataset.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--out" in result.stdout


def test_train_qlora_cli_help_does_not_require_torch():
    """--help must work even with zero GPU packages installed: argparse setup happens before
    any lazy torch/transformers/peft import in scripts/train_qlora.py."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_qlora.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_gen_dataset_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "cli_ds"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gen_dataset.py"),
            "--out", str(out_dir),
            "--n", "3",
            "--seed", "0",
            "--palettes", "8",
            "--payload-sizes", "16",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    manifest_path = out_dir / "manifest.jsonl"
    assert manifest_path.exists()
    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_gen_dataset_cli_rejects_invalid_palette(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gen_dataset.py"),
            "--out", str(tmp_path / "bad_ds"),
            "--n", "1",
            "--palettes", "3",  # not in VALID_PALETTES
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "invalid value" in result.stderr
