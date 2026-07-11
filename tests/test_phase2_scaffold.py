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

import importlib.util
import inspect
import json
import math
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
    build_prompt,
    format_output_text,
    generate_examples,
    iter_manifest,
    n_data_cells,
    pad_to_even_patch_grid,
    parse_output_text,
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
    _resolve_zero_shot_config,
    expected_max_new_tokens,
    teacher_forced_symbol_accuracy,
    zero_shot_symbol_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_qlora_module():
    """Load scripts/train_qlora.py as a module WITHOUT needing scripts/ to be a package (it has
    no __init__.py, by design -- see scripts/gen_dataset.py's identical layout). Importing the
    module itself must stay torch/transformers-free (see that script's own module docstring and
    test_train_qlora_cli_help_does_not_require_torch below) -- this is exercised directly here
    (not just via a --help subprocess) so the masking/collator/LoRA-target helper functions
    defined above `main()` are reachable for direct unit testing on CPU."""
    spec = importlib.util.spec_from_file_location(
        "train_qlora", REPO_ROOT / "scripts" / "train_qlora.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: dataclasses' own type-resolution machinery
    # looks the module up via sys.modules[cls.__module__] while CurriculumStage/
    # HeliogramVLCollator's @dataclass decorators run during exec_module itself.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_qlora = _load_train_qlora_module()

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
    """SYMBOL_ALPHABET must have one distinct character per symbol value for the LARGEST palette
    in VALID_PALETTES (256), so every palette is fully representable. If VALID_PALETTES grows
    past 256 without SYMBOL_ALPHABET keeping up, this fails -- exactly the drift it exists to
    catch. The first 64 characters stay the base64 alphabet for backward compatibility with any
    manifest.jsonl target string written before the palette range grew past 64."""
    assert len(SYMBOL_ALPHABET) == 256
    assert len(set(SYMBOL_ALPHABET)) == 256  # all distinct (target_to_symbols' lookup needs it)
    assert max(VALID_PALETTES) <= len(SYMBOL_ALPHABET)  # every palette fully representable
    # whitespace-safe: _extract_symbol_string strips whitespace via str.split(), so a whitespace
    # alphabet char would be silently dropped -- none may be whitespace.
    assert not any(ch.isspace() for ch in SYMBOL_ALPHABET)


def test_symbols_to_target_roundtrip():
    for palette in VALID_PALETTES:  # every palette, including 128/256, is now representable
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
    clean re-encode of the same payload, ONCE THAT RE-ENCODE IS PADDED THE SAME WAY
    generate_examples pads it (pad_to_even_patch_grid, D4 -- see that function's docstring) --
    and, when corruption_prob=0.0 (the default) AND no column padding was needed (width was
    already even), decoding the returned image with decode_pixels recovers the exact original
    payload (see pad_to_even_patch_grid's docstring's documented tradeoff: a COLUMN-padded image
    is not expected to round-trip through decode_pixels, only a row-padded or unpadded one)."""
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
        width_before_padding = clean_img.width // ex.patch_size
        clean_img = pad_to_even_patch_grid(clean_img, ex.patch_size, ex.palette)
        _, _, truth_symbols = extract_symbols(
            clean_img, palette=ex.palette, patch_size=ex.patch_size, subpatch=ex.subpatch
        )
        assert target_to_symbols(ex.target, ex.palette) == truth_symbols

        # Every image generate_examples emits has an even patch grid, regardless of whether
        # padding fired (D4's actual guarantee -- see test_generate_examples_images_always_have_
        # even_patch_grid below for a dedicated, broader test of this).
        assert (ex.image.width // ex.patch_size) % 2 == 0
        assert (ex.image.height // ex.patch_size) % 2 == 0

        assert ex.corruption == "clean"  # corruption_prob defaults to 0.0
        if width_before_padding % 2 == 0:  # no column was added -- decode_pixels still works
            recovered = decode_pixels(
                ex.image,
                palette=ex.palette,
                patch_size=ex.patch_size,
                nsym=ex.nsym,
                subpatch=ex.subpatch,
            )
            assert recovered == ex.payload


def test_generate_examples_corruption_prob_zero_is_always_clean():
    examples = list(generate_examples(
        10, palettes=VALID_PALETTES, payload_sizes=(16,), seed=3, corruption_prob=0.0
    ))
    assert all(ex.corruption == "clean" for ex in examples)


def test_generate_examples_corruption_prob_one_is_never_clean():
    examples = list(generate_examples(
        10, palettes=VALID_PALETTES, payload_sizes=(16,), seed=3, corruption_prob=1.0
    ))
    assert all(ex.corruption != "clean" for ex in examples)
    assert all(ex.corruption in DEFAULT_CORRUPTIONS for ex in examples)


def test_generate_examples_target_survives_corruption():
    """Even when the returned image IS corrupted, `target` still matches the clean, even-padded
    encode of the same payload -- ground truth is never re-derived from the (possibly corrupted)
    image."""
    for ex in generate_examples(
        6, palettes=VALID_PALETTES, payload_sizes=(48,), seed=5, corruption_prob=1.0
    ):
        clean_img = encode(
            ex.payload,
            palette=ex.palette,
            patch_size=ex.patch_size,
            nsym=ex.nsym,
            seed=0,
            subpatch=ex.subpatch,
        )
        clean_img = pad_to_even_patch_grid(clean_img, ex.patch_size, ex.palette)
        _, _, truth_symbols = extract_symbols(
            clean_img, palette=ex.palette, patch_size=ex.patch_size, subpatch=ex.subpatch
        )
        assert target_to_symbols(ex.target, ex.palette) == truth_symbols


def test_generate_examples_images_always_have_even_patch_grid():
    """D4 of the Phase-2 scaffold review: EVERY image generate_examples/write_dataset emits has
    both patch-grid dimensions even, regardless of palette/payload_size/corruption -- so its
    pixel dimensions are always an exact multiple of patch_size * 2 (the alignment Qwen's
    smart_resize requires to be a no-op; see heliogram/dataset.py's "PROCESSOR RESIZE HAZARD"
    module-docstring note). Exercises palettes (2, 4) specifically -- unlike DEFAULT_PALETTES
    (64/128/256, whose width is even by construction for this project's actual payload sizes),
    small palettes commonly produce an ODD width (see pad_to_even_patch_grid's docstring), so
    this is the case that actually exercises the column-padding branch, not just the row one."""
    for ex in generate_examples(
        30,
        palettes=(2, 4, 8, 64, 128, 256),
        subpatches=(1,),
        payload_sizes=(16, 48, 128, 1024),
        seed=11,
        corruption_prob=0.5,
    ):
        width = ex.image.width // ex.patch_size
        height = ex.image.height // ex.patch_size
        assert width % 2 == 0, (ex.palette, ex.image.size)
        assert height % 2 == 0, (ex.palette, ex.image.size)
        assert len(ex.target) == n_data_cells(width, height, ex.subpatch)


def test_pad_to_even_patch_grid_is_noop_when_already_even():
    img = encode(b"already even width and height", palette=256, patch_size=14, subpatch=1)
    width, height = img.width // 14, img.height // 14
    assert width % 2 == 0 and height % 2 == 0  # sanity: this payload/palette IS already even
    padded = pad_to_even_patch_grid(img, 14, 256)
    assert padded is img  # true no-op: same object, not merely equal pixels


def test_pad_to_even_patch_grid_preserves_decode_pixels_when_only_height_padded():
    """A HEIGHT-only pad (width already even) is provably safe for decode_pixels: the new row is
    a pure suffix beyond every real-or-already-zero-padded symbol in row-major order."""
    payload = bytes(range(16))
    img = encode(payload, palette=64, patch_size=14, nsym=32, subpatch=1)
    width = img.width // 14
    assert width % 2 == 0  # palette=64 keeps width even for this payload size (see dataset.py)
    padded = pad_to_even_patch_grid(img, 14, 64)
    assert (padded.height // 14) % 2 == 0
    recovered = decode_pixels(padded, palette=64, patch_size=14, nsym=32, subpatch=1)
    assert recovered == payload


def test_pad_to_even_patch_grid_column_padding_breaks_decode_pixels_by_design():
    """Documented tradeoff (see pad_to_even_patch_grid's docstring): a COLUMN pad interleaves one
    extra symbol into the row-major stream, breaking decode_pixels for any payload carrying real
    content across multiple rows -- expected and fine for THIS module's actual use (VLM training
    data), not for decode_pixels. This test pins that documented behavior so a future change to
    the padding scheme cannot silently "fix" it without the docstring/test being updated too."""
    payload = bytes(range(16))
    img = encode(payload, palette=2, patch_size=14, nsym=32, subpatch=1)
    width = img.width // 14
    assert width % 2 == 1  # palette=2 at this payload size produces an odd width (see dataset.py)
    padded = pad_to_even_patch_grid(img, 14, 2)
    assert (padded.width // 14) % 2 == 0
    with pytest.raises(HeliogramDecodeError):
        decode_pixels(padded, palette=2, patch_size=14, nsym=32, subpatch=1)


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
    manifest_path = write_dataset(
        tmp_path / "ds3", 2, palettes=VALID_PALETTES, payload_sizes=(16,), seed=4
    )
    for record in iter_manifest(manifest_path):
        assert Path(record["image_path"]).is_absolute()
        assert Path(record["image_path"]).exists()


# --- random_payload (shared with heliogram.harness's private helper, by construction) --------


def test_random_payload_deterministic():
    assert random_payload(0, 16) == random_payload(0, 16)
    assert random_payload(0, 16) != random_payload(1, 16)
    assert len(random_payload(0, 32)) == 32


# --- heliogram.vlm: pure parsing / framing, no model required ---------------------------------


def test_extract_symbol_string_falls_back_to_whole_text():
    assert _extract_symbol_string("  ABCD  \n") == "ABCD"


def test_extract_symbol_string_delegates_to_dataset_parse_output_text():
    """D5(b)/D5(d): heliogram.vlm._extract_symbol_string is now a thin wrapper around
    heliogram.dataset.parse_output_text -- not an independent reimplementation -- so the two can
    never silently drift apart."""
    for text in ["ABCD", "  AB\nCD  ", "row1\nrow2\nrow3", ""]:
        assert _extract_symbol_string(text) == parse_output_text(text)


def test_extract_symbol_string_backtick_hazard_is_fixed():
    """THE regression test for D5(d)'s actual bug: SYMBOL_ALPHABET includes the backtick
    character (index 89, reachable once palette > 64) -- a correct transcription of palette=128
    DATA can legitimately contain a run of three consecutive backticks. An EARLIER version of
    this parser preferred the contents of a fenced code block (```...```) -- indistinguishable
    from a legitimate backtick-symbol run -- and would have silently truncated everything after
    the "fence" it thought it found. The current parser (strip whitespace, nothing else) does
    not: a run of backtick DATA survives intact, wherever it falls in the response."""
    assert "`" in SYMBOL_ALPHABET[64:128]  # sanity: backtick IS in-range once palette > 64
    response_text = "AB\n```\nCD"  # a row-per-line response whose 2nd row is three backticks
    assert parse_output_text(response_text) == "AB```CD"  # nothing truncated
    assert _extract_symbol_string(response_text) == "AB```CD"  # same via the vlm.py entry point


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


@pytest.mark.parametrize("palette", [8, 128, 256])
def test_qwen_vl_decoder_full_pipeline_without_a_model(palette):
    """End-to-end proof that QwenVLDecoder's non-model plumbing (prompt -> [pretend the model
    echoed the target perfectly] -> parse -> RS/framing) reconstructs the original payload,
    without ever touching _generate/torch. Includes palette=128/256 to exercise the extended
    SYMBOL_ALPHABET end to end (prompt's alphabet slice + target<->symbol round trip)."""
    payload = b"end-to-end parse+framing check, no model involved"
    img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=14)
    target = symbols_to_target(symbols, palette)
    width, height = img.width // 14, img.height // 14

    decoder = QwenVLDecoder(model=object(), processor=object(), palette=palette, subpatch=1)
    # The prompt must list exactly `palette` alphabet characters -- for palette=128/256 this only
    # works because SYMBOL_ALPHABET was extended past 64 (it used to truncate silently to 64).
    prompt = decoder._build_prompt(width, height)
    assert f'"{SYMBOL_ALPHABET[:palette]}"' in prompt
    # Simulate a perfect model response: the row-per-line, fence-free output contract (D5(d)),
    # built via the same canonical format_output_text the training path uses.
    fake_response = format_output_text(target, width, subpatch=1)
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


# --- D5(a): max_new_tokens sizing --------------------------------------------------------------


def test_expected_max_new_tokens_matches_n_data_cells_plus_newlines_times_margin():
    width, height, subpatch = 64, 5, 1
    cells = n_data_cells(width, height, subpatch)  # 64 * 4 = 256
    newlines = height - 2  # format_output_text inserts height-2 '\n' separators
    assert expected_max_new_tokens(width, height, subpatch, margin=3.0) == math.ceil(
        (cells + newlines) * 3.0
    )
    # default margin is 3.0, matching QwenVLDecoder's own default token_margin
    assert expected_max_new_tokens(width, height, subpatch) == expected_max_new_tokens(
        width, height, subpatch, margin=3.0
    )


def test_expected_max_new_tokens_scales_with_margin():
    small = expected_max_new_tokens(64, 5, margin=1.0)
    big = expected_max_new_tokens(64, 5, margin=3.0)
    assert small < big


def test_expected_max_new_tokens_fixes_the_original_undersizing_bug():
    """The actual regression this exists to fix: a P=256, ~4096-byte payload's grid needs far
    more than the old fixed default of max_new_tokens=2048 -- see QwenVLDecoder's docstring."""
    payload = random_payload(0, 4096)
    img = encode(payload, palette=256, patch_size=14, nsym=32, subpatch=1)
    width, height = img.width // 14, img.height // 14
    budget = expected_max_new_tokens(width, height, subpatch=1)
    assert budget > 2048  # the old fixed default would have truncated this
    assert budget >= n_data_cells(width, height, 1)  # comfortably covers the real character count


def test_qwen_vl_decoder_default_max_new_tokens_is_none():
    """D5(a): max_new_tokens defaults to None (meaning "size it dynamically per image"), not a
    fixed int -- see QwenVLDecoder.__init__'s docstring."""
    decoder = QwenVLDecoder(model=object(), processor=object(), palette=8, subpatch=1)
    assert decoder.max_new_tokens is None
    assert decoder.token_margin == 3.0


def test_zero_shot_symbol_error_default_max_new_tokens_is_none():
    sig = inspect.signature(zero_shot_symbol_error)
    assert sig.parameters["max_new_tokens"].default is None


def test_resolve_zero_shot_config_payload_size_default_is_1024():
    """D5(a): the bare-config fallback payload_size shrank from an earlier 4096 to 1024 -- see
    zero_shot_symbol_error's docstring for why (a stock model's free-running zero-shot eval
    should have a realistic chance of finishing a correct transcription within budget)."""
    assert _resolve_zero_shot_config({"palette": 8})["payload_size"] == 1024
    # an explicit override still wins:
    assert _resolve_zero_shot_config({"palette": 8, "payload_size": 4096})["payload_size"] == 4096


# --- D5(b): prompt unification -----------------------------------------------------------------


def test_qwen_vl_decoder_build_prompt_delegates_to_canonical_dataset_build_prompt():
    """D5(b): QwenVLDecoder._build_prompt must produce the EXACT SAME string as
    heliogram.dataset.build_prompt for the same arguments -- not an independently-written prompt
    that merely happens to describe the same task."""
    for palette, subpatch in [(8, 1), (128, 1), (256, 1)]:
        decoder = QwenVLDecoder(
            model=object(), processor=object(), palette=palette, subpatch=subpatch
        )
        for width, height in [(64, 3), (256, 6)]:
            assert decoder._build_prompt(width, height) == build_prompt(
                palette, width, height, subpatch
            )


def test_train_qlora_imports_canonical_prompt_functions_from_dataset():
    """D5(b): scripts/train_qlora.py's training-target construction must call the SAME
    heliogram.dataset.build_prompt/format_output_text heliogram.vlm.QwenVLDecoder uses, not its
    own independently-written prompt string (the ORIGINAL bug this fixes: the two files' prompts
    used to be two different, hand-written strings describing the same task)."""
    assert train_qlora.build_prompt is build_prompt
    assert train_qlora.format_output_text is format_output_text
    # and the source of _build_hf_dataset actually calls build_prompt(...), not a literal string:
    source = inspect.getsource(train_qlora._build_hf_dataset)
    assert "build_prompt(" in source
    assert "format_output_text(" in source


def test_format_output_text_and_build_prompt_agree_on_line_count():
    for palette, width, height, subpatch in [(8, 4, 3, 1), (256, 64, 5, 1)]:
        symbols = [(i * 7) % palette for i in range(n_data_cells(width, height, subpatch))]
        target = symbols_to_target(symbols, palette)
        response_text = format_output_text(target, width, subpatch)
        prompt = build_prompt(palette, width, height, subpatch)
        assert response_text.count("\n") == height - 2  # (height-1) lines joined by '\n'
        assert f"{height - 1} lines" in prompt
        assert parse_output_text(response_text) == target


# --- D5(c): teacher_forced_symbol_accuracy guard rails (no model touched) ----------------------


def test_teacher_forced_symbol_accuracy_requires_a_model():
    img = encode(b"x", palette=8)
    with pytest.raises(ValueError, match="model"):
        teacher_forced_symbol_accuracy(None, None, img, target="AA", palette=8)


def test_teacher_forced_symbol_accuracy_rejects_target_length_mismatch():
    img = encode(b"a real payload for a real grid", palette=8, patch_size=14, subpatch=1)
    with pytest.raises(ValueError, match="data cells"):
        teacher_forced_symbol_accuracy(
            object(), object(), img, target="TOOSHORT", palette=8, patch_size=14, subpatch=1
        )


# --- D1: scripts/train_qlora.py label masking (pure Python, no torch) --------------------------


def test_mask_prompt_tokens_masks_only_the_prompt_prefix():
    """The core D1 regression test: given a fake batch (plain int list, standing in for a
    tokenized prompt+response sequence) and a prompt length, everything before prompt_len must
    become -100 (ignored by the loss) and everything from prompt_len on must be UNCHANGED (the
    real token ids -- these are what the model is actually trained to predict)."""
    input_ids = [10, 11, 12, 13, 20, 21, 22]  # first 4 = prompt+image tokens, last 3 = response
    labels = train_qlora._mask_prompt_tokens(input_ids, prompt_len=4)
    assert labels == [-100, -100, -100, -100, 20, 21, 22]
    assert len(labels) == len(input_ids)


def test_mask_prompt_tokens_handles_the_full_prompt_and_empty_prompt_edges():
    input_ids = [1, 2, 3]
    assert train_qlora._mask_prompt_tokens(input_ids, prompt_len=0) == [1, 2, 3]
    assert train_qlora._mask_prompt_tokens(input_ids, prompt_len=3) == [-100, -100, -100]


def test_mask_prompt_tokens_rejects_out_of_range_prompt_len():
    with pytest.raises(ValueError):
        train_qlora._mask_prompt_tokens([1, 2, 3], prompt_len=4)
    with pytest.raises(ValueError):
        train_qlora._mask_prompt_tokens([1, 2, 3], prompt_len=-1)


def test_mask_prompt_tokens_never_leaves_a_single_real_token_id_in_the_masked_prefix():
    """A large-image-placeholder-block regression check: even a long "prompt" (simulating
    hundreds of image-placeholder tokens) must be entirely -100, proving the huge image-token
    block D1 exists to fix would be fully masked out, not just the text prompt."""
    fake_prompt_and_image_tokens = list(range(500))  # stands in for text + image-placeholder ids
    fake_response_tokens = [9001, 9002, 9003]
    input_ids = fake_prompt_and_image_tokens + fake_response_tokens
    labels = train_qlora._mask_prompt_tokens(input_ids, prompt_len=500)
    assert labels[:500] == [-100] * 500
    assert labels[500:] == fake_response_tokens


# --- D2: scripts/train_qlora.py collator padding (pure Python, no torch) -----------------------


def test_pad_sequences_pads_to_the_longest_sequence_by_default():
    padded = train_qlora._pad_sequences([[1, 2, 3], [1, 2], [1]], pad_value=-100)
    assert padded == [[1, 2, 3], [1, 2, -100], [1, -100, -100]]
    assert all(len(row) == 3 for row in padded)


def test_pad_sequences_respects_an_explicit_max_len():
    padded = train_qlora._pad_sequences([[1, 2], [1]], pad_value=0, max_len=5)
    assert padded == [[1, 2, 0, 0, 0], [1, 0, 0, 0, 0]]


def test_pad_sequences_noop_when_all_equal_length():
    padded = train_qlora._pad_sequences([[1, 2], [3, 4]], pad_value=-1)
    assert padded == [[1, 2], [3, 4]]


def test_heliogram_vl_collator_constructs_without_torch():
    """HeliogramVLCollator must be constructible (and its non-__call__ surface introspectable)
    without importing torch -- only __call__ing it on a real batch does. See its docstring."""
    collator = train_qlora.HeliogramVLCollator(pad_token_id=0)
    assert collator.pad_token_id == 0
    assert collator.label_pad_token_id == -100
    assert "torch" not in sys.modules


def test_pad_token_id_prefers_pad_token_falls_back_to_eos():
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 42

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    assert train_qlora._pad_token_id(FakeProcessor()) == 42

    FakeTokenizer.pad_token_id = 7
    assert train_qlora._pad_token_id(FakeProcessor()) == 7


def test_pad_token_id_raises_without_either():
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = None

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    with pytest.raises(ValueError):
        train_qlora._pad_token_id(FakeProcessor())


# --- D3: scripts/train_qlora.py LoRA target modules (pure Python, no torch) ---------------------


def test_lora_merger_target_modules_included_by_default():
    targets = train_qlora._build_lora_target_modules(include_vision_blocks=False)
    for module in train_qlora.LORA_TARGET_MODULES:
        assert module in targets
    for module in train_qlora.LORA_MERGER_TARGET_MODULES:
        assert module in targets
    for module in train_qlora.LORA_VISION_BLOCK_TARGET_MODULES:
        assert module not in targets  # opt-in only, not included by default


def test_lora_vision_block_target_modules_only_with_flag():
    targets = train_qlora._build_lora_target_modules(include_vision_blocks=True)
    for module in train_qlora.LORA_VISION_BLOCK_TARGET_MODULES:
        assert module in targets
    # still includes the always-on sets too:
    for module in train_qlora.LORA_TARGET_MODULES + train_qlora.LORA_MERGER_TARGET_MODULES:
        assert module in targets


def test_train_qlora_cli_help_documents_include_vision_blocks_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_qlora.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--include-vision-blocks" in result.stdout
    assert "--eval-examples-per-stage" in result.stdout
    assert "--eval-symbol-accuracy-examples" in result.stdout


# --- D4: processor-alignment guard rails (pure Python, no torch) -------------------------------


def test_assert_processor_alignment_passes_for_even_patch_grid():
    img = encode(b"even grid check", palette=64, patch_size=14, subpatch=1)
    img = pad_to_even_patch_grid(img, 14, 64)
    train_qlora._assert_processor_alignment(img, 14)  # must not raise


def test_assert_processor_alignment_rejects_odd_patch_grid():
    img = encode(bytes(range(16)), palette=2, patch_size=14, nsym=32, subpatch=1)
    width = img.width // 14
    assert width % 2 == 1  # sanity: this combination IS odd (unpadded)
    with pytest.raises(ValueError, match="smart_resize"):
        train_qlora._assert_processor_alignment(img, 14)


def test_identity_pixel_bounds_equals_exact_pixel_count():
    img = encode(b"bounds check", palette=64, patch_size=14, subpatch=1)
    img = pad_to_even_patch_grid(img, 14, 64)
    bounds = train_qlora._identity_pixel_bounds(img)
    assert bounds == {
        "min_pixels": img.width * img.height,
        "max_pixels": img.width * img.height,
    }


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
