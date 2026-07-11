"""Tests for heliogram.ocr_eval -- the CPU-testable core of the typography READABILITY gate
(does a stock VLM actually OCR dense ascii85 typeset text; see that module's docstring for the
full argument and heliogram/typography.py for the GEOMETRIC gate this module builds on).

Everything in this file runs on the same CPU-only, no-torch environment as
tests/test_phase2_scaffold.py -- it is a hard requirement (see heliogram/ocr_eval.py's module
docstring, mirroring heliogram/vlm.py's) that the module needs no torch/transformers merely to be
imported or to have its non-model code paths exercised. Accordingly, this file never loads a real
VLM: everywhere `evaluate_ocr` would need one, the tests below only exercise the guard rail that
fires BEFORE a model is touched (model=None -> ValueError) plus the pure-Python rendering/
parsing/recovery code that does not need one.
"""

from __future__ import annotations

import random

import pytest

from heliogram.codec import HeliogramDecodeError
from heliogram.ocr_eval import (
    ASCII85_ALPHABET_NOTE,
    OcrConfig,
    OcrExample,
    build_ocr_prompt,
    char_error_rate,
    evaluate_ocr,
    expected_max_new_tokens,
    levenshtein,
    parse_ocr_response,
    recover_payload_from_transcription,
    render_ocr_example,
)
from tests.conftest import assert_import_stays_torch_free


def _payload(seed: int = 0, size: int = 64) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


# --- import boundary --------------------------------------------------------------------------


def test_import_stays_torch_free():
    """Same CPU-only-by-default import boundary heliogram.vlm/heliogram.typography hold to:
    importing heliogram.ocr_eval must never pull in torch or transformers as a side effect."""
    assert_import_stays_torch_free("heliogram.ocr_eval")


# --- levenshtein / char_error_rate -------------------------------------------------------------


def test_levenshtein_identical_strings_is_zero():
    assert levenshtein("abcdef", "abcdef") == 0
    assert levenshtein("", "") == 0


def test_levenshtein_one_substitution():
    assert levenshtein("kitten", "kitfen") == 1


def test_levenshtein_known_classic_pair():
    # textbook example: kitten -> sitting is edit distance 3 (k->s, e->i, +g)
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_pure_insertion():
    assert levenshtein("abc", "abcde") == 2


def test_levenshtein_pure_deletion():
    assert levenshtein("abcde", "abc") == 2


def test_levenshtein_empty_vs_nonempty():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_symmetric():
    a, b = "heliogram", "heliogrmm"
    assert levenshtein(a, b) == levenshtein(b, a)


def test_char_error_rate_identical_is_zero():
    assert char_error_rate("abcdef", "abcdef") == 0.0


def test_char_error_rate_one_substitution_out_of_six():
    assert char_error_rate("abcdef", "abcxef") == pytest.approx(1 / 6)


def test_char_error_rate_can_exceed_one():
    # hypothesis much longer than reference -> CER > 1.0, a valid (if bad) result
    assert char_error_rate("ab", "abcdefgh") > 1.0


def test_char_error_rate_empty_reference_raises():
    with pytest.raises(ValueError, match="non-empty"):
        char_error_rate("", "anything")


# --- render_ocr_example: determinism + ground-truth roundtrip ----------------------------------


def test_render_ocr_example_is_deterministic():
    payload = _payload(seed=1, size=128)
    a = render_ocr_example(payload, 10, apply_rs=False)
    b = render_ocr_example(payload, 10, apply_rs=False)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.image.size == b.image.size
    assert a.ground_truth_text == b.ground_truth_text
    assert a.total_patches == b.total_patches
    assert a.bits_per_patch == b.bits_per_patch


def test_render_ocr_example_differs_for_different_payloads():
    a = render_ocr_example(_payload(seed=1, size=128), 10, apply_rs=False)
    b = render_ocr_example(_payload(seed=2, size=128), 10, apply_rs=False)
    assert a.image.tobytes() != b.image.tobytes()
    assert a.ground_truth_text != b.ground_truth_text


def test_render_ocr_example_returns_ocr_example_dataclass():
    ex = render_ocr_example(_payload(size=64), 12, apply_rs=False)
    assert isinstance(ex, OcrExample)
    assert ex.font_size_px == 12
    assert ex.apply_rs is False
    assert ex.nsym is None
    assert ex.total_patches > 0
    assert ex.bits_per_patch > 0


def test_render_ocr_example_raw_ground_truth_a85_decodes_to_payload():
    payload = _payload(seed=3, size=256)
    ex = render_ocr_example(payload, 10, apply_rs=False)
    import base64

    assert base64.a85decode(ex.ground_truth_text) == payload


def test_render_ocr_example_rs_ground_truth_rs_decodes_to_payload():
    payload = _payload(seed=4, size=256)
    ex = render_ocr_example(payload, 10, apply_rs=True, nsym=32)
    assert ex.nsym == 32
    recovered = recover_payload_from_transcription(
        ex.ground_truth_text, apply_rs=True, nsym=32
    )
    assert recovered == payload


@pytest.mark.parametrize("font_size", [14, 12, 10, 8])
def test_render_ocr_example_across_font_sizes(font_size):
    payload = _payload(seed=5, size=128)
    ex = render_ocr_example(payload, font_size, apply_rs=True, nsym=32)
    recovered = recover_payload_from_transcription(
        ex.ground_truth_text, apply_rs=True, nsym=32
    )
    assert recovered == payload


# --- perfect-transcription roundtrip (parse_ocr_response + recover_payload_from_transcription) --


def test_perfect_transcription_roundtrip_raw():
    payload = _payload(seed=6, size=200)
    ex = render_ocr_example(payload, 10, apply_rs=False)
    # simulate a "perfect" model response: the exact ground truth, plus some benign whitespace
    # wrapping (what a real row-per-line rendering would look like once transcribed)
    fake_response = "\n".join(
        ex.ground_truth_text[i : i + 20] for i in range(0, len(ex.ground_truth_text), 20)
    )
    parsed = parse_ocr_response(fake_response)
    assert parsed == ex.ground_truth_text
    recovered = recover_payload_from_transcription(parsed, apply_rs=False)
    assert recovered == payload


def test_perfect_transcription_roundtrip_rs():
    payload = _payload(seed=7, size=200)
    ex = render_ocr_example(payload, 10, apply_rs=True, nsym=32)
    fake_response = "  " + "\n".join(
        ex.ground_truth_text[i : i + 15] for i in range(0, len(ex.ground_truth_text), 15)
    ) + "\n"
    parsed = parse_ocr_response(fake_response)
    assert parsed == ex.ground_truth_text
    recovered = recover_payload_from_transcription(parsed, apply_rs=True, nsym=32)
    assert recovered == payload


def test_parse_ocr_response_strips_all_whitespace_kinds():
    assert parse_ocr_response(" a\tb\nc \r\n d ") == "abcd"


# --- recover_payload_from_transcription: failure modes -----------------------------------------


def test_recover_raises_on_invalid_ascii85():
    with pytest.raises(HeliogramDecodeError, match="ascii85"):
        recover_payload_from_transcription("not \x00 valid ascii85 at all \x01", apply_rs=False)


def test_recover_rs_raises_on_too_short_stream():
    import base64

    short_text = base64.a85encode(b"abc").decode("ascii")  # 3 bytes, well under the 5-byte header
    with pytest.raises(HeliogramDecodeError, match="5-byte"):
        recover_payload_from_transcription(short_text, apply_rs=True, nsym=32)


def test_recover_rs_raises_on_garbled_frame():
    """A transcription that a85-decodes fine but is not a real RS-framed stream must raise, not
    silently return wrong bytes -- the whole point of recover_payload_from_transcription's
    contract (mirrors heliogram.codec.decode_pixels' DATA HONESTY note)."""
    import base64

    garbage = bytes(random.Random(42).getrandbits(8) for _ in range(64))
    text = base64.a85encode(garbage).decode("ascii")
    with pytest.raises(HeliogramDecodeError):
        recover_payload_from_transcription(text, apply_rs=True, nsym=32)


def test_recover_raw_never_raises_on_well_formed_ascii85_but_returns_the_decoded_bytes():
    # apply_rs=False has no framing to validate -- any well-formed ascii85 decodes to *some*
    # bytes, which is the correct (if not payload-matching) behavior; it's the RS path that adds
    # the frame-validation failure surface exercised above.
    import base64

    arbitrary = bytes(random.Random(99).getrandbits(8) for _ in range(32))
    text = base64.a85encode(arbitrary).decode("ascii")
    assert recover_payload_from_transcription(text, apply_rs=False) == arbitrary


# --- build_ocr_prompt ---------------------------------------------------------------------------


def test_build_ocr_prompt_mentions_char_count_and_ascii85():
    prompt = build_ocr_prompt(123)
    assert "123" in prompt
    assert "ascii85" in prompt


def test_build_ocr_prompt_default_alphabet_note_is_ascii85_note():
    prompt = build_ocr_prompt(10)
    assert ASCII85_ALPHABET_NOTE in prompt


def test_build_ocr_prompt_accepts_custom_alphabet_note():
    prompt = build_ocr_prompt(10, alphabet_note="CUSTOM NOTE MARKER")
    assert "CUSTOM NOTE MARKER" in prompt
    assert ASCII85_ALPHABET_NOTE not in prompt


# --- expected_max_new_tokens ---------------------------------------------------------------------


def test_expected_max_new_tokens_scales_with_margin():
    assert expected_max_new_tokens(100, margin=3.0) == 300
    assert expected_max_new_tokens(0) == 1  # max(1, ...) floor, never zero-or-negative


# --- evaluate_ocr: model=None/processor=None must raise, never fabricate -----------------------


def test_evaluate_ocr_raises_on_model_none():
    with pytest.raises(ValueError, match="model"):
        evaluate_ocr(None, object(), [OcrConfig(font_size_px=10, payload_size=32)])


def test_evaluate_ocr_raises_on_processor_none():
    with pytest.raises(ValueError, match="model"):
        evaluate_ocr(object(), None, [OcrConfig(font_size_px=10, payload_size=32)])


def test_evaluate_ocr_raises_on_both_none():
    with pytest.raises(ValueError):
        evaluate_ocr(None, None, [OcrConfig(font_size_px=10, payload_size=32)])


def test_ocr_renders_are_smart_resize_identity():
    """REGRESSION (the confounded first GPU run): every OCR render MUST be 28px-aligned so the
    model's mandatory smart_resize is the identity -- otherwise the processor silently resamples
    the canvas and the model grades a blurred image, not the rendering (the whole readability
    measurement is then meaningless). render_ocr_example uses align=2 for exactly this. Pin it
    across font sizes, both ECC variants, at the payload size the runner defaults to."""
    from heliogram.corruption import QWEN_GENEROUS_MAX_PIXELS, qwen_smart_resize_dims

    payload = bytes(random.Random(0).getrandbits(8) for _ in range(256))
    for font_size in (14, 12, 10, 8):
        for apply_rs in (False, True):
            ex = render_ocr_example(payload, font_size, apply_rs=apply_rs)
            w, h = ex.image.size
            assert w % 28 == 0 and h % 28 == 0, (
                f"OCR render fs={font_size} rs={apply_rs} is {w}x{h}, not 28px-aligned -- "
                "smart_resize will resample it"
            )
            h2, w2 = qwen_smart_resize_dims(
                h, w, min_pixels=28 * 28, max_pixels=QWEN_GENEROUS_MAX_PIXELS
            )
            assert (w, h) == (w2, h2), (
                f"OCR render fs={font_size} rs={apply_rs} is NOT smart_resize-identity: "
                f"{w}x{h} -> {w2}x{h2}"
            )


def test_typography_default_geometry_unchanged_by_align_param():
    """The align parameter must default to 1 so heliogram.typography's own pinned geometry numbers
    (test_typography.py) are untouched -- only the OCR path opts into align=2."""
    from heliogram.typography import render_typeset_density

    payload = bytes(range(64))
    default = render_typeset_density(payload, 12, apply_rs=True)
    explicit1 = render_typeset_density(payload, 12, apply_rs=True, align=1)
    assert default.image.size == explicit1.image.size
    assert default.total_patches == explicit1.total_patches
    assert default.bits_per_patch == explicit1.bits_per_patch
