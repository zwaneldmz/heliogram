"""heliogram.ocr_eval -- Phase-2 readability gate for the "dense typeset glyphs" pivot (Option 2).

CONTEXT (see heliogram/typography.py's module docstring for the full argument, which this module
builds directly on top of): the color-block codec is measured dead as compression (its net
ceiling, 6.996 bits/patch, is strictly below the measured text-context bars in
heliogram/data/text_baselines.json -- ascii85 at 8.374 bits/token is the strongest measured
encoding and therefore the honest bar). heliogram.typography answers the CHEAP question first:
does RS-framed ascii85 text, rendered as typeset glyphs at the codec's own 14px patch grid, clear
that 8.374 bits/token bar *geometrically* -- i.e. assuming perfect legibility? It measured YES,
at every font size down to 4px, and specifically at 12px and below for the harder ascii85 bar.

THIS MODULE ANSWERS THE NEXT QUESTION, THE ONE typography.py'S DOCSTRING EXPLICITLY DEFERS: can a
STOCK (not fine-tuned) Qwen2.5-VL tower actually OCR ascii85 text small enough (<=12px) to be
RECOVERABLE -- i.e. does real readability reach the font sizes where the geometric density
economics work? Geometric density is a model-free upper bound; this module measures the thing it
assumes away (perfect legibility) against a real model, exactly the same "geometric gate first,
then readability gate" two-step heliogram.baselines' `rendered_text_density` and
heliogram.typography's own module docstring both call out as the necessary-but-not-sufficient
relationship between the two questions.

DATA HONESTY (read this first before trusting any number this module produces, and before
touching `evaluate_ocr` -- same discipline as heliogram/vlm.py's module docstring, which this
module mirrors deliberately): nothing in this module has been run against a real model in this
repository -- there is no GPU here. `evaluate_ocr` REQUIRES a real, already-loaded `model`/
`processor` (e.g. from `transformers`); if either is missing it raises ValueError immediately
rather than fabricating a number or silently falling back to a geometric estimate. Every torch/
transformers import in this file is local to the one function that actually needs it (`_generate`
below, called only from inside `evaluate_ocr` after the model/processor presence check has
already passed) -- this module's own top-level imports are exactly as light as
heliogram.typography's (pillow/numpy/reedsolo + stdlib), so `import heliogram.ocr_eval` never
requires torch/transformers/peft/bitsandbytes.

What's actually implemented vs. what's untested:

- `levenshtein`/`char_error_rate`: plain Python edit-distance arithmetic, exact and CPU-tested
  (tests/test_ocr_eval.py) -- no model involved, nothing to be untested about.
- `render_ocr_example`: thin wrapper around `heliogram.typography.render_typeset_density` (reused,
  not duplicated -- see that function's docstring for the actual rendering/layout logic) that
  additionally recovers the ground-truth ascii85 TEXT the image typeset, using the exact same
  deterministic `base64.a85encode(stream)` construction `render_typeset_density` uses internally
  (`stream` via `heliogram.typography._rs_frame`, the SAME RS-framing helper, not a
  reimplementation). This is plain, deterministic code, CPU-tested for determinism and roundtrip.
- `recover_payload_from_transcription`: feeds a transcription through ascii85-decode and (if
  `apply_rs`) the SAME Reed-Solomon/framing contract `heliogram.codec.encode`/`decode_pixels` use
  (version byte + 4-byte length + payload), reusing `heliogram.codec.CODEC_VERSION` and raising
  `heliogram.codec.HeliogramDecodeError` -- the project's one canonical "recovery failed"
  exception -- on any failure. Plain code, CPU-tested (perfect-transcription roundtrip, and that
  garbage raises rather than returning wrong bytes).
- `build_ocr_prompt`/`parse_ocr_response`: prompt wording and response parsing, following the same
  "one canonical wording" convention `heliogram.dataset.build_prompt`/`parse_output_text`
  establish (see that module's "PROMPT/OUTPUT-CONTRACT UNIFICATION" note) -- UNTESTED against real
  model output (instruction-following quirks only a real run can surface), but the parsing side
  (whitespace-stripping) is plain code and CPU-tested.
- `_generate` and the `model.generate(...)` call inside `evaluate_ocr`: follows the documented HF
  Qwen2-VL/Qwen2.5-VL chat-template + processor + generate pattern, MIRRORING
  `heliogram.vlm.QwenVLDecoder._generate` byte-for-byte in structure (chat template,
  `processor(text=..., images=...)`, `model.generate(...)`, slice off the input tokens,
  `batch_decode`) -- but, like that function, has never been run against a real processor in this
  environment. Treat it as a documented starting point, not a verified integration.

See `scripts/run_typography_ocr.py` for the GPU runner that actually calls `evaluate_ocr` against
a loaded stock model, and RUNBOOK-GPU.md section 3.5 for the exact command and the three-way
verdict rule (pivot REAL / needs FINE-TUNING / DEAD) this experiment is designed to decide.
"""

from __future__ import annotations

import base64
import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from PIL import Image
from reedsolo import RSCodec

from .codec import CODEC_VERSION, PATCH_SIZE, HeliogramDecodeError
from .dataset import random_payload
from .typography import DEFAULT_NSYM, _rs_frame, render_typeset_density

__all__ = [
    "ASCII85_ALPHABET_NOTE",
    "DEFAULT_OCR_FONT_SIZES_PX",
    "DEFAULT_OCR_PAYLOAD_SIZE",
    "levenshtein",
    "char_error_rate",
    "OcrExample",
    "render_ocr_example",
    "build_ocr_prompt",
    "parse_ocr_response",
    "recover_payload_from_transcription",
    "expected_max_new_tokens",
    "OcrConfig",
    "OcrResult",
    "evaluate_ocr",
]

# Representative font sizes to actually spend GPU generation budget on: heliogram.typography
# measured the geometric ascii85 bar (8.374 bits/token) clears from 12px down; 14px is kept as
# the "should read trivially" control (bigger than the codec's own patch grid unit) so a total
# failure at 14px would be a strong signal the tower can't OCR this alphabet/layout AT ALL, not
# merely "not small enough" -- see scripts/run_typography_ocr.py's verdict rule. Deliberately
# narrower than heliogram.typography.DEFAULT_FONT_SIZES_PX (which also sweeps 6px/4px): those
# sizes are already known to be a geometric-only exercise (real OCR at 4-6px monospace text is
# not a serious readability bet for a general-purpose VLM tower) and would only burn GPU budget
# for a near-certain "illegible" result -- run them explicitly via --font-sizes if wanted.
DEFAULT_OCR_FONT_SIZES_PX = (14, 12, 10, 8)

# Small on purpose: this experiment measures READABILITY (does the tower resolve the glyphs at
# all), not payload capacity -- heliogram.typography's DEFAULT_PAYLOAD_SIZE (4096B) would RS-frame
# to a multi-hundred-character ascii85 string whose generation alone (at a 3x token margin, see
# expected_max_new_tokens below) costs real GPU money before the readability question is even
# answered. 256B keeps a single trial's max_new_tokens in the low hundreds regardless of variant.
DEFAULT_OCR_PAYLOAD_SIZE = 256

# ascii85 (base64.a85encode, Python's default foldspaces=False/adobe=False) alphabet: printable
# ASCII '!' (0x21) through 'u' (0x75), 85 symbols, plus the standard 'z' shorthand for four
# consecutive zero bytes (always enabled) and the optional 'y' shorthand for four consecutive
# spaces (only emitted with foldspaces=True, which heliogram.typography/this module never pass --
# included in the note anyway since a model seeing 'y' data characters should not "correct" them
# out of caution, and because a caller's own alphabet_note override may render text produced with
# foldspaces=True). No digits-only, letters-only, or punctuation-only subrun should be treated as
# more or less likely than any other -- every character in this set is a normal, equally-valid
# data symbol.
ASCII85_ALPHABET_NOTE = (
    "The text uses the ascii85 (base64.a85encode) alphabet: printable ASCII characters from '!' "
    "(0x21) through 'u' (0x75), plus the special shorthand character 'z' (standing in for four "
    "all-zero bytes) and, occasionally, 'y' (four consecutive spaces). Every one of these "
    "characters is an equally valid data symbol -- do not 'correct' an unfamiliar-looking run of "
    "punctuation into something more plausible-looking."
)


# --------------------------------------------------------------------------------------------
# Character error rate: pure Python, exact, no model involved.
# --------------------------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Exact Levenshtein (single-character insert/delete/substitute) edit distance between `a`
    and `b`. Plain iterative DP, O(len(a)*len(b)) time, O(min(len(a), len(b))) memory (via the
    row-swap below) -- no external dependency, deterministic, and small enough that its own
    correctness is checked directly in tests/test_ocr_eval.py against known-answer pairs rather
    than trusted by construction."""
    if a == b:
        return 0
    # Iterate the DP over the SHORTER string so the O(min(len)) row stays small.
    if len(a) < len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution (or match)
            )
        prev = curr
    return prev[lb]


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein edit distance between `hypothesis` and `reference`, normalized by
    `len(reference)` -- the standard character error rate (CER) definition.

    Raises ValueError (never divides by zero, never fabricates a number) if `reference` is
    empty -- CER is undefined for an empty ground truth; callers should never construct one (a
    real ascii85 transcription target is never empty for a non-empty payload), so hitting this
    guard means something upstream constructed a bad example, worth surfacing loudly rather than
    silently returning 0.0 or inf.
    """
    if len(reference) == 0:
        raise ValueError(
            "char_error_rate requires a non-empty reference string (dividing by len(reference) "
            "would divide by zero) -- pass the real ground-truth transcription, never an empty "
            "placeholder"
        )
    return levenshtein(reference, hypothesis) / len(reference)


# --------------------------------------------------------------------------------------------
# Rendering: thin wrapper over heliogram.typography, adding the ground-truth transcription text.
# --------------------------------------------------------------------------------------------


@dataclass
class OcrExample:
    """One rendered OCR probe example: the image a VLM would actually be shown, the exact
    ground-truth ascii85 text it typesets (the transcription target), and the payload/geometric
    metrics needed to score a model's response against it and cross-reference the geometric bars.
    """

    image: Image.Image
    ground_truth_text: str  # exactly what render_typeset_density typeset -- the OCR target
    payload: bytes
    font_size_px: int
    apply_rs: bool
    nsym: Optional[int]  # None when apply_rs is False, matching TypesetDensity's convention
    total_patches: int
    bits_per_patch: float  # geometric (model-free, perfect-legibility) upper bound -- see
    # heliogram.typography's DATA HONESTY note; this is NOT a measurement of what the model below
    # actually reads.


def render_ocr_example(
    payload: bytes,
    font_size_px: int,
    *,
    apply_rs: bool = False,
    nsym: int = DEFAULT_NSYM,
    patch_size: int = PATCH_SIZE,
) -> OcrExample:
    """Render `payload` as ascii85 typeset text at `font_size_px`, reusing
    `heliogram.typography.render_typeset_density` for the actual rendering/layout/geometric-
    density arithmetic (NOT reimplemented here), and additionally recover the exact ground-truth
    text that was typeset -- the transcription target a VLM's OCR output should be scored
    against.

    The ground-truth text is recomputed via the SAME deterministic construction
    `render_typeset_density` uses internally (`base64.a85encode(stream)`, `stream` via
    `heliogram.typography._rs_frame` when `apply_rs=True`, raw `payload` bytes otherwise) rather
    than duplicating any drawing/layout logic -- `render_typeset_density` itself does not return
    the text, only the rendered image and geometric metrics, so this is the minimal extra step
    needed to get a scorable ground truth, not a parallel reimplementation of rendering.
    """
    density = render_typeset_density(
        payload, font_size_px, apply_rs=apply_rs, nsym=nsym, patch_size=patch_size
    )
    stream = _rs_frame(payload, nsym) if apply_rs else bytes(payload)
    ground_truth_text = base64.a85encode(stream).decode("ascii")
    assert len(ground_truth_text) == density.rendered_len, (
        "ground-truth text length disagrees with what render_typeset_density actually typeset -- "
        "the stream construction above has drifted from typography.render_typeset_density's own; "
        "this is an internal consistency bug, not a user-facing condition"
    )

    return OcrExample(
        image=density.image,
        ground_truth_text=ground_truth_text,
        payload=bytes(payload),
        font_size_px=font_size_px,
        apply_rs=apply_rs,
        nsym=nsym if apply_rs else None,
        total_patches=density.total_patches,
        bits_per_patch=density.bits_per_patch,
    )


# --------------------------------------------------------------------------------------------
# Prompt / response parsing -- the canonical OCR wording, mirroring heliogram.dataset's
# "one canonical wording" convention for build_prompt/parse_output_text.
# --------------------------------------------------------------------------------------------


def build_ocr_prompt(n_chars: int, alphabet_note: str = ASCII85_ALPHABET_NOTE) -> str:
    """THE canonical OCR transcription prompt for a rendered ascii85 typography example -- the
    single wording this module's `evaluate_ocr` and `scripts/run_typography_ocr.py` both use, not
    an independently-written string per call site (same rationale as
    `heliogram.dataset.build_prompt` being THE canonical prompt for the pixel-grid task: a model
    is prompt-sensitive, so drift between call sites is a silent, avoidable capability tax).

    `n_chars` tells the model exactly how long a correct transcription is, mirroring
    `heliogram.dataset.build_prompt` telling the model exactly how many grid cells to produce.
    `alphabet_note` documents the ascii85 alphabet (see `ASCII85_ALPHABET_NOTE`) so the model does
    not "autocorrect" unfamiliar punctuation runs into more plausible-looking text.
    """
    return (
        "This image shows a block of small monospace text. The text is a single ascii85 "
        f"(base64.a85encode) encoded data string, exactly {n_chars} characters long, wrapped "
        "onto multiple lines purely for layout -- line breaks are NOT part of the data. "
        f"{alphabet_note} Transcribe EXACTLY the visible characters, in reading order (left to "
        "right, top to bottom), with no separators: concatenate every line's characters back "
        "into one continuous string, collapsing out all whitespace and line breaks (they are "
        "layout only). Output nothing else: no explanation, no code fence, no leading or "
        f"trailing whitespace -- just the exact {n_chars}-character transcription."
    )


def parse_ocr_response(text: str) -> str:
    """Strip ALL whitespace (spaces and newlines alike) from a model's raw text response, leaving
    the transcription as a single logical string ready for `recover_payload_from_transcription`.

    Mirrors `heliogram.dataset.parse_output_text`'s whitespace-strip approach, and for the same
    reason: `build_ocr_prompt` tells the model line breaks are layout only, not data, so the
    correct parse is "concatenate every non-whitespace character", nothing more -- no fenced-
    code-block detection or other heuristic, since ascii85's alphabet can legitimately contain any
    printable ASCII punctuation character a fence-detector might otherwise mistake for a
    delimiter (the same backtick-ambiguity rationale `heliogram.dataset.parse_output_text`'s
    docstring documents for the pixel-grid task's SYMBOL_ALPHABET).
    """
    return "".join(text.split())


def recover_payload_from_transcription(
    transcription: str, *, apply_rs: bool, nsym: int = DEFAULT_NSYM
) -> bytes:
    """Recover the original payload bytes from a (parsed) OCR transcription.

    `apply_rs=False`: plain `base64.a85decode(transcription)`, returned as-is -- any decode
    failure (malformed ascii85, e.g. a mistranscribed character) raises
    `heliogram.codec.HeliogramDecodeError` immediately; this function NEVER returns bytes it
    cannot fully verify came from a successful decode, mirroring `heliogram.codec.decode_pixels`'s
    "raise rather than silently return wrong bytes" contract.

    `apply_rs=True`: ascii85-decodes to the RS-framed ecc byte stream, then RS-decodes and strips
    the frame using the EXACT SAME contract `heliogram.codec.encode`/`decode_pixels` use (version
    byte + 4-byte big-endian length + payload, `reedsolo.RSCodec(nsym)`) -- reusing
    `heliogram.codec.CODEC_VERSION` and raising `heliogram.codec.HeliogramDecodeError` on any
    failure, the SAME exception `heliogram.vlm._payload_from_symbols`/`heliogram.codec.
    decode_pixels` raise, so callers get one exception type to catch across every heliogram
    payload-recovery path.

    DATA HONESTY / RS MIS-CORRECTION CAVEAT (same as `heliogram.codec.decode_pixels`'s own
    docstring): bounded-distance Reed-Solomon decoding GUARANTEES correction only up to
    `floor(nsym/2)` byte errors per RS_NSIZE-byte chunk, and GUARANTEES detection (raising rather
    than returning anything) only up to `nsym - floor(nsym/2)` errors. Beyond that, RS decoding
    can MIS-CORRECT: land on a different, wrong, but internally-consistent codeword and return
    wrong bytes without raising. The version-byte and length-self-consistency checks below make a
    false accept astronomically unlikely, not impossible -- do not treat "did not raise" as an
    unconditional correctness proof for arbitrarily corrupted transcriptions.
    """
    try:
        stream = base64.a85decode(transcription)
    except Exception as exc:  # binascii.Error and friends -- never silently return garbage bytes
        raise HeliogramDecodeError(
            f"ascii85 decode of the transcription failed: {exc}"
        ) from exc

    if not apply_rs:
        return stream

    if len(stream) < 5:
        raise HeliogramDecodeError(
            "ascii85-decoded stream is shorter than the 5-byte RS framing header (version + "
            "4-byte length) -- transcription is too short/garbled to have carried a real frame"
        )
    try:
        decoded_message, _, _ = RSCodec(nsym).decode(stream)
    except Exception as exc:  # reedsolo raises ReedSolomonError on uncorrectable input
        raise HeliogramDecodeError(f"Reed-Solomon decode failed: {exc}") from exc

    decoded_message = bytes(decoded_message)
    if len(decoded_message) < 5:
        raise HeliogramDecodeError("RS-decoded message shorter than the 5-byte framing header")
    if decoded_message[0] != CODEC_VERSION:
        raise HeliogramDecodeError(
            f"unsupported/corrupted codec version byte {decoded_message[0]!r} (expected "
            f"{CODEC_VERSION}) -- see this function's DATA HONESTY note on RS mis-correction risk"
        )
    payload_len = struct.unpack(">I", decoded_message[1:5])[0]
    if len(decoded_message) < 5 + payload_len:
        raise HeliogramDecodeError(
            "RS-decoded message shorter than its own declared payload length -- frame is "
            "internally inconsistent, refusing to return a possibly-wrong slice"
        )
    return decoded_message[5 : 5 + payload_len]


def expected_max_new_tokens(n_chars: int, margin: float = 3.0) -> int:
    """Size a generation token budget from the ACTUAL expected transcription length, the same
    "size it dynamically, not with a fixed guess" fix `heliogram.vlm.expected_max_new_tokens`
    documents (D5(a) of the Phase-2 scaffold review) for the pixel-grid task.

    `margin` (default 3.0x `n_chars`): a correct ascii85 transcription is one character per
    output position, but ascii85's alphabet includes punctuation characters that are NOT
    guaranteed single-token in an arbitrary BPE vocabulary (the same reasoning
    `heliogram.vlm.expected_max_new_tokens`'s docstring gives for `SYMBOL_ALPHABET`'s
    punctuation range) -- an UNVERIFIED (no GPU/real tokenizer here) safety factor, generous
    enough to survive some worst-case multi-token characters without wasting excessive GPU time
    generating far past a correct response.
    """
    return max(1, math.ceil(n_chars * margin))


# --------------------------------------------------------------------------------------------
# evaluate_ocr: the one function that reaches a real model. model=None/processor=None raises
# ValueError immediately -- see this module's DATA HONESTY note.
# --------------------------------------------------------------------------------------------


@dataclass
class OcrConfig:
    """One (font size, payload size, ECC variant) cell to evaluate."""

    font_size_px: int
    payload_size: int
    apply_rs: bool = False
    nsym: int = DEFAULT_NSYM
    patch_size: int = PATCH_SIZE


@dataclass
class OcrResult:
    """One `OcrConfig`'s worth of `evaluate_ocr` results, averaged over `trials` payloads.
    `raw_transcriptions`/`raw_ground_truths` keep every trial's actual model output and target --
    there is no other record of what a real stock model actually transcribed."""

    config: OcrConfig
    trials: int
    mean_cer: float
    exact_match_rate: float
    decode_success_rate: float
    bits_per_patch: float  # geometric reference figure for this config, see OcrExample
    raw_transcriptions: List[str] = field(default_factory=list)
    raw_ground_truths: List[str] = field(default_factory=list)


def _generate(
    model: object,
    processor: object,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> str:
    """Run one VLM generation call. ALL torch imports are local to this function -- it is the
    only place in this module that requires torch, and it is only ever reached from `evaluate_ocr`
    after the model/processor presence check has already passed.

    MIRRORS `heliogram.vlm.QwenVLDecoder._generate` exactly (same chat-template + processor +
    generate + slice-off-input-tokens + batch_decode sequence) -- deliberately not a
    reimplementation with its own quirks, so whatever adjustment a real GPU run finds necessary
    for one applies equally to the other. UNTESTED against a real processor in this environment
    (no GPU here) -- see this module's docstring.
    """
    import torch  # lazy: heavy GPU dep, see module docstring

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt")

    target_device = device or getattr(model, "device", None)
    if target_device is not None:
        inputs = {
            k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0]


def evaluate_ocr(
    model: object,
    processor: object,
    configs: Sequence[OcrConfig],
    *,
    n_trials: int = 3,
    seed: int = 0,
    max_new_tokens: Optional[int] = None,
) -> List[OcrResult]:
    """Phase-2 typography readability gate: run a STOCK (not fine-tuned) model directly over
    rendered ascii85 typography examples and measure OCR transcription quality against ground
    truth -- the readability half of the "does the pivot survive" question
    `heliogram.typography` deliberately leaves open (see this module's docstring).

    DATA HONESTY (the entire point of this function, same contract as
    `heliogram.vlm.zero_shot_symbol_error`): `model`/`processor` MUST be real, already-loaded
    objects (e.g. a stock `Qwen/Qwen2.5-VL-7B-Instruct` loaded via `transformers`). Passing
    `model=None` or `processor=None` raises ValueError immediately -- there is no fallback path,
    default value, or cached result that could return an invented number. Every number in the
    returned `OcrResult`s comes from an actual call to `model.generate(...)` (via `_generate`).
    This repo's CPU pytest environment has no GPU, so this function is only ever unit-tested here
    with the `model=None`/`processor=None` guard, never with a real forward pass -- see
    tests/test_ocr_eval.py.

    For each `OcrConfig` in `configs`: renders `n_trials` examples (`render_ocr_example`, payloads
    seeded from `seed` -- deterministic in the *inputs*, not in the model's output), transcribes
    each with the model, and measures:
      - `mean_cer`: mean `char_error_rate(ground_truth_text, transcription)` across trials.
      - `exact_match_rate`: fraction of trials where the parsed transcription equals the
        ground-truth text exactly, character for character.
      - `decode_success_rate`: fraction of trials where
        `recover_payload_from_transcription(transcription, apply_rs=config.apply_rs,
        nsym=config.nsym)` equals the original payload bytes exactly -- the metric that actually
        answers "would this transcription have recovered the real payload", not merely "how close
        was it textually" (a single dropped character can be textually near-perfect CER-wise yet
        fail every downstream recovery, especially without RS).

    `max_new_tokens` (default `None`) means "size it per-example from the actual expected
    transcription length" via `expected_max_new_tokens`, the same `None`-means-dynamic-sizing
    contract `heliogram.vlm.QwenVLDecoder`/`zero_shot_symbol_error` use; pass an explicit int only
    to override with a fixed budget.
    """
    if model is None or processor is None:
        raise ValueError(
            "evaluate_ocr requires a real, already-loaded model and processor (got model=None or "
            "processor=None) -- it never fabricates results. Load a stock "
            "Qwen2.5-VL-7B-Instruct (or similar) via transformers first; see "
            "scripts/run_typography_ocr.py's docstring for the expected package versions."
        )

    results: List[OcrResult] = []
    for config in configs:
        transcriptions: List[str] = []
        ground_truths: List[str] = []
        cers: List[float] = []
        exact_matches = 0
        decode_successes = 0
        bits_per_patch = 0.0

        for trial in range(n_trials):
            payload = random_payload(seed + trial, config.payload_size)
            example = render_ocr_example(
                payload,
                config.font_size_px,
                apply_rs=config.apply_rs,
                nsym=config.nsym,
                patch_size=config.patch_size,
            )
            bits_per_patch = example.bits_per_patch

            prompt = build_ocr_prompt(len(example.ground_truth_text))
            trial_max_new_tokens = max_new_tokens or expected_max_new_tokens(
                len(example.ground_truth_text)
            )
            raw_text = _generate(
                model, processor, example.image, prompt, max_new_tokens=trial_max_new_tokens
            )  # real forward pass -- see docstring

            transcription = parse_ocr_response(raw_text)
            transcriptions.append(transcription)
            ground_truths.append(example.ground_truth_text)

            cers.append(char_error_rate(example.ground_truth_text, transcription))
            if transcription == example.ground_truth_text:
                exact_matches += 1

            try:
                recovered = recover_payload_from_transcription(
                    transcription, apply_rs=config.apply_rs, nsym=config.nsym
                )
                if recovered == example.payload:
                    decode_successes += 1
            except HeliogramDecodeError:
                pass

        results.append(
            OcrResult(
                config=config,
                trials=n_trials,
                mean_cer=(sum(cers) / len(cers)) if cers else 0.0,
                exact_match_rate=(exact_matches / n_trials) if n_trials else 0.0,
                decode_success_rate=(decode_successes / n_trials) if n_trials else 0.0,
                bits_per_patch=bits_per_patch,
                raw_transcriptions=transcriptions,
                raw_ground_truths=ground_truths,
            )
        )
    return results
