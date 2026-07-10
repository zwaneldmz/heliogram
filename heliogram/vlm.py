"""heliogram.vlm -- Phase-2 VLM decoder plug point (GPU work, no GPU in this repo's CPU env).

DATA HONESTY (read this first): nothing in this module has been run against a real model in
this repository -- there is no GPU here. `QwenVLDecoder.__call__` and `zero_shot_symbol_error`
both REQUIRE a real, already-loaded `model`/`processor` (e.g. from `transformers`); if either is
missing they raise immediately rather than fabricating a number or falling back to
`decode_pixels`. Every torch/transformers import in this file is local to the one method that
actually needs it (`QwenVLDecoder._generate`) -- this module's own top-level imports are exactly
as light as `heliogram.codec`'s (pillow/numpy/reedsolo + stdlib), so `import heliogram.vlm` (and
`import heliogram`, which re-exports it) never requires torch/transformers/peft/bitsandbytes.

What's actually implemented vs. what's untested:

- Parsing a model's text response into a symbol list, and feeding that symbol list through the
  exact Reed-Solomon/framing layer `heliogram.codec.decode_pixels` uses (`_payload_from_symbols`
  below) -- this is plain code, testable and tested on CPU (see tests/), no model required.
- The actual prompt wording and the `processor(...)`/`model.generate(...)` call sequence in
  `_generate` -- this follows the documented Hugging Face Qwen2-VL/Qwen2.5-VL chat-template
  pattern, but has never been run against the real model/processor. Treat it as a documented
  starting point to adjust once a GPU and the real `Qwen/Qwen2.5-VL-7B-Instruct` processor are
  on hand, not as a verified integration.

See `scripts/train_qlora.py` for how a real `model`/`processor` pair would be produced, and the
README's "Phase 2 (GPU)" section for the end-to-end flow.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from PIL import Image
from reedsolo import RSCodec

from .codec import (
    CODEC_VERSION,
    PATCH_SIZE,
    VALID_SUBPATCHES,
    HeliogramDecodeError,
    bits_per_symbol,
    encode,
    extract_symbols,
    rs_encoded_length,
)
from .dataset import SYMBOL_ALPHABET, random_payload, target_to_symbols

__all__ = ["QwenVLDecoder", "ZeroShotResult", "zero_shot_symbol_error"]


def _check_subpatch(patch_size: int, subpatch: int) -> None:
    if subpatch not in VALID_SUBPATCHES:
        raise ValueError(f"subpatch must be one of {VALID_SUBPATCHES}, got {subpatch!r}")
    if patch_size % subpatch != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be evenly divisible by subpatch ({subpatch})"
        )


def _payload_from_symbols(symbols: Sequence[int], palette: int, nsym: int) -> bytes:
    """Feed a symbol list through the exact same RS/framing layer `decode_pixels` uses, without
    re-deriving those symbols from pixels -- this is the seam that lets a VLM-produced symbol
    list (instead of `extract_symbols`' pixel classification) reach the same Reed-Solomon decode
    and frame-header parsing `heliogram/codec.py` already implements.

    Reuses `heliogram.codec`'s bit-packing helper (`_symbols_to_bytes`, imported here even
    though it is a private name) rather than re-deriving the MSB-first symbol-bits-to-bytes
    scheme independently: `heliogram/codec.py` is out of scope to edit in this slice and does
    not expose a public "symbols -> payload" entry point separate from the full
    pixel-sampling `decode_pixels` (which starts from an image, not an arbitrary symbol list).
    Duplicating that bit-packing arithmetic here would be exactly the kind of RS/framing
    reimplementation this module is meant to avoid.
    """
    from .codec import _symbols_to_bytes  # local: see docstring for why this private import

    bps = bits_per_symbol(palette)
    stream = _symbols_to_bytes(list(symbols), bps)

    if len(stream) < 5:
        raise HeliogramDecodeError("recovered stream shorter than the 5-byte framing header")

    payload_len = struct.unpack(">I", stream[1:5])[0]
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    if len(stream) < ecc_len:
        raise HeliogramDecodeError(
            f"recovered stream too short ({len(stream)}B) for the framed message "
            f"({ecc_len}B expected) -- VLM transcription likely incomplete or malformed"
        )

    ecc_bytes = bytes(stream[:ecc_len])
    try:
        decoded_message, _, _ = RSCodec(nsym).decode(ecc_bytes)
    except Exception as exc:  # reedsolo raises ReedSolomonError on uncorrectable input
        raise HeliogramDecodeError(f"Reed-Solomon decode failed: {exc}") from exc

    decoded_message = bytes(decoded_message)
    if len(decoded_message) < message_len:
        raise HeliogramDecodeError("decoded message shorter than the expected framing")
    if decoded_message[0] != CODEC_VERSION:
        raise HeliogramDecodeError(
            f"unsupported/corrupted codec version byte {decoded_message[0]!r} "
            f"(expected {CODEC_VERSION})"
        )
    return decoded_message[5 : 5 + payload_len]


def _extract_symbol_string(text: str) -> str:
    """Best-effort extraction of the transcribed symbol string from a model's raw text
    response: prefer the contents of the first fenced code block (the prompt asks the model to
    wrap its answer in one), else fall back to the whole response with all whitespace stripped.

    UNTESTED against real model output (see module docstring) -- this is a plain heuristic;
    revisit it once real VLM responses are on hand, since instruction-following quirks (extra
    prose, multiple code blocks, stray punctuation) are exactly the kind of thing that cannot be
    anticipated without actually running the model.
    """
    fence = re.search(r"```(?:[^\n`]*\n)?(.*?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    return "".join(body.split())  # drop all whitespace/newlines


class QwenVLDecoder:
    """Phase-2 decoder plug point: wraps a (ideally QLoRA fine-tuned, see
    `scripts/train_qlora.py`) Qwen2.5-VL-style model to transcribe a heliogram symbol grid
    directly from pixels, in place of `decode_pixels`' nearest-neighbor pixel classifier.

    Callable with `(img, palette=, patch_size=, nsym=, subpatch=)` -- the same keywords
    `heliogram.codec.decode()` forwards to any `decoder=` callable -- so
    `decode(img, palette=P, subpatch=k, decoder=QwenVLDecoder(model=..., processor=...,
    palette=P, subpatch=k))` runs end to end: build a prompt describing the grid, call the
    model, parse its text response back into a symbol list (via
    `heliogram.dataset.target_to_symbols` -- the exact inverse of the compact target-string
    encoding `heliogram.dataset`/`scripts/train_qlora.py` use for ground truth/training
    targets), then feed that symbol list through `_payload_from_symbols` above (the same
    RS/framing layer `decode_pixels` uses -- no separate RS implementation for the model path).

    The constructor's `palette`/`subpatch`/`patch_size` are this decoder's fixed configuration
    (what the wrapped model was fine-tuned/prompted for); `__call__` validates that whatever
    `decode()` passes at call time matches, raising ValueError on a mismatch rather than
    silently using one value or the other. `nsym` is NOT required to match between construction
    and call time: it only affects the final RS-decode step (after transcription), not the
    prompt or parsing, so callers may freely vary it per call, exactly like `decode_pixels`.

    DATA HONESTY: calling an instance with `model=None` (the default) raises RuntimeError
    pointing at `scripts/train_qlora.py` -- there is no bundled model, and no code path here
    invents a result. See the module docstring for what is/isn't tested.
    """

    def __init__(
        self,
        model: object = None,
        processor: object = None,
        palette: int = 8,
        subpatch: int = 1,
        patch_size: int = PATCH_SIZE,
        nsym: int = 32,
        max_new_tokens: int = 2048,
        device: Optional[str] = None,
    ) -> None:
        bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
        _check_subpatch(patch_size, subpatch)
        self.model = model
        self.processor = processor
        self.palette = palette
        self.subpatch = subpatch
        self.patch_size = patch_size
        self.nsym = nsym
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _require_model(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "QwenVLDecoder has no fine-tuned model/processor loaded (model=None or "
                "processor=None). There is no model shipped in this repo -- fine-tune one with "
                "scripts/train_qlora.py (GPU required; see that script's top-of-file docstring "
                "for hardware expectations), then load the resulting checkpoint and pass it as "
                "model=/processor= here. To measure a STOCK (not fine-tuned) model instead, use "
                "zero_shot_symbol_error() with a real loaded model/processor."
            )

    def _build_prompt(self, width: int, height: int) -> str:
        """Instruction prompt asking the model to transcribe the data-cell grid. `width`/
        `height` are PATCH grid dimensions (as returned by `extract_symbols`), so the model is
        told exactly how many characters to produce. UNTESTED wording -- see module docstring."""
        n_data_cells = width * (height - 1) * self.subpatch * self.subpatch
        alphabet_prefix = SYMBOL_ALPHABET[: self.palette]
        subcell_note = (
            f" Each data patch is itself subdivided into a {self.subpatch}x{self.subpatch} "
            "grid of solid-color sub-cells (top-left, top-right, ... row-major); read those "
            "sub-cells in row-major order before moving to the next patch."
            if self.subpatch > 1
            else ""
        )
        return (
            "This image is a heliogram-encoded data grid. Row 0 (the top row) is a CALIBRATION "
            f"row cycling through a {self.palette}-color palette, in color-index order "
            f"0..{self.palette - 1}. Every row below it is DATA: each cell is one solid color "
            f"from that same {self.palette}-color palette." + subcell_note + " Read the data "
            "cells in row-major order (left to right, top to bottom) and classify each one's "
            "color against the calibration row's colors to get its color index (0.."
            f"{self.palette - 1}). Output exactly {n_data_cells} characters: for cell with color "
            f"index i, output the character at position i (0-indexed) of this exact string: "
            f'"{alphabet_prefix}". Wrap your answer in a single fenced code block ' + "(```"
            " ... ```) and output nothing else before or after it -- no explanation, no extra "
            "whitespace inside the block."
        )

    def _generate(self, img: Image.Image, prompt: str) -> str:
        """Run one VLM generation call. ALL torch/transformers imports are local to this method
        -- it is the only place in this module that can require those packages, and it is only
        ever reached after `_require_model` has confirmed a real model/processor were supplied.
        UNTESTED (see module docstring): this follows the documented HF Qwen2-VL/Qwen2.5-VL
        chat-template + processor + generate pattern, but has never been run against the real
        processor in this environment (no GPU here). `qwen_vl_utils.process_vision_info` may be
        worth using instead of the raw `images=[img]` call once this runs for real -- see
        requirements-gpu.txt.
        """
        import torch  # lazy: heavy GPU dep, see module docstring

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat_text], images=[img], return_tensors="pt")

        target_device = self.device or getattr(self.model, "device", None)
        if target_device is not None:
            inputs = {
                k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]

    def _parse_symbols(self, text: str) -> List[int]:
        """Parse the model's raw text response into a symbol list, reusing
        `heliogram.dataset.target_to_symbols`. Raises HeliogramDecodeError (not a bare ValueError)
        on any failure, consistent with `decode_pixels`' error contract."""
        candidate = _extract_symbol_string(text)
        try:
            return target_to_symbols(candidate, self.palette)
        except ValueError as exc:
            raise HeliogramDecodeError(
                f"could not parse the model's response into a valid symbol sequence: {exc}"
            ) from exc

    def __call__(
        self,
        img: Image.Image,
        palette: int = 8,
        patch_size: int = PATCH_SIZE,
        nsym: int = 32,
        subpatch: int = 1,
    ) -> bytes:
        self._require_model()
        if palette != self.palette or subpatch != self.subpatch or patch_size != self.patch_size:
            raise ValueError(
                f"QwenVLDecoder was constructed for palette={self.palette}, subpatch="
                f"{self.subpatch}, patch_size={self.patch_size}, but was called with "
                f"palette={palette}, subpatch={subpatch}, patch_size={patch_size}. Construct a "
                "decoder matching the image's actual encoding, or call decode()/decode_pixels "
                "with matching palette=/subpatch=/patch_size= arguments."
            )
        width = img.width // patch_size
        height = img.height // patch_size
        prompt = self._build_prompt(width, height)
        text = self._generate(img, prompt)
        symbols = self._parse_symbols(text)
        return _payload_from_symbols(symbols, palette=palette, nsym=nsym)


@dataclass
class ZeroShotResult:
    """One config's worth of `zero_shot_symbol_error` results, averaged over `trials`
    payloads. `raw_responses` keeps every raw model response for manual inspection -- there is
    no other record of what the stock model actually said."""

    palette: int
    subpatch: int
    patch_size: int
    payload_size: int
    trials: int
    symbol_error_rate: float
    decode_success_rate: float
    raw_responses: List[str] = field(default_factory=list)


def zero_shot_symbol_error(
    model: object,
    processor: object,
    configs: Sequence[Dict[str, object]],
    n_trials: int = 3,
    seed: int = 0,
    max_new_tokens: int = 2048,
) -> List[ZeroShotResult]:
    """Phase-2 "Step 8": run a STOCK (not fine-tuned) model directly over heliogram-encoded
    images and measure raw per-symbol transcription error against ground truth, the same way
    `heliogram.harness` measures `symbol_error_rate` for `decode_pixels` -- ground truth is read
    straight off the clean image at encode time via `extract_symbols` (exact by construction).

    DATA HONESTY (the entire point of this function): `model`/`processor` MUST be real, already
    -loaded objects (e.g. a stock `Qwen/Qwen2.5-VL-7B-Instruct` loaded via `transformers`,
    exactly as `scripts/train_qlora.py` would load it, just without the LoRA fine-tune step).
    Passing `model=None` or `processor=None` raises ValueError immediately -- there is no
    fallback path, default value, or cached result that could return an invented number. Every
    number in the returned `ZeroShotResult`s comes from an actual call to `model.generate(...)`
    (via `QwenVLDecoder._generate`, reused here so the exact same prompt/parsing code measured
    is the exact same code `QwenVLDecoder` uses for real decoding -- not a separate, untested
    measurement path). This repo's CPU pytest environment has no GPU, so this function itself is
    only ever unit-tested here with the `model=None`/`processor=None` guard, never with a real
    forward pass -- see tests/test_phase2_scaffold.py.

    `configs` is a sequence of dicts, each with at least a `"palette"` key (subset of
    `heliogram.codec.VALID_PALETTES`) and optionally `"subpatch"`, `"patch_size"`, `"nsym"`,
    `"payload_size"` (defaults: 1, `PATCH_SIZE`, 32, 48). Returns one `ZeroShotResult` per
    config, in the same order, each averaged over `n_trials` random payloads (seeded from
    `seed`, deterministic in the *inputs* generated -- not in the model's output, which is
    outside this function's control).
    """
    if model is None or processor is None:
        raise ValueError(
            "zero_shot_symbol_error requires a real, already-loaded model and processor (got "
            "model=None or processor=None) -- it never fabricates results. Load a stock "
            "Qwen2.5-VL-7B-Instruct (or similar) via transformers first; see "
            "scripts/train_qlora.py's docstring for the expected package versions."
        )

    results: List[ZeroShotResult] = []
    for cfg in configs:
        palette = int(cfg["palette"])  # type: ignore[arg-type]
        subpatch = int(cfg.get("subpatch", 1))  # type: ignore[arg-type]
        patch_size = int(cfg.get("patch_size", PATCH_SIZE))  # type: ignore[arg-type]
        nsym = int(cfg.get("nsym", 32))  # type: ignore[arg-type]
        payload_size = int(cfg.get("payload_size", 48))  # type: ignore[arg-type]

        decoder = QwenVLDecoder(
            model=model,
            processor=processor,
            palette=palette,
            subpatch=subpatch,
            patch_size=patch_size,
            nsym=nsym,
            max_new_tokens=max_new_tokens,
        )

        symbol_errors = 0
        symbol_total = 0
        successes = 0
        responses: List[str] = []
        for trial in range(n_trials):
            payload = random_payload(seed + trial, payload_size)
            img = encode(
                payload,
                palette=palette,
                patch_size=patch_size,
                nsym=nsym,
                seed=0,
                subpatch=subpatch,
            )
            _, _, truth = extract_symbols(
                img, palette=palette, patch_size=patch_size, subpatch=subpatch
            )

            width = img.width // patch_size
            height = img.height // patch_size
            prompt = decoder._build_prompt(width, height)
            text = decoder._generate(img, prompt)  # real forward pass -- see docstring
            responses.append(text)

            try:
                observed = decoder._parse_symbols(text)
            except HeliogramDecodeError:
                observed = []

            n = min(len(truth), len(observed))
            symbol_errors += sum(1 for i in range(n) if truth[i] != observed[i])
            symbol_errors += abs(len(truth) - len(observed))  # length mismatch counts as error
            symbol_total += len(truth)

            try:
                if _payload_from_symbols(observed, palette=palette, nsym=nsym) == payload:
                    successes += 1
            except HeliogramDecodeError:
                pass

        results.append(
            ZeroShotResult(
                palette=palette,
                subpatch=subpatch,
                patch_size=patch_size,
                payload_size=payload_size,
                trials=n_trials,
                symbol_error_rate=(symbol_errors / symbol_total) if symbol_total else 0.0,
                decode_success_rate=(successes / n_trials) if n_trials else 0.0,
                raw_responses=responses,
            )
        )
    return results
