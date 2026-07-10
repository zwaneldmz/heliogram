"""heliogram.encoder -- Phase-2 frozen-encoder pixel-gradient plug point (GPU work, no GPU here).

WHY THIS MODULE EXISTS (handoff M4/A12): expose a frozen vision encoder WITH GRADIENTS ENABLED
ON THE INPUT PIXELS -- the plug point a future tile-pixel optimizer would attach to, to optimize
actual tile pixel values directly against a real ViT/VLM's own embedding space (backprop through
frozen weights straight onto the pixels), rather than against a discrete, hand-searched palette.
`heliogram.instruments.learned_alphabet` (CPU, runs here) answers a related but DIFFERENT and
much narrower question: it optimizes the codec's PALETTE COLORS against `heliogram.codec.
decode_pixels`/`extract_symbols` -- the model-free reference pixel classifier -- via a seeded
coordinate-descent search with no notion of gradients or embeddings at all. `FrozenEncoderHandle`
below is the OTHER end of that same idea: were a real GPU and a real, already-loaded, FROZEN
ViT/VLM vision tower on hand, this class is where per-pixel gradients from a loss defined on that
model's own embeddings would flow back to the actual tile pixels, enabling first-order (gradient)
pixel optimization instead of `learned_alphabet`'s zeroth-order (hill-climb) color search. See
that module's own DATA HONESTY section for exactly where the line between the two sits.

DATA HONESTY (read this first, mirrors heliogram/vlm.py's own section): nothing in this module
has been run against a real model in this repository -- there is no GPU here.
`FrozenEncoderHandle.encode_pixels` and `.embed_with_pixel_grad` both REQUIRE a real,
already-loaded `model` (e.g. a frozen ViT/VLM vision tower loaded via `transformers`); if `model`
is None (the default) they raise RuntimeError immediately via `_require_model` -- exactly
`heliogram.vlm.QwenVLDecoder._require_model`'s pattern -- rather than fabricating an embedding, a
gradient, or any other number. Every torch import in this file is local to the one method that
actually needs it (`_encode` below); this module's own top-level imports are exactly as light as
`heliogram.codec`'s (pillow/numpy + stdlib -- no reedsolo needed here, there is no RS/framing
involved in a raw pixel<->embedding plug point), so `import heliogram.encoder` never requires
torch/transformers/peft/bitsandbytes.

What's actually implemented vs. what's untested (same structure as heliogram/vlm.py's module
docstring):

- The `_require_model` guard rail and the plain-numpy/PIL pixel-batch normalization
  (`_to_pixel_tensor`'s shape/dtype handling before a torch tensor even exists) -- ordinary,
  testable-without-a-model Python (see tests/test_learned_alphabet.py's FrozenEncoderHandle
  checks).
- The actual `self.model(pixel_tensor)` call and which attribute the embeddings come out under
  (`_encode` takes `.last_hidden_state` off the result if present, else the raw return value) --
  this varies by which real vision tower ends up loaded (a stock CLIP/SigLIP ViT and Qwen2.5-VL's
  own ViT do not agree on this), so it is a documented, reasonable-attempt STARTING POINT to
  adjust once a real GPU and a real loaded encoder are on hand, not a verified integration --
  exactly the same caveat `heliogram/vlm.py`'s `_generate` carries for its own model call.
- Whether gradients computed this way are actually well-conditioned for pixel-space optimization
  (versus, say, needing a smoothed/regularized objective, a learned prior, or per-channel
  normalization statistics matched to the real encoder's own training data) is an open Phase-2
  research question this module does not and cannot answer without a real GPU run.

DESIGN NOTE on why `model(pixel_tensor)` is called directly instead of going through a
caller-supplied `processor` (accepted by the constructor for interface parity with
`QwenVLDecoder`, and in case a real model's `forward` needs processor-derived extra kwargs, but
otherwise UNUSED by `_to_pixel_tensor`): a typical Hugging Face image processor's resize/
normalize step runs as plain PIL/numpy code with no autograd tracing at all, so composing it
before pixel gradients are needed would sever the gradient path back to the raw per-pixel values
-- exactly the thing this module exists to preserve. `_to_pixel_tensor` is deliberately the LAST
non-differentiable step: any resize/crop must happen before it (on the plain PIL/numpy tile);
everything after it (dtype cast, [0, 1] scaling, moving to `device`, and any per-channel
mean/std normalize a caller composes before the frozen model call) should stay pure torch ops on
the tensor it returns, so gradients can flow all the way back to those raw pixel values.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

__all__ = ["FrozenEncoderHandle"]


def _to_pixel_tensor(tile_batch: object, torch_module: object) -> object:
    """Convert `tile_batch` into a single float32 NCHW torch tensor scaled to [0, 1].

    Accepts a single `PIL.Image`, a single `(H, W, 3)` uint8 array, a sequence of either, or an
    already-stacked `(N, H, W, 3)` uint8 array -- the same "one tile or a batch of tiles" shapes
    `heliogram.codec.encode`'s own `PATCH_SIZE x PATCH_SIZE x 3` patches naturally come in.
    Pixel values are assumed to already be uint8 in `[0, 255]` (e.g. straight off a
    heliogram-encoded `PIL.Image`, or a candidate tile from
    `heliogram.instruments.learned_alphabet`) -- this function does not try to guess a scale from
    the data, it always divides by 255.0.

    This is the LAST plain-numpy (non-autograd) step before a torch tensor exists -- see the
    module docstring's DESIGN NOTE for why everything downstream of this function must stay pure
    torch ops for pixel gradients to mean anything.
    """
    if isinstance(tile_batch, Image.Image):
        arr = np.asarray(tile_batch.convert("RGB"), dtype=np.uint8)[None, ...]
    elif isinstance(tile_batch, (list, tuple)):
        frames = [
            np.asarray(t.convert("RGB"), dtype=np.uint8)
            if isinstance(t, Image.Image)
            else np.asarray(t, dtype=np.uint8)
            for t in tile_batch
        ]
        arr = np.stack(frames, axis=0) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    else:
        arr = np.asarray(tile_batch, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[None, ...]

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            "tile_batch must resolve to an (N, H, W, 3) pixel array (a single (H, W, 3) tile, a "
            f"PIL.Image, or a sequence of either are all also accepted) -- got array shape "
            f"{arr.shape!r}"
        )

    nchw = np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2))).astype(np.float32) / 255.0
    return torch_module.from_numpy(nchw)


class FrozenEncoderHandle:
    """Phase-2 plug point (handoff M4/A12): a frozen vision encoder with gradients enabled on
    the INPUT PIXELS, for optimizing actual tile pixels (not just a discrete palette's colors)
    directly against a real ViT/VLM vision tower's own embedding space.

    Constructed with a real, already-loaded, FROZEN `model` (its own weights are never updated
    here -- only the input pixels are meant to move) and, optionally, a `processor` (accepted
    for interface parity with `heliogram.vlm.QwenVLDecoder`'s model/processor contract; NOT
    currently used by the pixel-conversion path itself -- see the module docstring's DESIGN
    NOTE for why) and a `device` string/object to move tensors to before the forward pass.

    DATA HONESTY: every method that would need to actually run the model raises RuntimeError via
    `_require_model` when `model is None` (the default) -- there is no model shipped in this
    repo, and no GPU in this environment to load one. See the module docstring for what is/isn't
    implemented and tested here.
    """

    def __init__(
        self,
        model: object = None,
        processor: object = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "FrozenEncoderHandle has no frozen encoder loaded (model=None). There is no "
                "model shipped in this repo, and no GPU in this environment to load one -- load "
                "a real, already-frozen ViT/VLM vision tower yourself (e.g. a stock CLIP/SigLIP "
                "image encoder, or the vision tower of Qwen/Qwen2.5-VL-7B-Instruct loaded via "
                "transformers, mirroring heliogram.vlm.QwenVLDecoder's model/processor "
                "contract) and pass it as model= (and, if your model's forward needs it, "
                "processor=) here. To optimize palette colors WITHOUT a real model, on CPU, see "
                "heliogram.instruments.learned_alphabet instead -- it answers a related but "
                "narrower question (a discrete color search against decode_pixels/"
                "extract_symbols, not a gradient against a real vision encoder's embeddings); "
                "see this module's own docstring for exactly how the two relate."
            )

    def encode_pixels(
        self, tile_batch: object, requires_grad: bool = False
    ) -> Tuple[object, object]:
        """Run `tile_batch` through the frozen encoder and return `(embeddings, pixel_tensor)`.

        `pixel_tensor` is the actual torch tensor the frozen model was called with -- a LEAF
        tensor (`pixel_tensor.requires_grad == requires_grad`, `pixel_tensor.grad_fn is None`),
        so it IS "a handle for backprop to pixels": with `requires_grad=True`, call `.backward()`
        on any scalar loss computed from `embeddings` and then read `pixel_tensor.grad` for the
        gradient of that loss with respect to every input pixel. With the default
        `requires_grad=False` (a plain forward pass, no autograd bookkeeping), `pixel_tensor.grad`
        will always be `None` after any `.backward()` call elsewhere.

        Raises RuntimeError via `_require_model` if `model` is None -- see the module docstring
        and `_require_model`'s own message for what to do instead. Beyond that guard rail, this
        method is UNTESTED against a real model in this repository (no GPU here) -- see the
        module docstring's "what's actually implemented vs. untested" section.
        """
        self._require_model()
        return self._encode(tile_batch, requires_grad=requires_grad)

    def embed_with_pixel_grad(self, tile_batch: object) -> Tuple[object, object]:
        """`encode_pixels(tile_batch, requires_grad=True)` -- the actual pixel-gradient plug
        point this module exists for (handoff M4/A12). See `encode_pixels`'s docstring for the
        returned `(embeddings, pixel_tensor)` contract, and the module docstring for what is (and
        is not) implemented/tested here."""
        return self.encode_pixels(tile_batch, requires_grad=True)

    def _encode(self, tile_batch: object, requires_grad: bool) -> Tuple[object, object]:
        """ALL torch imports are local to this method -- it is the only place in this module that
        can require torch, and it is only ever reached after `_require_model` has confirmed a
        real model was supplied.

        UNTESTED (see module docstring): calls `self.model(pixel_tensor)` and takes
        `.last_hidden_state` off the result if present, else uses the raw return value -- a
        reasonable-attempt default that varies by which real vision tower ends up loaded, not a
        verified integration.
        """
        import torch  # lazy: heavy GPU dep, see module docstring

        pixels = _to_pixel_tensor(tile_batch, torch)
        if self.device is not None:
            pixels = pixels.to(self.device)
        # Detach + re-wrap as a fresh leaf so `.requires_grad_` below always creates a genuine
        # leaf tensor (grad_fn is None) regardless of whatever `.to(device)` did internally --
        # the "handle for backprop to pixels" this method returns must be a leaf, or
        # `pixel_tensor.grad` would never populate no matter what `.backward()` is called on.
        pixels = pixels.detach().clone()
        if requires_grad:
            pixels.requires_grad_(True)

        with torch.set_grad_enabled(requires_grad):
            raw_output = self.model(pixels)
        embeddings = getattr(raw_output, "last_hidden_state", raw_output)
        return embeddings, pixels
