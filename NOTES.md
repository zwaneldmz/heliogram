# NOTES — prior art sweep (2026-07-10)

Survey of existing work relevant to heliogram's thesis: *measure and maximize the
data-channel capacity of ViT patches for arbitrary symbolic/binary payloads, so that
encoded images become a cheaper context medium for self-hosted VLMs.*

## 1. Optical / visual-text context compression (render language as pixels)

These projects render **natural-language text** into images and rely on the VLM's
OCR-like ability to read it back. They establish that vision tokens can carry more
text than text tokens, but the payload is always language glyphs.

- **DeepSeek-OCR: Contexts Optical Compression** (Wei, Sun, Li — Oct 2025).
  https://arxiv.org/abs/2510.18234
  The paper that popularized "optical compression." DeepEncoder + 3B MoE decoder.
  Reported ~97% OCR decoding precision at <10x text-token:vision-token compression,
  degrading to ~60% at 20x. This is the empirical anchor: one vision token can carry
  roughly 10 text tokens of *rendered prose* before accuracy collapses.
- **DeepSeek-OCR-2** (Jan 2026). Successor with DeepEncoder V2 / "Causal Visual Flow"
  (semantics-aware reordering of image segments instead of raster scan); 91.09% on
  OmniDocBench v1.5 at 256–1,120 vision tokens per page. Coverage:
  https://intuitionlabs.ai/articles/deepseek-ocr-optical-compression
- **Glyph: Scaling Context Windows via Visual-Text Compression** (Zhipu/THU-CoAI,
  Oct 2025). https://arxiv.org/abs/2510.17800 — code: https://github.com/thu-coai/Glyph
  Renders long text into images, continual-pretrains + RL-posttrains a VLM on it.
  3–4x token compression at accuracy comparable to a Qwen3-8B text baseline; a
  128K-context VLM stretched to ~1M-token text tasks under aggressive rendering.
  Notably uses an LLM-driven genetic search over *rendering configurations*
  (font, dpi, layout) — i.e., they optimize the encoder side, but only within the
  space of typeset human text.
- **Text or Pixels? It Takes Half** (Oct 2025). https://arxiv.org/abs/2510.18279
  Measures token efficiency of visual text input in multimodal LLMs; roughly half
  the tokens for the same text with existing off-the-shelf models.
- **LensVLM: Selective Context Expansion for Compressed Visual Representation of
  Text** (2026). https://arxiv.org/abs/2605.07019
  Inference framework: scan compressed (low-res) rendered pages, selectively
  re-expand only relevant regions via learned tools. Addresses the accuracy cliff
  when glyphs shrink below the encoder's effective resolution.
- **VTCBench: Can VLMs Understand Long Context with Vision-Text Compression?**
  (2025). https://arxiv.org/pdf/2512.15649 — benchmark for the paradigm; documents
  3x–576x claimed compression ratios across methods and where they break.
- **VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning** (2026).
  https://arxiv.org/pdf/2601.22069
- Curated list: https://github.com/bailynlove/Awesome-OCR-Vision-Based-Context-Compression

## 2. Capacity measurements of vision tokens

- **How Much Information Can a Vision Token Hold? A Scaling Law for Recognition
  Limits in VLMs** (Zhuang et al., Jan 2026). https://arxiv.org/abs/2602.02539
  The closest work to heliogram's question. Stress-tests VLMs by increasing
  *character count* per image and finds a three-regime phase transition (stable /
  instability / collapse), attributing instability to **spatial alignment
  sensitivity of ViT patch partitioning** and collapse to an information-capacity
  limit; fits a probabilistic scaling law for max characters per vision-token
  budget. Crucial differences from heliogram: (a) the payload is still rendered
  *text*, so measured capacity is confounded with OCR ability and font geometry;
  (b) it characterizes limits of an existing input distribution rather than
  *designing a code* to approach the channel capacity; (c) no ECC, no
  patch-aligned symbol layout, no model-free reference decoder.
- **Inference Optimal VLMs Need Only One Visual Token but Larger Models**
  (Nov 2024). https://arxiv.org/abs/2411.03312 — the opposite direction (how few
  tokens suffice for *semantic* tasks), useful as framing: semantic content needs
  few tokens, verbatim payloads need many; heliogram measures the verbatim side.
- **Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And More**
  (2025). https://arxiv.org/html/2502.03738 — patch-size scaling effects; relevant
  because heliogram's symbol size is pinned to the patch grid.

## 3. Pixel-native language models (tokenizer-free)

Precedent that ViT patches over rendered text are a viable substrate for language,
though again the alphabet is human glyphs:

- **PIXEL — Language Modelling with Pixels** (Rust et al., ICLR 2023).
  https://arxiv.org/abs/2207.06991 — renders text as fixed-size patches, ViT
  encoder, reconstructs masked patches; no vocabulary embedding at all.
- **CLIPPO: Image-and-Language Understanding from Pixels Only** (2022).
  https://arxiv.org/abs/2212.08045 — single shared encoder for images and rendered
  text; near-CLIP performance with half the parameters and no tokenizer.

## 4. Patch-aligned machine codes for VLMs

Searched for: "machine-readable visual code for VLMs", QR-decoding VLM experiments,
patch-aligned visual code schemes, learned visual codes as model input.
**No direct prior work found.** The nearest neighbors are:

- Classical 2D barcodes (QR, DataMatrix, Aztec) — designed for camera optics and
  arbitrary orientation, with finder patterns, quiet zones, and RS ECC. They are
  *not* aligned to a ViT patch grid, waste large area on localization structure a
  fixed preprocessing pipeline doesn't need, and there is no published evidence of
  VLMs decoding them natively (anecdotally they cannot, absent a tool call).
- The scaling-law paper above (2602.02539), which identifies patch-alignment
  sensitivity as a failure mode but does not exploit alignment as a design axis.
- Glyph's genetic search over rendering configs — optimizes the encoder within
  typeset text; heliogram removes the "typeset text" constraint entirely.

## The gap heliogram fills

Every system above uses **language rendered as glyphs** as the payload and a
model's OCR competence as the decoder. That conflates three things: the channel
(pixels -> ViT patches -> tokens), the code (fonts/layout evolved for human eyes),
and the decoder (learned OCR). Nobody has published:

1. A **patch-aligned symbolic code** — one solid-color symbol per 14x14 patch, a
   deterministic separable palette, a calibration row, and Reed–Solomon ECC —
   i.e., a code *designed for the channel* rather than inherited from typography.
2. A **model-free channel measurement**: encode arbitrary bytes, corrupt with the
   distortions real preprocessing applies (resize, JPEG, crop/pad), decode with a
   reference pixel decoder, and report **effective bits/patch after ECC overhead**.
   This isolates channel robustness from model ability, giving a clean upper bound
   before any fine-tuning.
3. The economic framing for **arbitrary binary/symbolic payloads**: base64 in a
   text context costs ~6 bits per text token; if one patch (~1 vision token in
   self-hosted VLMs where the operator controls preprocessing) reliably carries
   more than ~6 bits through the corruption suite, images are the cheaper context
   medium for raw data — independent of whether the payload is language.

Honest caveats the prior art imposes on us:

- 2602.02539's phase-transition result predicts that per-patch capacity claims
  must be validated under *misalignment* — our corruption suite (resize/crop)
  covers exactly that, and negative results at high palette sizes are expected
  and publishable.
- DeepSeek-OCR's 10x figure is about *semantic-adjacent OCR of prose*, not exact
  byte recovery; heliogram's numbers are not comparable to it and should never be
  marketed as "beating" it.
- Phase 1 (this repo) measures the channel with a pixel decoder. Whether a
  fine-tuned open VLM can actually realize that capacity through its encoder is
  Phase 2, gated on GPU access — no model numbers are claimed until then.
