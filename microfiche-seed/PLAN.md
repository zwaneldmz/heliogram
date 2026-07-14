# microfiche — build plan

Goal: a library + CLI that turns an agent transcript into a **handoff packet** (text
spine + rendered image tier + TOC), and the evidence that this beats summarization at
equal token budgets.

Guiding principle, inherited from heliogram: **measure the economic claim before building
on it.** Every phase below has a deliverable and a kill criterion.

---

## Phase 0 — Go/no-go: recovery-at-budget experiment

The one number that matters: *at a fixed token budget B for "old context", how much of the
discarded region can the agent still use?* Compression ratio is vanity; recovery is the
product.

**Build (minimal, throwaway-quality allowed):**
- `render_pages(text, cfg) -> list[PNG]` — deterministic text→image renderer
  (monospace font, fixed font-size/dpi/page-geometry in `cfg`).
- `vision_cost(png, model) -> int` — vision-token cost per target model
  (Qwen2.5-VL's 28×28-per-token rule; Claude's ~(w·h)/750; make the model table pluggable).
- Eval harness over transcripts (synthetic + a few real agent sessions):
  - **Needle QA**: factoid questions whose answers live only in the discarded region.
  - **Exact-string recall**: file paths, identifiers, hashes, error messages — scored
    byte-exact, not fuzzy.
  - **Task resumption**: give the agent a follow-up task requiring old context; score
    completion.

**Conditions, all at equal budget B:**
1. Raw text, truncated to B (baseline floor).
2. LLM summary of the old region, ≤ B tokens (the incumbent).
3. Rendered pages of the old region, vision cost ≤ B.
4. Hybrid: text spine (~20% of B) + rendered pages (~80% of B).

**Deliverables:** `RESULTS.md` with recovery-vs-budget curves per condition and model;
`results.csv`.

**Kill criterion:** if (3) and (4) don't beat (2) on needle QA at 2–4× effective
compression on at least one strong VLM, stop — publish the negative result and the
harness, heliogram-style.

## Phase 1 — Renderer + token accounting (production-quality)

- Deterministic, seedable renderer with a versioned config (font, size, weight, dpi,
  margins, line wrap, syntax-neutral code layout). Rendering config travels **inside**
  the packet so pages are reproducible.
- **Patch alignment**: page dimensions snapped so the target model's preprocessor
  (`smart_resize` etc.) is a no-op — no resampling between us and the vision tower.
  (Direct lesson from heliogram's corruption suite: resampling is where information dies.)
- Resolution sweep: chars-per-vision-token vs. transcription accuracy; pick default
  operating points per model ("safe" and "dense").
- Content-aware handling: prose vs. code vs. tables get different layout presets;
  exact-string-risky content is flagged for the spine instead of the image tier.

**Deliverable:** `microfiche.render` module + calibration report per supported model.

## Phase 2 — Packet format + tiered memory

- **Packet spec (versioned, on-disk = a directory):** `spine.md`, `pages/*.png`,
  `toc.json` (page → turn range, topics, hash of source text), `manifest.json`
  (render config, model targets, source transcript hash).
- **Splitter**: rules + small-model pass that routes content to spine vs. image tier
  (decisions, paths, IDs, open TODOs → spine; discussion, tool output, exploration → pages).
- **Tiering policy**: recent turns = text; older = dense pages; oldest = low-res pages or
  summary. Policy is a pure function of (transcript, budget) — testable.
- **Promotion API**: `promote(packet, page_id) -> text` — transcribe a page back to text
  (via the VLM itself), verified against `toc.json` source hashes where available.

**Deliverable:** `microfiche.packet` + round-trip property tests (pack → read → promote
→ compare to source).

## Phase 3 — Integration + CLI

- CLI: `microfiche pack transcript.jsonl --budget 20000 --model qwen2.5-vl`,
  `microfiche read packet/ --question "..."`, `microfiche promote packet/ --page 7`.
- Compaction-hook adapter for at least one real harness (Claude Agent SDK / Claude Code
  PreCompact hook): on compaction, emit a packet instead of (or alongside) the summary.
- Orchestrator pattern: subagent briefing packets (parent packs, child reads).

**Deliverable:** end-to-end demo — a long session survives compaction and answers
needle questions a summary-only baseline fails.

## Phase 4 — Robustness + multi-model matrix

- Model matrix: Qwen2.5-VL (self-hosted), Claude, at least one more API VLM; publish the
  recovery/cost table per model.
- Failure-mode tests: harnesses that strip images, providers that recompress (JPEG) or
  resize uploads, prompt-cache behavior with image blocks.
- Adversarial/safety note: rendered context is model-legible but bypasses text-level
  filters; document the injection surface and mitigation (run safety checks on rendered
  text *before* packing, treat promoted text as untrusted).

**Deliverable:** `THREATMODEL.md` + robustness section in RESULTS.

---

## Non-goals

- Arbitrary-byte payloads through vision tokens (heliogram measured this: no).
- Steganography or filter evasion as a feature.
- Perfect fidelity in the image tier — that's what the spine and promotion are for.

## Risks (ranked)

1. **Reasoning-over-images degradation** — the model may *transcribe* pages well but
   *use* them poorly. Phase 0's task-resumption condition exists to catch this early.
2. **Exact-string corruption** — `1`/`l`, `0`/`O` in identifiers. Mitigated by the
   spine split + byte-exact scoring in Phase 0.
3. **Harness plumbing** — image blocks stripped or recompressed in real agent stacks.
   Phase 4, but check the target harness's behavior during Phase 3.
4. **Economics drift** — provider pricing/tokenization of images changes the 4× premise.
   Token accounting is a pluggable table, not a constant.

## Repo skeleton (Phase 0 end-state)

```
microfiche/
  microfiche/            # package: render.py, cost.py, packet.py (stub), eval/
  tests/
  spec/                  # packet format + renderer config specs
  RESULTS.md  results.csv
  README.md  PLAN.md  LICENSE (Apache-2.0)
```
