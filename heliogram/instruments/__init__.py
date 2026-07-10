"""heliogram.instruments -- durable, gate-independent measurement/guard tooling for heliogram.

heliogram.harness's capacity sweep measures the codec/channel's raw throughput under corruption
and is scoped to this project's Phase-2 Decision Gate (see the README's "Roadmap / Phase-2
boundary" section) -- its relevance depends on which side of that gate the project lands on.
The modules under this package are different on purpose: they measure or guard things whose
value does NOT depend on that gate. A pre-ingest guard that flags a heliogram-like payload
outside a trusted allow-list, for instance, is useful the moment ANY heliogram-encoded image
might reach a model's input -- whether the codec ships as-is, gets a fine-tuned VLM reader, or
never leaves Phase 1 -- clean images, corrupted images, and a future learned-alphabet variant
alike. Durable tooling first, capability work second: see each module's own docstring for the
specific instrument and which handoff item it answers.

This `__init__.py` is deliberately minimal: no re-exports, no imports of any kind beyond this
docstring. Importing `heliogram.instruments` by itself does nothing but register the package, so
it (and every individual module underneath it) stays exactly as import-light as heliogram.codec
-- pillow/numpy/reedsolo + stdlib only, never torch/transformers/peft/bitsandbytes at module
scope. Import the specific module you need, e.g. `from heliogram.instruments.foreign_tile import
guard`, rather than expecting this file to re-export it.
"""
