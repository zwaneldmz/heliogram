# heliogram instruments — data formats v0

Companion to [`format-v0.1.md`](format-v0.1.md) (the codec wire format). This file pins the two
*data* formats the gate-independent instruments expose to the outside world. Everything else the
instruments compute (saliency maps, fingerprints, detector TPR/FPR) is a plain in-memory
dataclass with no persistence contract, so it is not pinned here — only the formats a *stored*
or *submitted* artifact must obey are normative.

Scope note (same as the rest of Phase 1): every number these instruments produce comes from the
model-free reference decoder / channel measurements, **not** a VLM, unless a real model is passed
to a model-requiring entry point (which raises without one — it never fabricates). See the
README's "Instruments" section.

## 1. Injection-benchmark submission record (`heliogram.instruments.injection_bench`)

The behavioral-injection benchmark's **submission format** is one JSON object per result,
versioned by the module-level `RESULTS_FORMAT_VERSION` (currently `1`). `write_results()` emits
JSONL (one record per line); `read_results()` parses it back; `InjectionResult.to_record()` /
`from_record()` are the exact inverse pair.

Each record is:

```
{
  "version":        int,     # == RESULTS_FORMAT_VERSION at write time; from_record() raises on an unknown version
  "payload_name":   str,     # which BEHAVIORAL_PAYLOADS entry
  "category":       str,     # one of BEHAVIORAL_CATEGORIES (persona / schema / tool_call)
  "palette":        int,     # codec palette the payload was encoded at
  "subpatch":       int,
  "payload_size":   int,     # bytes
  "patch_size":     int,     # px
  "trials":         int,
  "influence_rate": float,   # fraction of trials the behavior fired -- ONLY ever produced by a real model run
  "note":           str
}
```

`version` is the compatibility gate: bump `RESULTS_FORMAT_VERSION` and `to_record()`/
`from_record()` together whenever the field set changes, so an old reader rejects a newer record
loudly (`from_record()` raises `ValueError`) instead of silently mis-parsing it. `influence_rate`
is never present in a record produced without a real model in the loop — `measure_behavioral_
capacity()` raises `ValueError` on `model=None`/`processor=None` rather than emitting a fabricated
rate.

## 2. Foreign-tile allow-list entry (`heliogram.instruments.foreign_tile`)

The pre-ingest guard trusts a sequence of `AllowListEntry`, each naming one codec configuration
the guard should treat as legitimate:

```
AllowListEntry(palette=8, patch_size=14, subpatch=1)
```

An image is **not** flagged foreign iff it either fails the cheap patch-structure filter (a
natural image — high within-patch color variance) or decodes cleanly under at least one
allow-list entry via `decode_pixels`. `nsym` is not an allow-list field (it is assumed shared,
defaulting to the codec's `nsym=32`, since `decode_pixels` needs it but `AllowListEntry` mirrors
only the fields that change the *pixels*). A patch-structured tile that no entry decodes is
foreign. This is a structural heuristic, not a learned classifier — see the module's
`DEFAULT_VAR_THRESHOLD` comment for the tunable ceiling and its upgrade path.
