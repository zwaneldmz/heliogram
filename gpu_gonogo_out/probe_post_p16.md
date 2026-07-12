# Frozen-encoder linear-probe report (Phase-2 Step 0)

Model (frozen vision tower): `Qwen/Qwen2.5-VL-3B-Instruct [arch=qwen2_5_vl probe-stage=merged probe-head=linear]`. Probe: linear softmax readout over merged-token embeddings, oracle labels, train-statistics standardization only.

**Scope honesty:** a PASS here means the frozen embeddings linearly carry the symbols -- it de-risks, but does not replace, the fine-tune (the LM still has to learn to read them). A FAIL on clean images means no LINEARLY-DECODABLE per-patch signal survives to this tap point; a higher-capacity/nonlinear probe run at the same tap point could still differ (untested here -- see docs/FINDINGS.md Section 5), so treat a FAIL as strong, not absolute, negative evidence.

| palette | corruption | probe symbol error | train error | RS budget | chance | verdict |
|---|---|---|---|---|---|---|
| 16 | clean | 0.7358 | 0.2332 | 0.0627 | 0.9375 | at/near chance -- no LINEARLY-DECODABLE per-patch signal survives to this tap point (if this happens on CLEAN images, check the token-order assumption first, then treat the LM-token branch as unsupported by this probe for this tower; a higher-capacity/nonlinear probe is untested and could differ) |
