# Frozen-encoder linear-probe report (Phase-2 Step 0)

Model (frozen vision tower): `Qwen/Qwen2.5-VL-3B-Instruct [arch=qwen2_5_vl probe-stage=pre_merger probe-head=linear]`. Probe: linear softmax readout over merged-token embeddings, oracle labels, train-statistics standardization only.

**Scope honesty:** a PASS here means the frozen embeddings linearly carry the symbols -- it de-risks, but does not replace, the fine-tune (the LM still has to learn to read them). A FAIL on clean images means no LINEARLY-DECODABLE per-patch signal survives to this tap point; a higher-capacity/nonlinear probe run at the same tap point could still differ (untested here -- see docs/FINDINGS.md Section 5), so treat a FAIL as strong, not absolute, negative evidence.

| palette | corruption | probe symbol error | train error | RS budget | chance | verdict |
|---|---|---|---|---|---|---|
| 16 | clean | 0.1344 | 0.0016 | 0.0627 | 0.9375 | above RS budget but far below chance -- partial signal; not decodable end-to-end at this operating point via a linear readout |
| 16 | jpeg_q70 | 0.1902 | 0.0120 | 0.0627 | 0.9375 | above RS budget but far below chance -- partial signal; not decodable end-to-end at this operating point via a linear readout |
