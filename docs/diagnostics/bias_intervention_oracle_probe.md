# Bias-Intervention Measurements: Oracle Ceiling and Training-Free Gated-DC Probe

Both scored with the EXACT tools/test.py mIoU pipeline (same PD filter with
cfg.TEST.PD, same double bilinear interpolation to original resolution, same final
F.softmax, same argmax, and the same intersect/union accumulator as
utils/test_mIoU: num_classes=C+1=21, ignore_index=255, reduce_zero_label on
GT, IoU=intersect/union with nan->0, avg=sum(IoU)/C=20). Directly comparable to
the official 0.8536. Interventions operate on the post-softmax probability tensor
(the exact tensor test.py argmaxes at line 203).

VOC2012 val, 1449 images.

## Baseline sanity check

Reproduced baseline mIoU (no intervention) = **0.8536** (official reference 0.8536).

## 1. Oracle ceiling (USES GT -- upper bound, NOT a method)

Per image, set the final probability of every class ABSENT from that image's GT to
-inf (perfect hallucination suppression), keep present classes, argmax + mIoU.

Oracle mIoU = **0.8998**  (vs 0.8536; ceiling gain = +0.0462).

This is the theoretical maximum any perfect anti-hallucination method could reach on
VOC with this backbone/checkpoint (it only removes false positives from absent
classes; it cannot fix errors among present classes).

## 2. Training-free gated-DC probe (NO GT -- candidate method)

Per image, per class c: u_c = spatial-mean over ALL pixels of the final prob map
(GT-free "DC"); gap_c = p95_c - u_c (peakedness). If gap_c < tau (diffuse => likely
bias) subtract lambda * u_c from that class's whole map; peaked classes untouched.
argmax + mIoU.

| threshold | strength | mIoU | |
|---|---|---|---|
| tau=0.02 | lambda=0.5 | 0.8536 |  **<== BEST**
| tau=0.02 | lambda=1.0 | 0.7342 |
| tau=0.05 | lambda=0.5 | 0.8536 |
| tau=0.05 | lambda=1.0 | 0.7342 |
| tau=0.1 | lambda=0.5 | 0.8536 |
| tau=0.1 | lambda=1.0 | 0.7334 |

Best cell: **tau=0.02, lambda=0.5** -> mIoU **0.8536** (vs baseline 0.8536, delta +0.0000; vs official 0.8536, delta +0.0000).

Note: the "best" cell is the smallest-lambda cell, which is numerically INERT
(u_c is prob-scale and tiny, so lambda=0.5 flips essentially no argmax decisions ->
it reproduces baseline exactly). Larger lambda=1.0 does flip decisions but HURTS
(gap-based gating also catches some genuine low-peak present classes, and the
DC-subtraction magnitude is miscalibrated), dropping mIoU to ~0.73.

## Honest read

Baseline reproduces at 0.853617 == official 0.8536 (comparability
confirmed). Oracle ceiling is 0.8998, so the total headroom a perfect
anti-hallucination method could claim is +0.0462 mIoU over
this baseline. The best training-free gated-DC probe cell reaches 0.8536, which
does NOT beat the baseline (delta
+0.0000) and captures
0.0%
of the baseline->oracle gap. Naive per-image gated-DC removal on the final
probability tensor is NOT a sufficient
training-free step -- at best it is inert, at worst it hurts. This is a legitimate
negative result: the sizeable +0.0462 baseline->oracle gap is
real and worth pursuing, but it demands a smarter image-dependent correction than
crude DC subtraction (the paper direction), not a training-free heuristic on the
post-hoc probabilities. (Space choice: interventions are post-softmax on the exact
tensor test.py argmaxes; logit-space DC is not viable here because test.py's PD
filter sets suppressed-class logits to -100, so a logit-space mean-DC subtraction
would resurrect PD-killed classes.)
