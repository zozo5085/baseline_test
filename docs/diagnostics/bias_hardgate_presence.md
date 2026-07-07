# Hard Presence-Gating with Unsupervised Presence Estimators

Mirrors the ORACLE operator (hard-suppress non-present classes to -inf, exactly as
the GT oracle does) but replaces the GT present-set with an UNSUPERVISED per-image
estimate. GT is used ONLY to score precision/recall of the estimator, never inside
the method. Scored with the exact tools/test.py mIoU pipeline (post-softmax tensor,
same PD/interp/argmax/accumulator). VOC val, 1449 images.

Reference points: baseline 0.8536 (== official 0.8536), GT-oracle ceiling
0.8998 (gap to close = +0.0462).

## 1. Peakedness hard-gate: present := gap_c > theta

| threshold | mIoU | vs 0.8536 | vs oracle 0.8998 |
|---|---|---|---|
| theta=0.01 | 0.5835 | -0.2701 | -0.3164 |  **<== BEST**
| theta=0.02 | 0.5835 | -0.2701 | -0.3164 |
| theta=0.04 | 0.5835 | -0.2701 | -0.3164 |
| theta=0.06 | 0.5835 | -0.2701 | -0.3164 |
| theta=0.08 | 0.5835 | -0.2701 | -0.3164 |
| theta=0.12 | 0.5793 | -0.2743 | -0.3205 |

## 2. Peak-prob hard-gate: present := p95_c > phi

| threshold | mIoU | vs 0.8536 | vs oracle 0.8998 |
|---|---|---|---|
| phi=0.3 | 0.8535 | -0.0001 | -0.0464 |  **<== BEST**
| phi=0.5 | 0.8517 | -0.0019 | -0.0481 |
| phi=0.7 | 0.8312 | -0.0224 | -0.0686 |

## 3. Oracle-gap captured by best cell

(mIoU_method - 0.8536) / (0.8998 - 0.8536):
- gap-gate best (theta=0.01): mIoU 0.5835 -> -584.2% of oracle gap
- peak-gate best (phi=0.3): mIoU 0.8535 -> -0.3% of oracle gap

## 4. Estimator quality vs GT present set (per-image, averaged)

| estimator (best cell) | precision | recall |
|---|---|---|
| gap_c > 0.01 | 0.5921 | 0.6885 |
| p95_c > 0.3 | 0.6919 | 0.9276 |

(precision averaged over 1068/1447 images with >=1 estimated class; recall over
1449 images with >=1 GT class.)

## Honest read

Hard peakedness-gating does NOT clear 0.8536
(best gap-gate 0.5835, -584.2% of the
oracle gap). The best presence estimator has precision 0.692 /
recall 0.928 vs the GT present set. The
bottleneck is the ESTIMATOR (imperfect present-set recovery caps the gain well below the oracle): the hard-gate operator is
correct (it IS the oracle operator), so the ceiling this probe reaches is set by how
well peakedness/peak recovers the true present classes.
