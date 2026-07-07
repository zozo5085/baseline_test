# ReCLIP++ Baseline and Feature Fusion Research Notes

## 1. Current Research Position

This project currently uses ReCLIP++ as the baseline for CLIP-Unsupervised Semantic Segmentation, C-USS.

The current principles are:

- Preserve the original ReCLIP++ Bias Rectification pipeline.
- Avoid directly damaging CLIP's final semantic feature.
- First reproduce the author's baseline locally before adding new modules.
- Design improvements based on observed qualitative failure cases, rather than stacking plug-and-play modules blindly.

A reasonable research direction is:

> Under the ReCLIP++ bias-rectified C-USS framework, analyze how CLIP ViT intermediate and final-layer features behave around object boundaries, class confusion regions, and spatially inconsistent predictions, then design selective feature refinement that does not damage the CLIP semantic anchor.

## 2. Baseline Reproduction Status

The local VOC baseline has been reproduced.

Author reported result:

```text
ReCLIP++ VOC mIoU: 0.854
```

Local reproduction:

```text
ReCLIP++ VOC mIoU: 0.8451
```

Difference:

```text
0.854 - 0.8451 = 0.0089
```

This gap is acceptable. Possible reasons include:

- random seed
- PyTorch / CUDA / CLIP package version
- pseudo-label details
- checkpoint saving/loading details
- test run finishing with `1448/1449`
- floating point and interpolation differences

For later ablation tables, report both:

```text
Reported ReCLIP++: 85.4
Local ReCLIP++ baseline: 84.51
Proposed variants: compare mainly against the local baseline
```

## 3. File and Experiment Layout

Baseline reproduction:

```text
config/voc_train_baseline_reproduce_cfg.yaml
config/voc_test_baseline_reproduce_cfg.yaml
model/model.py
experiments/reproduce/voc_reclippp_baseline/
```

DFF2d feature fusion:

```text
config/voc_train_feature_fusion_cfg.yaml
config/voc_test_feature_fusion_cfg.yaml
model/model_feature_fusion.py
experiments/voc_dff2d_fusion/
```

Class hallucination gate ablation:

```text
config/voc_train_dff2d_classgate_cfg.yaml
config/voc_test_dff2d_classgate_cfg.yaml
experiments/voc_dff2d_classgate/
```

Visualization tools:

```text
tools/make_segmentation_summary.py
tools/dump_baseline_feature_maps.py
tools/dump_feature_maps.py
```

## 4. Important VOC Visualization Fix

The initial baseline visualization looked much worse than the author's qualitative results. The main cause was not a model error, but an incorrect visualization rule for VOC background.

VOC label definition:

```text
0: background
1..20: foreground classes
255: ignore
```

The ReCLIP++ VOC config uses:

```yaml
REDUCE_ZERO_LABEL: True
```

Therefore, the evaluator converts background into ignore and does not include background pixels in mIoU.

However, the model prediction has only 20 foreground classes and no background channel. A raw argmax assigns every background pixel to one foreground class. If the whole image is overlaid, the background looks very noisy.

For author-style VOC qualitative visualization, use:

```text
valid_visual_region = GT foreground
prediction[~valid_visual_region] = 255
```

This is visualization-only. It must not be used as an mIoU improvement.

The current `tools/make_segmentation_summary.py` supports:

```text
--pred_mask auto
```

For VOC with `REDUCE_ZERO_LABEL=True`, this automatically uses GT foreground masking for prediction visualization.

To inspect raw background flooding, use:

```text
--pred_mask all
```

## 5. Preliminary DFF2d Result and Observation

Current DFF2d config:

```yaml
FEATURE_FUSION:
  ENABLE: True
  LAYERS: [6, 9, 12]
  MODE: 'dff2d'
  INIT_GAMMA: 0.01
  PRESERVE_LOSS_WEIGHT: 0.01
```

This corresponds to:

```text
L6 + L9 + L12 + DFF2d
```

DFF2d does not directly replace L12. It is applied in a residual form:

```text
Fout = F12 + gamma * DFF2d(L6, L9, L12)
```

Design intent:

- L12 remains the CLIP semantic anchor.
- L9 and L6 provide intermediate/low-level spatial and boundary cues.
- gamma is initialized small to avoid damaging CLIP semantic features.
- preserve loss constrains the fused feature to stay close to L12.

Preliminary DFF2d test result:

```text
mIoU: 0.4151
```

This is much lower than the local baseline `0.8451`. It suggests that direct full-image L6/L9/L12 DFF2d fusion likely damages CLIP semantic features or injects low-level noise from L6/L9 into the final semantic space.

Therefore, blindly adding heavier fusion modules is not recommended at this stage.

## 6. Main Failure Patterns from Visualization

From baseline and previous ReCLIPPP2026 qualitative results, the major failure patterns are:

```text
1. Class confusion near object boundaries
2. Holes inside foreground objects
3. Broken thin structures
4. Background leakage into foreground
5. Fragmented noisy regions after intermediate-layer fusion
```

In the error map:

```text
green = prediction is correct
red = prediction is wrong
black = ignored background
```

Thus, red/green alternation around object contours indicates boundary misalignment or class confusion around fine structures.

## 7. Does the Author Baseline Have L6/L9/L12 Feature Maps?

The author baseline mainly uses the final ViT value feature for prediction:

```text
VisionTransformer -> L12 value feature -> projection -> output_q
```

The original baseline does not use L6/L9/L12 fusion.

However, we can still hook or capture intermediate features for observation:

```text
L6 PCA / norm / attention
L9 PCA / norm / attention
L12 PCA / norm / attention
```

This does not modify the baseline prediction.

Current command:

```powershell
python tools\dump_baseline_feature_maps.py `
  --cfg config\voc_test_baseline_reproduce_cfg.yaml `
  --checkpoint experiments\reproduce\voc_reclippp_baseline\best_weight.pth `
  --out_dir experiments\feature_maps_baseline_reproduce `
  --num 8
```

This tool uses a debug-capable wrapper and forces:

```text
fusion_mode = l12_only
```

Therefore it can observe baseline L6/L9/L12 features without changing the baseline output.

## 8. Feature Maps to Inspect Next

For object boundary class confusion, inspect:

```text
Image
GT
Prediction
Error map
GT boundary band
L6 PCA
L9 PCA
L12 PCA
L6/L9/L12 attention
class logit heatmap
bias logit heatmap
uncertainty map
```

Interpretation logic:

```text
If L6 has clear boundaries but L12 is blurry:
  Intermediate layers contain useful spatial detail.
  Boundary-aware selective fusion may help.

If L12 is semantically correct but L6/L9 introduce noise:
  Full-image fusion is harmful.
  A reliability gate is required.

If bias logits are too strong around boundaries:
  ReCLIP++ bias rectification may over-suppress boundary pixels.

If absent classes have high class-logit responses:
  Class presence gating or class purification is needed.
```

## 9. Recommended Next Method Direction

Avoid full-image fusion. Use:

```text
Boundary-aware selective fusion
```

Core concept:

```text
F12 = semantic anchor
F9, F6 = spatial detail
U = uncertainty map from entropy / top1-top2 margin
B = boundary map from feature gradient or prediction gradient
G = reliability gate from [U, B, feature cues]

Fout = F12 + gamma * G * Detail(F6, F9, F12)
```

This allows L6/L9 to refine only uncertain or boundary-adjacent regions, instead of damaging the full CLIP semantic representation.

## 10. Absent-Class Problem and Class Gate

CLIP can activate classes that are not actually present in the image because of image-text pretraining priors.

A class presence gate can be designed:

```text
P_img(c) = global image-text CLIP score
P_pix(c) = top-k pixel confidence
P_area(c) = high-confidence area ratio
P_margin(c) = top1-top2 class margin

class_gate(c) = reliability score
```

Application:

```text
logit_c = logit_c + log(class_gate(c))
```

This can reduce absent-class activation.

A preliminary `CLASS_GATE` ablation config has already been implemented, but it is not yet the main method.

## 11. Suggested Ablation Order

Stage 1: baseline and simple variants

```text
A. ReCLIP++ local baseline
B. model_feature_fusion + L12 only
C. L9 + L12 selective fusion
D. L6 + L12 selective fusion
E. L6 + L9 + L12 DFF2d full fusion
```

### Stage 1 Results (VOC val, mIoU)

| Variant | mIoU | SAVE_DIR | Date | Note |
|---|---|---|---|---|
| A. ReCLIP++ local baseline | 0.8451 | experiments/reproduce/voc_reclippp_baseline/ | 2026-07-05 | reproduction of paper baseline |
| B. L12 only | not run | — | — | — |
| C. L9 + L12 selective fusion (v1, VOID) | ~~0.4125~~ | experiments/voc_l9l12_selective/ | 2026-07-06 | VOID — trained/tested under the parity-bug module (see diagnosis below) |
| C. L9 + L12 selective fusion v2 (fixed module) | **0.6897** | experiments/voc_l9l12_selective_v2/ | 2026-07-07 | 50 epochs, gamma9=0.20 fixed; clean identity-verified module; queue log E01_test_l9l12_v2_20260707_035911.log. GENUINE negative result — see "Fusion line verdict" below. |
| D. L6 + L12 selective fusion | CANCELLED (recommended) | — | — | same fixed-gamma training design as C v2; expected to reproduce C's failure mode |
| E. L6 + L9 + L12 DFF2d full fusion | 0.4151 | — | 2026-07-05 | known failure (see §5) |

Per-class IoU for C (verbatim, 21 values, class-name mapping unverified):

```text
[0.26803517 0.61151217 0.         0.39510102 0.         0.80677777
 0.35201197 0.29875481 0.         0.86118226 0.44037268 0.83425966
 0.8705477  0.72464389 0.         0.         0.92131626 0.
 0.8659734  0.         0.        ]
avg: 0.4125 (1448/1449 images)
```

**Observation (2026-07-06):** C (0.4125) lands almost exactly on E's failure value
(0.4151), with the same signature (many exact-zero classes) — despite C being the
"selective" fix designed to avoid E's failure. This suggests the failure is not about
*how much* fusion, but something structural shared by both runs (e.g., test-time gamma
override vs learned gamma, fusion path clobbering the projected features, or a
train/test config mismatch). Diagnose before launching D — D would likely reproduce
the same failure. Hypotheses unverified as of this entry.

### Failure diagnosis of C (2026-07-07, in progress)

Diagnostic run 1 — same C checkpoint, `--fusion_gamma9 0.0` (fusion additive term
zeroed): **mIoU 0.4070**, same zero-class signature. Log:
`experiments/voc_l9l12_selective/test_gamma0_console.log`.
Conclusion: the additive fusion term is NOT the culprit; failure persists with zero
fusion content.

Code finding: with fusion enabled, `forward` replaces the raw ViT value features `v`
with `apply_safe_layer_fusion(...)` output (model_feature_fusion.py:544-545), which is
`normalize_feature_map(f12)` even at gamma=0 — channel-wise standardization + L2 norm
(model_feature_fusion.py:19-24) — before feeding the FROZEN CLIP-initialized
`self.proj` (model_feature_fusion.py:549). `proj` expects raw CLIP feature statistics;
destroying channel means/scales before it plausibly breaks text alignment for many
classes (the exact-zero-IoU signature). Prime suspect.

Diagnostic runs 2/3/4 (2026-07-07, all completed, full 1448/1449, baseline checkpoint
`experiments/reproduce/voc_reclippp_baseline/best_weight.pth` in every run):

| Run | Module | Fusion | mIoU | Log |
|---|---|---|---|---|
| A | model_feature_fusion | ENABLE=True, `l12_only` (normalize path, zero fusion content) | **0.2160** | experiments/diag_l12only_console.log |
| B | model_feature_fusion | ENABLE=False (raw `v` passthrough) | **0.2838** | experiments/diag_fusionoff_console.log |
| C | model.model (original) | n/a | **0.8451** (exact reproduction) | experiments/diag_baseline_modelmodel_console.log |

Readout:
- C proves the test pipeline (tools/test.py, data, env) did NOT drift.
- B proves `model_feature_fusion` is NOT behavior-equivalent to `model.model` even with
  fusion fully disabled — a parity bug in its non-fusion inference path. This alone
  accounts for 0.8451 → ~0.28.
- A (0.2160 < B 0.2838) shows the normalize-and-replace path adds further damage on top
  of the parity bug.
- Consequence: ALL model_feature_fusion results to date (DFF2d 0.4151, l9l12 selective
  0.4125) were measured through a broken module and say nothing about the fusion ideas
  themselves. Parity must be fixed and B re-run to 0.8451 before any fusion experiment
  is interpretable.

Root cause found and FIXED (2026-07-07): model_feature_fusion normalized the reference
prompt per-class (`prompt.norm(dim=-1, keepdim=True)`) while model.model uses a single
global Frobenius norm (`prompt.norm()`, model.py:473). The rescaled/rebalanced
`bias_logits` broke the checkpoint's logit calibration regardless of fusion flags.
Fixed at model_feature_fusion.py:570 to match model.model exactly. A line-level diff
(agent, 2026-07-07) found all other inference-path differences behavior-preserving.

Post-fix re-runs (baseline checkpoint, full 1448/1449):

| Run | Fusion | mIoU pre-fix | mIoU post-fix |
|---|---|---|---|
| B | ENABLE=False (parity check) | 0.2838 | **0.8451** (exact parity restored) |
| A | `l12_only` (normalize-and-replace f12, zero fusion content) | 0.2160 | **0.7393** |

Remaining finding from A: `normalize_feature_map` (channel standardization + L2) applied
to f12 before the frozen CLIP `proj` still costs ~0.106 mIoU on its own. Design
consequence for the fusion rework: never feed a re-normalized f12 to `proj`; compute
residuals (a9/a6) in scale-equalized space if needed, but add them to the RAW f12 with
no final re-normalization, so that gamma=0 is exactly the identity/baseline path.

Status: DFF2d 0.4151 and l9l12 selective 0.4125 were both measured (and the l9l12
checkpoint TRAINED) under the broken module — both numbers are void as evidence about
fusion. Fusion math reworked 2026-07-07 (gamma=0 = exact identity, verified: l12_only
on baseline ckpt = 0.8451 exactly); l9l12 retrain (v2 configs, SAVE_DIR
`experiments/voc_l9l12_selective_v2/`) pending user launch.

### Fusion line verdict (2026-07-07, closed)

Evidence chain on the FIXED module (gamma=0 = exact identity, parity verified):

| Experiment | mIoU | Meaning |
|---|---|---|
| C v2 in-training eval (best epoch) | ~0.8034 | looked healthy during training |
| C v2 formal test, gamma9=0.20 (E01) | 0.6897 | −0.1554 vs baseline; broad per-class degradation, not the old zero-class bug signature |
| C v2 checkpoint, test with gamma9=0 | 0.6773 | fusion term OFF is no better → damage lives in the TRAINED WEIGHTS (prompt/pe_proj/decoder co-adapted to fused features in a way that hurts the formal whole-image + rectification test pipeline) |
| baseline ckpt, TEST-TIME-ONLY fusion gamma9=0.05 | 0.8455 | +0.0004 — noise-level |
| baseline ckpt, test-time-only gamma9=0.10 | 0.8431 | −0.0020 |
| baseline ckpt, test-time-only gamma9=0.20 | 0.8378 | −0.0073, monotone decline |

Conclusion: the L9 residual carries no exploitable test-time signal (monotone decline
with gamma, break-even only at gamma→0), AND training with a fixed gamma actively
damages the learned parameters relative to the formal test pipeline (train-eval 0.80
vs test 0.69, while baseline goes the other way: 0.7966 train-eval → 0.8451 test).
The uncertainty-gated selective L9/L12 fusion line is CLOSED as a genuine, cleanly
measured negative result (usable as an ablation row in the paper). D (L6+L12) is
recommended cancelled — same design, no reason to expect a different outcome.
Remaining active improvement lines: test-time refinement (below, already +0.0087)
and CBR/IABR rectification redesign (research_notes section 15).

### Test-time refinement track (no retraining, E1 from RECOMMENDATIONS.md)

| Variant | mIoU | Delta vs baseline | Date | Evidence |
|---|---|---|---|---|
| baseline (model.model) | 0.8451 | — | 2026-07-07 re-verified | experiments/diag_baseline_modelmodel_console.log |
| + PG-CP-SFP + SP-DTLR legacy profile (`model.model_sfp_dtlr`) | **0.8538** | **+0.0087** | 2026-07-07 | experiments/voc_sfp_dtlr_eval/smoke_console.log (full 1448/1449; refinement executed once per image, 1449/1449); reproduced exactly after parameterization refactor (legacy_repro_console.log) |
| + SFP+DTLR generalization profile (TOP_FRACTION 0.75, SIGMA_S_REL 2.3, DTLR_BETA 1.0, PROXY_LAMBDA 1.0, STRUCTURE_CLASSES []) | **0.8520** | **+0.0069** | 2026-07-07 | experiments/voc_sfp_dtlr_gen_eval/gen_console.log (full 1448/1449) |

### IABR track (2026-07-07, closed — negative result)

IABR (image-adaptive bias rectification, ported from
othermodel_guide/model_author_reclippp_iabr.py into model/model_iabr.py; zero-init
identity verified at 0.8451 exactly before training):

| Stage | mIoU | Note |
|---|---|---|
| zero-init identity check (baseline ckpt) | 0.8451 | exact — port verified clean |
| 50-epoch training, in-training eval | peak ~0.7831 @ epoch ~4, then monotone decline to ~0.62 | textbook drift: adaptive branches leave identity and keep hurting |
| formal test of best_weight | **0.8001** | −0.0450 vs baseline; experiments/queue_logs/IABR_formal_test_*.log |

Verdict: under ReCLIP++'s unsupervised training signal, the input-adaptive scale
drifts away from identity and never recovers — even the best epoch tests 0.045 below
baseline. Combined with the fusion-line result, the pattern of this project is now
clear and paper-worthy: TRAINING-TIME additions (feature fusion, adaptive
rectification) drift under the unsupervised objective (train-eval looks fine, formal
test degrades), while TEST-TIME refinements are robust and transfer across
checkpoints. The author's own lack of recorded IABR numbers is consistent with this.

### Method direction A — image-dependent residual bias (diagnostic, 2026-07-07)

Premise: ReCLIP++ subtracts a static, image-INDEPENDENT bias
(bias_logits = pe_proj(pos_emb) @ prompt.T, identical for every image). Diagnostic
(tools/analyze_bias_residual.py, official ckpt, full 1449 val, decompose each class's
final per-pixel value into spatial-mean "DC" u_c and 95th-pct peak; GT used to label
present/absent — analysis only, method will be unsupervised). Report+figure:
docs/diagnostics/bias_residual_diagnostic.md, bias_residual.png; raw arrays
experiments/diag_official854_eval/bias_residual_raw.npz.

| measure | value | reading |
|---|---|---|
| (a) absent-class DC mean | 0.0050 (std 0.037) vs present 0.612 (0.360) | average residual small — PD filter already suppresses most hallucination |
| (b) absent-class DC across-image std | median 0.031; top: person 0.073, sofa 0.066, dining-table 0.054, boat 0.050, sheep 0.049 | residual is strongly IMAGE-DEPENDENT, concentrated in confusable classes — exactly what a static term cannot track |
| (c) peakedness gap | present 0.197 vs absent 0.006 | clean separator: present=peaked, bias=diffuse → gate the method on peakedness |
| (d) Pearson r(per-image absent-DC, per-image error) | 0.730 (n=1449) | strong — residual bias predicts which images are hard |
| (e) FP pixels from absent-elevated-DC classes | 7.15% | ceiling for a pure anti-hallucination method on VOC: modest |

Verdict (my read): premise SUPPORTED. (Correction to an earlier under-read: the 7.15%
"elevated-DC" figure is a narrow subset; the true anti-hallucination headroom is the
oracle below, which is LARGE.)

Intervention results (tools/analyze_bias_intervention.py, same test.py-exact eval,
baseline reproduces 0.8536171 ≡ 0.8536; report docs/diagnostics/):

| intervention | mIoU | vs 0.8536 | note |
|---|---|---|---|
| ORACLE: GT-suppress absent classes → -inf | **0.8998** | **+0.0462** | true ceiling of perfect anti-hallucination; the operator works at recall=1 |
| hard gate, present := gap_c > theta (best) | 0.5835 | −0.270 | catastrophic; 381 imgs lose all classes |
| hard gate, present := p95 prob > 0.3 (best) | 0.8535 | −0.0001 | ties; estimator recall 0.928 / prec 0.692 |

Diagnosis: the gating OPERATOR is settled (it is the oracle op, reaching 0.8998 at
recall=1). The bottleneck is estimator RECALL under a HARD gate — hard suppression
zeroes the IoU of every present class the estimator misses, and even 7% misses erase
the whole gain. p95 peak-height estimates presence far better than relative peakedness.

DERIVED METHOD STATEMENT (evidence-driven, not ad hoc): to capture the +0.0462 the
method must be a SOFT, calibrated, high-recall, image-conditioned class-presence gate
(soft so a missed present class is attenuated, not zeroed). Presence-signal options,
in increasing novelty/depth: (i) dense p95 peak — high recall but circular; (ii) the
repo's DISABLED class_gate = z_global @ text — a signal INDEPENDENT of the dense
prediction (soft log-domain); (iii) photometric-consistency presence (a present class
is detected stably under photometric jitter, a hallucination flickers) — unifies the
image-dependent-bias and photometric-instability failure modes into one method. All
training-free (no drift). VOC headroom +0.0462; larger expected on Context/ADE/COCO.

### Official checkpoint track (2026-07-07) — GOAL ACHIEVED

Upstream repo (dogehhh/ReCLIP) releases official checkpoints for ALL five datasets.
VOC ReCLIP++ checkpoint downloaded to `experiments/official/voc_reclippp_854/`.

| Variant (all on the OFFICIAL checkpoint) | mIoU | Evidence |
|---|---|---|
| official ckpt, plain eval (paper claims 85.4) | **0.8536** | experiments/queue_logs/DIAG_official854_*.log — our eval pipeline matches the authors' to 0.0004; the 0.8451 self-trained gap is training env (RTX 5090 + torch nightly) + seed |
| official ckpt + SFP+DTLR legacy profile | **0.8590** | experiments/queue_logs/SFPDTLR_official_legacy_*.log (+0.0054 over official) |
| official ckpt + SFP+DTLR generalization profile (no VOC hacks) | **0.8582** | experiments/queue_logs/SFPDTLR_official_gen_*.log (+0.0046) |

Both variants EXCEED the published 85.4, including the de-VOC-ified generalization
profile.

**Refiner comparison — PAMR (2026-07-07, fair, all on official ckpt 0.8536):**
The standard training-free refiner PAMR (Araslanov & Roth CVPR 2020, faithful
visinf/1-stage-wseg port, model/model_pamr.py; NUM_ITER=0 identity verified 0.8536 in
both token and full-res modes) was benchmarked at its designed setting AND at our hook.

| PAMR setting | mIoU | vs 0.8536 |
|---|---|---|
| token grid / 10 iter | 0.5843 | −0.269 |
| token grid / 3 iter | 0.7250 | −0.129 |
| token grid / 1 iter | 0.8128 | −0.041 |
| full image res / 10 iter (PAMR's designed setting) | 0.8361 | −0.018 |

Every PAMR configuration — including full-resolution where PAMR is designed to run —
scores BELOW the baseline; it uniformly hurts. SFP+DTLR (0.8590) is the only refiner
that improves the already-sharp ReCLIP++ rectified logits. PAMR's aggressive
appearance-diffusion bleeds across semantically-correct-but-appearance-different
regions; DTLR is gentler and confidence-guided (SFP purification precedes it). This
is a fair comparison (PAMR given its best resolution) and it strengthens the paper's
refiner claim. DenseCRF comparison pending (pydensecrf not installed; numpy-2
incompatible on this env — needs a decision before touching the environment). The refinement gain is checkpoint-independent (+0.0087/+0.0069 on our
0.8451; +0.0054/+0.0046 on the official 0.8536 — smaller headroom at higher base, as
expected). Official checkpoints for Context/ADE20K/Cityscapes/COCO-Stuff also exist
upstream — this removes the "per-dataset trained baselines" blocker for Stage 3
cross-dataset validation (remaining blockers: dataset images + text embeddings).

Parameterization (2026-07-07): all SFP/DTLR constants moved to a `MODEL.SFP_DTLR`
config block (config/configs.py:41-56; defaults = legacy verbatim values, so configs
without the block are behavior-identical — verified by exact 0.8538 reproduction).
Generalization profile removes every VOC-specific device flagged by the review
(fractional top-k, grid-relative sigma_s, no overshoot, no extrapolation, no VOC class
indices) and keeps +0.0069 over baseline on VOC — the gain does NOT depend on the
VOC-tuned hacks. Cross-dataset transfer still unproven until Stage 3 data exists.

Port notes: new module `model/model_sfp_dtlr.py` wraps model.model's RECLIPPP
(checkpoint loads with 0 missing / 0 unexpected keys; zero new parameters; model.model
untouched). Hyperparameters copied verbatim from
`othermodel_guide/model_1/model_lrab_v1_voc_final_862.py` (sfp_topk=800,
conf_thd=0.97, logit_beta=0.55, proxy_lambda=2.0, dtlr_beta=1.20, sigma_s=70,
sigma_r=1.5, structure_classes=(4,8,10)). The source's attribute-residual stage
(chair/diningtable hack, 862:1358-1615) is EXCLUDED — its 0.8564 reference number
included that stage.

Generalization review completed 2026-07-07 — full report:
`docs/othermodel_guide/sfp_dtlr_generalization_review.md`. Key findings:
- BLOCKER-level insight: at VOC eval the token grid is ~588 tokens < sfp_topk=800, so
  the top-k selection NEVER binds — SFP degenerates to "smooth every token with
  max-softmax < 0.97". Behavior flips qualitatively on datasets yielding > 800 tokens
  (e.g., Cityscapes 2048-wide). topk must become a token-count fraction.
- PG-CP-SFP verdict: transferable-with-parameterization (topk→fraction; conf_thd is
  class-count-sensitive — 171 COCO-Stuff classes would flag nearly all tokens;
  proxy_lambda=2.0 extrapolates past the proxy, should be ≤ 1; the 5x5 averaging
  erases thin structures unless boundary-gated).
- SP-DTLR verdict: transferable-with-parameterization, verging VOC-biased
  (sigma_s=70 ≈ global at a ~30-wide token grid, must be grid-relative;
  dtlr_beta=1.20 > 1 overshoots; structure_classes=(4,8,10) = hardcoded VOC
  bottle/chair/table indices — must be removed or made dataset-config).
- Porting caveat: in the source, SFP ran during TRAINING too (checkpoint co-adapted);
  our +0.0087 was measured as pure test-time add-on anyway, but sfp_logit_beta=0.55
  need not be optimal for our baseline — worth a small sweep.
- Clean list verified: divisions guarded, softmax dims correct, no NaN paths;
  the excluded attribute-residual stage is confirmed the most VOC-biased component.

Cross-dataset evaluation constraint (checked 2026-07-07): data/ holds only VOCdevkit
images; ade/cityscapes/coco/context have pseudo-label JSONs only — no images, and
text/ has only voc_ViT16_clip_text.pth. Stage 3 needs dataset downloads + per-dataset
text embeddings + per-dataset trained baselines before any transfer test can run.
Cheapest intermediate check: a parameterized (dataset-relative) SFP+DTLR variant that
keeps VOC ≥ 0.85, plus the review's suggested 3-image before/after visual dump once
any second dataset is available.

Stage 2: boundary confusion

```text
F. Boundary-aware DFF2d
G. Boundary-aware DFF2d + preserve loss
H. Boundary-aware DFF2d + class gate
I. Boundary-aware DFF2d + bias modulation
```

Stage 3: cross-dataset validation

```text
VOC
Cityscapes
Pascal Context
ADE20K
COCO Stuff
```

## 12. Suggested Evaluation Metrics

Besides mIoU, add:

```text
Boundary IoU
Boundary F-score
boundary-band error ratio
connected component count
false positive class count
foreground leakage ratio
class confusion matrix on boundary band
```

These metrics directly evaluate:

```text
red/green boundary alternation
boundary class confusion
object holes
broken thin structures
```

## 13. Short-Term Conclusions

Current conclusions:

```text
1. The local ReCLIP++ baseline is close to the author's reported VOC result.
2. The earlier VOC visualization mismatch mainly came from ignored-background rendering, not a major model error.
3. Direct L6/L9/L12 full DFF2d fusion significantly reduces mIoU.
4. Intermediate features should not be injected globally.
5. The next direction should be boundary-aware or reliability-aware selective fusion.
6. Feature-map analysis should first compare baseline L6/L9/L12 PCA with error maps to identify where boundary confusion begins.
```

Recommended next diagnostic tool:

```text
tools/diagnose_boundary_confusion.py
```

Suggested output:

```text
Image / GT / Prediction / Error
Boundary band
L6 PCA / L9 PCA / L12 PCA
uncertainty map
class logit heatmap
bias logit heatmap
```

This tool can directly support method design and paper-quality qualitative analysis.

## 14. Baseline Feature Map Observations (2026-07-06)

Source: `experiments/feature_maps_baseline_reproduce/` (8 VOC val samples, baseline weight, fusion_mode=l12_only).

Per-layer behavior observed from L6/L9/L12 PCA and CLS attention:

```text
L12: semantically grouped but spatially coarse. Jagged patch-level boundaries.
     Salt-and-pepper outlier patches (isolated off-color tokens) scattered in
     homogeneous regions (sky in 2007_000033/000061, grass in 2007_000175).
L9:  best object-level grouping with cleaner spatial coherence than L12
     (airplane 000033, sheep 000175, TV/tower 000187, faces 000323).
     Fewer outlier patches than L12.
L6:  boundaries follow true image edges best (sheep contour, plane outline),
     but semantics are weak and illumination/texture-driven
     (night train 2007_000123 dominated by lighting, not object identity).
CLS attention: L9/L12 attention is extremely sparse spike-like spots, often on
     background/corners (consistent with ViT artifact tokens), not object coverage.
     L6 attention spreads along object contours in some samples (000129, 000175).
```

Implications:

```text
1. L9 is the most valuable intermediate layer: semantic + spatial detail.
   -> Supports running C (L9+L12 selective) before D (L6+L12).
2. L6 is only useful near boundaries; global injection imports illumination bias.
   -> Explains DFF2d full-fusion failure (0.4151) and supports boundary-gated use of L6.
3. L12 outlier patches are a distinct error source; a selective gate should
   also suppress/repair these tokens, not only refine boundaries.
```

## 15. Proposed Next Method: Content-Conditioned Bias Rectification (CBR) (2026-07-06)

Grounding in ReCLIP++ (arXiv 2408.06747) and the local implementation:

```text
Paper bias model:      bias_logits = pe @ prompt.t()          (model/model.py:476)
Rectification:         output = output_q - bias_logits        (model/model.py:477)
pe = pe_proj(positional_embedding)                            (model/model.py:475)
Fusion hook is BEFORE rectification (model_feature_fusion.py:544-549);
class_gate modifies output_q, not the bias branch (model_feature_fusion.py:556-564).
```

Key observation: the paper's space-preference bias `pe` depends ONLY on position —
it is the SAME bias map for every image. But §14 feature-map evidence shows the
dominant remaining errors (boundary confusion, L12 artifact tokens) are
IMAGE-DEPENDENT spatial bias, which a position-only `pe` cannot express.

Proposal (stays inside the paper's rectification framework):

```text
bias_logits = (pe + alpha * h(v.detach())) @ prompt.t()

h     = lightweight projection (1-2 layer) from visual feature map to pe's space
alpha = learnable scalar, init 0  -> exact ReCLIP++ at initialization
v.detach() -> no gradient into CLIP features; semantic anchor untouched
```

Candidate directions considered:

```text
A. CBR (above): image-conditioned space bias in the bias branch.  [RECOMMENDED]
B. Boundary-aware selective fusion (Stage 2 F): feature branch, depends on
   Stage 1 C/D results.
C. Artifact-token suppression only: subsumed by A if h sees the outlier tokens.
```

Ablation position: CBR applies on top of the plain baseline first (isolate its
contribution), independent of the selective-fusion axis; combinable later (Stage 2 I).
