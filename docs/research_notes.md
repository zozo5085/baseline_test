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
| A. ReCLIP++ local baseline | 0.8451 | experiments/voc_reclippp_baseline/ | 2026-07-05 | reproduction of paper baseline |
| B. L12 only | not run | — | — | — |
| C. L9 + L12 selective fusion | **0.4125** | experiments/voc_l9l12_selective/ | 2026-07-06 | 50 epochs; test cfg GAMMA9 0.20, GATE_TEMP 10.0; result from console (log file was not appended). 8/21 classes have IoU exactly 0. |
| D. L6 + L12 selective fusion | not run (queued) | experiments/voc_l6l12_selective/ | — | ON HOLD pending C failure diagnosis |
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
