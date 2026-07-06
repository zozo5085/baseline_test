# SFP + DTLR Generalization Review (adversarial)

Reviewed: `othermodel_guide/model_1/model_lrab_v1_voc_final_862.py`
Date: 2026-07-07. Goal: does PG-CP-SFP (Stage 1) + SP-DTLR (Stage 2) transfer beyond VOC,
or does it buy VOC mIoU by exploiting VOC-specific structure? Job = find problems.
All line numbers verified by reading source. Eval regime verified from
`config/voc_test_baseline_reproduce_cfg.yaml` + `utils/preprocess.py:279-315` +
`tools/test.py:147-204`: whole-image inference (NO slide window), SCALE=[2048,336]
(short edge 336), patch 16, batch 1, NUM_CLASSES=20, REDUCE_ZERO_LABEL, background via
PD threshold not via SFP.

## Eval-resolution fact (load-bearing for the whole review)
Whole image resized to short-edge 336. Typical VOC image (~500x375) -> ~448x336 ->
token grid ceil(448/16) x ceil(336/16) = 28 x 21 = **~588 tokens** (range ~590-880 across
aspect ratios; only very elongated images exceed ~800). This single fact drives F1/F2.

## Findings

| # | Mech | file:line | Issue | Severity | Adaptation |
|---|---|---|---|---|---|
| F1 | SFP | 619, 870, 858-873 | `sfp_topk=800` is a FIXED ABSOLUTE token count. Selection is `k=min(800, valid.sum())`. At VOC eval the whole token grid is ~588 (<800), so **the 800 cap NEVER binds and the top-k / margin ranking is completely inert** — every "valid" token is selected. "Top-800 unreliable tokens" is a misnomer at this resolution: it is "all low-confidence tokens". Selectivity only appears once token_count>800 (higher-res datasets), so the mechanism's qualitative behavior flips with token count. | blocker (transfer) | Replace 800 with a fraction of valid tokens: `k = round(top_p * valid.sum())`, e.g. top_p≈0.6, OR a percentile on `sfp_score`. Makes selectivity resolution-invariant. |
| F2 | SFP | 860-861, 620, 622-623 | "Unreliable" = `sfp_score > -1e9` (min_score=-1e9 -> ALWAYS true, filter inert) AND `conf < 0.97` under temperature `conf_scale=10`. Both the score gate and (via F1) the top-k are inert, so **selection collapses to a single test: max-softmax<0.97**. That threshold's selectivity is class-count-dependent: with 171 classes (COCO-Stuff) baseline max-prob is lower, so far more tokens qualify -> near-total purification. | transfer-risk | Make `conf_thd`/`conf_scale` dataset-relative, or select by relative rank (percentile) not absolute conf. Re-tune `sfp_min_score` to actually gate, or delete it. |
| F3 | SFP | 906-942, 939-940 | The "proxy" = mean logits of local high-confidence (`>0.95`) kept neighbors in a 5x5 box (`proxy_kernel=5`). Target = `neigh_mean + lambda*(proxy_mean - neigh_mean)` with `sfp_proxy_lambda=2.00` -> `2*proxy_mean - neigh_mean`: an **EXTRAPOLATION past the proxy** (lambda>1). Assumes the 5x5 window is class-homogeneous with a confident core (true for object-centric VOC). On stuff/dense-class scenes the 5x5 straddles boundaries and the 2x overshoot pushes logits arbitrarily. | transfer-risk | Clamp `proxy_lambda<=1` (interpolate, no overshoot) or make it edge-gated; validate on a multi-class-dense image before trusting. |
| F4 | SFP | 887-946, 890-901 | Purification target is 8-neighbor / 5x5 logit averaging -> **spatial-coherence prior**: assumes large contiguous same-class regions. Hostile to thin structures (Cityscapes poles/riders/signs 1-2 tokens wide) and small regions (ADE/COCO): the neighborhood is dominated by the surrounding class and smooths the thin object AWAY. Shared root with F6/F9. | transfer-risk | Gate purification off inside high-boundary-density regions, or reduce beta where predicted-class run-length is short. |
| F5 | SFP | 1682 (no guard) vs 1686/1695/1707 (`not training`) | `sfp_logit_purify` runs in **BOTH train and eval**, and its blend (890-946) is OUTSIDE `torch.no_grad` -> **differentiable, trained through**. The 862 decoder/prompt were optimized WITH purify in the loop, so the checkpoint is co-adapted. Porting purify as a pure test-time add-on onto the ReCLIP++ baseline (trained WITHOUT it) is a **train/eval mismatch** — gains may not reproduce, or may need re-training. DTLR/attr/fbls are genuinely eval-only, so those DO port as pure post-hoc. | blocker | Either (a) port purify into training too and re-train, or (b) treat purify as eval-only from day one and re-tune beta on the frozen baseline; do NOT assume 862's beta=0.55 carries over. |
| F6 | DTLR | 665, 527-528, 1035 | `sigma_s=70.0`, `sigma_r=1.50` -> gain `sigma_s/sigma_r=46.7`. The domain transform runs at **token-grid resolution (~30 wide)**, so a spatial sigma_s=70 >> grid width => the filter propagates **near-globally** along low-gradient paths (only hard RGB edges stop it). On VOC's few large regions this is benign smoothing; the same sigma at a larger token grid becomes a LOCAL filter. sigma_s is not scaled to grid size -> its effective radius (as a fraction of the image) drifts with token count. | transfer-risk (parameterize) | Set sigma_s as a fraction of token-grid width (e.g. sigma_s = c * W_tokens), not an absolute 70. |
| F7 | DTLR | 664, 1081, 1100 | `sfp_dtlr_beta=1.20`. Update = `output + beta*mask*(filtered-output)` -> beta>1 **OVERSHOOTS past the filtered logits**. Aggressively snaps outliers toward (and beyond) the edge-smoothed value. On thin structures this amplifies erasure (compounds F4/F6). | transfer-risk | Constrain beta<=1 for stuff/thin datasets; expose as dataset knob. |
| F8 | DTLR | 682, 1065-1068, 681 | `sfp_dtlr_structure_classes=(4,8,10)` = VOC bottle/chair/diningtable, **hardcoded indices**. On any other dataset these integers protect arbitrary/wrong classes. `gain_thd=0.00` (design note recommended 0.03-0.05, 677-679 — shipped looser). This is VOC label-space leakage inside Stage 2 (independent of the excluded Stage 3 attribute block). | transfer-risk | Disable structure protection on non-VOC, or remap to that dataset's structured classes; do not carry the (4,8,10) tuple. |
| F9 | DTLR | 553-582 | `_filter_horizontal/_filter_vertical` are Python for-loops over W then H (sequential recursion). Correct but O(W)+O(H) sequential python per iter. Fine at token grid ~30; **if ever applied at image resolution (Cityscapes 2048) it is ~2048 sequential steps/direction -> unusably slow**. Constrains DTLR to the low-res token grid, which in turn forces F6. | minor (perf/scope) | Keep at token grid; if higher-res filtering wanted, vectorize the recurrence or use a separable approximation. |

## Per-mechanism verdicts

- **PG-CP-SFP (Stage 1): transferable-with-parameterization.** No hardcoded class indices
  in the purify body (checked 795-995) — the VOC bias is in the *knobs*, not the label space.
  Must-change knobs: `sfp_topk` -> fraction/percentile (F1); `conf_thd`/`conf_scale`
  dataset-relative (F2); `proxy_lambda<=1` (F3); boundary-gate the smoothing (F4). Also
  resolve the train/eval co-adaptation (F5) before trusting numbers. As-shipped it is NOT
  "selective" at VOC resolution — it near-globally smooths, which only looks good because
  VOC regions are large and object-centric.
- **SP-DTLR (Stage 2): transferable-with-parameterization, verging on VOC-biased.** Two
  knobs are resolution-coupled (sigma_s F6, must become grid-relative) and one (beta>1, F7)
  is an aggressive VOC tuning; plus hardcoded VOC class indices in the structure guard (F8).
  With sigma_s made grid-relative, beta<=1, and structure protection disabled/remapped, the
  edge-preserving idea itself is dataset-agnostic. Left as-is it will erode thin structures
  on Cityscapes and over-smooth dense-class scenes.
- Neither mechanism is *fundamentally* VOC-biased in its math (unlike the excluded Stage 3
  attribute-residual block at 1358-1615, which hardcodes chair/table indices (8,10), eta
  40/32, and hand-written attribute phrases 746-769 — correctly excluded from the port).

## Checked and found CLEAN (so silence != unchecked)
- Numerical: every division guarded — `neigh_count.clamp_min(1.0)` (900), `proxy_count.clamp_min(1.0)` (937), `(score_max-score_min).clamp_min(1e-6)` (850), `update_count.clamp_min(1.0)` (958), `selected_count.clamp_min(1.0)` (1111); DTLR `max(sigma_r,1e-6)` (527), `max(sigma_h,1e-6)` (540), `max(feedback,1e-12)` (543), guide `(max-min).clamp_min(1e-6)` (519). No division-by-zero path found.
- Softmax dims correct: `softmax(output*scale, dim=1)` over class channel [B,C,H,W] at 824, 1059-1060; `conf=max(dim=1)`; `topk(prob,k=2,dim=1)` needs C>=2, holds for every target dataset.
- top-k bounds guarded: `k=min(topk, valid.sum())`, `if not valid.any(): continue` (867-870) — no k>N crash.
- Stage 1 purify has NO hardcoded class indices (grepped 795-995) — clean w.r.t. label space; the 20/21 background handling lives in `tools/test.py` PD thresholding, not in SFP.
- Mask hand-off consistent: DTLR reuses exactly `sfp_last_outlier_mask` cached by purify (876, 1008), interp `nearest` (1019-1023) is a no-op when grids match (they do). No shape mismatch at VOC.
- `sfp_score` resolution handling (816-821) is a no-op at eval (score and logits share the token grid); would degrade gracefully via nearest-interp otherwise.

Zero findings would have been suspicious for a VOC-tuned test-time module; the 9 above
concentrate exactly where expected (absolute counts, absolute spatial sigmas, absolute
class indices, and a train/eval co-adaptation), which suggests reasonable coverage of the
two ported mechanisms. Stages 3/4 were only spot-checked (excluded from port).

## Facts vs inference
- VERIFIED (read in source, cited): all constants (800/0.97/10/0.95/2.00/0.55; 70/1.5/1.20;
  (4,8,10); beta>1 overshoot; lambda>1 extrapolation; purify runs in train+eval and is
  differentiable; python-loop recurrence; guards; softmax dims).
- INFERENCE (labeled assumption): token count ~588-880 is COMPUTED from SCALE/patch, not
  measured on a run — but F1's conclusion (top-k inert) holds for any token_count<800, a
  wide margin. Harm on stuff/thin structures (F3/F4/F6/F7) is reasoned from the operators'
  spatial-averaging nature, not measured; recommend a 3-image sanity dump on ADE/Cityscapes
  before/after each stage to confirm.
