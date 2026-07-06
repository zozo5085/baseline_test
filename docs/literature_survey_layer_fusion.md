# Literature Survey — Multi-Layer CLIP Fusion, Gated Selective Fusion, and Test-Time Logit Refinement

Date: 2026-07-06. Method: 5 parallel web-search angles + 1 adversarial verification pass (all critical claims re-fetched from primary sources: arXiv HTML/abs pages, official GitHub source code). Works covered in depth: 25+. Emphasis 2023–2026, plus seminal ancestry (2011–2020).

**Our components under evaluation:**
- **(a)** Uncertainty-gated residual fusion of CLIP ViT layers 6/9/12 inside ReCLIP++'s learned bias-rectification U-USS pipeline (gamma=0-at-init identity residual; gate derived from anchor logits).
- **(b)** Test-time edge-preserving domain-transform logit refinement (DTLR/SFP-style) stacked on the U-USS pipeline.

Legend: claims tagged [verified] were confirmed against the primary source (paper HTML/PDF or official code); [inferred] are this survey's reasoning; [unverified] could not be confirmed and are flagged.

---

## VERDICT (a) — Uncertainty-gated selective layer fusion in U-USS: **novelty survives, with mandatory related-work citations**

**No prior work found that gates fusion of intermediate CLIP ViT layers by uncertainty (or boundary signals) in any segmentation setting — let alone inside a learned bias-rectification U-USS pipeline.** An adversarial search specifically trying to kill this claim (12+ query families, 10+ full-paper fetches, source-code inspection where papers were paywalled) found the space split into two non-overlapping camps:

1. **Multi-layer fusion, but static/architectural (no confidence gating):**
   - **ITACLIP** (arXiv:2411.12044, CVPR-W 2025): unweighted average of final-layer attention with mean of layers 7/8/10 attention — `Attn = (Attn_L + mean(Attn_l'))/2`. No gating. [verified]
   - **ResCLIP** (arXiv:2411.15851, CVPR 2025): aggregates intermediate-layer q-k attention (ablation best: **layers 6→9**) and blends with final-block attention using a **fixed scalar** λ=0.5. No per-pixel gating. [verified]
   - **GCLIP** (arXiv:2502.06818, 2025): fuses attention of auto-identified "global-token-emerging" blocks (block 6–7 on ViT-B/16) with last-block Q-Q attention by plain averaging. [verified]
   - **MiddleCLIP** (Neurocomputing 2026): official code shows **fixed linear interpolation** with config-file scalar weights (`(1-λ)·mid + λ·shallow`, VOC: shallow layer 5, mid layer 10, λ=0.8/0.2); grep of the code for entropy/uncertainty/boundary/gate found nothing. [verified via code; paper PDF paywalled]
   - **SAN** (arXiv:2302.12242, CVPR 2023): taps CLIP layers stem/3/6/9 into a trained side adapter by **simple element-wise addition** — authors explicitly leave sophisticated fusion to future work. Supervised OVSS, not U-USS. [verified]

2. **Uncertainty-gated fusion, but not across ViT layers:**
   - **DR-Seg** (arXiv:2604.02010, 2026, no venue): "Uncertainty-Guided Adaptive Fusion" — pixel-wise softmax **entropy** from the original CLIP branch modulates fusion of two **branches** (CLIP-semantic vs DINO-structure), for **trained** open-vocabulary **remote-sensing** segmentation. Closest match on the gating-signal axis. [verified]
   - **MROVSeg** (arXiv:2408.14776): learned per-location "scale attention" gate over CLIP layers {stem,6,12,18} — a real learned gate over layers, but trained supervised, gate is a multi-resolution "trust" weight, not uncertainty-derived, no identity-residual design. Closest match on the fusion-mechanism axis. [verified]
   - **NERVE** (arXiv:2511.08248, 2025): entropy gates **refinement strength** (random-walk propagation), not layer fusion. [verified at abstract level]
   - **CrispFormer** (arXiv:2511.19765, WSSS): predicted aleatoric variance gates a residual logit correction — same "uncertainty gates a residual" skeleton, but SegFormer decoder features, weakly-supervised, not CLIP layers. [verified at abstract level]

**Delta of our component (a) over the closest priors (MROVSeg + DR-Seg):** no single work combines (i) gate signal = uncertainty derived from anchor logits (vs MROVSeg's trained scale-attention, DR-Seg's softmax entropy of a branch), (ii) fusion axis = depth-wise CLIP ViT layers 6/9/12 of one trunk (vs DR-Seg's two model branches), (iii) gamma=0-at-init identity residual guaranteeing the baseline pipeline is preserved at initialization, and (iv) setting = learned bias-rectification U-USS (ReCLIP++ family) — all prior fusion works live in training-free OVSS or supervised OVSS, none inside a bias-rectification pipeline. No follow-up to ReCLIP/ReCLIP++ extending its bias framework was found at all (citation-graph check, non-exhaustive).

**Residual risks:** (1) MiddleCLIP's paper text itself is paywalled — the no-gating conclusion rests on its official code, which is strong but not the paper's prose; (2) CaR / PnP-OVSS / GEM internals were checked via abstracts only (no gating surfaced, but absence-of-evidence); (3) the generic "gated residual fusion" pattern (Gated-SCNN 2019, GFF AAAI 2020) is old — the paper must claim the *combination*, not gated fusion per se. **ITACLIP, ResCLIP, GCLIP, MiddleCLIP, SAN, MROVSeg, DR-Seg must all be cited and distinguished** — a reviewer will find them.

---

## VERDICT (b) — DTLR/SFP-style test-time logit refinement on U-USS: **novelty survives as a combination claim, not a mechanism claim**

The mechanism itself is old and well-known; the paper cannot claim edge-preserving logit filtering as new. The ancestry chain [verified]:

- **Gastal & Oliveira, "Domain Transform for Edge-Aware Image and Video Processing"** (SIGGRAPH 2011) — the filter itself.
- **Krähenbühl & Koltun, DenseCRF** (NeurIPS 2011) — the canonical edge-aware post-processor for segmentation scores (DeepLab-era standard).
- **Chen, Barron, Papandreou, Murphy, Yuille** (CVPR 2016, arXiv:1511.03328) — **the direct predecessor**: domain-transform filtering of CNN segmentation logits, but with a **discriminatively trained** task-specific edge map (end-to-end, not test-time-only).
- **Barron & Poole, Fast Bilateral Solver** (ECCV 2016) — test-time-only edge-aware drop-in DenseCRF replacement.
- **PAMR** (Araslanov & Roth, CVPR 2020) — parameter-free RGB-affinity iterative mask refinement; **the de-facto standard post-processor in training-free CLIP OVSS** (used/compared by SCLIP-family baselines; specific per-paper attribution [inferred] from FreeDA's comparison framing — verify in each baseline's code before citing as fact).

**No paper matching the literal acronyms "DTLR" or "SFP" was found in the segmentation literature** despite targeted searches — if these names come from a specific paper, we could not locate it; treat them as internal names and cite the Chen et al. 2016 lineage instead. [verified-absence, searches enumerated in agent logs]

**Delta of our component (b):** nobody found applies an **untrained, test-time-only domain-transform filter to the logits of a CLIP-based unsupervised segmentation pipeline**. Chen et al. 2016 trains the edge predictor; PAMR uses RGB-affinity averaging (not a domain transform) and lives in training-free OVSS rather than learned U-USS; CLIP-DINOiser refines features (learned convs), not logits. The honest framing: "we port a classical, well-understood edge-aware filter (domain transform) as a training-free logit-refinement stage for C-USS, replacing/complementing PAMR/DenseCRF" — an engineering/empirical contribution, positioned against PAMR as the standard alternative. **Do not oversell as a new mechanism.** A head-to-head vs PAMR and DenseCRF on the same checkpoints would make this contribution solid.

**Residual risk:** SCLIP/ClearCLIP/NACLIP full texts resisted PDF extraction; whether any of them already uses a domain-transform (rather than PAMR) post-processor was **not fully ruled out** — check their repos for `pamr`/`dt`/`crf` modules before submission (10-minute job).

---

## ReCLIP++ canonical citation [verified via arXiv + official repo BibTeX]

Two separate publications — cite the right one:
- Conference: Wang, Jingyun and Kang, Guoliang. **"Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation."** CVPR 2024, pp. 4102–4112. (No "++" in the CVPR title.)
- Journal (extended): Wang & Kang. **"ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation."** IJCV 2025. arXiv:2408.06747, DOI 10.1007/s11263-025-02566-5.

Rectifies class-preference bias (learnable Reference prompt) and space-preference bias (positional-embedding projection); category "C-USS" alongside MaskCLIP+, CLIPpy, ReCo, CLIP-S4.

---

## Annotated works by theme

### Theme 1 — Multi-layer / intermediate CLIP-ViT feature or attention fusion

| Work | Venue/Year | URL | What it does | vs. ours |
|---|---|---|---|---|
| ITACLIP | CVPR-W 2025 | https://arxiv.org/abs/2411.12044 | Training-free OVSS; averages final-layer attention with mean of layers 7/8/10 attention, unweighted | Static average, no gate, attention-level not feature-level, training-free not U-USS |
| ResCLIP | CVPR 2025 | https://arxiv.org/abs/2411.15851 | RCS: mean of intermediate q-k attentions (best range 6→9) blended into final block at fixed λ=0.5; SFR refines by semantic masks | Fixed scalar blend of attention maps; no uncertainty gate, no learned pipeline |
| GCLIP | arXiv 2025 | https://arxiv.org/abs/2502.06818 | Finds "global tokens" emerge from block 6; averages block-6/7 attention with last-block Q-Q attention + FFN channel suppression | Plain averaging of attention; independently validates layer-6 as a semantically meaningful depth (useful citation for our layer choice) |
| MiddleCLIP | Neurocomputing 2026 | https://www.sciencedirect.com/science/article/abs/pii/S0925231226016875 | Training-free OVSS fusing shallow (l.5) + middle (l.10) features by fixed linear interpolation (code-verified, λ from config) | No gating (code-verified); training-free; nearest "middle-layer" branding — must be cited and distinguished |
| SAN | CVPR 2023 | https://arxiv.org/abs/2302.12242 | Side adapter taps CLIP layers stem/3/6/9, fused by element-wise addition (ablation: 21.1→27.8 mIoU) | Supervised OVSS, static addition into a trained side net; validates that mid layers help dense prediction |
| MROVSeg | arXiv 2024 | https://arxiv.org/abs/2408.14776 | Learned per-location scale-attention gate over CLIP layers {stem,6,12,18} for multi-resolution OVSS | Closest on mechanism: a true learned gate over layers — but supervised, gate = resolution-trust not uncertainty, no identity-residual |
| CLIPSelf | ICLR 2024 | https://arxiv.org/abs/2310.01403 | Self-distills image-level features into dense features | No multi-layer fusion |
| CLIP Surgery | PR 2025 (arXiv 2023) | https://arxiv.org/abs/2304.05653 | Dual-path v-v attention surgery for explainability/OVSS | No cross-layer fusion |

**Final-layer-only negative evidence [verified]:** SCLIP (ECCV 2024, https://arxiv.org/abs/2312.01597), ClearCLIP (ECCV 2024, https://arxiv.org/abs/2407.12442), NACLIP (WACV 2025, https://arxiv.org/abs/2404.08181), GEM (CVPR 2024, https://arxiv.org/abs/2312.00878), CLIPtrase (ECCV 2024, https://arxiv.org/abs/2407.08268), ProxyCLIP (ECCV 2024, https://arxiv.org/abs/2408.04883, fuses DINO↔CLIP across models, not layers), MaskCLIP (ECCV 2022, https://arxiv.org/abs/2112.01071) — all modify only the final block's attention/features. This is the standard limitation our fusion is positioned against.

### Theme 2 — Gating / uncertainty-driven selection (any axis)

| Work | Venue/Year | URL | Gate signal → what is gated | Overlap rating |
|---|---|---|---|---|
| DR-Seg | arXiv 2026 (no venue) | https://arxiv.org/abs/2604.02010 | Pixel-wise softmax entropy of CLIP branch → fusion of CLIP branch vs DINO-refined branch; trained; remote sensing | PARTIAL (signal axis) — entropy-gated fusion exists, but branch-wise, trained, different domain |
| MROVSeg | arXiv 2024 | https://arxiv.org/abs/2408.14776 | Learned scale-attention → multi-layer multi-resolution fusion | PARTIAL (mechanism axis) |
| NERVE | arXiv 2025 | https://arxiv.org/abs/2511.08248 | Prediction entropy → strength of random-walk label refinement | LOOSE — gates refinement, not fusion |
| CrispFormer | arXiv 2025 (WSSS) | https://arxiv.org/abs/2511.19765 | Predicted aleatoric variance → residual logit correction + multi-scale fusion gate | LOOSE — same skeleton, SegFormer decoder, not CLIP layers |
| Uncertainty-Gated Region Retrieval | arXiv 2025 | https://arxiv.org/abs/2512.18082 | Per-region uncertainty → auxiliary retrieval | LOOSE — same "gate auxiliary info by confidence" pattern |
| GFF | AAAI 2020 | https://ojs.aaai.org/index.php/AAAI/article/view/6805 | Learned gate maps → cross-level CNN feature fusion | ANCESTRY — pre-ViT generic gated multi-level fusion |
| Gated-SCNN | ICCV 2019 | https://arxiv.org/abs/1907.05740 | Boundary stream gates → shape/texture fusion | ANCESTRY — boundary-gated fusion concept |

### Theme 3 — Test-time / post-hoc logit and mask refinement

| Work | Venue/Year | URL | Mechanism | Test-time only? |
|---|---|---|---|---|
| Gastal & Oliveira | SIGGRAPH 2011 | https://dl.acm.org/doi/10.1145/2010324.1964964 | Domain transform: 1D isometric warp → linear-time edge-aware filtering | Yes (signal processing) |
| DenseCRF | NeurIPS 2011 | https://arxiv.org/abs/1210.5644 | Fully-connected CRF, Gaussian edge potentials, mean-field | Yes (inference) |
| Chen et al. | CVPR 2016 | https://arxiv.org/abs/1511.03328 | Domain transform on CNN logits with a trained task-specific edge map | **No** — edge net trained end-to-end (key delta vs ours) |
| Fast Bilateral Solver | ECCV 2016 | https://arxiv.org/abs/1511.03296 | Bilateral-space least squares on scores; 8× faster DenseCRF replacement | Yes |
| PAMR | CVPR 2020 | https://openaccess.thecvf.com/content_CVPR_2020/papers/Araslanov_Single-Stage_Semantic_Segmentation_From_Image_Labels_CVPR_2020_paper.pdf | Parameter-free RGB-affinity iterative label averaging; de-facto standard in training-free CLIP OVSS [attribution per-baseline: inferred] | Yes |
| CLIP-DINOiser | ECCV 2024 | https://arxiv.org/abs/2312.12359 | 2 light convs learn DINO-like patch affinity to pool CLIP features | No — learned; feature-level not logit-level |
| CaR (CLIP as RNN) | CVPR 2024 | https://arxiv.org/abs/2312.07661 | Recurrent inference-time mask/query filtering with frozen VLM | Yes, but iterative re-inference, not edge-aware filtering |
| NERVE | arXiv 2025 | https://arxiv.org/abs/2511.08248 | Entropy-guided random-walk propagation of labels | Yes |

**"DTLR" / "SFP" acronyms: not found in the literature** despite direct searches — cite the mechanism by lineage (Gastal & Oliveira 2011; Chen et al. 2016), not by these acronyms.

### Theme 4 — U-USS / bias-rectification context and layer-analysis foundations

- **ReCLIP++ / ReCLIP** — see canonical citation above. No methodological follow-ups extending its bias framework found (Semantic Scholar citation check, non-exhaustive) → the fusion/refinement extension space is open. [verified, coverage-limited]
- **MaskCLIP** (ECCV 2022, https://arxiv.org/abs/2112.01071) — earliest training-free dense CLIP; value-embedding readout, final layer only.
- **DenseCLIP** (CVPR 2022, https://arxiv.org/abs/2112.01518) — fine-tuned pixel-text matching; no bias framing, no fusion.
- **TCL** (CVPR 2023, https://arxiv.org/abs/2212.00785) — text-grounded contrastive learning; TL-OVSS baseline in ReCLIP++.
- **ReCo** (NeurIPS 2022, https://arxiv.org/abs/2206.07045) — retrieve-and-co-segment; C-USS baseline alongside ReCLIP++.
- **PnP-OVSS** (CVPR 2024, https://arxiv.org/abs/2311.17095) — cross-attention + salience dropout, training-free; no fusion/gating surfaced.
- **FreeDA** (CVPR 2024, https://arxiv.org/abs/2404.06542) — diffusion-augmented prototypes; evidence that PAMR is the standard refinement comparison point.
- **Raghu et al., "Do ViTs See Like CNNs?"** (NeurIPS 2021, https://arxiv.org/abs/2108.08810) — ViT final third of layers becomes CLS-centric; motivates why last-layer-only dense readout loses spatial signal.
- **LHT-CLIP** (arXiv 2025, https://arxiv.org/abs/2510.23894) — [verified at HTML level] visual discriminability follows an **inverted-U across CLIP-ViT depth** (declines sharply in the last 2–3 blocks of ViT-B/16 due to anomalous dominant tokens) while text alignment rises monotonically. **Strongest published motivation for fusing mid-depth (6/9) with final (12) layers.**
- **Evidential/logit-space uncertainty** (arXiv:2605.20543) — argues logit-space uncertainty beats post-softmax entropy for control signals (softmax couples classes, discards evidence magnitude) — useful citation to justify a gate computed from raw anchor **logits**. [verified at abstract level]
- **Training-free OVSS survey** (arXiv:2505.22209, May 2025) — 30+ methods taxonomized; good coverage checkpoint for the related-work section.

---

## Related-work paragraph draft (English, adapt freely)

> **Intermediate-layer features in CLIP-based segmentation.** Most training-free adaptations of CLIP to dense prediction operate exclusively on the final transformer block, reformulating its self-attention to recover spatial locality (SCLIP [Wang et al., ECCV 2024]; ClearCLIP [Lan et al., ECCV 2024]; NACLIP [Hajimiri et al., WACV 2025]; GEM [Bousselham et al., CVPR 2024]). Yet analyses of ViT representations show that the final blocks trade patch-level discriminability for global text alignment (Raghu et al., NeurIPS 2021; LHT-CLIP [Zhou et al., 2025]), motivating a recent line of work that re-injects intermediate-layer information: ITACLIP averages the attention maps of layers 7/8/10 into the final block (Aydın et al., 2025), ResCLIP blends aggregated intermediate cross-correlation attention with the final block at a fixed ratio (Yang et al., CVPR 2025), GCLIP fuses attention from the blocks where global tokens first emerge (Wang et al., 2025), MiddleCLIP interpolates shallow and middle features with fixed scalar weights (Jin et al., Neurocomputing 2026), and SAN taps layers 3/6/9 into a trained side adapter by simple addition (Xu et al., CVPR 2023). In all of these, fusion is *static*: layer contributions are fixed scalars or unconditionally learned weights, applied uniformly across all pixels. Confidence-adaptive fusion has appeared only outside the layer axis — DR-Seg gates the combination of a CLIP branch and a DINO-refined branch by prediction entropy for remote-sensing segmentation (Feng et al., 2026), and MROVSeg learns resolution-trust weights over multi-scale crops (2024). In contrast, we introduce an *uncertainty-gated* residual fusion of intermediate CLIP layers inside a learned bias-rectification pipeline for unsupervised semantic segmentation: a per-pixel gate derived from the anchor logits selectively admits intermediate-layer detail only where the rectified prediction is unreliable, and a zero-initialized residual design guarantees the pipeline reduces exactly to its bias-rectified baseline at initialization.
>
> **Test-time refinement of segmentation logits.** Edge-preserving post-processing of segmentation scores has a long lineage: fully-connected CRFs with Gaussian edge potentials (Krähenbühl & Koltun, NeurIPS 2011) were the standard refinement stage of the DeepLab era, later accelerated or replaced by the fast bilateral solver (Barron & Poole, ECCV 2016) and by domain-transform filtering (Gastal & Oliveira, SIGGRAPH 2011), which Chen et al. (CVPR 2016) applied to CNN logits with a discriminatively trained edge predictor. In training-free CLIP segmentation, the prevailing lightweight refiner is PAMR (Araslanov & Roth, CVPR 2020), an RGB-affinity iterative averaging scheme, while CLIP-DINOiser instead learns a feature-level affinity pooling (Wysoczańska et al., ECCV 2024). We revisit the classical domain transform as a *purely test-time, training-free* logit-refinement stage for CLIP-based unsupervised segmentation, requiring no learned edge predictor, and evaluate it against the PAMR/DenseCRF alternatives on identical checkpoints.

---

## Coverage and limitations of this survey

- Works substantively assessed: 25+ (counts: layer-fusion angle 16, gating angle 10 fetched, refinement angle 9, context angle 13, foundations angle 10; overlapping).
- Unresolved: (1) MiddleCLIP paper prose (paywalled; conclusion from official code); (2) whether SCLIP/ClearCLIP/NACLIP repos contain any domain-transform post-processing (PDF extraction failed; check their code); (3) CaR/PnP-OVSS/GEM internals checked at abstract level only; (4) ReCLIP++ citation-graph check was truncated, not exhaustive.
- arXiv IDs of form 25xx/26xx are 2025/2026 preprints; venue "arXiv only" means no peer-reviewed venue found as of 2026-07-06.
