# Journal 10-Page Experiment Plan — Reliability-Guided Logit Purification for Generalizable CLIP-Based Semantic Segmentation

Compiled 2026-07-08. Consolidates `docs/JOURNAL_EXTENSION_PLAN.md` (positioning + formal/exploratory
rules) and `docs/GENERALIZATION_PROTOCOL.md` (what is formally claimable), against all results in
`docs/method_results.csv` and `docs/research_notes.md §11`. All mIoU verbatim (4 decimals).

LGAK (`docs/LGAK_IMPLEMENTATION_REVIEW.md`) is **archived as a separate new-direction MVP** and is
**excluded from the journal main experiments** (see §10).

---

## 0. Status and the central strategic decision (read first)

The journal was positioned around one thesis (`JOURNAL_EXTENSION_PLAN.md:17`): *reliability-guided
logit purification (SFP/DTLR) generalizes across datasets*. **Our own results, produced under that
plan's own protocol, do not yet support that thesis as a main claim.** The one non-VOC datapoint we
have (PASCAL Context) shows dataset-agnostic SFP/DTLR is a **negative** delta, and it stayed negative
after we removed its VOC-calibration confound in code:

- Context baseline no-TTA **0.1980** → agnostic SFP/DTLR no-TTA **0.1929** (**−0.0051**).
- After the entropy-gate de-confound: **0.1945** (+0.0016 recovered, but **still −0.0035** vs base).

By the plan's own **Decision Rule** (`JOURNAL_EXTENSION_PLAN.md:229`): *"if it only works on VOC,
downgrade SFP/DTLR to a VOC-specific observation."* We are currently in that branch — with **one
untested confound remaining**: the Context base is under-trained (8-epoch, mIoU 0.198, not converged).

This forces a framing choice. Both are legitimate journal papers; they need different experiments:

**Framing A — "Generalization audit" (defensible with data we nearly have).** The contribution is
the **dataset-agnostic evaluation protocol + de-VOC methodology + the honest finding that what
transfers is parameter-free (flip-TTA), while VOC-tuned logit purification does not**. flip-TTA is
the clean generalizable positive (VOC +0.0065, Context +0.0048). The entropy-normalized reliability
gate is a methodological tool. SFP/DTLR appears as a VOC-effective component whose cross-dataset
limitation is reported honestly. **Lower risk; publishable on existing + Top-3 results.**

**Framing B — "SFP/DTLR generalizes" (the original thesis; needs a positive non-VOC result).**
Requires SFP/DTLR to improve **≥1 non-VOC dataset** without VOC tuning. That is only possible if the
remaining confound (under-trained Context base) is the cause — i.e. a **converged** Context base or
**ADE20K** turns the delta positive. If Top-3 experiment #1/#2 come back positive, Framing B is alive;
if negative, Framing A is the honest paper.

**Recommendation:** run Top-3 #1 and #2 (§11) first — they decide which framing is real — and default
to **Framing A** unless they turn positive. Do not write the SFP-generalizes claim until a non-VOC
positive exists. This is a research-direction call; the plan below supports **both** so the decision
can wait for data.

---

## 1. Journal claims (期刊主張)

**Working title** (`JOURNAL_EXTENSION_PLAN.md:24`): *Reliability-Guided Logit Purification for
Generalizable CLIP-Based Semantic Segmentation.*

**Primary contribution (framing-independent, safe):**
- C1. A **dataset-agnostic evaluation protocol** for test-time refinement on frozen CLIP USS:
  a hard formal/exploratory split, a De-VOC checklist that converts every VOC-tuned constant to a
  ratio/normalized quantity, and an **entropy-normalized reliability gate**
  `H_norm = entropy(softmax(logits·s))/log(C)` that removes the class-count operating-point confound
  of absolute confidence thresholds. Validated: on VOC the gate is a near-no-op (0.8579 vs 0.8582);
  on 59-class Context it recovers the over-flagging (+0.0016).
- C2. A **generalization study** of ReCLIP++-style refinement: **parameter-free flip-TTA transfers**
  (VOC +0.0065, Context +0.0048), whereas **VOC-tuned multi-scale and the legacy VOC-profiled
  SFP/DTLR do not** — quantified under one fixed protocol with reproduced baselines.

**Conditional contribution (only if Top-3 #1/#2 give a non-VOC positive — Framing B):**
- C3. Dataset-agnostic reliability-guided logit purification improves CLIP dense prediction **beyond
  VOC** (≥2 datasets), i.e. the refinement is a generalizable method, not a VOC setting.

**Explicit non-claims (see §9):** no multi-scale main number; no VOC-val-tuned hyperparameter as a
result; SFP/DTLR is **not** claimed generalizable unless C3's evidence exists.

---

## 2. Result inventory — four buckets

### 2A. Formal-claimable (usable as main results)  — 已有結果哪些可用

VOC (reproduced baseline **0.8536**; full val 1449; PD 1.0):

| result | mIoU | Δ vs base | class |
|---|---|---|---|
| ReCLIP++ baseline no-TTA | 0.8536 | — | formal |
| baseline + flip-TTA | 0.8601 | **+0.0065** | formal (standard-TTA table) |
| SFP+DTLR (legacy) | 0.8590 | +0.0054 | formal |
| SFP+DTLR gen (dataset-agnostic) | 0.8582 | +0.0046 | formal |
| SFP+DTLR gen + entropy-gate | 0.8579 | +0.0043 | formal (de-VOC ablation) |
| SFP+DTLR gen + flip | 0.8639 | +0.0103 | formal |
| Method A (trainable presence head) | 0.8565 | +0.0029 | formal (secondary) |

PASCAL Context (in-repo 8-ep base; full val 5105; PD 0.85) — **preliminary: base not converged**:

| result | mIoU | Δ | class |
|---|---|---|---|
| baseline no-TTA | 0.1980 | — | formal-preliminary |
| baseline + flip-TTA | 0.2028 | **+0.0048** | formal-preliminary |
| agnostic SFP+DTLR no-TTA | 0.1929 | −0.0051 | formal-preliminary (negative) |
| agnostic SFP+DTLR + flip | 0.1955 | −0.0025 | formal-preliminary |
| agnostic SFP+DTLR + entropy-gate no-TTA | 0.1945 | −0.0035 | formal-preliminary (de-VOC) |
| agnostic SFP+DTLR + entropy-gate + flip | 0.1973 | −0.0055 | formal-preliminary |

### 2B. Exploratory-VOC (appendix/diagnostic only — NOT main claim)  — 不能當 main claim

| result | mIoU | why exploratory |
|---|---|---|
| SFP+DTLR + ms(1.0,1.25) + flip | 0.8661 | ms=1.25 selected on VOC val |
| Method A + ms(1.0,1.25) + flip | 0.8643 | VOC-selected multi-scale |
| TTA ms(1.0,1.25) + flip (baseline) | 0.8637 | VOC-selected multi-scale |
| TTA ms(1.0,1.25,1.5) + flip | 0.8599 | ms sweep; 1.5 *hurts* (peak=1.25) |

Oracle upper bound (reference only, not a method): image-level perfect FP removal **0.8998** (+0.0462).

### 2C. Negative findings (preserve — they strengthen the "conservative logit-level" argument)

See §8.

### 2D. Future / new-direction (out of journal main scope)

LGAK-MVP (language-guided pre-similarity feature refinement): identity 0.8536 verified, not yet
short-run trained. §10.

---

## 3. Five-dataset matrix — what must be filled  — 五資料集要補什麼

Fixed protocol per dataset: reproduced ReCLIP++ baseline no-TTA / +flip / agnostic SFP-DTLR no-TTA /
agnostic SFP-DTLR +flip (`JOURNAL_EXTENSION_PLAN.md:110-115`).

| dataset | classes | baseline | +flip | SFP no-TTA | SFP +flip | status / blocker |
|---|---|---|---|---|---|---|
| PASCAL VOC | 20 | ✅0.8536 | ✅0.8601 | ✅0.8582 | ✅0.8639 | **complete (formal)** |
| PASCAL Context | 59 | ⚠0.1980 | ⚠0.2028 | ⚠0.1929 | ⚠0.1955 | **preliminary** — 8-ep base, re-run on converged base (Top-3 #1) |
| ADE20K | 150 | ❌ | ❌ | ❌ | ❌ | needs **in-repo rectification base + text emb + pseudo** (Top-3 #2) |
| Cityscapes | 19 | ❌ | ❌ | ❌ | ❌ | needs in-repo base; raw at `D:\ReCLIPv3\datasets` |
| COCO-Stuff | 171 | ❌ | ❌ | ❌ | ❌ | needs in-repo base; largest, slowest |

**Blocker (verified, `GENERALIZATION_PROTOCOL.md §6`):** the per-dataset checkpoints in
`D:\ReCLIPv3\experiments` are a **different (fusion-modified) architecture** (`clip.visual.*` +
`fuse_alpha`), incompatible with this repo's vanilla model. Every non-VOC dataset needs an **in-repo**
rectification base trained here. Per plan `:117`, record missing datasets as blockers, do **not** tune
VOC further to compensate. **Minimum for any generalization claim: VOC + ≥2 non-VOC (Context + ADE).**

Per-dataset preparation checklist (learned from Context, `research_notes.md §11`):
1. Verify GT **class order** matches the text/pseudo order (the Context labels were VOC-20-first, not
   alphabetical — a silent mismatch that gave mIoU 0.0027 until fixed). **Check first for every dataset.**
2. Build `text/<set>_ViT16_clip_text.pth` in GT order; regenerate pseudo aligned to train.txt.
3. Pick the eval `TEST.PD` per class count (PD 1.0 is VOC-specific; Context needed 0.85 — PD 1.0
   prunes all 59 classes → 0.0021 collapse). Document PD choice as dataset metadata, not a tuned knob.

---

## 4. Main results table design  — 實驗表格清單

**Table 1 — Cross-dataset formal main result (no-TTA).** Rows = 5 datasets; columns = {reproduced
baseline, dataset-agnostic SFP/DTLR, Δ}. no-TTA only. This is the paper's spine.

**Table 2 — Standard flip-TTA (separate table, per `JOURNAL_EXTENSION_PLAN.md:53`).** Rows = 5
datasets; columns = {baseline+flip, SFP/DTLR+flip, Δ vs each no-TTA}. Flip is the clean generalizable
positive — keep it in its own table so it never contaminates the no-TTA main claim.

**Table 3 — De-VOC transfer (the key comparison, `plan:136`).** Columns = {legacy VOC-tuned SFP/DTLR,
dataset-agnostic gen, gen + entropy-gate}; rows = VOC and each non-VOC dataset. Shows the claim
survives (or does not) without the legacy VOC profile. We already have the VOC row (0.8590 / 0.8582 /
0.8579) and the Context row (n/a legacy / 0.1929 / 0.1945).

**Table 4 — Appendix, exploratory VOC.** All multi-scale rows (§2B), clearly labelled "VOC-selected,
not a formal claim."

Every table: report the **reproduced** baseline in the same setting (not the paper's number), the Δ,
and (for non-VOC) the PD used.

---

## 5. Ablation table design  — ablation table 設計

**Ablation 1 — component build-up (fixed dataset-agnostic settings, no TTA).** On VOC and Context:

```
Base (reproduced ReCLIP++)
+ URD / selected unreliable tokens (SFP selection only)
+ Proxy-guided purification (CP-SFP)
+ DTLR (RGB-guided domain-transform logit refine)
+ SFP + DTLR combined
[appendix] + attribute residual (VOC-only, excluded from main)
```
Purpose: show which sub-component carries the VOC gain and which one fails to transfer. Each row on
both datasets exposes where the VOC/non-VOC divergence originates.

**Ablation 2 — De-VOC transfer (dataset-agnostic-ization).** Per §4 Table 3: legacy → gen → gen+
entropy-gate, on VOC + non-VOC. This is the methodological ablation: it isolates how much of the VOC
gain was VOC-calibration artifact vs genuine refinement.

**Ablation 3 — reliability-gate design.** absolute CONF_THD (legacy) vs entropy-normalized gate, with
the **flagged-token fraction** reported per dataset — the quantitative evidence that the class-count
confound is real (59-class over-flagging) and that the gate removes it.

**Ablation 4 — TTA (separate).** no-TTA / flip / [appendix] multi-scale. Never merged with the method
ablation.

Current coverage: Ablation 2 VOC row ✅, Context row ✅ (preliminary). Ablations 1/3 need per-component
runs (cheap — all test-time, no training) and the fraction logging already added to `model_sfp_dtlr`.

---

## 6. Qualitative figure design  — qualitative figure 設計

- **Fig. Q1 — before/after logit heatmaps.** Per-class similarity map before vs after SFP/DTLR on
  2–3 images per dataset; overlay the SFP-selected unreliable-token mask. Shows *where* purification
  acts. (Debug export already exists in `model_sfp_dtlr`: `sfp_score`, `sfp_outlier_mask`,
  `sfp_confidence`, `cpsfp_delta`.)
- **Fig. Q2 — reliability/entropy maps.** `H_norm` map on a 20-class (VOC) vs 59-class (Context)
  image side by side, with the legacy-0.97 flagged region vs the entropy-gate flagged region — the
  visual argument for the de-VOC gate.
- **Fig. Q3 — failure gallery.** Boundary-band errors, thin/small objects, and class hallucinations
  before/after — including a case where SFP/DTLR *hurts* on Context (supports the honest limitation).
- **Fig. Q4 — flip-TTA effect.** Prediction disagreement map between the two flip views and the
  averaged result — why the free symmetric average helps and generalizes.

Each figure: same 2–3 anchor images reused across datasets for consistency; include VOC + ≥1 non-VOC.

---

## 7. Runtime / cost table design  — runtime / cost table 設計

**Table R1 — per-method cost** (measure on one fixed GPU, RTX 5090, batch 1, VOC input):

| method | trainable params | train time | test-time overhead vs baseline | peak VRAM | notes |
|---|---|---|---|---|---|
| ReCLIP++ baseline | 0 (frozen infer) | — | 1.0× | measure | reference |
| + flip-TTA | 0 | — | ~2.0× (2 forwards) | measure | parameter-free |
| + SFP/DTLR (gen) | 0 | — | measure (+conv/domain-transform) | measure | test-time only |
| + entropy-gate | 0 | — | ≈ SFP (one extra softmax-entropy) | measure | negligible added |
| + multi-scale (appendix) | 0 | — | ~N× (N scales) | measure | exploratory |
| Method A (secondary) | 4 | ~1.5 h (15 ep) | ≈1.0× | measure | trainable head |

Columns to actually fill: FPS (img/s), ms/img for the refinement stage alone, and the multiplier vs
the raw CLIP forward. SFP/DTLR is test-time-only (no training cost) — a selling point for a *practical*
refinement. Report the domain-transform iteration count's effect on cost.

---

## 8. Failure / negative findings section  — failure / negative findings

A dedicated section (strengthens "conservative logit-level refinement > uncontrolled feature-level
modification", `plan:161`):

| negative finding | evidence | lesson |
|---|---|---|
| **SFP/DTLR does not transfer to Context** | 0.1929 (−0.0051), de-confounded 0.1945 (−0.0035) | VOC gain is partly VOC-specific; logit-purification generalization is not free |
| Feature-level fusion collapses | full-image L6/L9/L12 DFF2d **0.4151** (vs 0.8451) | uncontrolled feature modification destroys CLIP alignment |
| Trainable-from-scratch drift | IABR retrain **0.8001** | retraining under the unsupervised objective drifts below baseline |
| Hand-tuned class gate collapses | CLASS_GATE sweep **0.097** on 20-img | VOC-val hand-sweeps are brittle and non-general |
| Photometric flicker ≠ absent-class signal | Method C cov AUC **0.186** | a plausible reliability cue that empirically fails |
| PAMR over-smooths sharp logits | (to reproduce/quantify) | post-processing can hurt already-sharp ReCLIP++ |
| Multi-scale 1.5 hurts | 0.8637 → **0.8599** | scale is a knob, not a free gain |
| PD 1.0 is VOC-specific | Context baseline @ PD1.0 **0.0021** | absolute prune thresholds are class-count dependent |

The through-line: **parameter-free, symmetric, class-count-invariant operations transfer; VOC-tuned or
feature-destructive ones do not.** This is the paper's honest backbone.

---

## 9. What can and cannot be a main claim  — 明確區分

**CAN be main claim (formal):**
- Reproduced ReCLIP++ baseline (no-TTA) per dataset.
- flip-TTA improvement (no-TTA → flip), any dataset — it is parameter-free and transfers.
- The de-VOC protocol, the entropy-normalized gate, and the flagged-fraction confound analysis.
- Dataset-agnostic SFP/DTLR **delta per dataset** — reported as measured, **including the negative
  Context delta** (a result, not hidden).

**CANNOT be main claim:**
- Any multi-scale number (0.8661/0.8643/0.8637/0.8599) — VOC-selected knob → appendix only.
- Any VOC-val-tuned hyperparameter, VOC class-index hack, or chair/table attribute residual.
- **"SFP/DTLR generalizes"** — currently refuted on the one non-VOC dataset; only becomes claimable if
  Top-3 #1/#2 produce a non-VOC positive (Framing B / contribution C3).
- Context numbers as *converged* results — the 8-ep base is under-trained; label preliminary until
  Top-3 #1.
- The oracle 0.8998 as a method (it is an upper bound only).

---

## 10. LGAK — future work (out of journal main)  — LGAK 標記為 future/new-direction

Per user decision (2026-07-08), LGAK is archived as a separate new-direction MVP and is **not** part
of the journal main experiments. It attacks a different stage (pre-similarity **feature formation**,
not post-logit refinement). Status: identity α=0 = 0.8536 verified, smoke passed, short run not run.
It appears in the journal only as a one-line **future-work** pointer ("language-guided pre-similarity
feature refinement is a promising orthogonal direction"), never as a result. Full record:
`docs/LGAK_IMPLEMENTATION_REVIEW.md`. If the journal generalization succeeds it stays future work; if
the SFP/DTLR generalization fails (Framing A), LGAK is the natural *next paper*, not this one.

---

## 11. Top-3 priority next experiments  — 下一步最優先 3 個實驗

Ordered by decision-value (each resolves whether Framing B is alive):

**#1 — Converged PASCAL Context base, re-run the full protocol (incl. entropy-gate).**
Why first: it removes the **last remaining confound** on the pivotal negative result. The current
Context delta (−0.0035 de-confounded) was measured on an 8-epoch under-trained base (0.198). Train
Context to convergence (~30–50 ep, in-repo, new SAVE_DIR), then re-run baseline no-TTA/flip + agnostic
SFP/DTLR (gen + entropy-gate) no-TTA/flip. Outcome decides the framing: **positive → Framing B alive;
still negative → Framing A confirmed** and SFP/DTLR is honestly downgraded. Cost: ~half day. Highest
information per GPU-hour.

**#2 — ADE20K in-repo rectification base + 4-step protocol.**
Why: the decisive **second non-VOC dataset**; no generalization claim (positive or negative) is
credible on a single non-VOC point. Prepare per §3 checklist (verify class order first — the Context
trap), train an in-repo base, run the fixed 4-step. 150 classes stress-tests the entropy-gate's
class-count invariance the hardest. Cost: ~1 day (training + eval). If #1 and #2 are both negative,
Framing A is locked and the SFP-generalizes claim is dropped for this paper.

**#3 — Diagnostics + flip-TTA consolidation on VOC + Context (low compute, high writing value).**
Why: turns the strongest existing result (flip-TTA) into a defensible table + the "why" figures, and
produces the required diagnostics for §6/§8 regardless of #1/#2. Run: boundary-band error, small/thin-
object mIoU, false-positive class count, and the before/after logit + entropy heatmaps (debug export
already exists). Also fills Ablation 1 (component build-up) — all test-time, no training. This is the
safe parallel track that de-risks the paper even if #1/#2 kill Framing B.

Deprioritized until #1/#2 resolve: Cityscapes + COCO-Stuff bases (expensive; only needed to *broaden*
a claim that must first be shown to exist on ADE).

---

## 12. Mapping to the journal writing structure

| Journal section (`JOURNAL_EXTENSION_PLAN.md:167`) | Fed by |
|---|---|
| 1 Introduction | C1/C2 (§1); the "what transfers" thesis (§0) |
| 2 Related work | CLIP dense prediction, training-free OVSS/C-USS, logit refinement, reliability-guided |
| 3 Method | ReCLIP++ formulation; URD/SFP; proxy purification; dataset-agnostic SFP/DTLR + entropy-gate (C1); optional components separated |
| 4 Experiments | Tables 1–4 (§4), Ablations 1–4 (§5), Runtime R1 (§7), Qualitative Q1–Q4 (§6) |
| 5 Discussion | what generalizes (flip-TTA), what is VOC-specific + removed (multi-scale, legacy SFP profile), limitations (§8, §9) |
| 6 Conclusion | C1/C2 confirmed; C3 conditional on Top-3; LGAK future work (§10) |

---

## 13. One-line summary

The journal's safe spine is a **rigorous dataset-agnostic generalization study**: reproduced
baselines + a de-VOC protocol + an entropy-normalized reliability gate, showing **parameter-free
flip-TTA transfers while VOC-tuned logit purification does not** (an honest negative). Whether it can
additionally claim **SFP/DTLR generalizes** hinges entirely on Top-3 #1 (converged Context) and #2
(ADE20K). Run those before committing the thesis. LGAK stays future work.
