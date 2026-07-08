# Generalization Protocol — formal-claimable vs exploratory

Written 2026-07-08. Purpose: draw a hard line between **dataset-agnostic results we
can formally claim** and **VOC-tuned findings that are exploratory only**. Any
hyperparameter chosen by looking at VOC val is exploratory. A formal claim requires:
fix every hyperparameter under a dataset-agnostic definition, then evaluate across
datasets with **no per-dataset tuning**.

---

## 1. TTA policy

| Setting | Status | Reason |
|---|---|---|
| no-TTA | **formal** | parameter-free |
| horizontal-flip TTA | **formal** | parameter-free, symmetric, dataset-agnostic |
| multi-scale (any scale set) | **exploratory only** | scale set is a knob |
| ms = 1.25 specifically | **exploratory only, never a main result** | 1.25 was selected on VOC val |

Multi-scale may appear only as a labeled exploratory ablation. The formal main number
per dataset is baseline-or-method under **no-TTA and flip-TTA**.

## 2. SFP+DTLR dataset-agnostic definition ("gen")

Absolute, VOC-tuned constants are replaced by ratios / normalized quantities so the same
config transfers to any dataset unchanged:

| Knob | VOC-tuned (exploratory) | Dataset-agnostic (formal) |
|---|---|---|
| top-k tokens | absolute `TOPK` | `TOP_FRACTION` = fraction of token count |
| spatial sigma | absolute pixels | `DTLR_SIGMA_S_REL` = grid-relative ratio |
| structural classes | VOC class indices | `DTLR_STRUCTURE_CLASSES = []` (none hand-picked) |
| logit overshoot | `DTLR_BETA` / `PROXY_LAMBDA` > 1 | `DTLR_BETA = 1.0`, `PROXY_LAMBDA = 1.0` (no overshoot) |
| prune thresholds | absolute | class-count / entropy / margin normalized |

**Fixed agnostic values used on every dataset (no re-tuning):**
`TOP_FRACTION 0.75, DTLR_SIGMA_S_REL 2.3, DTLR_BETA 1.0, PROXY_LAMBDA 1.0,
DTLR_STRUCTURE_CLASSES []`. These are frozen before cross-dataset evaluation.

**De-VOC audit caveat (2026-07-08, code-verified over all 662 lines of
`model/model_sfp_dtlr.py`).** The five items above fully close the project's De-VOC
checklist. But four gates remain **absolute VOC-inherited constants with no ratio-based
code path**: `CONF_THD 0.97`, `CONF_SCALE 10.0`, `PROXY_CONF_THD 0.95`, `PROXY_KERNEL 5`
(plus an always-on 3×3 CP-SFP kernel with no config key). Max-softmax confidence is
mechanically lower on a 59-class dataset than a 20-class one, so the same `0.97` cutoff
flags a systematically different fraction of tokens as "unreliable" on Context vs VOC —
a class-count operating-point shift unrelated to genuine per-image reliability. This
**confounds the cross-dataset comparison** and cannot be fixed in yaml (no `_REL`/percentile
sibling exists). A fully dataset-agnostic version requires a **code change**: an
entropy-normalized confidence gate (`H_norm = entropy(prob)/log(C)`, `C = num classes`,
already runtime-available) and grid-relative kernel sizing. Until then, Context SFP/DTLR
numbers are reported with this caveat and the confound is listed in limitations; the four
constants are written explicitly in `config/context_test_sfp_dtlr_gen_cfg.yaml` so the
inheritance is auditable rather than silent.

**RESOLVED (partial, 2026-07-08).** The entropy-normalized confidence gate was implemented
(`ENTROPY_GATE`, `model/model_sfp_dtlr.py`; see §6). It replaces the `CONF_THD` /
`PROXY_CONF_THD` max-prob gates with `H_norm = entropy(softmax(logits·CONF_SCALE))/log(C)`,
taus frozen from VOC C=20 and applied unchanged to all datasets. It removed the class-count
confound (recovered +0.0016 / +0.0018 on Context, no-op on VOC: 0.8579 vs 0.8582). The
kernel items (`PROXY_KERNEL 5`, the always-on 3×3 CP-SFP kernel) were left absolute — after
the confidence-gate fix Context SFP was still negative, so SFP/DTLR was downgraded (§6) and
the kernel de-VOC was not pursued.

## 3. Method A (trainable soft presence-calibration head)

Formal claim requires: fix the head's training recipe (init = identity, LR, epochs)
once, train the per-dataset rectification, evaluate — **no VOC-only tuning of the head
hyperparameters**. VOC identity-init verification (0.8536 exact) is a sanity check, not a
result.

## 4. Cross-dataset evaluation

Datasets in scope: VOC, PASCAL Context (59), ADE20K (150), COCO-Stuff (171),
Cityscapes (19). Each requires, in **this repo's architecture**:
1. a compatible rectification checkpoint (trained here — see §6 caveat),
2. text embeddings (`text/<set>_ViT16_clip_text.pth`),
3. a config carrying the agnostic SFP+DTLR block.

Report, per dataset, `delta = method − baseline` under the identical fixed protocol
(no-TTA and flip-TTA columns).

## 5. Result classification

**Formal-claimable**
- Baseline no-TTA / flip-TTA per dataset.
- Agnostic SFP+DTLR (gen settings above) delta per dataset.
- Method A delta with fixed hyperparameters, measured across ≥2 datasets.

**Exploratory only — NOT a formal claim**
- Any multi-scale result (ms = 1.25 and all others).
- Any SFP / DTLR setting selected by looking at VOC val.
- Any VOC-val threshold or CLASS_GATE sweep (e.g. the LOG_BIAS_SCALE hand-sweep that
  collapsed to 0.097).
- Any single-dataset (VOC-only) improvement presented as a general method.

## 6. Status ledger (2026-07-08)

- **VOC**: all test-time-refinement / TTA numbers gathered so far are VOC-val and are
  reclassified **exploratory** per §5 (see `docs/method_results.csv`). Baseline
  no-TTA / flip-TTA on VOC remain formal.
- **PASCAL Context**: DONE (`experiments/context_vanilla_run2/`, 8-ep base, PD 0.85).
  Formal full-val: baseline no-TTA **0.1980**, baseline flip **0.2028 (+0.0048)**,
  agnostic SFP+DTLR no-TTA **0.1929 (−0.0051)**, SFP+DTLR flip **0.1955**. First non-VOC
  datapoint: **flip-TTA generalizes** (+0.0048, cf. VOC +0.0065); **agnostic SFP/DTLR does
  NOT** (−0.0051, looks VOC-specific). Two confounds keep this from cleanly refuting the
  SFP concept: under-trained base (0.198) and the VOC-calibrated CONF_THD/kernel gates
  (see caveat in §2 — a fully-de-VOC'd entropy-gate SFP was not tested). Note: the shipped
  Context labels are in **VOC-20-first class order**, not the alphabetical
  `pascal_context_classes` order — text embeddings + pseudo had to be reordered to GT
  order before any of this trained (see `AUTONOMOUS_SESSION_2026-07-08.md`).
  - **PASCAL Context — entropy-gate de-confound (2026-07-08):** the §2 CONF_THD confound was
    fixed *in code*, not yaml. An entropy-normalized reliability gate
    `H_norm = entropy(softmax(logits·CONF_SCALE)) / log(C) ∈ [0,1]` (`ENTROPY_GATE`,
    `model/model_sfp_dtlr.py`) replaces the absolute max-prob `CONF_THD`/`PROXY_CONF_THD`
    gates; the two taus are frozen from the VOC C=20 operating point (normalized-entropy
    equivalents of 0.97/0.95) and applied **unchanged** to every dataset (no Context-val
    tuning). Result: **the confound was real but explains only ~1/3 of the gap.** VOC
    preserved — gated SFP **0.8579** vs non-gated 0.8582 (−0.0003), a near-no-op on 20
    classes, confirming the fix is genuinely dataset-agnostic. Context recovered part of the
    loss but stays negative: gated no-TTA **0.1945** (+0.0016 vs non-gated 0.1929, still
    −0.0035 vs baseline 0.1980); gated flip **0.1973** (+0.0018 vs 0.1955, still −0.0055 vs
    baseline flip 0.2028). Per the pre-registered Decision Rule (JOURNAL_EXTENSION_PLAN
    req 5), corrected Context SFP remains a negative delta →
    **SFP/DTLR is downgraded to VOC-effective-but-not-generalizable.** Caveat: only the
    confidence-gate confound was removed; the under-trained-base confound (0.198, 8-ep) was
    deliberately left untested (no longer Context training, per user's de-confound-only
    scope). **flip-TTA remains the sole clean dataset-agnostic positive.** Configs:
    `config/{voc,context}_test_sfp_dtlr_entgate_cfg.yaml`.
- **ADE20K / COCO-Stuff / Cityscapes**: raw images + labels are local at
  `D:\ReCLIPv3\datasets`, but the per-dataset checkpoints in `D:\ReCLIPv3\experiments`
  are a **different (fusion-modified) architecture** (`clip.visual.*` + `fuse_alpha`),
  incompatible with this repo's vanilla model. Formal cross-dataset claims on these
  require in-repo rectification training first.

## 7. One-line rule

If a number depended on VOC val to pick a knob, it is exploratory. Everything formal must
survive being frozen and shipped unchanged to a dataset it never saw.
