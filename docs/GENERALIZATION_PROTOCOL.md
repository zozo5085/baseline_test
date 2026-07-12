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

## 6b. Ledger update (2026-07-09) — second confound removed, downgrade final

The under-trained-base confound was removed: Context base retrained to convergence
(`experiments/context_vanilla_converged/`, EPOCH 30, best ep17; recipe unchanged, only
epochs). Formal full-val (PD 0.85): baseline no-TTA **0.2412**, flip **0.2473 (+0.0061)**,
agnostic SFP gen **0.2353 (−0.0059)**, entropy-gate **0.2367 (−0.0045)**. Both confounds
now removed and SFP/DTLR is still (slightly more) negative → the §6 downgrade is **final**:
VOC-effective, not generalizable. Component ablation (Table IV) localizes the harm to the
CP-SFP neighbourhood rewrite; selection+DTLR-only scored **+0.0010** on Context — a
**post-hoc observation** that motivates §8 below but is not claimable without it.

## 7. One-line rule

If a number depended on VOC val to pick a knob, it is exploratory. Everything formal must
survive being frozen and shipped unchanged to a dataset it never saw.

## 8. PRE-REGISTERED: ADE20K DTLR-only confirmation (written 2026-07-09, BEFORE any ADE number exists)

No ADE20K evaluation of any kind has been run in this repo at the time of writing; this
section's git commit timestamp is the pre-registration evidence. Nothing below may be
changed after the first ADE number is produced.

### 8.1 Hypothesis

**H1 (primary).** On a converged in-repo ADE20K base, *selection + DTLR only*
(gen profile with `CPSFP_UPDATE = False`, i.e. `--sfp_disable cpsfp`) yields a
non-negative formal no-TTA mIoU delta vs. the same base.
Motivation (post-hoc, Table IV): the CP-SFP rewrite is the localized failure source
(all rewrite-containing configs negative on Context), while DTLR-only is positive on
VOC (+0.0042) and marginally positive on Context (+0.0010).

**H2 (secondary).** Flip-TTA yields a positive delta on ADE20K (would make it 3/3
datasets for the sole clean dataset-agnostic operator).

### 8.2 Frozen protocol (no ADE tuning of anything)

- Base: this repo's `tools/train.py`, recipe identical to the Context converged run
  (no method/LR changes; EPOCH 30; best epoch by the same per-epoch val criterion).
  New SAVE_DIR `experiments/ade_vanilla_converged/`.
- Gen profile exactly as §2: `TOP_FRACTION 0.75, DTLR_SIGMA_S_REL 2.3, DTLR_BETA 1.0,
  PROXY_LAMBDA 1.0, DTLR_STRUCTURE_CLASSES []`; entropy-gate taus frozen at 0.0745/0.1154.
- `TEST.PD = 0.85` (the frozen non-VOC default, as used for Context). Sole pre-declared
  fallback: if the **baseline** at PD 0.85 operationally collapses (mIoU < 0.01, the
  Context-PD1.0 failure mode), PD is set to 0.0 (prune disabled) **for all arms equally**;
  this decision is made from the baseline number alone, before any method number is run.
  No other PD value may be used. The endpoint is a *delta*, computed within one PD setting.
- Battery (fixed, formal): base no-TTA / base flip / SFP gen / SFP entgate / DTLR-only
  (each also +flip) + diagnostics (`tools/diag_metrics.py`) + runtime + flagged-fraction
  stats (`tools/sfp_stats_extract.py`). Every eval a new SAVE_DIR.
- Data-integrity gate BEFORE training (the Context lesson): GT class order vs. text
  embedding order vs. pseudo labels verified (top1-in-GT rate sanity); failure blocks
  training until fixed and documented.

### 8.3 Pre-registered endpoints

- **H1 CONFIRMED** iff: ADE DTLR-only no-TTA delta > 0 AND paired per-image bootstrap
  (`tools/bootstrap_significance.py`, 10k resamples, seed 0) one-sided p < 0.05.
  Consequence: "selection + DTLR is a mild dataset-agnostic positive" may be promoted to
  a formal cross-dataset claim (VOC + Context + ADE, with Context's +0.0010 explicitly
  labeled as not individually significant if that is what the bootstrap says).
- **H1 INCONCLUSIVE** iff delta > 0 but p ≥ 0.05: reported as an observation only; no
  formal claim; no further confirmation datasets are added to chase significance.
- **H1 REFUTED** iff delta ≤ 0: DTLR-only line is closed. Recorded as a formal negative.
  No component-level rescue attempts beyond this point.
- **H2**: sign test only (delta > 0 strengthens the flip claim to 3/3; delta ≤ 0 is an
  honest negative that bounds the flip claim to "2 of 3").
- **Anti-rescue clause**: whatever full SFP gen / entgate score on ADE, the §6/§6b
  downgrade is NOT reversed — a positive ADE delta cannot outweigh the demonstrated
  Context failure for a method claimed as dataset-agnostic. Their ADE numbers are audit
  datapoints only.
- All numbers verbatim 4 decimals into `JOURNAL_EXPERIMENT_INDEX.md` + `method_results.csv`
  regardless of outcome.

### 8.4 Pre-run Amendment (2026-07-11, before ADE training/evaluation)

Append-only amendment. Written BEFORE any ADE model has been trained and BEFORE any ADE
segmentation mIoU exists (item 7 below). §8.1 hypotheses, §8.2 battery, and §8.3 success
criteria are **unchanged in full** (item 8).

1. **Data-integrity gate: COMPLETED.** Class order (repo `ade_classes` == objectInfo150,
   0/150 mismatches), GT value range 0..150, text-embedding recipe verified against the
   shipped VOC weights, CLIP top1-in-GT 0.527 (~8x random). Evidence:
   `docs/ADE20K_PSEUDO_COMPARISON.md` + commits `890007b`, `70c24b3`.
2. **Old external pseudo file is DATA LEAKAGE.**
   `D:\ReCLIPPP2026\text\ade_pseudo_label.json` = GT-derived top-5 presence labels
   = invalid for unsupervised experiments. Banned from all formal / unsupervised training.
   (Also: the repo-shipped `text/ade_pseudo_label_ReCLIPPP.json` is order-broken on this
   machine — recall 0.235, shift-invariant — and equally unusable.)
3. **Official training input = locally regenerated `text/ade_pseudo_label.json`**
   (20210 lines, class idx 0..149, 0 empty/parse failures, full-set aligned GT recall
   0.583 as a post-hoc diagnostic; validation table in `docs/ADE20K_PSEUDO_COMPARISON.md`).
4. **Pseudo-label provenance**: generated ONLY from frozen CLIP ViT-B/16 image/text
   features (sliding-window crop voting against `text/ade_ViT16_clip_text.pth`);
   GT annotations were read only AFTER each label list was written, exclusively for
   integrity diagnostics (code audit in `docs/ADE20K_PSEUDO_COMPARISON.md`,
   `tools/ade_pseudo_regen.py:95-104`).
5. **`time.sleep(0.08)` removed from `tools/train.py`** (line 151 of the pre-amendment
   file). Reason: it is a pure per-iteration wall-clock delay (~27 min/epoch at ADE's
   20210 images); it touches no RNG state and no tensor computation.
6. **This change alters NO numerics**: image order, batch size, shuffle policy,
   augmentation, model architecture, loss, optimizer, LR schedule, epoch count,
   checkpoint selection, and evaluation protocol are all untouched.
7. **Timing attestation**: at the moment this amendment is committed, no ADE model has
   been trained in this repo and no ADE segmentation mIoU (of any arm) has been produced.
8. **§8.1 / §8.2 / §8.3 are unchanged** — hypotheses, frozen profile, PD rule, battery,
   endpoints, and the anti-rescue clause all stand exactly as pre-registered.
9. **Random seed fixed = 0** (none was set before this amendment): Python `random`,
   NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`, plus DataLoader determinism
   via `generator=torch.Generator().manual_seed(0)` and a `worker_init_fn` deriving
   per-worker seeds from `torch.initial_seed()` (NUM_WORKERS 4 unchanged). cuDNN
   algorithm-selection flags are NOT changed (forcing deterministic kernels would alter
   the numerics/perf envelope relative to the Context run).
10. **Reproducibility record**:
    - pseudo-label `text/ade_pseudo_label.json` SHA256
      `1255492DD3095B895DF598B126F172707BF79E532E4505C67295549EF2865BB6`
    - config `config/ade_train_converged_cfg.yaml` SHA256
      `88413186D9BEEC554BB4A382323D044E5D178E0EC6947B325158B2536CEA232C`
    - repo HEAD when this amendment was written: `70c24b3`; the training source snapshot
      = the commit CONTAINING this amendment ("ADE20K pre-run protocol amendment and
      reproducible training snapshot"), whose hash is recorded at launch in
      `experiments/ade_vanilla_converged/launch_info.txt` and `docs/SESSION_HANDOFF.md`.
    - environment: conda env `reclip5090`, Python 3.10.19,
      torch 2.11.0.dev20260214+cu128, CUDA 12.8, NVIDIA GeForce RTX 5090 (32 GB),
      Windows 10; launch command recorded in `launch_info.txt`.

#### 8.4 Addendum A (2026-07-11, pre-launch; appended after independent fresh-context verification flagged a commit-range ambiguity — still BEFORE any ADE training or ADE mIoU)

- The pre-run snapshot commit `60c14e9` necessarily bundles, besides this amendment's own
  changes (sleep removal + fixed seeding), the previously-UNCOMMITTED local-training
  infrastructure refactor of `tools/train.py` that had lived in the working tree since
  2026-07-06/07: optional `--distributed` (single-process default), `--model_module`
  dynamic model loading, `shuffle = train_sampler is None` (native shuffle when
  non-distributed), device handling, throttled validation prints.
- Item 6's "untouched" claims are therefore relative to **the code that produced the
  existing baselines** (the working tree), not to the stale last-committed upstream
  `train.py` — which was DDP-only (`shuffle=False` + `DistributedSampler`, `mp.spawn`
  with gloo) and did not even accept the recorded launch commands.
- Evidence the refactor predates every comparison baseline:
  (a) committed handoff text documents the uncommitted `tools/train.py(+86)` diff BEFORE
      the Context converged run (`git log -S "tools/train.py(+86)" -- docs/SESSION_HANDOFF.md`);
  (b) recorded launch commands from 2026-07-06..08 use `--model_module`
      (e.g. `model.model_presence`, `model.model_feature_fusion`) — an argument that does
      not exist in the pre-snapshot committed `train.py`
      (`git show 70c24b3:tools/train.py` contains no `--model_module`), so those runs can
      only have executed the refactored working tree;
  (c) `experiments/context_vanilla_converged/console.log` timestamps
      2026-07-08 21:50 → 2026-07-09 05:09 (single-process run).
- Consequently the ADE run and the Context converged baseline share the IDENTICAL
  sampler / single-process code path; the only code difference between the
  Context-baseline code and the ADE launch code is exactly items 5 + 9 (sleep removal,
  fixed seeding). No cross-arm confound is introduced. Independently, §8.1's H1/H2
  endpoints are within-ADE deltas (every arm on the same base/ckpt/code), so they are
  insensitive to any cross-dataset code question.
- Completeness note (verifier claim-3 caveat): the string `ReCLIPPP2026` appears in
  `tools/ade_pseudo_compare.py`'s docstring solely as the `--old` usage example of the
  ban-verification tool itself; it is not a load path in any training code or config.

### 8.5 OUTCOME (2026-07-12) — endpoints resolved exactly as pre-registered

Base trained (ep24 best, per-epoch val 0.1305, 17h20m, seed 0); battery run per §8.2 on
full val 2000, PD 0.85 (baseline 0.1332 ≥ 0.01 → no fallback, decided from the baseline
alone). All numbers verbatim in `JOURNAL_EXPERIMENT_INDEX.md §12` + `method_results.csv`;
logs in `experiments/journal_logs/ade_*`.

- **H1: REFUTED.** DTLR-only no-TTA 0.1250, delta **−0.0082** vs base 0.1332; paired
  bootstrap (10k, seed 0) CI [−0.0109, −0.0044], p < 0.0002 — significantly negative.
  Per §8.3: the DTLR-only line is **closed**, recorded as a formal negative; Context's
  +0.0010 is confirmed post-hoc noise; **no component-level rescue attempts** follow.
- **H2: sign positive, magnitude ≈ 0.** flip 0.1335, delta +0.0003, CI [−0.0027, +0.0020],
  p = 0.8548. Honest claim going forward: flip-TTA is **non-negative on 3/3** datasets and
  significantly positive on 2/3 (VOC +0.0065, Context +0.0061); the "similar magnitude"
  phrasing must be narrowed to non-negativity — ADE adds a boundary/FP improvement that
  nets to ~zero (diagnostics: bnd 0.7375→0.7340, FP 4.09→3.41, small-obj 0.1420→0.1396).
- **Anti-rescue clause applied:** full SFP gen −0.0100 (p<0.0002) and entgate −0.0080 —
  audit datapoints only; the §6/§6b downgrade stands, now reinforced by a third dataset.
- **New audit finding:** the Context-derived localization ("failure = CP-SFP rewrite;
  DTLR alone harmless") does NOT extend to ADE: DTLR-only (−0.0082) ≈ entgate (−0.0080)
  ≈ full (−0.0100), and DTLR-only's diagnostics are near-identical to full SFP
  (small-obj 0.1321 vs 0.1327). On a 150-class fine-grained label space the entire
  purification family erodes structure; the flagged-mass chain is monotone
  (unrel_ent 0.32→0.68→0.77; proxy support 0.99→0.74→0.50 across VOC→Context→ADE).
