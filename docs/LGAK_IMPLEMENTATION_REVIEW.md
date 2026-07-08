# LGAK Implementation Design Review

Review date: 2026-07-08. Reviewer: Claude (Opus 4.8), design-review only — **no code was
written or trained**. Sources reviewed: `docs/NEW_DIRECTION_LGAK_RESEARCH_PLAN.md`,
`C:/Users/NUTC2507/Documents/碩論/CLAUDE_LGAK_IMPLEMENTATION_PROMPT.md`, and the real
forward path of `model/model.py` (class `RECLIPPP`, verified line-by-line).

---

## 0. TL;DR verdict

**The plan is fundamentally sound and implementable.** The assumed insertion point exists
almost exactly as described, the feature is already in a depthwise-conv-ready spatial layout,
there is no oracle leak, dtype is clean, and there is a single forward path with existing
freeze/load precedents. The biggest implementation worries were **de-risked** by inspecting
the real code.

It **was** not ready to implement verbatim: two correctness issues (F1, F2), one bootstrap
issue (F4), one design fork (F3), and one maintenance caveat (F5). **All are now resolved**
(§10, decided 2026-07-08) — none were blockers; the concrete fixes are folded into the spec
below (§8–§9).

| # | Finding | Severity | Action |
|---|---|---|---|
| F1 | `feat` is reused in the decoder concat (`model.py:479`), not only in `output_q` — the plan's pseudocode misses this | **major (correctness)** | Decide scope; recommend refine feat for `output_q` only |
| F2 | LGAK inserts **after** unit-normalization (`:463`); `F + α·g·F_refine` breaks unit norm at α>0 → uncalibrated similarity | **major (correctness)** | Re-normalize after LGAK |
| F3 | `g = MLP(mean(T))` conditions on a **dataset-global** text mean — same for every image/pixel; very weak "language guidance" | design/novelty | Your decision (§4) |
| F4 | `α = 0` init gives exact identity but **zero gradient to the convs**; a 2–3 epoch run may under-train, not disprove | risk (accepted) | Keep α=0 (user decision); monitor α, warmup only if it stalls |
| F5 | LGAK inserts mid-forward, so `model_lgak.py` must **copy** the forward body (can't just append a head like `model_presence.py`) | maintenance | Mark the copied region; accept drift risk |
| — | Runtime overhead | non-issue | ~3% for MVP (§6) |

**Recommendation:** all four design forks are now resolved (§10, decided 2026-07-08). Apply
F2 + F1 as specified, keep α=0 (F4), global-mean gate (F3); the MVP is a clean, low-risk probe.
The spec is complete — implementation is gated only on the user's final go-ahead.

---

## 1. Forward-path validation — PASS (with one correction)

The plan's assumed insertion (`plan:77-86`, `prompt:36-43`) matches the real code at
`model/model.py:451-482`:

```python
462:  feat = self.proj(v)
463:  feat = feat / feat.norm(dim=1, keepdim=True)          # <-- unit-normalized here
      # ---- proposed LGAK insertion point ----
468:  output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None])...   # 1x1 conv = cosine sim
472:  prompt = self.text_encoder(cls_name_token); prompt = prompt / prompt.norm()
476:  bias_logits = pe @ prompt.t()
477:  output = torch.sub(output_q, bias_logits)...
479:  feature = torch.cat((feat, output), dim=1)            # <-- feat used AGAIN, plan misses this
480-482: decoder_conv2 / decoder_norm2 → returned prediction
```

Confirmed facts (all file:line-verified):
- **Layout is spatial `[B, C, H, W]`**, `C = 512` (`cfg.MODEL.TEXT_CHANNEL`), established at
  `model.py:200` (`reshape(B,h,w,-1).permute(0,3,1,2)`) and reconfirmed by the `dim=1`
  normalize at `:463`. `H, W` are in scope as `shape[0], shape[1]` (`:459`). **DWConv/PWConv
  run directly on `feat` — no reshape needed.** This removes the plan's single biggest risk.
- **Text is the full class set**, not per-image: `gt_cls_text_embeddings = zeroshot_weights`
  `[cnum, 512]` (`:455`, `:453`), loaded once (`tools/test.py:125`) and passed unchanged to
  every call. `mean(T)` = mean over the `cnum` class axis → `[512]`.
- **No oracle leak.** `gt_cls` (the real GT-derived list) is read only inside `if training:`
  (`:487-498`); the main compute path `:453-482` never touches it, and eval passes
  `gt_cls=[]` (`test.py:157`). The name `gt_cls_text_embeddings` is a **naming trap** — it is
  NOT derived from `gt_cls`; it is the static `zeroshot_weights`. Conditioning LGAK on it is
  safe. **Do not wire in the real `gt_cls`.**
- **dtype = fp32, no autocast** (repo-wide grep clean; `proj.weight` is fp32 at `:514`). The
  LGAK module in fp32 is consistent — no half/float mismatch.
- **One forward path**, reused by `test.py:184`, `train.py:162/190`, `test_tta.py:93`. No
  sliding-window inference for `RECLIPPP`. A single insertion at `:463→:468` covers train,
  eval, and TTA.

**Correction to the plan:** the pseudocode stops at `output = output_q - bias_logits`. The
real forward continues — `feat` is concatenated into the decoder at `:479`. See F1.

---

## 2. Required correctness fixes (before any training)

### F1 — `feat` feeds the decoder too, not just similarity (major)

`model.py:479` does `feature = torch.cat((feat, output), dim=1)` and runs a decoder head
(`decoder_conv2/norm2`). So the projected `feat` has **two consumers**: the similarity conv
(`:468`) and the decoder concat (`:479`). If LGAK rebinds `feat` in place, refined features
flow into **both** — which is broader than the "before similarity" claim and feeds the
decoder an input distribution it was never trained on (baseline decoder saw un-refined feat).

**Recommended (MVP):** refine into a **new variable** used only by the similarity conv, leave
the decoder concat on the original `feat`:
```python
feat_refined = lgak(feat, text)          # new tensor
output_q = F.conv2d(feat_refined, ...)   # similarity uses refined
...
feature = torch.cat((feat, output), 1)   # decoder keeps ORIGINAL feat
```
This isolates the "pre-similarity feature formation" effect (matching the paper claim,
`plan:148-149`) and preserves the decoder's training distribution. Identity at α=0 holds
either way (`feat_refined == feat`). If you instead *want* refined feat everywhere, that is a
legitimate variant — but it should be a **separate ablation row**, not the MVP default.

### F2 — re-normalize after LGAK, or similarity/calibration drift (major)

`feat` is unit-normalized at `:463`, so `output_q = conv2d(feat, unit_text)` is a genuine
**cosine similarity**, and the `bias_logits` subtraction (`:476-477`) is calibrated to that
unit scale. The MVP residual `F_out = F + α·g·F_refine` is **not unit-norm for α>0**, so:
- `output_q` silently becomes a scaled/uncalibrated dot product,
- the fixed `bias_logits` no longer matches the logit scale,
- any short-run degradation may come from **norm/calibration drift, not from the method**.

That last point is exactly the confound class we just spent a session removing on SFP/DTLR —
do not reintroduce it. **Fix:** re-normalize after the residual:
```python
F_out = (F + alpha * g * F_refine)
F_out = F_out / F_out.norm(dim=1, keepdim=True)
```
At α=0, `normalize(F) == F` (already unit) → **exact identity preserved**. At α>0, `output_q`
stays a true cosine and calibration is intact. This is the single most important fix.

---

## 3. The α=0 bootstrap (F4) — user decision: keep α=0

`α = nn.Parameter(torch.zeros(1))` gives exact identity, but at init the gradient to the
DWConv / PWConv / MLP params is `∂L/∂F_out · (α·…) = 0` — **only α itself receives gradient**
(its own grad `∝ g·F_refine ≠ 0`). So α drifts first and *then* unlocks the convs: a slow
start. In a 2–3 epoch short run this risks a **flat result that is really under-training**,
not absence of signal (cf. the Method-A γ=0 slow-start).

**Decision (2026-07-08): keep α=0 init in the first version — do NOT hand-set a non-zero α.**
- **Identity config** (`voc_test_lgak_identity_cfg.yaml`): `α=0` **and frozen**
  (`requires_grad=False`) → must reproduce baseline exactly (0.8536).
- **Training config**: `α=0` init but **learnable** (`requires_grad=True`); let it move on its
  own — no manual warm value.
- **In-scope mitigations:** F2 re-normalize keeps the α-unlock transition bounded (no jolt when
  α first leaves 0); **log the α trajectory** every smoke/short-run step.
- **Contingency (only if needed):** if α stays ≈0 through the short run, add a small-α warmup in
  a *second* iteration — not the first. And read a flat short-run as *possibly under-trained*,
  not a clean negative, before deciding.

---

## 4. Text-conditioning strength (F3) — user decision: A (global mean)

The MVP's `g = MLP(mean(T))` reduces the class vocabulary to **one dataset-global vector**,
identical for every image and pixel — a per-channel gate scaled by a dataset constant. The
"text-gated > image-only" proof (`plan:148`) may therefore come out near zero at MVP.

**Decision (2026-07-08): A — global-mean text gate for the first version.** Rationale (user):
the MVP's job is to confirm a language-conditioned refinement **does not break the baseline**
and shows any signal; the image-level present-class gate (option B) would introduce an oracle /
class-set dependence that risks making the method dataset-specific — deferred. Consequence to
keep in mind: a wash on "text-gated vs image-only" at MVP is *expected* and is not by itself a
refutation; stronger conditioning is a later variant.

### Success / failure thresholds (user-pre-registered, 2026-07-08)

Ordered gates — do **not** use VOC multi-scale or any VOC-tuned TTA as a success signal:
1. **Identity**: formal test with `α=0` **must equal baseline** (0.8536). Hard gate — if not,
   stop and debug, do not train.
2. **VOC no-TTA floor**: short-run formal mIoU **≥ baseline − 0.005**. Below that for two
   consecutive checkpoints → stop.
3. **Positive VOC → Context**: only if VOC shows a **positive** no-TTA delta, replicate on
   PASCAL Context to test generalization (same fixed protocol as the SFP work).
4. **Weak-signal branch**: if VOC mIoU is flat but **boundary / small-object** metrics improve,
   record as a *weak signal* — do **not** launch a long run on it.

---

## 5. Maintenance caveat (F5)

`model_presence.py:84-103` is the freeze-then-append precedent (subclass baseline →
`super().__init__()` → `load_state_dict(strict=False)` → freeze all → create head *after*
freeze so it stays trainable; `train.py:133` then optimizes `filter(requires_grad, …)`). LGAK
can reuse the freeze/load half of this pattern verbatim.

But `model_presence` hooks **after** `output_q` (`:119-124`), whereas LGAK must inject
**between** `:463` and `:468`. No existing module inserts pre-`output_q` on `feat`, so
`model_lgak.py` **cannot** just append a head — it must **override `forward` and copy the
`:453-482` compute body**, inserting LGAK mid-stream. This forks the forward from `model.py`
(drift risk if the baseline changes). Acceptable for a research branch; keep the copied region
minimal, clearly commented as "verbatim copy of model.py:453-482 + LGAK", and do **not** edit
`model.py` itself (hard-constraint 5 / prompt-constraint 6).

---

## 6. Runtime cost (the requested criterion)

MVP is layer-12-only, `feat = [B, 512, H, W]`. Token grid for VOC is ~21²–32² depending on
input scale; take H·W ≈ 1024 as an upper bound.
- DWConv 3×3, 512ch: `512·9·HW ≈ 4.7M` MAC
- PWConv 512→512: `512·512·HW ≈ 268M` MAC
- MLP(mean T): `~0.26M` MAC (negligible)
- **Total ≈ 0.27 GMAC ≈ 0.55 GFLOP per forward.**

Against a CLIP ViT-B/16 image encoder (~17.5 GFLOP, run once per forward), LGAK-MVP adds
**~3%** compute and one `[B,512,H,W]` activation (~2 MB fp32) — negligible latency/VRAM.
Multi-layer variants (L6+9+12, `plan:126-127`) scale to ~9% and require inserting inside the
ViT (far more invasive) — out of MVP scope. **Runtime is not a concern for the MVP.**
(Estimate assumes the ~1024-token grid; confirm the real H·W in the first smoke run.)

---

## 7. Hard-constraint compliance

| Constraint (plan §Hard / prompt §Hard) | Status | Note |
|---|---|---|
| Freeze CLIP image + text encoder | ✓ by design | reuse `model_presence.py:101-102` freeze |
| No distillation / no pixel-GT-mask loss | ✓ | reuse existing ReCLIP++ pseudo-label training loss; add **no new loss** |
| New module zero/identity-init | ✓ | α=0 → identity, **given F2 re-normalize** |
| Disabled path == baseline exactly | ✓ pending test | α=0 **and** decoder keeps original feat (F1) → formal mIoU must == 0.8536 |
| No offset AKConv / no `[B,K,C,H,W]` first | ✓ | MVP is residual DW+PW conv only |
| No destructive edits to `model.py` | ✓ | via copy-forward in `model_lgak.py` (F5) |
| New SAVE_DIR, no overwrite of ckpt/experiments | ✓ | `experiments/voc_lgak_mvp_run1/` |
| No VOC-val tuning as main claim | ✓ process | formal test only; dataset-agnostic; §4 threshold |

The one constraint that needs active care is **"train LGAK only"**: the loss must reach the
LGAK params through `output_q → feat_refined`. It does (F.conv2d is differentiable), but the
implementer must confirm in the smoke train (`prompt:117-122`) that **grad is non-zero on LGAK
params and exactly `None`/zero on all frozen baseline params**, and log the trainable-param
count (expected: only DWConv + PWConv + MLP + α).

---

## 8. Disambiguated insertion recipe (hand this to the implementation session)

A weaker implementing model would misread several underspecified points. Pin them down:

1. **Insert after `model.py:463`**, before `:468`. Produce `feat_refined`; feed it to
   `output_q` only (F1). Decoder concat at `:479` keeps original `feat`.
2. **Re-normalize** `feat_refined` (F2): `F_out = normalize(F + α·g·F_refine, dim=1)`.
3. **`mean(T)`** = `zeroshot_weights.mean(dim=0)` → `[512]`; `g = MLP(mean_T)` shaped to
   `[1, 512, 1, 1]` (per-channel gate). Use `gt_cls_text_embeddings` (= full class set), **not**
   the real `gt_cls`.
4. **α**: `nn.Parameter(torch.zeros(1))`. Identity config: `α=0`, **frozen** (`requires_grad
   =False`). Training config: `α=0` init, **learnable** — no manual non-zero value (F4).
   DWConv/PWConv standard init (inert at α=0). Log the α trajectory.
5. **Module**: `class TextGatedConvRefiner(nn.Module)` in `model/lgak.py`;
   `forward(image_feat[B,512,H,W], text_feat[cnum,512]) -> [B,512,H,W]`.
6. **Wrapper**: `model/model_lgak.py`, subclass `RECLIPPP`, freeze-then-append per
   `model_presence.py:100-103`, override `forward` copying `model.py:453-482` with the LGAK
   line injected. `--model_module model.model_lgak` → auto `strict=False` load (`test.py:136`).
7. **Configs**: `voc_train_lgak_mvp_cfg.yaml`, `voc_test_lgak_mvp_cfg.yaml`,
   `voc_test_lgak_identity_cfg.yaml`. New SAVE_DIR `experiments/voc_lgak_mvp_run1/`.
8. **"Layer 9/12" in the plan** ≠ multiple insertions for the MVP — the single `:463`
   insertion IS the top (post-proj) feature. Multi-layer variants are deferred and invasive.

---

## 9. Revised implementation sequence (still gated on your go-ahead)

0. Baseline formal test → confirm 0.8536 (`test.py`/`test_tta.py`, no-TTA).
1. Build `model/lgak.py` + `model/model_lgak.py` + 3 configs. **F2 re-normalize + F1 decoder
   split baked in.**
2. **Identity check**: `voc_test_lgak_identity_cfg.yaml` (α=0) formal mIoU **must == 0.8536**.
   If not, stop and debug — do not train.
3. **Smoke train** (2 iters): finite loss; grad only on LGAK params; log param count, α,
   feature norm before/after; assert no norm explosion (F2 makes this stable by construction).
4. **Short run** (2–3 ep, or longer per F4): formal test each epoch (not train.py val). Stop
   if formal mIoU < baseline − 0.005 for two consecutive checkpoints.
5. **Decide** against the §4 pre-registered positive threshold, then the ablation ladder
   (image-only vs text-gated; before-vs-after similarity), Context replication before any
   generalization claim.
6. Log to `research_notes.md`, `method_results.csv`, `updated.md`, `SESSION_HANDOFF.md`.

---

## 10. Decisions — RESOLVED (2026-07-08)

All four forks were decided by the user; this doc is now the authoritative MVP spec.

1. **F3 conditioning → A.** Global-mean text gate for v1; present-class/pseudo conditioning
   deferred (avoids oracle / class-set dependence). See §4.
2. **F1 scope → A.** Refined feature feeds `output_q` **only**; decoder concat (`:479`) keeps
   the original `feat`. No decoder-input or decoder-architecture change. See §2/F1.
3. **Success threshold → pre-registered** (§4): identity==baseline; VOC no-TTA ≥ baseline−0.005;
   positive VOC → Context replication; flat-VOC-but-boundary/small-object = weak signal (no long
   run). No VOC multi-scale / VOC-tuned TTA as a success signal.
4. **F4 α → keep α=0.** Identity config α=0 frozen; train config α=0 but learnable, no manual
   non-zero init; small-α warmup only as a later contingency if α stalls. See §3.

**Applied as defaults** (correctness/mechanics, not taste): **F2 re-normalize after LGAK**,
**F5 copy-forward in `model_lgak.py`** (no edit to `model.py`).

**Status: awaiting the user's final go-ahead to start writing LGAK code.** No code written yet.

---

## 11. Implementation status (2026-07-08) — code written, identity + smoke PASS

User gave the go-ahead to implement the MVP (constraints: MVP only; new files only; no
destructive edit to `model/model.py`; no overwrite of experiments; identity → smoke → report,
no short run yet).

**Files added** (no edit to `model/model.py`):
- `model/lgak.py` — `TextGatedConvRefiner`: `F_out = normalize(F + α·g·PWConv(DWConv(F)))`,
  `g = 1 + MLP(mean_c(T))` (dataset-global gate, F3=A), α zero-init. Eval fast-path returns F
  exactly when α==0 (bit-exact identity); training always runs the re-normalized path (F2).
- `model/model_lgak.py` — `RECLIPPP(_BaseRECLIPPP)`: freeze-then-append; copy of
  `model.py:451-511` forward with LGAK feeding `output_q` ONLY and the decoder concat keeping
  the original `feat` (F1=A).
- `config/voc_train_lgak_mvp_cfg.yaml`, `config/voc_test_lgak_mvp_cfg.yaml`,
  `config/voc_test_lgak_identity_cfg.yaml`; `config/configs.py` registers `MODEL.LGAK.*`.
- `tools/smoke_test_lgak.py`.

**Identity check — PASS.** Full-val VOC, `model.model_lgak` with α=0:
`the mIOU: 0.8536` == baseline `model.model` `0.8536` on the same config (exact). 7 missing keys
= the fresh LGAK params (strict=False). Evidence: `experiments/lgak_id_lgak/`,
`experiments/lgak_id_baseline/`.

**Smoke train (2 iters) — PASS.** `tools/smoke_test_lgak.py`:
- Only `lgak.*` trainable (398,465 params); baseline fully frozen — no gradient leaked.
- Finite loss (1.49 → 2.00); refined-feature norm `out = 1.0000` (F2 holds).
- **F4 bootstrap confirmed empirically:** iter 0 `α 0 → −6.2e-5` (α.grad 0.0062, conv+mlp grad
  **0.0**); iter 1 (α≠0) conv+mlp grad **1.9e-5** — α unlocks the convs exactly as predicted.

Every §1–§5 prediction held: identity exact (F1/F2), grad only on LGAK, F4 slow start observed.

**Next (gated on user go):** the 2–3 epoch short run per §9, evaluated by the §4 pre-registered
thresholds (formal no-TTA, no VOC-tuned TTA). **Not started.**
