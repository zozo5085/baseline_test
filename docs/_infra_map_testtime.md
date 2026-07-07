# Test-time Infra Map (2026-07-08, for autonomous inference-only experiments)

Persisted from an Explore pass (~55 files scanned). Baseline formal = **0.8536**.

## Formal test
- Cmd: `python tools/test.py --cfg <cfg> --model RECLIPPP --model_module <module>`
  (parser `tools/test.py:26-48`; defaults cfg=`voc_test_ori_cfg.yaml`, module=`model.model`).
- mIoU **printed to stdout only** (`tools/test.py:209-212`, `the mIOU:%.4lf`). No metrics file — redirect stdout to a log and grep `the mIOU:`.
- Saves per-image `.pt` to `cfg.SAVE_DIR` (test.py:204) like train.py → **each test needs fresh SAVE_DIR**.
- VOC val = **1449** images (`ImageSets/Segmentation/val.txt`). Print says "1448/1449" (off-by-one in idx), all 1449 processed.
- Baseline ref cmd: `tools/test.py --cfg config/voc_test_official854_cfg.yaml --model RECLIPPP --model_module model.model` → **0.8536**.

## Why per-epoch val is ~0.04 below formal (IMPORTANT)
- `tools/test.py:186-197` applies **TEST.PD** = hard whole-image class-presence prune (low-confidence classes suppressed). train.py per-epoch val (`train.py:190-196`) does NOT. Same ckpt → val 0.8039 vs formal 0.8442. So the gap is a **systematic eval-methodology difference**, not noise → per-epoch val trend is a valid signal, and +~0.04 projection to formal is reasonable.
- Baseline eval already contains a HARD class gate (TEST.PD). Any new presence/class gate stacks on top of it.

## SFP+DTLR (current best test-time, 0 params, inference-only)
- `model/model_sfp_dtlr.py`: Stage1 SFP purify (`sfp_logit_purify` :306-511) + Stage2 DTLR edge-preserving refine (`sfp_domain_transform_logit_refine` :513-635, parameter-free `DomainTransformRecursiveFilter` :50-157).
- Configs: `voc_test_sfp_dtlr_official_cfg.yaml` → **0.8590**; `voc_test_sfp_dtlr_gen_official_cfg.yaml` (TOP_FRACTION 0.75, DTLR_SIGMA_S_REL 2.3, DTLR_BETA 1.0) → **0.8582**. Tunables `MODEL.SFP_DTLR` (`configs.py:46-61`). LOAD_PATH = official ckpt.

## TTA — OFF, not wired (opportunity)
- `val_preprocess` (`utils/preprocess.py:279-336`) = single deterministic resize by `DATASET.SCALE` only. No flip, no multi-scale.
- flip/ratio jitter exist ONLY in train augmenter `preprocess()` (`utils/preprocess.py:130-276`), never called at eval.
- `cfg.TEST` (configs.py:87-90) has only BATCH_SIZE/PD/ReCLIP_PD — no scale/flip keys. TTA = new code.

## Presence / class gating (test-time knobs)
- `model_presence.py` gate (`:67-81`): `add = tanh(gamma)*scale*log(sigmoid((⟨z_global,text⟩-tau)*temp))` added to output_q. 4 params (tau/temp/scale/gamma) have **NO config keys** — set training-free only via edited state_dict + LOAD_PATH (test.py loads non-model.model with strict=False). identity (gamma=0) = 0.8536.
- **Soft `CLASS_GATE`** in `model_feature_fusion.py:565-573`: `class_gate = sigmoid((⟨z_global,text⟩-THRESHOLD)*TEMP)`, added as `LOG_BIAS_SCALE*log(...)` to output_q. Config floats `MODEL.CLASS_GATE` (configs.py:40-44). Independent of FEATURE_FUSION.ENABLE (:550 vs :566). Test-time toggle: `--class_gate --fusion_mode l12_only`. **Trainable-free, config-swept.**
- Hard-gate diagnostic (`tools/analyze_bias_intervention.py:206-221`, `docs/diagnostics/bias_hardgate_presence.md`): closed negative, best cell 0.8535 ≈ ties baseline.

## Datasets on disk
- `data/` = only `VOCdevkit/VOC2012` (17125 JPEGs). No Context/COCO-Stuff/ADE/Cityscapes images (only pseudo-label JSONs in `text/`). **Cross-dataset eval blocked without download.**

## Planned safe experiments (inference-only, frozen baseline, new SAVE_DIR)
1. Test-time soft CLASS_GATE sweep (config-only; l12_only baseline-equiv + gate) — test-time analog of Method A, probes oracle +0.046 FP-removal headroom.
2. TTA flip (+ multi-scale if model supports variable resolution) via new `tools/test_tta.py`.
3. Stack winner(s) on SFP+DTLR 0.8590.
