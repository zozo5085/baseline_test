# Journal Experiment Index — 可驗證實驗索引表

Built 2026-07-09. Every journal-relevant number with its provenance chain:
config → log → save_dir/ckpt → commit. Classes: **formal**(可進主表)/ **diagnostic**
(解釋用,不進主結果表)/ **exploratory**(附錄限定)/ **superseded / voided**(不可用)。

Conventions:
- All mIoU verbatim (4 decimals). Eval tool = `tools/test_tta.py` (`--scales 1.0` [+`--flip`]) unless noted.
- `journal_logs/` = `experiments/journal_logs/` (session tmp logs preserved 2026-07-09).
- "log not preserved" = run predates log-preservation practice; evidence = per-image `.pt` in
  save_dir + `docs/method_results.csv` row + recording commit. Re-runnable from config.
- VOC ckpt = `experiments/official/voc_reclippp_854/best_weight.pth`(official 854, protected)。
- Context converged ckpt = `experiments/context_vanilla_converged/best_weight.pth`
  (train cfg `config/context_train_converged_cfg.yaml`, EPOCH 30, best ep17 val 0.2341,
  train log `experiments/context_vanilla_converged/console.log`, commit `357c56f`).

---

## 1. VOC formal core (main Tables I/II/III) — ckpt = official 854

| # | Method | mIoU | class | config | log | save_dir | commit | main table? |
|---|---|---|---|---|---|---|---|---|
| V1 | ReCLIP++ baseline no-TTA | 0.8536 | formal | `config/voc_test_lgak_identity_cfg.yaml`(model.model) | `experiments/lgak_identity_eval/base.log` + `journal_logs/lgak_identity_RUN.log` | `experiments/lgak_id_baseline/`(1449 .pt) | `1fa591a`(本次重現);原始列 `5ebb07b` | ✅ I |
| V2 | baseline + flip-TTA | 0.8601 | formal | same, `--flip` | `journal_logs/voc_flip_regen.log`(本次重現 0.8601) | `experiments/voc_diag_flip/`(1449)+ 原始 `experiments/voc_tta_flip/` | `f6ad5a1`;原始 `5ebb07b` | ✅ II |
| V3 | SFP/DTLR legacy(VOC-tuned) | 0.8590 | formal | `config/voc_test_sfp_dtlr_official_cfg.yaml` | not preserved(pre-index session) | `experiments/voc_sfp_dtlr_official_eval/` | `5ebb07b` | ✅ III |
| V4 | SFP/DTLR dataset-agnostic(gen) | 0.8582 | formal | `config/voc_test_sfp_dtlr_gen_official_cfg.yaml` | not preserved;20-img 迴歸 gate 0.5611 見 `b00110d` message | `experiments/voc_sfp_dtlr_gen_official_eval/` | `5ebb07b` | ✅ I, III |
| V5 | SFP gen + entropy-gate | 0.8579 | formal | `config/voc_test_sfp_dtlr_entgate_cfg.yaml` | `experiments/entgate_eval/voc_gated.log` + `journal_logs/entgate_RUN.log` | `experiments/voc_sfp_entgate_eval/`(1449) | `4c16036` | ✅ III |
| V6 | SFP gen + flip | 0.8639 | formal | gen cfg + `--flip` | not preserved | `experiments/voc_sfpdtlr_tta_flip/` | `5ebb07b` | ✅ II |
| V7 | Method A(trainable presence head) | 0.8565 | formal(secondary) | train `config/voc_train_presence_cfg.yaml` | not preserved | `experiments/voc_presence_run1/`(含 best_weight.pth) | `5ebb07b` | ✅ I secondary — VOC-only,依 protocol 不得做跨資料集主張 |

## 2. Context converged formal (main Tables I/II/III) — ckpt = converged, PD 0.85

All: log 亦見 `journal_logs/conv_eval_RUN.log`;commit `357c56f`;eval `--load_path` 指 converged ckpt。

| # | Method | mIoU | class | config | log | save_dir | main table? |
|---|---|---|---|---|---|---|---|
| C1 | baseline no-TTA | 0.2412 | formal | `config/context_test_local_cfg.yaml` | `experiments/context_conv_eval/base_notta.log` | `experiments/context_conv_eval/base_notta/`(5105) | ✅ I |
| C2 | baseline + flip | 0.2473 | formal | same + `--flip` | `.../base_flip.log` | `.../base_flip/` | ✅ II |
| C3 | SFP/DTLR gen no-TTA | 0.2353 | formal(negative) | `config/context_test_sfp_dtlr_gen_cfg.yaml` | `.../sfp_notta.log` | `.../sfp_notta/` | ✅ I — 誠實負結果,audit 核心 |
| C4 | SFP gen + flip | 0.2383 | formal | same + `--flip` | `.../sfp_flip.log` | `.../sfp_flip/` | ✅ II |
| C5 | SFP + entropy-gate no-TTA | 0.2367 | formal(negative) | `config/context_test_sfp_dtlr_entgate_cfg.yaml` | `.../entgate_notta.log` | `.../entgate_notta/` | ✅ III |
| C6 | SFP + entropy-gate + flip | 0.2400 | formal | same + `--flip` | `.../entgate_flip.log` | `.../entgate_flip/` | ✅ II/III |

## 3. Component ablation (main Table IV) — commit `b00110d`, log `journal_logs/ablate_bench_RUN.log`

Config = gen cfg(§1 V4 / §2 C3 同檔)+ CLI `--sfp_disable {dtlr|proxy|cpsfp}`(開關預設 True,
迴歸 gate:gen 20-img 前後皆 0.5611 精確)。Context 列 + `--load_path` converged。

| # | Configuration | VOC | Context | class | save_dirs | main table? |
|---|---|---|---|---|---|---|
| A1 | Sel.+CP-SFP(−DTLR) | 0.8563 | 0.2345 | formal | `experiments/voc_ablate_nodtlr/`,`experiments/context_ablate_nodtlr/` | ✅ IV |
| A2 | Sel.+CP-SFP w/o proxy+DTLR(−proxy) | 0.8581 | 0.2366 | formal | `.../voc_ablate_noproxy/`,`.../context_ablate_noproxy/` | ✅ IV |
| A3 | Sel.+DTLR(−CP-SFP rewrite) | 0.8578 | 0.2422 | formal(表列)⚠ | `.../voc_ablate_nocpsfp/`,`.../context_ablate_nocpsfp/` | ✅ IV 表列;⚠ Context +0.0010 為 **post-hoc 觀察** — 不得升格 formal 泛化主張(需 ADE 預註冊確認) |

## 4. Runtime bench (main Table V) — commit `b00110d`, log `journal_logs/ablate_bench_RUN.log`

Tool `tools/bench_runtime.py`(RTX 5090,batch 1,fp32,50-img CUDA-synced forward;排除 I/O 與
final resize)。No save_dir(timing only)。全部 test-time-only、0 added trainable params。

| Method | VOC ms/img / FPS / VRAM | Context ms/img / FPS / VRAM | main table? |
|---|---|---|---|
| baseline | 14.7 / 67.98 / 712MiB | 24.3 / 41.20 / 923MiB | ✅ V |
| + flip | 29.1 / 34.39 / 715 | 47.8 / 20.91 / 928 | ✅ V |
| + SFP gen | 20.2 / 49.57 / 714 | 30.5 / 32.82 / 927 | ✅ V |
| + SFP gen + flip | 39.9 / 25.06 / 719 | 60.6 / 16.50 / 939 | ✅ V |
| + entropy-gate | 20.3 / 49.28 / 714 | 30.9 / 32.33 / 927 | ✅ V |

Method A row in Table V(4 params、~1.5h、forward=baseline)= 靜態資訊,來源 csv row(`5ebb07b`),
runtime 未另測(forward 與 baseline 同構 + 4 純量)。

## 5. Diagnostics (paper §Diagnostic Analysis,Table `tab:diagnostics`) — commit `f6ad5a1`, log `journal_logs/diag_RUN.log`

Tool `tools/diag_metrics.py`(band r3;small<1024px;FP≥max(50px,0.1%))。輸入 = §1/§2 的 pred
save_dirs + GT(`SegmentationClassContext` / VOC `SegmentationClass`)。Context N=1021(stride 5),
VOC N=725(stride 2)。**class = diagnostic — 不進主結果表**(解釋段+表用)。

| Setting | bnd err | small-obj | FP/img |
|---|---|---|---|
| Ctx base / +flip / SFP | 0.6925 / 0.6834 / 0.6897 | 0.1510 / 0.1528 / 0.1431 | 4.5406 / 4.0480 / 3.9902 |
| VOC base / +flip / SFP | 0.1522 / 0.1496 / 0.1513 | 0.7502 / 0.7386 / 0.7444 | 0.8055 / 0.8110 / 0.8055 |

VOC SFP 列 2026-07-09 補(preds = §1 V4 `experiments/voc_sfp_dtlr_gen_official_eval/`,
log `journal_logs/diag_voc_sfp.log`)— 同機制:small-obj 侵蝕(0.7502→0.7444)換 boundary 微改善,
VOC 粗物件統計使淨 delta 仍正。

**Figure** `fig_flip_diag.png`(JournalPaper 根目錄):tool `tools/diag_figure.py`,輸入 C1 vs C2
save_dirs,picks 2007_007688 / 2007_002611 / 2008_000032(log 見 f6ad5a1 session);diagnostic。

## 6. Context 8-epoch(superseded — 不進任何表)

Base ckpt `experiments/context_vanilla_run2/best_weight.pth`(8ep,未收斂)。原因:under-trained
base confound,已被 §2 converged 取代。Commit `4c16036`(csv 記錄)。

| Method | mIoU | save_dir | log |
|---|---|---|---|
| baseline no-TTA / flip | 0.1980 / 0.2028 | `experiments/context_eval/base_notta/`,`.../base_flip/` | not preserved(僅 csv+.pt) |
| SFP gen no-TTA / flip(PD0.85 corrected) | 0.1929 / 0.1955 | `.../sfp_notta_pd085/`,`.../sfp_flip_pd085/` | `journal_logs/context8ep_sfp_corrected.log` |
| SFP entgate no-TTA / flip | 0.1945 / 0.1973 | `experiments/context_sfp_entgate_eval/`,`..._flip_eval/` | `experiments/entgate_eval/ctx_gated_*.log` |
| baseline @ PD1.0 | 0.0021 | `.../base_pd1_notta/` | not preserved — diagnostic(PD1.0 = VOC-specific,prune 全 59 類) |

## 7. Exploratory VOC multi-scale(附錄限定 — 不得進主表)

原因:scale 1.25 由 VOC val 挑選 = validation-selection。Commit `5ebb07b`;logs not preserved。

| Method | mIoU | save_dir |
|---|---|---|
| SFP+DTLR + ms(1.0,1.25) + flip | 0.8661 | `experiments/voc_sfpdtlr_ms125_flip/` |
| Method A + ms + flip | 0.8643 | `experiments/voc_presence_run1_ms125_flip/` |
| baseline ms + flip | 0.8637 | `experiments/voc_tta_ms125_flip/` |
| baseline ms(1.0,1.25,1.5) + flip | 0.8599 | `experiments/voc_tta_ms1255_flip/` |

## 8. Reference / negative / sanity(不進主表;部分進 Negative Findings 文字)

| Item | value | class | provenance | main table? / 原因 |
|---|---|---|---|---|
| Oracle image-level FP removal | 0.8998 | reference bound | csv row(`5ebb07b`);非實作方法 | ❌ 上限參考,非方法 |
| IABR retrain | 0.8001 | negative | csv(`5ebb07b`);save_dir 未記錄 | ❌ negative finding 文字 |
| DFF2d full-image fusion | 0.4151 | **voided** | csv 註明「作廢(parity bug 污染)」 | ❌ voided;**tex 已改引 fusion v2 0.6897(2026-07-09)**,0.4151 不再出現於 paper |
| Fusion v2(identity-verified L9+L12 selective) | 0.6897 | negative(formal test) | `research_notes.md §11`(E01,2026-07-07);save_dir `experiments/voc_l9l12_selective_v2/`;vs 自訓 baseline 0.8451 | ❌ Negative Findings 文字(tex 引用中) |
| CLASS_GATE 手調 sweep | 0.097 | negative(subset) | csv(`5ebb07b`);20-img subset only | ❌ subset、手調;negative 文字需帶 subset 註記 |
| Method C photometric flicker | cov AUC 0.186 | negative | commit `cb9b4cb` message + `docs/research_notes.md` | ❌ negative finding 文字 |
| PAMR full-image res 10 iter | 0.8361 | negative(formal, official ckpt) | cfg `config/voc_test_pamr_official_fullres_cfg.yaml`;save_dir `experiments/voc_pamr_fullres_official_eval/`(1449 .pt);csv 2026-07-09;**recompute 驗證精確重現**(`tools/recompute_miou.py`) | ❌ Negative Findings 文字(tex 引用中) |
| PAMR token-grid 1/3/10 iter | 0.8128 / 0.7250 / 0.5843 | negative(diagnostic) | save_dirs `voc_pamr_token_iter1_eval/`,`voc_pamr_token_iter3_eval/`,`voc_pamr_official_eval/`(各 1449 .pt);0.8128 recompute 驗證 | ❌ 同上 |
| PAMR identity(0 iter) | 0.8536 | sanity | `experiments/diag_pamr_identity/` | ❌ sanity(wrapper 無副作用) |
| TTA identity check | 0.8536 | sanity | csv(`5ebb07b`);`experiments/voc_tta_identity/` | ❌ sanity(證明 test_tta==test.py) |
| Method A identity(γ=0)| 0.8536 | sanity | csv;`experiments/diag_presence_identity/` | ❌ sanity |
| Method A ep0(舊 run) | 0.8442 | superseded | csv;`experiments/voc_presence/` | ❌ 舊 run |
| LGAK-MVP identity(α=0) | 0.8536 | sanity(future work) | `1fa591a`;`experiments/lgak_id_lgak/`+`experiments/lgak_identity_eval/lgak.log` | ❌ LGAK = future work,不進 journal 主實驗 |

## 9. Verification status of `sections/4_experiments.tex`(2026-07-09 全數字溯源檢查)

- 主表/ablation/runtime/diagnostic/flip 全部數字 ↔ `method_results.csv` 或 `journal_logs/*` 一一對上(§1–§5)。
- 衍生量核驗:flip≈2×(29.1/14.7=1.98、47.8/24.3=1.97)、SFP +26–37%(1.255/1.374)、entropy gate <2%(0.5%/1.3%)、VRAM Δ≤16MiB(7/16)、"recovers roughly a quarter"(0.0014/0.0059≈24%,原文 "about a third" 已修正)。
- ~~[NEEDS VERIFICATION] 標註 2 處~~ **兩處已解(2026-07-09)**:①PAMR 句 — 數字原在 `research_notes.md`(2026-07-07 全 full-val),0.8361/0.8128 由 `tools/recompute_miou.py` 從 saved preds 精確重現後寫入 tex+csv(§8);②DFF2d 句 — 改引 fusion v2 0.6897(formal,§8),作廢的 0.4151 不再出現於 paper。tex 現無任何 \TODO。

## 10. SFP flagged-fraction stats(paper `tab:flagged_fraction`;diagnostic)— 2026-07-09

Tool `tools/sfp_stats_extract.py`(gen 設定單次 forward 同時記兩種 gate 比例;no-TTA,full val)。
JSON `experiments/sfp_stats_extract/{voc_gen,context_gen}.json`;log `journal_logs/sfp_stats_{voc,context}.log`。
VOC = official ckpt + gen cfg(§1 V4 同設定);Context = converged ckpt + gen cfg(§2 C3 同設定)。

| stat(mean over images) | VOC(C=20,N=1449) | Context(C=59,N=5105) |
|---|---|---|
| unrel_frac_conf(max-prob 0.97) | 0.3465 | 0.7254 |
| unrel_frac_ent(tau 凍結於 C=20) | 0.3168 | 0.6818 |
| ratio(實際 rewrite,capped) | 0.2605 | 0.5444 |
| proxy_available_ratio | 0.9932 | 0.7433 |
| h_norm_mean | 0.0890 | 0.2933 |

解讀:class-count confound 真實但小(gate 換掉只回收 ~4pt/38pt),與 mIoU 「recovers ~1/4」吻合;
Context 實際被 rewrite 的 token >54% 且 proxy 支撐少 26% → rewrite 侵蝕主導,呼應 Table IV/diagnostics。
