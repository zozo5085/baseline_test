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

## 11. Paired bootstrap significance(diagnostic/statistical;2026-07-10)

Tool `tools/bootstrap_significance.py`(per-image intersect/union 可加性 → paired image bootstrap;
10,000 resamples,seed 0;base/variant 用同一 resample indices)。Sanity:observed mIoU 全部逐字重現
(VOC 0.8536/0.8601/0.8582;Context 0.2412/0.2473/0.2353)。輸入 preds = §1 V1/V2/V4 與 §2 C1/C2/C3
的 save_dirs。JSON `experiments/bootstrap_significance/{voc,context}.json`;log
`journal_logs/bootstrap_{voc,context}.log`。

| dataset | comparison | delta | 95% CI | p(two-sided) |
|---|---|---|---|---|
| VOC | flip − base | +0.0065 | [+0.0022, +0.0111] | 0.0052 |
| VOC | SFP gen − base | +0.0046 | [+0.0025, +0.0069] | <0.0002 |
| Context conv. | flip − base | +0.0061 | [+0.0055, +0.0067] | <0.0002 |
| Context conv. | SFP gen − base | −0.0059 | [−0.0073, −0.0045] | <0.0002 |

四個 headline delta 全部顯著:flip 兩資料集顯著為正;SFP gen VOC 顯著為正、Context 顯著為負
(= audit 結論的統計背書)。尚未寫進 tex(建議一句 significance 註記,見 handoff 下一步)。

## 12. ADE20K formal core(2026-07-12;第三資料集,§8 預註冊 battery)

Base:`experiments/ade_vanilla_converged/best_weight.pth`(**epoch 24**,per-epoch val 0.1305,
EPOCH 30 recipe = Context converged 同款,seed 0,sleep 已除;train cfg
`config/ade_train_converged_cfg.yaml`,snapshot commit `60c14e9`,launch 記錄
`experiments/ade_vanilla_converged/launch_info.txt`,訓練 17h20m,curve 31 點見 handoff)。
Eval:`tools/test_tta.py`,PD 0.85(**gate:baseline 0.1332 ≥ 0.01 → 無 fallback,全 arm PD 0.85**),
full val 2000,全部同一 ckpt;battery log `journal_logs/ade_battery_arm1_base.log` +
`ade_battery_RUN.log`;configs commit `fe5a479`(先於一切 formal 數字)。

| # | Method | mIoU | Δ | class | config | save_dir | main table? |
|---|---|---|---|---|---|---|---|
| D1 | baseline no-TTA | 0.1332 | — | formal | `config/ade_test_local_cfg.yaml` | `experiments/ade_conv_eval/base_notta/` | ✅ I |
| D2 | baseline + flip | 0.1335 | +0.0003(n.s.,p=0.8548) | formal | 同上 + `--flip` | `.../base_flip/` | ✅ II |
| D3 | SFP gen no-TTA | 0.1232 | −0.0100(p<0.0002) | formal(negative,audit-only) | `config/ade_test_sfp_dtlr_gen_cfg.yaml` | `.../sfp_notta/` | ✅ I |
| D4 | SFP gen + flip | 0.1238 | −0.0097 vs flip base | formal(negative) | 同上 + `--flip` | `.../sfp_flip/` | ✅ II |
| D5 | entgate no-TTA | 0.1252 | −0.0080 | formal(negative) | `config/ade_test_sfp_dtlr_entgate_cfg.yaml` | `.../entgate_notta/` | ✅ III |
| D6 | entgate + flip | 0.1257 | −0.0078 vs flip base | formal(negative) | 同上 + `--flip` | `.../entgate_flip/` | ✅ II/III |
| D7 | **DTLR-only no-TTA(H1)** | 0.1250 | **−0.0082(p<0.0002)** | formal(negative)**★H1 REFUTED** | gen cfg + `--sfp_disable cpsfp` | `.../dtlronly_notta/` | ✅ IV(ADE 列) |
| D8 | DTLR-only + flip | 0.1253 | −0.0082 vs flip base | formal(negative) | 同上 + `--flip` | `.../dtlronly_flip/` | — |

**§8.3 預註冊判定(2026-07-12)**:
- **H1 REFUTED**:DTLR-only delta −0.0082 ≤ 0 且 bootstrap 顯著為負(CI[−0.0109,−0.0044],
  p<0.0002)→ DTLR-only 線**關閉**,記 formal negative;Context 的 +0.0010 確認為 post-hoc 巧合
  (預註冊防升格機制發揮作用);依條款不再做任何 component-level rescue。
- **H2**:符號為正(+0.0003)→ 字面上 3/3 非負;但 p=0.8548、CI 跨 0 → 誠實表述 = **flip-TTA
  3/3 資料集不傷、2/3(VOC/Context)顯著為正、ADE 幅度≈0**。「same sign similar magnitude」
  主張需縮限為 non-negative。
- **反 rescue**:full SFP −0.0100 / entgate −0.0080 皆顯著為負 = 第三資料集 audit 數據,
  Context 降級結論加固。
- **審計新發現**:Table IV 的「失敗定位於 CP-SFP rewrite」**不延伸到 ADE** —— ADE 上 DTLR-only
  (−0.0082)≈ entgate(−0.0080)≈ full(−0.0100),連 edge-aware 平滑本身都在細粒度標籤空間
  侵蝕結構(見 §12b diagnostics:DTLR-only small-obj 0.1321 vs base 0.1420)。

### 12b. ADE diagnostics(N=1000,stride 2;log `journal_logs/ade_diag_bootstrap.log`)

| Setting | bnd err ↓ | small-obj ↑ | FP cls/img ↓ |
|---|---|---|---|
| base / +flip | 0.7375 / 0.7340 | 0.1420 / 0.1396 | 4.0870 / 3.4080 |
| SFP gen / DTLR-only | 0.7449 / 0.7458 | 0.1327 / 0.1321 | 3.0180 / 3.0220 |

flip:邊界+FP 改善、small-obj 微蝕 → 淨 ≈ 0(似 VOC 型,非 Context 型);SFP/DTLR-only:FP 大降
(4.09→3.02)但邊界與 small-obj 齊蝕 → 淨負,且兩者幾乎同值(rewrite 有無不再是差異主因)。

### 12c. ADE flagged-fraction + runtime(logs `ade_stats_bench.log`;json `sfp_stats_extract/ade_gen.json`)

| stat | VOC(C=20) | Context(C=59) | ADE(C=150) |
|---|---|---|---|
| unrel_frac_conf / ent | 0.3465 / 0.3168 | 0.7254 / 0.6818 | **0.8081 / 0.7669** |
| rewrite ratio(capped) | 0.2605 | 0.5444 | **0.6064** |
| proxy_available_ratio | 0.9932 | 0.7433 | **0.5008** |
| h_norm_mean | 0.0890 | 0.2933 | 0.3517 |

單調惡化鏈完整:類別數/難度 ↑ → 被旗標 token ↑、proxy 支撐 ↓(0.99→0.74→0.50)→ 侵蝕主導。
Runtime(RTX 5090,batch1,50-img):base 48.9ms/20.4FPS、+flip 98.3、SFP 68.7、SFP+flip 114.9、
entgate 56.7;peakVRAM ≤1044MiB;全 test-time 0 params。

### 12d. ADE bootstrap(10k,seed 0;json `bootstrap_significance/ade.json`;base 0.1332 逐字重現)

| comparison | delta | 95% CI | p(two-sided) |
|---|---|---|---|
| flip − base | +0.0003 | [−0.0027, +0.0020] | 0.8548(n.s.) |
| SFP gen − base | −0.0100 | [−0.0130, −0.0060] | <0.0002 |
| DTLR-only − base | −0.0082 | [−0.0109, −0.0044] | <0.0002 |
