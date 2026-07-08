# updated.md — 重大更新紀錄

> 使用者慣例:每次重大改版在此記錄「改了什麼、為什麼、結果」。最新在上。

## 2026-07-08(深夜)— LGAK-MVP 實作,identity + smoke 通過,待 short run

- **改了什麼**:新方向 LGAK(language-guided adaptive kernel,frozen CLIP dense feature 的
  *pre-similarity* refinement)。先做 design review(`docs/LGAK_IMPLEMENTATION_REVIEW.md`)把計畫
  插入點對到真實 `model.py:462-482`,抓到 5 問題(F1 feat 也餵 decoder、F2 插在 normalize 後破壞
  單位範數、F3 mean(T) 語言條件弱、F4 α=0 conv 零梯度、F5 需 copy-forward),使用者拍板 4 決策後才實作。
  **MVP only**(無 offset / kernel-mixture / multi-layer)。
- **交付**:`model/lgak.py`(`TextGatedConvRefiner`:`F_out=normalize(F+α·g·F_refine)`,
  `g=1+MLP(mean_c(T))` 全域 mean gate,α 零初始)+ `model/model_lgak.py`(subclass baseline、
  copy-forward、LGAK **只餵 output_q**、decoder 保留原始 feat、freeze-then-append)+ 3 configs
  (`voc_{train,test}_lgak_mvp` + `voc_test_lgak_identity`)+ `tools/smoke_test_lgak.py` +
  `config/configs.py` 註冊 LGAK keys。**未動 `model/model.py`**。
- **為什麼**:F2 re-normalize 保 cosine 校準(避免重演 SFP 的範數/校準 confound);F1 只餵 output_q
  保 decoder 訓練分布;α=0 保精確 identity;凍 baseline 只訓 LGAK → 結構上不 drift。
- **驗收(全過)**:**identity(α=0,full-val VOC)= 0.8536 整 == baseline**(缺鍵僅 7 個 LGAK 參數);
  smoke = trainable **全在 `lgak.`**(398,465)、loss 有限、feat_norm out=1.0000、**F4 bootstrap 實測吻合**
  (iter0 只有 α 有梯度 grad 0.0062、conv=0;iter1 α≠0 後 conv grad=1.9e-5)。
- **狀態**:**尚未跑 2-3 epoch short run**(待使用者確認)。成功門檻已預註冊:identity==baseline ✓、
  VOC no-TTA ≥ baseline−0.005、有正 delta 才驗 Context、不拿 VOC-tuned TTA 當成功依據。

## 2026-07-07 — Method A 實作完成(trainable soft presence-calibration head),待訓練

- **背景轉向**:放棄救 IABR trainable,改在 ReCLIP++ baseline **rectification stage** 上設計
  受約束的小型可訓練 module。設計 3 案 A/B/C(`docs/TRAINABLE_BASELINE_METHODS.md`):
  A = soft presence calibration head(**主線**);B = reliability-guided bias scale(僅保留
  conservative ablation);C = photometric/prompt consistency gate(**前提被診斷否證**,
  cov AUC 0.186,只留 negative finding)。決策與計畫在 `docs/METHOD_A_IMPLEMENTATION_PLAN.md`。
- **Method A 交付**:`model/model_presence.py` 子類化 baseline RECLIPPP。`__init__` 從
  `PRESENCE.INIT_FROM`(官方 0.8536 ckpt)載入→**凍結整個 baseline**→只建 4 參數
  `PresenceHead`(tau/temp/scale/gamma)。forward 在 `output_q` 加
  `tanh(gamma)*scale*log(sigmoid((<z_global,text>-tau)*temp))`,**gamma zero-init → identity**。
  loss = baseline region loss + 非對稱 presence BCE(recall 導向)+ baseline-preserving reg。
  **為什麼這樣設計**:凍 baseline、只訓 4 純量 → 結構上不可能 drift,直接修掉
  IABR/fusion 的失敗模式(從頭 retrain 在無監督目標下漂移)。
- **驗收(全過)**:import OK;**identity(gamma=0)mIoU = 0.8536 整**(缺鍵只有 4 個 head 參數);
  smoke = trainable params **恰 4 個**、loss 有限、梯度落在 head。commit `5d11af2`,已 push `mine`。
- **狀態**:**尚未訓練**。下一步啟動 ablation 1(`zglobal`),建議 EPOCH 50→15(4 純量收斂快)。
  誠實預期:VOC 近飽和,最可能**打平 ~0.8536**(定位為 premise gate,真正 payoff 在難資料集)。

## 2026-07-07(清晨)— IABR 收案:0.8001,負結果(訓練期改良二連敗)

- 50-epoch 訓練:eval 峰值 0.7831(epoch ~4)後單調下滑到 ~0.62;best_weight
  正式 test = **0.8001**(baseline 0.8451,−0.045)。
- 判決:無監督訓練訊號下自適應 scale 漂移離開 identity 後不再回頭 — 與 fusion 線
  同型。**本專案浮現的核心發現:訓練期外掛(fusion / adaptive rectification)在無
  監督目標下會漂移(訓練 eval 好看、正式 test 劣化),測試期精修則穩健且跨
  checkpoint 移轉** — 這個對比本身就是論文的故事線。
- 作者自己沒記錄 IABR 數字,與此結果一致。

## 2026-07-07(清晨)— 達標:SFP+DTLR × 官方權重 = 0.8590 > 作者 85.4

**事件鏈**
- 發現上游 repo 有官方權重(五個資料集全套)。VOC ReCLIP++ 權重本地評測 = **0.8536**
  (作者稱 85.4)→ 評測管線與作者一致;0.8451 自訓差距 = 環境 + seed,非 bug。
- SFP+DTLR 疊官方權重:legacy = **0.8590**、泛化版(無 VOC hack)= **0.8582** —
  **雙雙超過發表數字 85.4**,免重訓。
- 附帶解鎖:官方也放了 Context/ADE20K/Cityscapes/COCO-Stuff 權重 → Stage 5 資料集
  驗證不再需要自訓 baseline,只剩資料集影像 + 文字嵌入兩個前置。
- 同時:IABR 訓練已啟動(experiments/voc_iabr/,~4-6h,監控中)。

## 2026-07-07(凌晨)— l9l12 v2 結果:0.6897,fusion 線關閉(乾淨負面結果)

**結果鏈**(全部在修復後模組上,identity 已驗證)
- E01 正式 test(gamma9=0.20)= **0.6897**(baseline 0.8451,−0.1554);佇列依規則自動停止。
- 同權重 test 關融合(gamma9=0)= 0.6773 → 傷害在訓練後權重,非 test 融合項。
- baseline 權重 + 只在 test 開融合:gamma 0.05/0.10/0.20 = 0.8455/0.8431/0.8378
  → L9 殘差無可利用訊號,單調遞減。
- **判決:selective L9/L12 fusion 線關閉**;這是可寫進論文的乾淨消融負結果
  (對比 v1 的 0.4125 是 bug 產物)。建議取消 E02/E03(l6l12,同設計)。
- 剩餘主線:SFP+DTLR 測試期精修(已 +0.0087)、CBR/IABR rectification 重設計(§15)。

## 2026-07-07(晚)— SFP+DTLR 參數化:去 VOC 化後仍 +0.0069(0.8520)

**做了什麼**
- 所有 SFP/DTLR 常數移入 `MODEL.SFP_DTLR` config 區塊(configs.py:41-56,預設 =
  舊逐字值)。legacy 驗收:**0.8538 完全重現**(參數化零行為改變)。
- generalization profile(`config/voc_test_sfp_dtlr_gen_cfg.yaml`):top-k 改比例
  0.75、sigma_s 改網格相對 2.3、beta 1.0、proxy_lambda 1.0、VOC 類別索引清空 —
  審查點名的 VOC 特化全數移除 → VOC **0.8520(+0.0069)**。
- 結論:增益不依賴 VOC hack,跨資料集移轉有基本面;實測仍待 Stage 3 資料集下載。

## 2026-07-07(晚)— SFP+DTLR 移植成功:VOC mIoU 0.8538(+0.0087,免重訓)

**做了什麼**
- 新模組 `model/model_sfp_dtlr.py`:包裝原版 model.model 的 RECLIPPP,forward 輸出
  疊 PG-CP-SFP 淨化 + SP-DTLR 濾波(超參數逐字照抄 862 檔;checkpoint 載入
  0 missing / 0 unexpected;不加任何參數;model.model 未動)。
- 排除來源檔的屬性殘差 stage(chair/diningtable VOC hack)— 對照組 0.8564 含該 stage。
- 全量評測:**0.8538** vs baseline 0.8451(1448/1449,每張圖精修都有執行)。
- config:`config/voc_test_sfp_dtlr_cfg.yaml`。
- 已知風險:sfp_topk=800 絕對值、structure_classes=(4,8,10) 寫死 VOC 類別索引 —
  泛化性審查(opus agent)進行中,目標五資料集皆有效。

**同日稍早**
- fusion 數學重做完成並驗收(gamma=0 = 精確 identity,l12_only = 0.8451 整;
  smoke PASSED)。l9l12 v2 重訓 config 就緒,等使用者啟動(~5-7h)。

## 2026-07-07 — C 失敗診斷:排除 fusion 加法項,鎖定 normalize 路徑

**做了什麼**
- 診斷測試 1:同一顆 l9l12 權重,`--fusion_gamma9 0.0` → **mIoU 0.4070**(vs 0.2 時
  0.4125),零類別特徵相同 → 融合加法項不是元凶。
- 程式碼追查:fusion 啟用時 `v` 會被換成 `normalize_feature_map(f12)`(逐通道標準化
  + L2,`model_feature_fusion.py:19-24,544-549`)才進凍結的 CLIP `proj` — gamma=0 也一樣。
  主嫌 = 標準化破壞 CLIP 特徵統計 → 文字對齊崩壞。
- 診斷測試 2/3(A:baseline ckpt + l12_only;B:baseline ckpt + fusion 全關)已啟動,
  預期 A 崩、B 維持 0.845 即可定案。詳見 research_notes.md §11「Failure diagnosis」。
- 新增診斷 config 兩份:`config/voc_test_diag_*_baselineckpt_cfg.yaml`。
- Git:新增 remote `mine` = https://github.com/zozo5085/baseline_test(使用者自有
  repo;origin 仍為上游,禁 push)。

**診斷結果(A/B/C 皆跑滿 1448/1449,同一顆 baseline 權重)**
- A(fusion 模組,l12_only 只走 normalize)= **0.2160**
- B(fusion 模組,fusion 全關)= **0.2838**
- C(原始 model.model)= **0.8451** 完全重現
- 結論:test 管線沒問題;`model_feature_fusion` 的**非融合路徑本身**就與 model.model
  不等價(parity bug),既有 DFF2d 0.4151 / l9l12 0.4125 全部是壞模組量出來的,
  不能作為 fusion 方向成敗的證據。修好 parity、B 重測回 0.8451 之前,fusion 實驗
  全數凍結。
- Git:已 push 至 https://github.com/zozo5085/baseline_test(remote `mine`)。

**Parity bug 已找到並修復(同日)**
- 元凶:`model_feature_fusion.py` 把 reference prompt 正規化寫成逐類 L2;原版
  `model.py:473` 是全域 Frobenius norm。`bias_logits` 尺度/類別權重被改變 →
  checkpoint 的 logit 校準崩壞。已改回一致(`model_feature_fusion.py:570`)。
- 修後驗收:B(fusion 全關)= **0.8451** 完全重現;A(l12_only normalize 路徑)
  = **0.7393** → normalize-and-replace 本身另吃 ~0.106,fusion 重做時 gamma=0
  必須是精確 identity(殘差加在原始 f12 上、不得再正規化)。
- DFF2d 0.4151、l9l12 0.4125 皆為壞模組下的產物,全部作廢;l9l12 需以修復後
  模組+重做的融合式重訓。

**下一步**
- 重做 fusion 數學(gamma=0 = identity)→ smoke → 重訓 l9l12;平行推進
  SFP+DTLR 測試期精修(RECOMMENDATIONS.md Tier 1)。

## 2026-07-06 — L9+L12 selective fusion 訓練+評測完成(結果:失敗)

**改了什麼**
- 完成 `l9_l12` selective fusion 訓練(50 epochs,`config/voc_train_l9l12_selective_cfg.yaml`,
  model_module `model.model_feature_fusion`),權重存 `experiments/voc_l9l12_selective/best_weight.pth`。
- 補跑評測(第一次 test 在 img_idx 161/1449 卡死,kill 後以
  `config/voc_test_l9l12_selective_cfg.yaml` 重跑完成)。
- 結果表建立於 `docs/research_notes.md` §11「Stage 1 Results」。

**結果**
- **mIoU 0.4125**(1448/1449 張;數字來自 console,共用 log 檔本次未被 append)。
- 對照:baseline 0.8451、full DFF2d 0.4151。
- 21 類中 8 類 IoU 恰為 0,失敗特徵與 full DFF2d 相同。

**為什麼重要 / 下一步**
- selective(C)與 full fusion(E)failure 值幾乎重合(0.4125 vs 0.4151),
  代表問題可能不是「融合量太多」,而是兩者共用的結構性問題
  (候選假設:test 期 GAMMA9=0.20 覆寫掉訓練學到的 gamma、fusion 路徑覆寫 projected
  features、train/test config 不一致 — 均未驗證)。
- **l6l12 訓練(D)暫停排隊**,先診斷 C 的失敗原因,否則 D 大概率重演同樣失敗。
- 診斷完成、Stage 1 表格補齊後,才進入 othermodel_guide 移植實驗
  (見 `docs/othermodel_guide/RECOMMENDATIONS.md`:Tier 1 = SFP+DTLR 免重訓精修)。

## 2026-07-08 — Test-time TTA(多尺度+翻轉)+ SFP+DTLR 疊加 + Method A 訓練完成(新最佳 0.8661)

**改了什麼**
- 新增 `tools/test_tta.py`:inference-only 通用 eval engine(多尺度 + 水平翻轉 TTA,
  平均 per-view softmax@原解析度;TEST.PD/存檔/mIoU 與 test.py 一致)。
  `--scales 1.0` no-flip 經 identity check == test.py(**0.8536 精確**)。不碰 CLIP 特徵、免訓練。
- 完成 Method A run1 訓練(`config/voc_train_presence_cfg.yaml`,LR 0.01,15 epoch,MODE zglobal,
  SAVE_DIR `experiments/voc_presence_run1/`,best = epoch 12;per-epoch val 峰 0.8201)。
- 新增 `tools/smoke_test_presence.py`、`config/voc_test_classgate_s2_cfg.yaml`、
  `docs/_infra_map_testtime.md`、`docs/AUTONOMOUS_SESSION_2026-07-08.md`、
  `docs/method_results.csv`(不同方法數據表:訓練時間 / 改動內容 / mIoU)。

**結果(formal test,VOC val 1449,mIoU;baseline = 官方 ckpt 0.8536)**

| 底模 \ TTA | alone | +flip | +ms(1.0,1.25)+flip |
|---|---|---|---|
| baseline | 0.8536 | 0.8601 | 0.8637 |
| SFP+DTLR | 0.8590 | 0.8639 | **0.8661** |
| Method A(run1,訓練) | 0.8565 | 0.8637 | 0.8643 |

- 🏆 新最佳 = **SFP+DTLR + ms(1.0,1.25) + flip = 0.8661**(baseline +0.0125,全 inference-only,不碰 CLIP 特徵)。
- flip 單獨 = 0.8601 已 > 前一個最佳 SFP+DTLR 0.8590。scale 峰值 = 1.25(加 1.5 掉到 0.8599)。
- Method A(訓練 4 參數 presence 頭)= 0.8565 > baseline +0.0029;訓練有效但被 test-time TTA 支配。

**為什麼重要 / 下一步**
- flip TTA 是最便宜、零超參、最穩健的乾淨提升;SFP+DTLR+flip(0.8639)最無爭議。加 TTA 後三底模收斂 0.864–0.866,底模差異被抹平。
- Method A 確認 image-level presence 校準方向在 VOC 有效(雖小),是 CLASS_GATE 手調(20 圖崩壞 0.097)的「正確學習版」。
- 誠實備註:多尺度集合(1.25)在 test set 上挑選,屬探索性;正式報告尺度應在 held-out 上定。
- 下一步(需使用者決定):(a)難資料集驗證(Context/ADE/COCO-Stuff,需下載);(b)把 flip/ms TTA 設為預設 eval 並在 held-out 定尺度;(c)Method A ablation 2(zglobal_dense)。
