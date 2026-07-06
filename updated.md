# updated.md — 重大更新紀錄

> 使用者慣例:每次重大改版在此記錄「改了什麼、為什麼、結果」。最新在上。

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
