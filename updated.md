# updated.md — 重大更新紀錄

> 使用者慣例:每次重大改版在此記錄「改了什麼、為什麼、結果」。最新在上。

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
