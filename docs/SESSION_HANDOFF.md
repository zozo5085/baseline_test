# Session 交接(2026-07-07 清晨)

> 給下一個 session:讀完本檔即可接續。詳細數據在 docs/research_notes.md §11、
> 事件紀錄在 updated.md、實驗佇列在 docs/EXPERIMENT_QUEUE.md。

## 一夜結果總覽(2026-07-06 深夜 → 07-07 清晨)

**主成果:VOC 超越作者發表數字(免重訓)**
- 官方 ReCLIP++ VOC 權重(上游 repo 釋出)本地評測 = 0.8536(作者稱 85.4)
- 官方權重 + SFP+DTLR legacy = **0.8590**;+ 泛化版(零 VOC hack)= **0.8582**
- 兩者皆超過 85.4;增益跨 checkpoint 成立(自訓 0.8451 上 +0.0087/+0.0069)

**修復的重大 bug**
- `model_feature_fusion.py` prompt 正規化與 model.model 不一致(逐類 L2 vs 全域
  Frobenius)→ 曾讓所有 fusion 實驗數字作廢;已修復並以 parity(0.8451 整)驗收。

**關閉的研究線(乾淨負結果,皆可入論文消融)**
- Selective fusion(l9l12 v2):正式 test 0.6897;test-time-only gamma 掃描單調遞減
  → L9 殘差無可利用訊號。E02/E03(l6l12)已取消。
- IABR:zero-init identity 驗收 0.8451 整;訓練後漂移(eval 峰值 0.7831 @ epoch 4
  → 一路下滑),best_weight 正式 test 0.8001。
- **浮現的論文故事線:訓練期外掛在無監督目標下漂移(train-eval 好看、test 劣化),
  測試期精修穩健且跨 checkpoint 移轉。**

## 資產位置

- 官方權重:`experiments/official/voc_reclippp_854/best_weight.pth`(上游 README 有
  五個資料集全套的 Google Drive 連結,其餘尚未下載)
- SFP+DTLR 模組:`model/model_sfp_dtlr.py`(參數化,`MODEL.SFP_DTLR` config 區塊;
  legacy 預設 = 逐字舊值,泛化 profile 見 `config/voc_test_sfp_dtlr_gen_*.yaml`)
- IABR 模組:`model/model_iabr.py`(保留,負結果已記錄)
- 文獻 survey:`docs/literature_survey_layer_fusion.md`(novelty 判決 + related work
  草稿);泛化性審查:`docs/othermodel_guide/sfp_dtlr_generalization_review.md`
- 佇列系統:`tools/run_experiment_queue.ps1` + `docs/EXPERIMENT_QUEUE.md` +
  `docs/EXPERIMENT_STATUS.md`(佇列在 E01 後依規則自停,E02/E03 已取消)

## 下一步(依優先序,皆待使用者確認)

1. **跨資料集驗證 SFP+DTLR 泛化版**(論文泛化主張的關鍵):
   前置 = 下載資料集影像(建議先 ADE20K)+ 對應官方權重 + 生成該資料集 CLIP 文字
   嵌入(voc 的生成方式參考 text/ 與上游 repo)。
2. **PAMR / DenseCRF 對照實驗**(文獻 survey 指出審稿必問;需實作,實作後各 ~2 分
   鐘評測,對象 = 官方權重)。
3. 論文寫作素材已齊:主結果表 + 兩條負結果消融 + related work 草稿。

## Git / 環境

- push 目標:remote `mine` = https://github.com/zozo5085/baseline_test(至 c1b56ab)。
  origin = 上游,永不 push。
- ML python:`C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`。
- GPU 目前空閒;個人機,啟動前查 `nvidia-smi --query-gpu=memory.used --format=csv,noheader`。
