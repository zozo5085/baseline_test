# Session 交接(2026-07-07)

> 給下一個 session:讀完本檔即可接續。前一版交接(2026-07-06 深夜)的事項已全數處理或更新如下。

## 任務背景

目標:超越 ReCLIP++ VOC mIoU 0.85。baseline 重現 = **0.8451**
(`experiments/reproduce/voc_reclippp_baseline/best_weight.pth`,以 `model.model` 測得,
2026-07-07 重驗過完全重現)。使用者確立的方向:**保留 ReCLIP++ 的 bias
rectification,用其他方法疊加改善**(參考 `docs/othermodel_guide/RECOMMENDATIONS.md`)。

## 2026-07-07 重大事件:model_feature_fusion parity bug(已修復)

- l9_l12 selective 補測 mIoU = 0.4125 → 診斷後發現不是 fusion 的錯:
  `model_feature_fusion.py` 把 reference prompt 正規化寫成逐類 L2,原版
  `model.py:473` 是全域 Frobenius norm → `bias_logits` 尺度被改變,無論 fusion
  開關都會崩。已修(`model_feature_fusion.py:570`,附防回歸註解)。
- 驗收:baseline ckpt + fusion 全關 = **0.8451** 完全重現;l12_only(normalize 路徑)
  = **0.7393** → `normalize_feature_map` 替換 f12 本身另吃 ~0.106。
- **作廢結果:DFF2d 0.4151、l9l12 selective 0.4125**(壞模組下訓練/測試)。
  fusion 成敗未定,需重做。
- 完整診斷紀錄:`docs/research_notes.md` §11「Failure diagnosis」;各 diag config
  在 `config/voc_test_diag_*.yaml`,log 在 `experiments/diag_*_console.log`。

## Git

- remote `mine` = https://github.com/zozo5085/baseline_test(使用者自有,已 push 至
  e7aa24a)。origin = dogehhh/ReCLIP 上游,**永遠不 push**。
- 慣例:大改版寫 `updated.md`、大版本 local commit + push mine。

## 未完成(接續順序)

1. **SFP+DTLR 移植(Tier 1,使用者已口頭傾向)**:從
   `othermodel_guide\model_1\model_lrab_v1_voc_final_862.py` 把 PG-CP-SFP 淨化 +
   SP-DTLR 濾波移植成可掛在 `model.model` + baseline ckpt 的測試期後處理,
   免重訓,直接測 mIoU(內文參考值 0.8564)。分析見
   `docs/othermodel_guide/lrab_analysis.md`。
2. **fusion 數學重做**:gamma=0 必須是精確 identity(殘差可在正規化空間算,
   但要加回原始 f12、不得再 normalize)→ smoke(`tools/smoke_test_fusion.py`)→
   使用者核准後重訓 l9_l12(~5-7h,給使用者終端機一行指令)。
3. l6l12 訓練:凍結中,等 2 完成再議。
4. 每次啟動 GPU 工作前:`nvidia-smi --query-gpu=memory.used --format=csv,noheader`
   (個人機,遊戲/桌布程式佔 VRAM)。

## 環境備忘

- ML python:`C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`(系統 python 無 torch)。
- test.py 的結果不一定寫進 `experiments/log_voc_rectification.txt`,以 console
  redirect 的獨立 log 為準;test.py 會把每張圖的 output 存成 SAVE_DIR 下的 .pt(輸出,非輸入)。
- 每實驗新 SAVE_DIR,不覆蓋 best_weight.pth。
