# 實驗佇列(EXPERIMENT QUEUE)

> 執行器:`tools/run_experiment_queue.ps1`(循序執行,先等現有實驗結束才開始)。
> 狀態即時記錄:`docs/EXPERIMENT_STATUS.md`。
> 鐵律:不刪任何 checkpoint;不覆蓋既有 log(全部 timestamp 新檔名);
> 訓練前檢查 SAVE_DIR 無既有 best_weight.pth,有就停止佇列。
> 佇列停止條件:log 出現 Traceback/OOM/RuntimeError、exit code 非 0、
> test mIoU 解析不到、或 test mIoU < 0.80(明顯低於 baseline 0.8451)。

## 正在跑(不受佇列管理,佇列會等它結束)

### E00 — l9l12 selective fusion v2 訓練
- **目的**:填 Stage 1 表 C 行(L9+L12 selective),parity 修復後首個乾淨 fusion 實驗。
- **假設**:uncertainty-gated L9 殘差注入可改善邊界,mIoU ≥ baseline 0.8451。
- **command**(使用者終端機,已於 2026-07-07 00:44 啟動):
  `python tools\train.py --cfg config\voc_train_l9l12_selective_v2_cfg.yaml --model RECLIPPP --model_module model.model_feature_fusion`
- **log**:使用者終端機 console + `experiments/log_voc_rectification.txt`(共用 append)
- **expected runtime**:~5–7 h(50 epochs)
- **success criteria**:訓練完成、`experiments/voc_l9l12_selective_v2/best_weight.pth` 存在、
  訓練中 eval 無崩壞(觀察值:epoch14 已達 0.8004,健康)。

## 佇列(依序自動執行)

### E01 — l9l12 v2 正式評測
- **目的**:取得 Stage 1 表 C 行的正式 mIoU(train-loop eval 與 test.py 流程不同,以此為準)。
- **假設**:mIoU ≥ 0.8451(超過 baseline → selective fusion 有效);0.80–0.8451 = 無效但無害;< 0.80 = 有害,停佇列診斷。
- **command**:`<PY> tools\test.py --cfg config\voc_test_l9l12_selective_v2_cfg.yaml --model RECLIPPP --model_module model.model_feature_fusion`
- **前置**:`experiments/voc_l9l12_selective_v2/best_weight.pth` 存在(E00 產物)。
- **log**:`experiments/queue_logs/E01_test_l9l12_v2_<timestamp>.log`
- **expected runtime**:~2–20 min
- **success criteria**:跑滿 1448/1449、解析出 `the mIOU:`、值記入 STATUS 並與 0.8451 比較。

### E02 — l6l12 selective fusion v2 訓練
- **目的**:填 Stage 1 表 D 行(L6+L12 selective),分離淺層(邊界層)貢獻。
- **假設**:L6 殘差(gamma6=0.05,boundary-gated)提供與 L9 不同的邊界訊息。
- **command**:`<PY> tools\train.py --cfg config\voc_train_l6l12_selective_v2_cfg.yaml --model RECLIPPP --model_module model.model_feature_fusion`
- **前置**:`experiments/voc_l6l12_selective_v2/` 內無 best_weight.pth(不覆蓋保護)。
- **log**:`experiments/queue_logs/E02_train_l6l12_v2_<timestamp>.log`
- **expected runtime**:~5–7 h(50 epochs)
- **success criteria**:exit 0、無 Traceback、best_weight.pth 產出、
  訓練中 eval 末段 ≥ 0.75(sanity,非正式數字)。

### E03 — l6l12 v2 正式評測
- **目的**:Stage 1 表 D 行正式 mIoU。
- **假設**:同 E01 判準。
- **command**:`<PY> tools\test.py --cfg config\voc_test_l6l12_selective_v2_cfg.yaml --model RECLIPPP --model_module model.model_feature_fusion`
- **前置**:`experiments/voc_l6l12_selective_v2/best_weight.pth` 存在(E02 產物)。
- **log**:`experiments/queue_logs/E03_test_l6l12_v2_<timestamp>.log`
- **expected runtime**:~2–20 min
- **success criteria**:同 E01。

`<PY>` = `C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`(script 內寫死絕對路徑)。

## 待排(需要人工決策或前置作業,不在自動佇列內)

| 候選 | 目的 | 前置 |
|---|---|---|
| E04 safe_l6_l9_l12 v2(全融合+雙 gate) | Stage 1 表 E 行 | 看 E01/E03 結果決定值不值得再花 5–7 h |
| E05 SFP+DTLR vs PAMR vs DenseCRF 對打 | 文獻 survey 要求的對照(審稿必問) | 需實作 PAMR/DenseCRF 掛載(免重訓,實作後各 ~2 min 評測) |
| E06 SFP+DTLR 疊在 fusion 權重上 | 兩條線疊加是否互補 | 需寫 model_sfp_dtlr 的 feature_fusion 版 wrapper;E01 結果 ≥ baseline 才值得 |
| E07 ADE20K 移轉測試(generalization profile) | 五資料集目標的第一個實測 | 下載 ADEChallengeData2016 + 生成 ADE 文字嵌入 + ADE baseline 權重 |

## 維護規則

- 新實驗:在本檔加條目 + 在 `tools/run_experiment_queue.ps1` 的 `$Experiments` 陣列加對應項(兩處同步)。
- 完成的實驗:結果由 script 寫入 STATUS;人工把正式 mIoU 移錄 `docs/research_notes.md` §11 表格。
