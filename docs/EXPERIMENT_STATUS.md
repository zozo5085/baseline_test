# 實驗狀態(EXPERIMENT STATUS)

> 本檔由 `tools/run_experiment_queue.ps1` 自動 append;人工修改請只加不刪。
> 佇列定義:`docs/EXPERIMENT_QUEUE.md`。baseline 對照 = 0.8451。

## 手動記錄

- 2026-07-07 00:44 E00(l9l12 v2 訓練)由使用者於自己終端機啟動;
  訓練中 eval 軌跡:0.4981 → 0.7107 → 0.7328 → … → 0.8004(epoch ~14,健康)。
- 2026-07-07 佇列系統建立(E01–E03 排入;E04–E07 待決策/前置)。

## 自動記錄(script append 區)

## Queue run started 2026-07-07 02:19:04
- GPU memory at start: 4451 MiB
- waiting for running ML process(es) to finish: pid 86444
- 2026-07-07 03:59:11: no running ML process, queue begins
- E01 started 2026-07-07 03:59:11; log: D:\ReCLIPP_Test\experiments\queue_logs\E01_test_l9l12_v2_20260707_035911.log
- E01 DONE 04:00:04: mIoU 0.6897 (baseline 0.8451, delta -0.1554)
- **QUEUE STOPPED after E01**: mIoU 0.6897 is clearly below baseline (< 0.8); diagnose before continuing
## Queue run ended 2026-07-07 04:00:05 (stopped early: True)

## 手動記錄(2026-07-07 凌晨,E01 後診斷)
- E01 = 0.6897(佇列正確自停)。診斷:同權重 gamma9=0 → 0.6773(傷害在訓練後權重);
  baseline 權重 test-time-only fusion gamma 0.05/0.10/0.20 → 0.8455/0.8431/0.8378(無可利用訊號)。
- 判決:fusion 線關閉;E02/E03 建議取消(待使用者確認)。診斷 log 在 experiments/queue_logs/DIAG_*.log。
