# SESSION HANDOFF — ReCLIP++ C-USS 研究(貼上即可接續)

> **給下一個 session 的你**:這份是可直接貼上的接手 prompt。讀完本檔後,**不要重新探索、不要重跑已完成的診斷**,直接從 §5「下一步」開工。所有數字用 4 位小數逐字沿用,不要四捨五入進記錄。基準線是 **官方 checkpoint 0.8536**,使用者的硬性要求:**所有任務結果一律要 > 0.8536**。
> 最後更新:2026-07-07(承接 d9cd3f55 session,已 compaction 一次)。

---

## 0. 一句話狀態

Method A(trainable soft presence-calibration head)**已實作完成、identity 0.8536 驗過、smoke 過、已 commit+push**(`5d11af2`)。**還沒開始訓練**。下一步就是啟動 Method A 訓練(或先由使用者確認 EPOCH)。目前**沒有任何 process 在跑**。

---

## 1. 各條線最新結果(逐字)

| 線 | 結果 | 狀態 | 備註 |
|---|---|---|---|
| **官方 baseline(ReCLIP++)** | **mIoU 0.8536** | ✅ 正式基準 | `experiments/official/voc_reclippp_854/best_weight.pth`(627.9 MB)。所有比較以此為準。 |
| 自訓 baseline | 0.8451 | 環境/seed 落差 | 無法完全復現官方 0.8536,已接受此差距,不要再耗時間追。 |
| **SFP+DTLR(test-time,官方 ckpt)** | **0.8590 / 0.8582** | ✅ 已達標(> 論文 85.4) | 主要達成成果。test-time refinement,無訓練。 |
| IABR(trainable,從頭訓) | 0.8001 | ❌ 負結果(已關線) | 從頭 retrain 在無監督目標下 drift。乾淨負結果保留。 |
| Feature fusion(DFF2d / L9L12) | 0.4151 / 0.4125 → **已作廢** | ❌ parity bug 汙染 | 舊數字因 prompt 正規化 parity bug 無效。修正後 reworked fusion 0.6897,仍輸,關線。 |
| PAMR(test-time) | 全設定皆輸 | ❌ 已關線 | 每種設定都低於 baseline。 |
| **Method A(presence head)** | identity **0.8536**(gamma=0 完全復現) | 🟡 待訓練 | 只有 4 個純量參數可訓,baseline 全凍結。**尚未訓練。** |

### 重要診斷數字(已完成,不要重跑)
- **Oracle 反幻覺天花板 = 0.8998(+0.0462)**:image-level 完美去除不存在類別的上限。Method A 的理論 headroom。
- 殘差 bias 與誤差相關 **r=0.73**(image-dependent → direction A 前提成立)。
- Hard gating 太脆(brittle)→ 改用 soft gate。
- **Method C(photometric consistency)前提被否證**:幻覺在光度上穩定,cov AUC **0.186**(存在類反而 flicker 更多)。Method C 不做主線,只留 negative finding。
- 診斷細節在 `docs/diagnostics/`:`bias_residual_diagnostic.md`、`bias_intervention_oracle_probe.md`、`bias_hardgate_presence.md`、`photometric_consistency_premise.md`。

---

## 2. Git 狀態 / 改過的檔案

- 分支 `main`,HEAD = **`5d11af2`**,**0 commits ahead of `mine/main`**(全部已 push)。
- Remote:`mine` = `https://github.com/zozo5085/baseline_test.git`(**push 目標**);`origin` = `dogehhh/ReCLIP`(**上游,永不 push**)。
- 近期 commit:`5d11af2` Method A head / `0d4de4b` A 計畫 / `cb9b4cb` 否證 C / `25498c5` 設計 3 method / `6cd340b` headroom / `fca9ac7` 殘差 bias 診斷。

**已追蹤但尚未 commit 的改動(`git diff --stat`,254 insertions):**
`README.md`(+34)、`model/model.py`(+17)、`tools/test.py`(+133)、`tools/train.py`(+86)、`utils/test_mIoU.py`(+23)。這些是先前 diagnostics/test-time 基礎設施改動,**非 Method A 需要**(Method A 用 `model_presence.py` 子類化,不改 `model.py`)。下次要 commit 前先 `git diff` 看清楚再決定是否納入。

**本 session 新建的關鍵檔案(已 commit):**
- `model/model_presence.py` — Method A 主交付。子類 `_BaseRECLIPPP`,`__init__` 從 `PRESENCE.INIT_FROM` 載官方 ckpt→凍結全 baseline→只建 `PresenceHead`(4 參數)。forward 在 `output_q` 加 `tanh(gamma)*scale*log(sigmoid((<z_global,text>-tau)*temp))`,gamma=0 → identity。
- `config/voc_train_presence_cfg.yaml`(訓練)、`config/voc_test_presence_identity_cfg.yaml`(identity 驗證)。
- `config/configs.py` 加了 `MODEL.PRESENCE` block(MODE/INIT_FROM/BCE_W/NEG_POS_W/REG_W)。
- `docs/TRAINABLE_BASELINE_METHODS.md`(A/B/C 設計)、`docs/METHOD_A_IMPLEMENTATION_PLAN.md`(實作計畫+驗收協議)。

**未追蹤、非我建立、放著不動**(使用者平行工作):`docs/DEEP_RESEARCH_*.md`、`docs/EXPERIMENT_*.md`、`docs/claude_a_prompts/`(CDAM/FreeCP/SOM;其中 FreeCP 的 SC 訊號可當 presence head 輸入特徵)、`docs/othermodel_guide/*`。**commit 時不要 `git add docs/` 全加**,會把這些也 stage 進去(上次踩過:得 `git reset`)。逐檔 add。

---

## 3. 正在跑的 process / PID

**無。** 檢查過 `Win32_Process Name='python.exe'` = 空;`nvidia-smi` = 14% util、7017/32607 MiB(背景遊戲/Wallpaper Engine 佔用,非訓練)。開訓練前確認 VRAM 夠即可,不要砍使用者的桌面程式。

---

## 4. 最重要的 checkpoint / log / output

- **官方 ckpt(唯一基準來源,勿覆蓋)**:`experiments/official/voc_reclippp_854/best_weight.pth`。Method A 的 `INIT_FROM` 指向它。
- Method A 訓練 SAVE_DIR:`experiments/voc_presence/`(乾淨,尚無 best_weight)。**新實驗一律新資料夾,絕不覆蓋既有 best_weight.pth。**
- identity 驗證產物:`experiments/diag_presence_identity/`(已得 0.8536,可留可清)。
- 共用訓練 log:`experiments/log_voc_rectification.txt`(各訓練 append)。
- SFP+DTLR 成果:`experiments/voc_sfp_dtlr_official_eval/`、`voc_sfp_dtlr_gen_official_eval/`。
- 主結果表 + 診斷:`docs/research_notes.md §11`。

---

## 5. 下一個 session 第一步(直接做)

**啟動 Method A 訓練(ablation 1:`zglobal`)。** 前置三關已全過(identity 0.8536 / trainable params=4 / smoke finite loss),不用重驗。

啟動指令:
```
python tools\train.py --cfg config\voc_train_presence_cfg.yaml --model RECLIPPP --model_module model.model_presence
```
用 ML 環境的 python:`C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`(系統 python 無 torch)。

**先決定 EPOCH**:目前 config 是 `EPOCH: 50`。但 train.py 有 `time.sleep(0.08)/img` + 每 epoch 全量 val,50 epoch ≈ **~7 小時**;而 4 個純量參數從 identity 起步**收斂很快**,建議降到 **~15 epoch(~2h)**。改 `config/voc_train_presence_cfg.yaml` 的 `EPOCH`/`MAX_EPOCH` 即可。**多小時訓練屬需使用者確認的等級 — 開跑前確認 EPOCH + ETA 一次。**

啟動後驗證「真的在跑」:`nvidia-smi` util > 0 **且** log iter counter 前進,才回報 running。掛 drift tripwire(訓練中定期跑 formal test,若掉破 baseline 就停)。

**訓練完成後**:
1. Formal test vs baseline **0.8536** / test-time soft-gate / oracle 0.8998 三者比較。
2. 依結果分叉:> 0.8536 → 做 ablation 2(`zglobal_dense` 模式,加 output_q dense 統計 + 考慮 FreeCP SC 訊號當輸入);打平 → 印證 VOC 飽和,轉難資料集(COCO-Stuff/ADE,需下載)。
3. mIoU(4 位小數)寫進 `docs/research_notes.md §11`,寫 `updated.md` 條目,逐檔 commit + push `mine`。

**誠實預期(已寫進計畫 F3)**:此低容量全域 z_global 閘門在 VOC **最可能打平 ~0.8536**(VOC 近飽和,hard-gate 已打平、photometric 已否證)。VOC run 定位為 **premise gate**,真正 payoff 預期在難資料集。打平**不代表方向失敗**,判準先講好。

---

## 6. 不能重複做 / 不能覆蓋的事項

- **絕不 push `origin`**(上游 dogehhh/ReCLIP)。push 只去 `mine`。
- **絕不覆蓋任何既有 `best_weight.pth`**,尤其 `experiments/official/voc_reclippp_854/`。每個實驗新 SAVE_DIR。
- **不要重跑已完成的診斷**(oracle 0.8998、殘差 r=0.73、photometric AUC 0.186)—— 都在 `docs/diagnostics/`。
- **不要 `git add docs/`**(會 stage 使用者平行檔案)—— 逐檔 add。
- **config 一律用 Write tool 寫**,勿用 PowerShell `Out-File`(BOM 會炸 yaml loader)。
- 已作廢的舊 fusion 數字(0.4151/0.4125)因 parity bug 無效,引用時要標註。
- 硬性約束(使用者定的,Method 設計不可違反):不 distillation、不改 CLIP image/text encoder、不替換 CLIP feature、不用 pixel-level GT mask、新模組 zero-init/identity-init、bias_logits 要有 baseline-preserving reg、優先 presence/reliability calibration 而非直接訓 dense mask、要能跨 VOC/Context/ADE/Cityscapes/COCO-Stuff。

---

## 7. 環境提醒

- Python(ML):`C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`(torch cu128, RTX 5090)。系統 `python` 無 torch。
- PowerShell 5.1:無 `&&`,用 `;`。變數/cwd 不跨 tool call 保留 → 一律絕對路徑。
- **TEMP/TMP 已改到 `D:\Temp`**。新 session 請確認**不要把大型 cache/dataset/checkpoint 放 C 槽**(下載資料集、torch cache、輸出權重都導向 D 槽)。
- 開 GPU 工作前 `nvidia-smi --query-gpu=memory.used --format=csv,noheader`;VRAM 緊時告知使用者、勿自行砍其桌面程式。
- 多小時訓練優先給使用者可貼上的一行指令在他自己終端機跑(他偏好看得到),或 `run_in_background` 導 log 到 `experiments/<name>/console.log`。

---

## 8. 貼上用的接手 prompt(複製下面這段給新 session)

```
讀 D:\ReCLIPP_Test\docs\SESSION_HANDOFF.md 接續 ReCLIP++ 研究。
現況:Method A(model/model_presence.py)已實作、identity 0.8536 驗過、已 push(5d11af2),
尚未訓練,無 process 在跑。第一步:確認 EPOCH(建議降 50→15)後啟動 Method A 訓練
(config/voc_train_presence_cfg.yaml, --model_module model.model_presence),掛 drift tripwire。
基準 0.8536,所有結果要 > 0.8536。勿 push origin、勿覆蓋 official ckpt、勿重跑已完成診斷。
```
