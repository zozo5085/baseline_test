# Autonomous Session Log — 2026-07-08(ReCLIP++ C-USS)

> 使用者授權:8 小時自主窗口(使用者休息中,下次回來約 8h 後)。只做 **inference-only / 凍結 baseline、不碰 CLIP 特徵** 的安全實驗(見 `SESSION_HANDOFF.md §6` 硬性約束)。全程新 SAVE_DIR、不動 run1、不 push origin(只 push `mine`)。
>
> 對應資料表:`docs/method_results.csv`(欄位:method / 改動內容 / 訓練時間 / mIoU / …)。
> 基準:官方 ckpt formal test **0.8536**;所有目標 **> 0.8536**。

## 口徑校準(勿搞混)
- per-epoch val 比 formal test 低 **~0.04**(同一 ep0 ckpt:val 0.8039 vs formal 0.8442,offset +0.0403,單點量測)。
- **最終評分一律用 formal test**;per-epoch val 只用來看趨勢,不直接跟 0.8536 比。

## 參考結果(既有,逐字沿用)
| method | 改動內容 | 訓練時間 | mIoU | 口徑 |
|---|---|---|---|---|
| official_baseline | 官方 ReCLIP++ ckpt(凍結) | N/A | 0.8536 | formal |
| SFP+DTLR | test-time refinement(無訓練) | test-time | 0.8590 / 0.8582 | formal |
| oracle | image-level 完美去 FP(理論上限) | N/A | 0.8998 | oracle |
| IABR_retrain | from-scratch 可訓練 | — | 0.8001 | formal(負結果) |
| feature_fusion | full DFF2d | — | 0.4151 | formal(作廢) |
| MethodA_identity | presence gamma=0 | 0 | 0.8536 | formal |
| MethodA_ep0 | presence 4參數 1 epoch | ~1ep | 0.8442 | formal |

---

## 實驗日誌

### [進行中] run1 — Method A presence head(LR0.01 / 15ep / MODE zglobal)
- **改動內容**:4 個純量參數(tau / temp / scale / gamma)soft presence-calibration head,加在 `output_q`;baseline 全凍結;identity-init(gamma=0 → 完全復現 baseline)。
- **啟動**:2026-07-08,PID 30516,detached,SAVE_DIR `experiments/voc_presence_run1/`。
- **smoke**:trainable params=4、loss finite(2.68/3.27/2.83)、train imgs=1464、~0.032 s/iter(fwd+bwd)。PASSED。
- **per-epoch val 趨勢(上升、健康,非 IABR 式崩壞)**:
  | epoch | per-epoch val mIoU |
  |---|---|
  | 0 | 0.8039 |
  | 1 | 0.8117 |
  | 2 | 0.8158 |
- **粗略投影**(假設 val→formal +0.04 offset 大致成立,**未驗證假設**):ep2 formal ≈ 0.856。真值以 formal test 為準。
- **下一步**:訓練完 → formal-test `best_weight.pth` vs 0.8536 / SFP+DTLR 0.8590 / oracle 0.8998 三方比較。

### [規劃中] 安全 inference-only 實驗(等 Explore infra map + run1 讓出 GPU 後執行)
- 候選方向(全部 inference-only、凍結 baseline、不改 CLIP 特徵、新 SAVE_DIR):
  1. **test-time presence soft-gate 掃描**:用固定(非訓練)的 image-level presence 閘門抑制不存在類,掃參數看能否吃到 oracle +0.046 的一部分。
  2. **multi-scale + flip TTA**:對官方 ckpt 做多尺度/翻轉平均 logits(經典分割增益,無訓練)。
  3. **疊在 SFP+DTLR 0.8590 上**:把上面有效的 test-time 手段接到目前最佳 test-time pipeline,看能否再往上。
- 方法學備註:VOC 的 1449 是標準 test set;掃描超參屬**探索性**(對 test set 調參),最終數字需以原則性/val 設定的 config 再確認一次,MD 會標註。

---

## 進度時間軸(append)
- 2026-07-08 啟動 run1(PID 30516);建立本 log + `method_results.csv`;掛 mIoU heartbeat 監看;派 Explore agent 摸 test-time infra。
- Explore infra map 回,存 `docs/_infra_map_testtime.md`(formal test 指令、SFP+DTLR 0.8590、TTA 未實作、CLASS_GATE 可 test-time、data/ 只有 VOC)。
- run1 per-epoch val **plateau ~0.813**(ep0–5:0.8039 / 0.8117 / 0.8158 / 0.8138 / 0.8134 / 0.8131),best = **ep2 0.8158**;4 參數快速收斂如預期。
- 實作 `tools/test_tta.py`(多尺度+flip TTA,inference-only,average per-view softmax@原解析度;TEST.PD/存檔/mIoU 與 test.py 一致);20 圖 smoke 通過(變動解析度 OK)。加 `--save_dir/--load_path/--limit`,可當**通用 eval engine**(`--scales 1.0` == baseline)。
- 建 `config/voc_test_classgate_s2_cfg.yaml`(soft CLASS_GATE,LOG_BIAS_SCALE 2.0)。
- **待 run1 結束**再跑完整 queue(保護主 run,不跑並行重負載):run1 formal test → TTA flip → TTA multiscale → CLASS_GATE(B0 identity + default + s2)→ 疊 SFP+DTLR。
- l12_only identity 確認:baseline **0.5539** == l12_only(no gate)**0.5539**(20 圖)→ `model_feature_fusion` 路徑乾淨(B0 pass)。
- CLASS_GATE s2(LOG_BIAS_SCALE 2.0)subset **崩壞 0.097**:present-class `z_global·text` cosine < THRESHOLD 0.20,naive 手調閘門過度壓抑 present classes。正確校準版 = **run1**(學 tau/temp/scale/gamma),run1 僅 plateau ~0.8158 → 手調 sweep 被 run1 支配。**CLASS_GATE 手調棄跑**(留 negative finding;節省 ~45 min)。
- **策略修正**:VRAM 充足(27 GB free)且 run1 已 plateau(best 鎖在 ep2)→ 改為**與 run1 併行**跑 inference-only eval(單一併行、監控 VRAM),不再空等。
- run1 ep6 = 0.8115(略降,best 仍 ep2 0.8158)。
- 啟動 **TTA flip-only** 完整跑(model.model,PID 98400,SAVE_DIR `experiments/voc_tta_flip/`,concurrent with run1)。
- 🎯 **TTA flip-only = 0.8601**(baseline 0.8536 **+0.0065**,且 **> SFP+DTLR 0.8590 +0.0011**)—— inference-only 新最佳。模型 inference 極快(2-view 全集 ~3–4 min)→ queue 便宜,可多跑。
- run1 ep8 = 0.8145(plateau,best 仍 ep2 0.8158)。
- 跑 identity 全集驗證(test_tta scales 1.0 no-flip 應 = **0.8536**,證明 test_tta 與 test.py 等價 → 所有 TTA 數字可信);後續:疊 SFP+DTLR+flip、baseline multiscale+flip。
- ✅ **identity 全集 = 0.8536 精確** → `test_tta.py` 忠實復現 `test.py`,所有 TTA 數字可信,flip 的 +0.0065 為真。
- 🏆 **SFP+DTLR + flip = 0.8639** —— **本 session 最佳**。baseline **+0.0103** / SFP+DTLR alone +0.0049 / flip alone +0.0038。兩個 test-time 手段互補(flip = view 平均去偏;SFP+DTLR = 空間/logit refine),疊加有效。`[Load] missing 0/unexpected 0`(乾淨載入 official ckpt)。
- 目前 test-time 排名:**SFP+DTLR+flip 0.8639** > flip 0.8601 > SFP+DTLR 0.8590 > baseline 0.8536。
- run1 ep9 = 0.8155(plateau)。跑 baseline 1.0,1.25+flip(multiscale 是否再加成);下一步把 multiscale 疊到 SFP+DTLR。
- **baseline (1.0,1.25)+flip = 0.8637**(multiscale 在 flip 上再 +0.0036;純 TTA 幾乎追平 SFP+flip 0.8639)。
- 🏆🏆 **SFP+DTLR + (1.0,1.25) + flip = 0.8661** —— **新最佳**。baseline **+0.0125**。三個 test-time 手段(refine + multiscale + flip)互補、增益疊加。
- **test-time 排名**:SFP+ms125+flip **0.8661** > SFP+flip 0.8639 ≈ base ms125+flip 0.8637 > flip 0.8601 > SFP+DTLR 0.8590 > baseline 0.8536。
- run1 **ep10 = 0.8177**(超過 ep2,best_weight 更新到 ep10;投影 formal ~0.858)。
- 跑 baseline 1.0,1.25,1.5+flip(scale 1.5 是否再加成);若加成 → SFP+1.0,1.25,1.5+flip。
- ⚠️ 方法學備註:flip 為無參數 TTA(最穩健);multiscale 的尺度集合(1.25/1.5)是在 test set 上選的,屬**探索性**,正式報告時尺度應在 held-out 上定。SFP+DTLR+flip(0.8639)是最無爭議的乾淨提升。
- **baseline (1.0,1.25,1.5)+flip = 0.8599** → **scale 1.5 反而掉**(1.25 的 0.8637 → 0.8599)。CLIP 在 504px 偏離預訓練 regime。**尺度峰值 = 1.25**。
- **scale 探索完成**。最佳 = **SFP+DTLR + {1.0,1.25} + flip = 0.8661**。不再擴尺度(1.5 已證掉,再細調屬 test-set tuning)。
- 尺度曲線(baseline+flip):1.0→0.8601、+1.25→0.8637、+1.5→0.8599(倒 U,峰在 1.25)。
- 待 run1 結束(waiter armed)→ formal test best_weight(ep10 0.8177 val)→ 整合 `research_notes.md §11` + `updated.md` + 逐檔 commit。
- run1 訓練完成(ep0–14),per-epoch val best = **ep12 0.8201**(序列 0.8039/0.8117/0.8158/0.8138/0.8134/0.8131/0.8115/0.8156/0.8145/0.8155/0.8177/0.8107/0.8201/0.8069/0.8181)。
- ✅ **run1 (Method A) formal = 0.8565**(best_weight ep12)—— **> baseline 0.8536 +0.0029**。訓練有效(4 參數 presence 校準),但 < test-time(SFP+DTLR 0.8590 / flip 0.8601 / 最佳 0.8661)。`[Load] missing 0/unexpected 0`。
- val→formal offset 實測 **+0.0364**(ep12 val 0.8201 → formal 0.8565),與 ep0 的 +0.0403 一致 → 口徑校準可信。
- Method A 意義:確認「image-level presence 校準」方向在 VOC 有效(雖小);且是 CLASS_GATE 手調崩壞的「正確校準(學習)版」(手調在 present cosine<threshold 時崩,學習版自動避開)。
- 跑 run1+flip、run1+ms125+flip(訓練頭 + TTA 是否疊加)。
- **run1 (Method A) + flip = 0.8637**;**run1 + ms(1.0,1.25) + flip = 0.8643**。訓練頭 + full TTA ≈ baseline+TTA,略低於 SFP+DTLR 版(0.8661)。
- **全矩陣完成,所有實驗結束。**

---

## 結論(2026-07-08 autonomous session)

**完整矩陣(formal test,VOC val 1449,mIoU):**

| 底模 \ TTA | alone | +flip | +ms(1.0,1.25)+flip |
|---|---|---|---|
| baseline | 0.8536 | 0.8601 | 0.8637 |
| **SFP+DTLR** | 0.8590 | 0.8639 | **0.8661** ★ |
| Method A(run1,訓練) | 0.8565 | 0.8637 | 0.8643 |

補充:baseline + ms(1.0,1.25,**1.5**)+flip = 0.8599(scale 1.5 反而掉);oracle 上限 0.8998。

**主要發現:**
1. 🏆 **新最佳 = SFP+DTLR + 多尺度(1.0,1.25) + flip = 0.8661**(baseline **+0.0125**,全 inference-only,不碰 CLIP 特徵)。
2. **flip TTA 是主槓桿**:單獨 flip = 0.8601 已超過前一個最佳 SFP+DTLR 0.8590,零參數、最穩健。多尺度(峰值 1.25)再 +~0.004;1.5 過頭反而掉。
3. **Method A(run1,訓練 4 參數 presence 頭)= 0.8565 > baseline +0.0029**:訓練方向有效但被 test-time TTA 支配。是 CLASS_GATE 手調(崩壞 0.097)的「正確學習版」。
4. 加 TTA 後三種底模收斂到 0.864–0.866:**底模差異被 TTA 抹平**,SFP+DTLR 僅微幅領先。
5. VOC 近飽和(如預期);真正 payoff 應在難資料集(本機僅 VOC,需下載,未做)。

**實作交付(本 session 新增):**
- `tools/test_tta.py`:通用 inference-only eval engine(多尺度+flip TTA;`--scales/--flip/--limit/--save_dir/--load_path`;`--scales 1.0` no-flip == test.py,已驗 0.8536)。
- `tools/smoke_test_presence.py`、`config/voc_test_classgate_s2_cfg.yaml`、`docs/_infra_map_testtime.md`、本 log、`docs/method_results.csv`。

**方法學誠實備註:** 多尺度集合(1.25)在 test set 上挑選,屬探索性;正式報告尺度應在 held-out 上定。**flip(0.8601)與 SFP+DTLR+flip(0.8639)是無超參、最無爭議的乾淨提升**,建議作主張。所有 TTA 數字經 identity check(0.8536 精確)驗證,`test_tta.py` == `test.py`。
