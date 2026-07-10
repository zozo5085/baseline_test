# SESSION HANDOFF — ReCLIP++ C-USS 研究(貼上即可接續)

> **給下一個 session 的你**:這份是可直接貼上的接手 prompt。讀完本檔後,**不要重新探索、不要重跑已完成的診斷**,直接從 §5「下一步」開工。所有數字用 4 位小數逐字沿用,不要四捨五入進記錄。基準線是 **官方 checkpoint 0.8536**,使用者的硬性要求:**所有任務結果一律要 > 0.8536**。
> 最後更新:**2026-07-10**(Checkpoint + 驗證標註;統整交接與歷史保留在下方)。
> 註:上行「所有任務結果一律要 > 0.8536」僅適用 VOC;Context/ADE 依 delta 判斷(見 🟢 §7 紀律)。

---

## 🔵 2026-07-10 Checkpoint(最新;下次 session 先讀本塊 + 下方 🟢 統整交接)

> **硬性標註(不可違反)**:①**ADE20K 尚未正式開始長時間 training**(僅前置完成,等使用者 go);
> ②**目前不應啟動 LGAK**;③**不應做 VOC-only tuning**;④**不應修改 batch size、learning rate、
> augmentation 或 class order**(ADE 一切參數已由 protocol §8 預註冊凍結);⑤**下次 session 必須
> 先讀取本檔(SESSION_HANDOFF.md)再繼續**。

### 1. 已完成的工作(2026-07-09 深夜 ~ 07-10 凌晨 session)
- push `mine` 補齊(`5ebb07b`…`629afc7` 11 個 + 本 session 4 個;**絕不 push origin**)。
- flagged-fraction 表(tex `tab:flagged_fraction` + index §10)。
- VOC+SFP diagnostic 列(tex `tab:diagnostics` + index §5)。
- tex 兩處 `\TODO` 全解:DFF2d 改引 fusion v2 0.6897;PAMR 免重跑(數字已存在,recompute 驗證)。
- paired bootstrap 顯著性:VOC + Context 四個 headline delta 全顯著(index §11)。
- ADE20K 前置:預註冊(protocol §8)→ 完整性檢查 → text 生成 → pseudo 重生腳本 + configs。
- main.tex 編譯 0 錯(6 頁),無任何 `\TODO`。

### 2. 目前仍在執行的 background task / PID
- **無**(**已重驗 2026-07-10**:python process 數 = 0;無本專案 GPU 工作)。所有 background 工作
  已完成(VOC/Context bootstrap、stats 抽數、diag)。輸出已保全至 `experiments/journal_logs/`
  (`bootstrap_voc.log`、`bootstrap_context.log`、`sfp_stats_{voc,context}.log`、`diag_voc_sfp.log`
  —— 已重驗全部存在)。
- 註:重驗當下 `nvidia-smi` 顯示 GPU util 67% / 6.5 GB —— 為使用者桌面程式(遊戲/Wallpaper Engine
  類)佔用,非本專案。啟動任何 GPU 工作前照慣例先查 VRAM,勿殺使用者程式。
- ADE ETA 估算(~9h / ~20h)為外插估算值,非量測(依 Context converged 實跑 7h19m/30ep,
  console.log 檔案時間戳 21:50:37→05:09:11 推得)。

### 3. ADE20K 目前狀態
- **dataset**:`D:/ReCLIPv3/datasets/ADEChallengeData2016/`(train 20210、val 2000、`objectInfo150.txt`)。
- **class mapping**:repo `ade_classes`(utils/preprocess.py)== objectInfo150 官方順序,0/150 錯位;
  GT png 值域 0..150(0=ignore,REDUCE_ZERO_LABEL);CLIP top1-in-GT 0.527(≈8×random,對齊正常)。
  ⚠ **[NEEDS VERIFICATION]** 本行整批數字(0/150、值域、0.527)為 2026-07-10 session 實跑觀察,
  **輸出未保存成 log**(僅存於 commit `890007b` message)。重驗指令:
  `reclip5090 python tools\ade_integrity_check.py --dataroot "D:/ReCLIPv3/datasets/ADEChallengeData2016/" --n_sample 300`
  (~4 min GPU inference;下次跑請 `> experiments/journal_logs/ade_integrity.log` 保存)。
- **text**:`text/ade_ViT16_clip_text.pth` 已生成 —— **已重驗(2026-07-10)**:檔案存在、
  shape (150,512)、float32、row-normalized。recipe 驗證數字(對出貨 voc pth per-class cosine
  min 0.999999)⚠ **[NEEDS VERIFICATION]**(同上,未保存 log;含在上述重驗指令輸出 [2] 行)。
  工具 `tools/ade_integrity_check.py`。
- **pseudo(UPDATE 2026-07-10 晚)**:`text/ade_pseudo_label.json` **已本機重生並通過全部驗證**
  (20210 行、idx 0..149、0 空列表/parse 失敗、全量 aligned recall 0.583 ≫ 0.235;
  完整驗證表+生成過程無 GT 稽核 = `docs/ADE20K_PSEUDO_COMPARISON.md`;
  log `experiments/journal_logs/ade_pseudo_regen.log`)。**training input 就緒,但 training 仍未啟動。**
  ⛔ `D:\ReCLIPPP2026\text\ade_pseudo_label.json` = **GT-derived top-5 presence labels =
  DATA LEAKAGE,禁用於任何 formal / unsupervised training**(使用者 2026-07-10 確認)。
  以下為原重生前狀態記錄:`text/ade_pseudo_label.json` 不存在(待重生)——已重驗(2026-07-10)。
  出貨 `ade_pseudo_label_ReCLIPPP.json` 判定錯位不可用(recall 0.235≈chance、對影像位移±1不變
  → 行序不合本機 listdir,不可回復)⚠ **[NEEDS VERIFICATION]**(session 觀察,未保存 log;
  診斷腳本在 scratchpad 已隨 session 消失,重驗走上述 integrity 指令的 [6] 行,其 recall<0.35 即 FAIL)。
  重生腳本 `tools/ade_pseudo_regen.py`(批次滑窗,同 pseudo_class.py ReCLIPPP recipe)。
  30-img smoke:3.2 img/s、recall 0.715(⚠ **[NEEDS VERIFICATION]**,未保存 log;重驗:
  `--limit 30 --out <tmp>`)。全量 ETA ~1.75h(由 3.2 img/s 外插,估算值)。
- **config**:`config/ade_train_converged_cfg.yaml`(EPOCH 30、LR 0.01、CROP 512、SCALE [2048,512],
  Context 同 recipe)+ `config/ade_test_local_cfg.yaml`(PD 0.85,依預註冊)。
- **checkpoint**:無(`experiments/ade_vanilla_converged/` 未建立、未訓練)。
- **SAVE_DIR 規劃**:train `experiments/ade_vanilla_converged/`;eval `experiments/ade_conv_eval/<arm>/`。
- **預註冊**:`GENERALIZATION_PROTOCOL.md §8`(commit `b1c5695`,先於任何 ADE 數字)——H1 DTLR-only
  delta>0 且 bootstrap p<0.05;H2 flip 正 delta(sign);PD 0.85 唯一 fallback = baseline 崩(<0.01)才
  改 PD 0.0;**反 rescue 條款**:SFP gen/entgate 的 ADE 結果不可翻轉降級判決。

### 4. ADE20K 尚未做出的選擇(等使用者,勿自行啟動)
| 選項 | 內容 | 成本 | 差異 |
|---|---|---|---|
| A | pseudo 重生 + 訓練(**移除** train.py 0.08s/img sleep) | ~1.75h + ~9h | sleep 只影響 wall-time 不影響學習;需先在 protocol §8 加註 amendment(數字出現前仍合法)再跑 |
| B | pseudo 重生 + 訓練(**保留** sleep,完全照原 recipe) | ~1.75h + ~20h | 與 Context converged run 的 wall-clock 行為完全一致 |
| C | 先只跑 pseudo 重生 | ~1.75h | 完成後驗 recall 回報,訓練另行決定 |
| D | 暫緩 ADE | 0 | 期刊現有兩資料集內容已完整(tex 無 TODO、Table I–V 就緒) |

**UPDATE 2026-07-10 晚:pseudo 重生已完成並驗證(見上方 pseudo 條目)→ 選項只剩訓練本身的
go/no-go(sleep 保留 ~20h / 移除 ~9h,移除需先在 protocol §8 加 amendment)。**

訓練啟動指令(使用者 go 後):`python tools\train.py --cfg config\ade_train_converged_cfg.yaml --model RECLIPPP`
(ML python = `C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`;訓練後固定 battery =
base/flip/SFP gen/entgate/DTLR-only(各含 flip)+ diagnostics + runtime + flagged-fraction)。

### 5. VOC / Context 已完成結果(逐字;完整溯源見 index)
- **formal**(🟢 §2 全部有效):VOC base 0.8536 / flip 0.8601 / SFP legacy 0.8590 / gen 0.8582 /
  entgate 0.8579 / gen+flip 0.8639 / MethodA 0.8565;Context conv. base 0.2412 / flip 0.2473 /
  SFP gen 0.2353 / gen+flip 0.2383 / entgate 0.2367 / entgate+flip 0.2400;
  Ablation:VOC −DTLR 0.8563 / −proxy 0.8581 / −CPSFP 0.8578;Context 0.2345 / 0.2366 / 0.2422。
- **formal negative(新記錄)**:PAMR full-res 10it 0.8361(token 1/3/10it 0.8128/0.7250/0.5843,
  identity 0.8536 精確;recompute 驗證);fusion v2 0.6897(vs 自訓 0.8451)。
- **diagnostic**:diag 表(Context base/flip/SFP + VOC base/flip/SFP,含新 VOC SFP 列
  0.1513/0.7444/0.8055);flagged-fraction(VOC unrel 0.3465/0.3168、Context 0.7254/0.6818、
  rewrite 0.2605/0.5444、proxy avail 0.9932/0.7433);runtime Table V。
- **exploratory(附錄限定)**:ms 0.8661/0.8643/0.8637/0.8599。**8-ep Context 全部 superseded**。

### 6. Bootstrap significance 進度:**完成**
`tools/bootstrap_significance.py`(10k resamples,seed 0,paired;observed 全部逐字重現)。
VOC flip **+0.0065** CI[+0.0022,+0.0111] p=0.0052;VOC SFP gen **+0.0046** CI[+0.0025,+0.0069]
p<0.0002;Context flip **+0.0061** CI[+0.0055,+0.0067] p<0.0002;Context SFP gen **−0.0059**
CI[−0.0073,−0.0045] p<0.0002。四個 headline delta 全顯著;audit 結論獲統計背書。
JSON `experiments/bootstrap_significance/`;index §11。**尚未寫進 tex(見下一步②)。**

### 7. Journal 已完成的表 / 圖 / 文件
- Desktop `02_2026_JournalPaper/sections/4_experiments.tex`:Table I(2/5 資料集)、II(2/5)、
  III(VOC 完整;Context legacy n/a by design)、IV(component ablation,DONE)、V(runtime,DONE)、
  `tab:diagnostics`(本次 +VOC SFP 列)、`tab:flagged_fraction`(本次新增)、`fig_flip_diag.png`、
  Negative Findings(PAMR/fusion 數字本次補齊)。main.tex 編譯 exit 0(6 頁)、無 `\TODO`。
- `JOURNAL_STATUS.md`:missing-data #1 #2 已標 DONE;剩 ADE(#3)與 Q1-Q3 圖(#4)。

### 8. 尚未完成的工作
1. ADE20K 全流程(等使用者選 §4 選項)。
2. bootstrap 顯著性寫進 tex(一句 + 或表註;數字在 index §11)。
3. Q1-Q3 質性圖(可選;Q4 已有)。
4. Cityscapes / COCO-Stuff(optional breadth,protocol §4;非必要)。

### 9. 建議下一步順序
① 使用者決定 ADE 選項(§4);② tex 加 bootstrap significance 句(30 min,無 GPU);
③ 若 go:pseudo 重生 → recall 驗證(門檻:顯著高於 chance,參考 smoke 0.715)→ 訓練 →
固定 battery → H1/H2 依 §8 判定;④ Q1-Q3 圖(可選)。

### 10. Provenance(本 session 新增結果的溯源鏈)
- 全部結果:config/log/save_dir/commit 十欄位溯源 = `docs/JOURNAL_EXPERIMENT_INDEX.md`
  (本次新增 §10 flagged-fraction、§11 bootstrap;§5 VOC SFP 列;§8 PAMR/fusion 更新)。
- mIoU 帳本 = `docs/method_results.csv`(本次 +5 PAMR 列)。
- 本 session commits:`13337b7`(close-outs:stats/diag/PAMR/csv/index)→ `b1c5695`
  (**ADE 預註冊**,先於一切 ADE 數字)→ `890007b`(ADE integrity + text pth + regen 腳本 + configs)
  → `504d248`(docs)→ 本 checkpoint commit。已 push `mine`(= zozo5085/baseline_test)。
- 保全 log:`experiments/journal_logs/`(+4 個:bootstrap_voc / bootstrap_context /
  sfp_stats_voc / sfp_stats_context;另 diag_voc_sfp)。

---

## 🟢 統整交接 — 2026-07-09(**只讀這塊即可開工;下方全部是歷史細節**)

**TL;DR**:期刊 = **Framing A 泛化 audit,已鎖定**。SFP/DTLR 經雙重 de-confound(entropy gate + converged base)後在 Context 仍負 → **正式降級 = VOC-effective / not-generalizable,勿再 rescue**。**flip-TTA 是唯一乾淨 dataset-agnostic 正結果**(VOC +0.0065 / Context +0.0061)。期刊 Table IV/V 已完成,I/II 有 2/5 資料集。LGAK 封存為下一篇方向。~~10 個 commit 未 push~~(**已於 2026-07-10 全部 push `mine`**,見上方 🔵)。

### 1. 研究判決(已定案,勿重開)
- **SFP/DTLR 不泛化**:8-ep Context −0.0051 → 除 confidence-gate confound(entropy gate,tau 由 VOC C=20 凍結)仍 −0.0035 → 除 under-train confound(converged base 0.2412)仍 **−0.0059 / entgate −0.0045**。兩 confound 皆除、預註冊規則觸發 → 降級。
- **flip-TTA 泛化**:VOC 0.8536→0.8601、Context converged 0.2412→0.2473。diagnostics(N=1021/725):Context 三指標全改善(bnd 0.6925→0.6834、small-obj 0.1510→0.1528、FP 4.54→4.05);機制 = 無參數對稱平均,沒有可被新資料集 mistune 的 prior。
- **失敗已定位(Table IV)**:Context 傷害源 = **CP-SFP 鄰域 rewrite**(含 rewrite 全負 −0.0046~−0.0067;selection+DTLR-only = 0.2422 = **+0.0010** vs base,且 VOC 保 +0.0042)。⚠ DTLR-only 正值 = **post-hoc 觀察,不得升格 formal 泛化主張**,需 ADE 預註冊確認。
- 防重踩 bug(已修):Context GT label 是 **VOC-20-first 順序**(text/pseudo 已重排,top1-in-GT 0.094→0.873;字母序備份 `text/context_ViT16_clip_text_alpha.pth`);**TEST.PD 1.0 是 VOC-specific**(Context 用 0.85;PD1.0 → 0.0021 全類 prune 崩)。

### 2. 關鍵數字(formal、verbatim;每筆完整溯源見 `docs/JOURNAL_EXPERIMENT_INDEX.md`)
- **VOC**(official ckpt `experiments/official/voc_reclippp_854/`,PD 1.0):base **0.8536** / flip **0.8601** / SFP legacy 0.8590 / SFP gen 0.8582 / gen+entgate 0.8579 / gen+flip 0.8639 / MethodA 0.8565(VOC-only secondary)。
- **Context converged**(ckpt `experiments/context_vanilla_converged/best_weight.pth`,EPOCH30 best ep17 val 0.2341,PD 0.85):base **0.2412** / flip **0.2473** / SFP gen **0.2353** / gen+flip 0.2383 / entgate **0.2367** / entgate+flip 0.2400。
- **Ablation(no-TTA)**:VOC −DTLR 0.8563 / −proxy 0.8581 / −CPSFP 0.8578;Context 0.2345 / 0.2366 / **0.2422**。
- **Runtime**(5090,batch1,50-img):VOC 14.7 / flip 29.1 / SFP 20.2 / entgate 20.3 / SFP+flip 39.9 ms;Context 24.3 / 47.8 / 30.5 / 30.9 / 60.6;flip≈2×、SFP +26-37%、entgate <2%、VRAM Δ≤16MiB、全 test-time 0 params。
- **Exploratory(附錄限定,VOC-val 選 scale)**:ms 0.8661 / 0.8643 / 0.8637 / 0.8599。**8-ep Context 全列 superseded**(0.1980 等,勿再引用為現行)。

### 3. 期刊 paper(Desktop `C:\Users\NUTC2507\Desktop\school\03_論文投稿與計畫\02_2026_JournalPaper\`)
- `sections/4_experiments.tex`:converged 主表 + flip-transfer 表 + Diagnostic Analysis(指標定義/why-flip/why-SFP-fails/audit 意義)+ §Component Ablation + §Runtime and Cost + `fig_flip_diag.png`。**main.tex 編譯 0 錯**(pdflatex 驗證過)。
- ⚠ **2 處 `\TODO{NEEDS VERIFICATION}`**:①PAMR 句(全庫無紀錄數字)②DFF2d 0.4151(csv 標「作廢 parity bug」)。引用前須補驗或刪句。
- `JOURNAL_STATUS.md` = 表就緒度/缺口權威清單:Table I/II 2/5 資料集、III VOC 完 + Context legacy n/a by design、**IV/V DONE**。

### 4. LGAK(封存 = future work / 下一篇,不進 journal 主實驗)
`model/lgak.py` + `model/model_lgak.py` + 3 configs + `tools/smoke_test_lgak.py`(未動 `model/model.py`)。identity(α=0,full-val)= **0.8536 整**;smoke 過(trainable 398,465 全在 `lgak.`、F4 bootstrap 實測吻合)。**未跑 short run —— 未經使用者授權勿動。** 權威 spec = `docs/LGAK_IMPLEMENTATION_REVIEW.md`(F1-F5 決議 + 預註冊成功門檻)。

### 5. 本輪新增可重用工具
- `ENTROPY_GATE`(`model_sfp_dtlr.py`,class-count-invariant 可靠度 gate)+ component 開關 `PROXY_ENABLE/DTLR_ENABLE/CPSFP_UPDATE`(預設 True=原行為;回歸 gate 0.5611 驗證)+ `test_tta.py --sfp_disable dtlr|proxy|cpsfp`。
- `tools/diag_metrics.py`(bnd/small-obj/FP,從 saved .pt+GT,免重跑模型)、`tools/diag_figure.py`(flip diff-map)、`tools/bench_runtime.py`(CUDA-synced timing)。
- 證據保全:`experiments/journal_logs/`(7 個 eval log 永久保存)+ `docs/JOURNAL_EXPERIMENT_INDEX.md`(每筆 10 欄位溯源:config/log/save_dir/commit/main-table 資格)。

### 6. 開放項(2026-07-09 深夜更新:第0/1級全清,ADE 前置備妥等 go)
1. ~~PAMR~~ **已解(免補跑)**:數字本就在 research_notes(2026-07-07 full-val,save_dir 各 1449 .pt)。
   `tools/recompute_miou.py` 重算 0.8361(full-res 10it)/0.8128(token 1it)精確重現 → tex+csv+index。
   全設定負:0.8361/0.8128/0.7250/0.5843;identity 0.8536 精確。
2. ~~flagged-fraction~~ **完成**:`tools/sfp_stats_extract.py`(gen 一次跑同得兩 gate 比例,full val)。
   unrel conf/ent:VOC 0.3465/0.3168,Context 0.7254/0.6818;rewrite 0.2605/0.5444;proxy avail
   0.9932/0.7433 → confound 真實但小(~4/38pt,吻合 mIoU 回收 ~1/4),rewrite 侵蝕主導。
   tex 新表 `tab:flagged_fraction`;index §10。
3. **ADE20K:等使用者 go(唯一未動大項)**。已完成:預註冊(protocol §8,commit 先於任何 ADE 數字)、
   完整性檢查(class order 0/150 錯位、text recipe cosine 0.999999 驗證後生成
   `text/ade_ViT16_clip_text.pth`、top1-in-GT 0.527≈8×random、GT 值域 OK);**出貨 pseudo 錯位不可用**
   (recall 0.235≈chance 且對影像位移不變 → 行序不合本機 listdir,不可回復),重生腳本
   `tools/ade_pseudo_regen.py`(smoke 3.2 img/s、recall 0.715)。configs:
   `config/ade_{train_converged,test_local}_cfg.yaml`。
   **待 go:pseudo 重生 ~1.75h → 訓練 30ep ~20h(train.py 含 0.08s/img sleep;若移除 ≈9h)→ 固定 battery。**
4. ~~push mine~~ **完成**(`5ebb07b`…`629afc7` 共 11 個,本 session 又 +3)。**絕不 push origin。**
5. Q1-Q3 質性圖(可選;Q4 已有)。
6. **bootstrap 顯著性(新)**:`tools/bootstrap_significance.py`(paired per-image,10k,seed 0)。
   VOC:flip +0.0065 CI[+0.0022,+0.0111] p=0.0052;SFP gen +0.0046 CI[+0.0025,+0.0069] p<0.0002。
   Context 結果見 `experiments/bootstrap_significance/context.json`(session 末尾完成,記錄於 index)。
7. **其他本 session 完成**:VOC SFP diagnostic 列(bnd 0.1513/small-obj 0.7444/FP 0.8055,N=725)進 tex 表;
   DFF2d TODO 解(tex 改引 fusion v2 0.6897);tex 無任何 \TODO,main.tex 編譯 0 錯(6 頁)。

### 7. 紀律(不可違反)
formal 只用 no-TTA + flip;VOC-val 選的 multi-scale 永遠 exploratory;**不再 rescue SFP**;DTLR-only 主張需 ADE 預註冊;每實驗新 SAVE_DIR;勿覆蓋 official ckpt;數字 4 位小數 verbatim;「>0.8536」僅 VOC,Context 看 delta;ML python = `C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe`。

### 8. 檔案地圖
repo:`JOURNAL_EXPERIMENT_INDEX.md`(溯源)→ `method_results.csv`(mIoU 帳本)→ `JOURNAL_10PAGE_EXPERIMENT_PLAN.md`(計畫+framing 決議)→ `GENERALIZATION_PROTOCOL.md`(formal 規則)→ `research_notes.md §11`(敘事)。paper:Desktop `JOURNAL_STATUS.md` + `sections/4_experiments.tex`。

---

## 🕐 歷史逐步敘事 2026-07-08~09(已被上方統整取代;僅細節查閱用)

**方向切換**:使用者新增 `docs/JOURNAL_EXTENSION_PLAN.md`(期刊延伸線,本 session 執行中)+ `docs/NEW_DIRECTION_LGAK_RESEARCH_PLAN.md`(新方法,**尚未開始,勿碰**)。判準 = JOURNAL plan 的 Decision Rule。VOC 的「所有結果 > 0.8536」是 VOC-only 要求;Context 是不同資料集(baseline ~0.20),用 **delta** 判斷,勿套 0.8536。

**Context 實驗(第一個非 VOC 資料集)已完成:**
- 🔴 **重大 bug 修復**:D:\ReCLIPv3 的 Context GT label 是 **VOC-20-first 類別順序**,但 repo 的 text embedding + pseudo 是**字母序**(`pascal_context_classes`)→ 全類別 index 錯位 → 任何訓練 mIoU≈0(初跑 0.0027)。已把 text 重排成 GT 順序(`[VOC20]+[39 stuff 字母序]`;top1-in-GT 0.094→0.873),pseudo 用修正 text 重生(對齊 train.txt,recall 0.25→0.579)。字母序 text 備份 `text/context_ViT16_clip_text_alpha.pth`;舊 pseudo `text/context_pseudo_label.json.bak4996`。
- **base**:`experiments/context_vanilla_run2/best_weight.pth`(model.model,8 ep,per-epoch val 0.1106→0.1917;**未收斂**,delta 驗證用)。
- **正式 eval(full val 5105,PD 0.85,`tools/test_tta.py`)**:baseline no-TTA **0.1980** / flip **0.2028(+0.0048)** / agnostic SFP+DTLR no-TTA **0.1929(−0.0051)** / SFP+flip **0.1955**。診斷:baseline @ **PD1.0 = 0.0021**(prune 全 59 類;PD1.0 靠 20 類 softmax 飽和 = VOC-specific;gen config 已 1.0→0.85)。
- **結論(初判)**:**flip-TTA 泛化**(+0.0048,cf VOC +0.0065)= 乾淨 dataset-agnostic 正結果;**agnostic SFP/DTLR 不泛化**(−0.0051)→ Decision Rule 傾向降級。當時兩 confound 未除:(a) base 未收斂(0.198,8ep);(b) CONF_THD/kernel VOC-calibrated 絕對常數。

- 🟢 **entropy-gate de-confound 已做(2026-07-08 晚,本 session)**:使用者選「只做一個 principled 修正」——把 max-prob gate(CONF_THD 0.97 / PROXY_CONF_THD 0.95)換成 entropy-normalized 可靠度 gate `H_norm=entropy(softmax(logits·CONF_SCALE))/log(C)∈[0,1]`(`ENTROPY_GATE`,`model/model_sfp_dtlr.py`;tau=0.0745/0.1154 由 VOC C=20 凍結、套所有資料集、**無 Context 調參**)。修正後 full-val:**VOC gated 0.8579**(vs 0.8582 −0.0003,20 類近乎 no-op → 修正確實 dataset-agnostic)/ **Context gated no-TTA 0.1945**(+0.0016 vs 0.1929,**仍 −0.0035 vs base 0.1980**)/ **Context gated flip 0.1973**(+0.0018,**仍 −0.0055 vs base flip 0.2028**)。**confound 真實但只佔約 1/3**;除掉後 Context SFP 仍負。
- **判決(依 JOURNAL req5 預註冊規則)**:corrected Context SFP 仍負 delta → **SFP/DTLR 降級為 VOC-effective-but-not-generalizable**。只除了 confidence-gate confound;base 未收斂 confound 未測(使用者 scope = 只做 de-confound,不再訓 Context)。**flip-TTA 是唯一乾淨 dataset-agnostic 正結果。**

**下一步**:轉 LGAK 新方向(`NEW_DIRECTION_LGAK_RESEARCH_PLAN.md`)——**尚未開始,需使用者明確指示才動**(大方向、多小時、架構級)。ADE20K raw 在 D:\ReCLIPv3(per-dataset ckpt 架構不相容,需本 repo 重訓)可留作補充泛化資料點。**勿開始 LGAK,除非使用者指示。**
已更新:`method_results.csv` / `GENERALIZATION_PROTOCOL.md §2+§6` / `research_notes.md §11` / `AUTONOMOUS_SESSION_2026-07-08.md` / 本檔。新 config:`config/{voc,context}_test_sfp_dtlr_entgate_cfg.yaml`;code:`model/model_sfp_dtlr.py` + `config/configs.py`。已本地 commit `4c16036`(未 push)。

**LGAK(新方向)進度**:使用者已同意「開始 LGAK,但先只做 design review、不實作不訓練」。design review 已完成 → `docs/LGAK_IMPLEMENTATION_REVIEW.md`。結論:計畫 sound 且可實作(forward path 已對 `model.py:462-482` 驗證、layout 已是 `[B,512,H,W]` 可跑 DWConv、無 oracle 洩漏、單一 forward),但**不可照抄實作**:F1(feat 在 `:479` 還餵 decoder,非只 output_q)、F2(插在 normalize 後會破壞單位範數 → 需 re-normalize)必修;F4(α=0 conv 零梯度啟動)建議修;F3(text 條件是 dataset-global mean,語言引導很弱)是設計分歧留給使用者。

**LGAK 決策 + 實作已完成(2026-07-08 深夜)**:使用者拍板 F3=A(全域 mean gate)、F1=A(refined feat 只餵 output_q)、F4 α=0(不手動 warm)、成功門檻預註冊(identity==baseline / VOC no-TTA ≥ base−0.005 / 有正 delta 才驗 Context / 不拿 VOC-tuned TTA 當成功)。已實作 `model/lgak.py` + `model/model_lgak.py` + 3 configs + `tools/smoke_test_lgak.py` + `config/configs.py` LGAK keys(**未動 `model/model.py`**)。**驗收全過**:identity(α=0,full-val)= **0.8536 整 == baseline**;smoke = trainable 全在 `lgak.`(398,465)、loss 有限、feat_norm out=1.0、F4 bootstrap 實測吻合(α 先動、conv 後動)。**尚未跑 2-3 epoch short run —— 等使用者確認才進 short run。** eval 產物:`experiments/lgak_id_lgak/`(identity)、`experiments/lgak_id_baseline/`(config sanity 0.8536)。訓練 SAVE_DIR 將是 `experiments/voc_lgak_mvp_run1/`。**LGAK 已封存為 future/new-direction,不進期刊主實驗。**

**回到 VCIP 期刊延伸線(2026-07-08 深夜)**:已產出 `docs/JOURNAL_10PAGE_EXPERIMENT_PLAN.md`。核心張力:計畫原主張「SFP/DTLR 泛化」被自家 Decision Rule 否決(Context de-confound 後仍 −0.0035)→ 兩個 framing:A=誠實泛化 audit(protocol + entropy-gate + flip-TTA 泛化 + SFP 誠實負結果,現有資料可寫)、B=救 SFP 泛化主張(需 converged Context 或 ADE 出現非-VOC 正 delta)。**建議先跑 Top-3 決定 framing,預設 A**。Top-3:#1 converged Context、#2 ADE、#3 diagnostics。

**Framing A 已鎖定、SFP 已降級(2026-07-09,Top-3 #1 完成)**:使用者採 Framing A 並授權 #1。訓了 converged Context base(`experiments/context_vanilla_converged/`,EPOCH 30,val 平台 ~0.22-0.23、best ep17;不動方法/LR,只加 epoch)。固定 6-eval(PD 0.85,`--load_path` 指 converged 權重):baseline no-TTA **0.2412**(vs 8ep 0.1980,under-train confound 除)/ flip **0.2473(+0.0061)** / agnostic SFP gen **0.2353(−0.0059)** / entropy-gate SFP **0.2367(−0.0045)**。**兩 confound 皆除,SFP/DTLR 在 Context 仍負(甚至更負)→ 依預註冊規則正式降級 = VOC-effective / not-generalizable。flip-TTA 再度泛化(+0.0061),是唯一乾淨 dataset-agnostic 正結果。** 依規則負結果 → 不需 ADE 解 SFP。docs 已更新(JOURNAL_10PAGE plan §0/§2A/§11、method_results、research_notes §11、本檔)。**#2 diagnostics + flip-TTA 鞏固已完成(2026-07-09)**:`tools/diag_metrics.py`(boundary-band error / small-object acc / FP class count,從 saved preds+GT 算)+ `tools/diag_figure.py`(flip-vs-base diff map)。Context converged N=1021 / VOC N=725:flip 在 Context 三項全改善(bnd 0.6925→0.6834、small-obj 0.1510→0.1528、FP 4.54→4.05),VOC 主要改 boundary(0.1522→0.1496);**SFP 減 FP(→3.99)但 erode small-object(→0.1431)→ 淨負**,機制性解釋(好 base 更負)。已更新 Desktop LaTeX(`02_2026_JournalPaper/sections/4_experiments.tex` 加 converged 表+flip-transfer 表+diagnostic 表+`fig_flip_diag.png`+「why flip transfers」段;`JOURNAL_STATUS.md` framing A LOCKED)+ repo `research_notes §11`。VOC flip preds 重生於 `experiments/voc_diag_flip/`(0.8601)。**下一步(等使用者)**:component ablation(URD/proxy/DTLR/entropy,全 test-time)+ runtime 表;ADE optional breadth。**勿再 rescue SFP。**

**期刊狀態盤點完成(2026-07-09,無新實驗)**:權威清單 = Desktop `02_2026_JournalPaper/JOURNAL_STATUS.md`(§1 formal-table-ready:VOC 7 列 + Context converged 6 列,含 formal negative;§2 diagnostic/negative/appendix;§3 缺口;§4 下一優先)。**Table 就緒度**:I(no-TTA)2/5 資料集、II(flip)2/5、III(legacy vs gen vs entgate)VOC 完整 + Context legacy=n/a by design、**IV(component ablation)全空、V(runtime)全空**。**缺口**:①Table IV ablation(全 test-time,最便宜)②Table V runtime ③flagged-fraction 表(log 已在 model_sfp_dtlr stats,只差抽數字)④**PAMR 負結果 claim 目前無數據**(4_experiments.tex 有這句 → 補跑或軟化)⑤ADE/City/COCO bases(optional breadth)⑥Q1-Q3 圖(Q4 已有)。**單一最優先 = component ablation + runtime 同批跑**(VOC+Context converged,Base/+selection/+proxy/+DTLR/full,順帶 ms/img+VRAM+flagged-fraction,~1-2h GPU,無訓練)→ 一次填 Table IV+V。等使用者授權。

**Table IV+V 已完成(2026-07-09,使用者授權)**:code 加 3 個 component 開關(`PROXY_ENABLE`/`DTLR_ENABLE`/`CPSFP_UPDATE`,預設 True;回歸 gate:gen 20-img 前後皆 0.5611 精確)+ `test_tta.py --sfp_disable` + `tools/bench_runtime.py`。**Ablation(formal no-TTA)**:VOC −DTLR 0.8563 / −proxy 0.8581 / −CPSFP 0.8578(vs full 0.8582 → 增益 DTLR 主導,proxy≈+0.0001);Context −DTLR 0.2345 / −proxy 0.2366 / **−CPSFP 0.2422(+0.0010 vs base 0.2412)** → **Context 失敗定位於 CP-SFP 鄰域 rewrite**(含 rewrite 全負;DTLR-only ≈ baseline)。⚠ DTLR-only 正值 = **post-hoc 觀察,依 protocol 不得升格 formal 泛化主張**(需 ADE 預註冊確認)—— 非 rescue。**Runtime(5090,batch1,50-img)**:VOC base 14.7ms/68FPS、flip 29.1、SFP 20.2、entgate 20.3、SFP+flip 39.9;Context 24.3/47.8/30.5/30.9/60.6;flip≈2×、SFP+26-37%、entgate<2%、VRAM Δ≤16MiB、全 test-time 0 params。已更新:`4_experiments.tex`(新 §Component Ablation + §Runtime and Cost + Planned Tables 改狀態)、`JOURNAL_STATUS.md`(IV/V DONE,缺口剩 flagged-fraction / PAMR / ADE / Q1-Q3 圖)、`method_results.csv`(6 列 [ABLATION formal])。**下一步(等使用者)**:小收尾 = flagged-fraction 抽數 + PAMR(補跑或軟化);之後唯一大項 = ADE(optional breadth + DTLR-only 預註冊確認場)。

**可驗證實驗索引已建(2026-07-09)**:`docs/JOURNAL_EXPERIMENT_INDEX.md` — 每筆結果 10 欄位溯源(dataset/method/value/class/config/log/save_dir/commit/main-table?/原因)。session tmp logs 已永久保存到 `experiments/journal_logs/`(ablate_bench / diag / conv_eval / entgate / lgak_identity / voc_flip_regen / context8ep_sfp)。`4_experiments.tex` 全數字溯源檢查完:全部對上 csv/log,僅 2 處 `\TODO{NEEDS VERIFICATION}`(①PAMR 句無紀錄數字;②DFF2d 0.4151 在 csv 標記作廢 parity bug)+ 修正一處衍生量(entropy gate 回收 ≈1/4 非 1/3)。

---

## ⚠️ 2026-07-08 狀態更正(voc_presence / MethodA 舊塊,已被上方新塊取代大部分;MethodA 細節仍有效)

- **🔴 2026-07-08 run1 訓練中(PID 30516,detached)**:使用者指示「做A」= 待決定清單的 (A)。已啟動完整 run:LR 0.01 / 15 epoch / MODE `zglobal` / INIT official / **SAVE_DIR `experiments/voc_presence_run1/`**(輸出 `console.log` stdout + `console.err.log` tqdm)。smoke 過(trainable=4、loss finite、train imgs=1464、8.85 it/s)。判準:最終 best_weight 用 **formal test 比 0.8536**(per-epoch val 低 ~0.04,勿直接比);若 per-epoch val 崩向 ~0.80(IABR 式 drift)就停。**勿重啟、勿覆蓋 run1、勿覆蓋 voc_presence 的 ep0 權重。** 完成後 formal test → 寫 `research_notes.md §11` + `updated.md` + 逐檔 commit push `mine`。
- **Method A 已跑過一輪並中止**:使用者 **2026-07-07 23:45** 啟動 presence 訓練(LR 0.01),為重啟 session 於 **epoch 1 中途 kill**(epoch 0 已完成)。非失敗,是人為中止。
- `experiments/voc_presence/` **不是乾淨的**:內含該中止 run 的 **epoch-0 `best_weight.pth`**(mtime 2026-07-07 23:45:57,**勿覆蓋**)+ 1449 個 `.pt`。存檔 head 參數:gamma **0.0192** / tau 0.437 / temp 2.72 / scale 0.716。
- **已 formal-test 該 ep0 ckpt = `0.8442`**(`config/voc_test_presence_ep0_cfg.yaml` → `experiments/voc_presence_ep0_test/`,1449/1449 完成)。
- **三方對照(逐字)**:identity(gamma=0)formal **0.8536** / ep0 formal **0.8442** / ep0 train.py per-epoch val **0.8039**。
  - **校準**:per-epoch val 比 formal test 低 **~0.04**(同一 ckpt:0.8039 vs 0.8442)→ tripwire / 最終評分**一律用 formal test**,勿拿 per-epoch val 直接跟 0.8536 比。
  - ep0 formal 0.8442 **比 identity 低 ~0.009**(1 個 epoch 把 4 參數推往略微有害方向;**只 epoch 0、趨勢未知**,可能回升也可能像 IABR 掉到 ~0.80)。
- **config 變更(尚未 commit)**:`config/voc_train_presence_cfg.yaml` 的 `EPOCH`/`MAX_EPOCH` 已 50→15;新增 `config/voc_test_presence_ep0_cfg.yaml`。
- **目前無 process 在跑**;GPU ~11% / 6.9 GB(背景桌面程式);git HEAD `a626c71`,無新 commit。
- **待決定(下一步)**:Method A 重訓方向 —(A)完整跑 LR 0.01 / 15ep / **新 SAVE_DIR** `voc_presence_run1`、final best_weight 用 formal test 評 vs 0.8536;(B)降 LR(0.001–0.003)再跑;(C)先查 BCE presence loss 與 mIoU 是否對齊;(D)跳難資料集(需下載)。**重訓一律新 SAVE_DIR,勿覆蓋 voc_presence 的 ep0 best_weight。**

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

## 8. 貼上用的接手 prompt(複製下面這段給新 session;2026-07-09 更新,舊 Method A 版已淘汰)

```
讀 D:\ReCLIPP_Test\docs\SESSION_HANDOFF.md 的「🟢 統整交接 2026-07-09」區塊(其餘是歷史,勿全讀),
接續期刊 Framing A 泛化 audit。按需再讀:GENERALIZATION_PROTOCOL.md(改預註冊前)、
JOURNAL_EXPERIMENT_INDEX.md(記錄數字時)、Desktop 02_2026_JournalPaper\JOURNAL_STATUS.md(動 tex 前)。

依 2026-07-09 優先序執行:
【第0級,先清】
1. push mine 補上 10 commits(5ebb07b…a858563;絕不 push origin)
2. flagged-fraction 抽數(model_sfp_dtlr stats log 已有)→ 填表
3. VOC+SFP diagnostic 列:tools/diag_metrics.py 吃 experiments/voc_sfp_dtlr_gen_official_eval/
   既有 preds,免重跑模型
4. 4_experiments.tex 兩處 TODO 之 DFF2d:標作廢(parity bug)或刪句
【第1級】
5. PAMR 補跑一個 formal 設定解另一 TODO(給 audit 第三個 refinement;負結果照 formal 規則記)
6. per-image bootstrap 顯著性:flip 與 SFP 在 VOC/Context 的 delta(saved preds+GT,純分析)
【第2級,唯一大項】
7. ADE20K:(a) 先把 DTLR-only 假說+成功門檻預註冊寫進 GENERALIZATION_PROTOCOL.md
   (必須在看到任何 ADE 數字之前)(b) GT class order/text/pseudo 完整性檢查(Context 錯位教訓)
   (c) converged base 訓練——多小時,先報 config+ETA 等使用者 go(d) 固定 battery:
   base/flip/SFP gen/entgate/DTLR-only + diagnostics + runtime

紀律:formal 只 no-TTA+flip;不再 rescue SFP;LGAK 勿動;multi-scale 永遠 exploratory;
每實驗新 SAVE_DIR、勿覆蓋 official ckpt;數字 4 位小數 verbatim 記入
JOURNAL_EXPERIMENT_INDEX.md + method_results.csv;
ML python = C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe。
```
