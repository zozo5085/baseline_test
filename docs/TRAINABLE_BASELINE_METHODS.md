# Trainable Improvement Modules on the ReCLIP++ Baseline

> 設計文件(2026-07-07)。三個 **從 ReCLIP++ baseline rectification stage 出發** 的
> 小型、受約束、可訓練 improvement module。不基於 IABR。目標:改善 class
> hallucination / image-dependent bias / photometric instability / local uncertainty。
> 尚未實作、尚未實驗。

---

## 0. 硬性限制(全部方法共同遵守)

1. 不使用 distillation。
2. 不修改 CLIP image / text encoder(兩者永遠 frozen)。
3. 不直接替換 CLIP feature(`feat` / `v` 不得被覆寫)。
4. 不使用 pixel-level GT segmentation mask。**只允許 image-level 類別存在標籤**
   (`gt_cls`,即 ReCLIP++ 本來就在用的訊號)與自監督訊號。
5. 新增 module 必須 zero-init / identity-init,初始輸出 **逐位元等價** baseline。
6. 不允許自由大幅改動 `bias_logits`;必須有 baseline-preserving regularization。
7. 優先訓練 **class-presence / reliability calibration**,而非直接訓練 dense mask。
8. 方法必須 dataset-agnostic:VOC / Context / ADE20K / Cityscapes / COCO-Stuff 通用
   (無寫死類別索引、無絕對 token 數、閾值一律類別數/解析度相對)。

---

## 1. ReCLIP++ baseline 解剖(所有插入點的依據)

`model/model.py` `RECLIPPP.forward`(inference path,行號為 baseline):

```
v, shape, z_global, k, positional_embedding = self.vit(image)      # 459  CLIP frozen
feat = self.proj(v); feat = feat / feat.norm(dim=1, keepdim=True)  # 462-463  dense visual [B,512,H,W]
output_q = conv2d(feat, text_embeddings)                            # 468  cosine logits [B,HW,C]  (pre-bias)
prompt = text_encoder(cls_name_token); prompt = prompt/prompt.norm()# 472-473
pe = pe_proj(positional_embedding)                                  # 475
bias_logits = pe @ prompt.t()                                       # 476  STATIC, image-independent [1,HW,C]
output = output_q - bias_logits                                     # 477  ← rectification 的核心減法
feature = cat(feat, output); decoder_conv2; decoder_norm2           # 479-482  → 最終 logits
```

**可用張量(input features 的來源)**
| 張量 | 形狀 | 性質 | 對方法的價值 |
|---|---|---|---|
| `z_global` | [B,512] | 全域 CLIP 影像嵌入,**baseline 未使用** | 獨立於 dense 預測的 image-level presence 訊號(方法 A 主力) |
| `feat` | [B,512,H,W] | normalized dense visual | 不得覆寫(限制 3);可唯讀取統計 |
| `output_q` | [B,HW,C] | pre-bias cosine logits | per-class 空間統計(peak/margin/entropy)→ reliability |
| `bias_logits` | [1,HW,C] | 靜態 bias | 方法 B 縮放的對象 |
| `prompt` / text emb | [C,512] | 類別文字嵌入 | presence 的文字側 |

**baseline 的 7 個 trainable tensor**:`text_encoder.prompt_token`、`pe_proj.{weight,bias}`、
`decoder_conv2.{weight,bias}`、`decoder_norm2.{weight,bias}`。CLIP visual/text、`proj`、
`logit_scale` 全 frozen。

**baseline 訓練 loss(關鍵:本來就無 pixel mask)** — `model.py:487-508`:對每個 image-level
present class `i ∈ gt_cls`,用 gumbel-hard 把像素指派到各類,pool 被指派給 `i` 的區域特徵,
強制其 CLIP-text 分類為 `i`。即 **image-level tag 驅動的 region-prototype CE**。我們的新
loss 可沿用同一個 image-level 訊號,完全不碰 pixel GT。

---

## 2. 共同設計原則(直接來自今晚 IABR / fusion 的失敗教訓)

今晚兩次可訓練嘗試都以同一種方式失敗:**在無監督訊號下,高容量自適應模組漂移 ——
train-eval 好看、formal test 崩壞**(fusion:train 0.80 / test 0.69;IABR:峰值 epoch4
後單調下滑)。因此本文件所有方法遵守五條抗漂移原則:

- **P1 低容量**:新增參數是 per-class 純量 / 低秩,而非 IABR 那種 dense per-pixel field。
  容量小 → 無法擬合空間雜訊。
- **P2 well-posed loss**:訓練目標是 **image-level presence calibration / consistency**
  (有明確最優解),而非去 fit 雜訊 pseudo-mask。這是與 IABR 最本質的差異。
- **P3 zero-init + baseline-preserving reg**(限制 5、6):所有干預乘上 `tanh(γ)`,`γ=0`
  初始 → 逐位元等價 baseline;loss 含 `λ_reg·‖干預量‖`,只有訊號夠強才偏離 baseline。
- **P4 soft、非硬砍**:presence gate 用 log-domain 軟加,漏掉的 present class 只被輕微
  衰減、不歸零(今晚證實硬 gate 在 recall<1 時災難性,見 §0 diagnostics)。
- **P5 drift tripwire**:訓練中每隔數 epoch 跑 **formal `test.py` 評測**(非只看 in-training
  eval),若 formal-test 低於 baseline 0.8536 且 train-loss 仍下降 → 判定漂移,立即中止。

**共同評測底線**:identity check(`γ=0` → 必須 formal test = 0.8536 整)→ smoke(2 iter,
finite loss/grad)→ 訓練 → formal test。所有方法都有對應的 **test-time 版本**做對照
(訓練版必須贏過 hand-tuned test-time 版,才證明「訓練」有價值)。

---

## 3. Method A — Trainable Soft Presence Calibration Head

**動機失敗機制**:class hallucination + image-dependent bias。今晚 oracle 顯示完美
anti-hallucination = +0.0462(VOC 0.8998),而瓶頸是「高 recall 的 soft presence 估計」。
A 把一個 **校準過的 image-level presence log-prior** 學進來。

- **插入位置**:`output_q`(model.py:468)之後、減 bias(477)之前:
  `output_q' = output_q + tanh(γ) · a(x)_c`,`a(x)_c` 是 per-class(broadcast 到空間)的
  presence log-prior。放在 bias 前,讓後續 rectification 與 decoder 一起吸收。
- **input features**:主力 = `z_global`(**獨立於 dense 預測**),取 per-class
  `s_c = ⟨norm(z_global), norm(text_c)⟩`(image-text 相似度);輔以 `output_q` 的 per-class
  空間統計(max、mean、margin)當第二視角。全部是 per-class 純量。
- **trainable parameters**:一個 **跨類別共享** 的 tiny calibration head `g_φ`(對每個類別
  的特徵向量 `[s_c, max_c, mean_c, margin_c]` → presence logit `a_c`),參數量 ~數百;外加
  一個純量 gate `γ`。**共享權重 → 與類別數無關 → 天生跨資料集(限制 8)**。
- **zero/identity-init**:`γ=0` → `tanh(γ)=0` → `output_q' ≡ output_q` → baseline。`g_φ` 可
  任意初始,因為被 `γ` 關住。
- **loss(無 pixel mask)**:image-level multi-label **BCE**:head 預測 `a_c`,用 `gt_cls`
  的存在/缺席當標籤。這是純粹的 presence calibration(限制 7)。加:
  - baseline-preserving reg `λ_reg · ‖tanh(γ)·a‖²`(限制 6);
  - **非對稱權重**:present class 被壓(`a_c<0`)的懲罰 ≫ absent class 被留的懲罰 → 偏向
    high-recall,避免壓掉真實類別。
  可選:保留 baseline gumbel loss 讓 decoder 一致(或凍 decoder,只訓 head+γ)。
- **避免 drift**:P1(per-class 純量,無 dense 場)+ P2(BCE 有明確最優)+ P3 + P4。容量
  比 IABR 的 dense `local_gate` 小數個量級。
- **避免壓掉真實類別**:soft log 加(非 −inf)+ 非對稱 recall-oriented loss + `γ` 約束。
- **最小實驗流程**:凍 baseline 7 參數,只訓 `g_φ`+`γ`,image-level BCE + reg,50 epoch。
  identity check `γ=0`→0.8536;每 5 epoch formal test(P5)。
- **test-time 對照**:test-time 版 = 直接啟用 `model_feature_fusion` 那個停用的 `class_gate`
  (`sigmoid((z_global·text−thr)·temp)` 手調 thr/temp)。A 用**學的**校準取代手調 →
  直接量「訓練 vs 手調」的增益。
- **預期改善資料集**:類別多、混淆重的 **COCO-Stuff(171)/ ADE20K(150)** 最大;VOC 朝
  +0.0462 天花板小幅前進。
- **風險 / 失敗判準**:若學出的 gate 只打平(如今晚 hand-tuned peak-gate 0.8535),代表
  `z_global` 獨立訊號太弱 → 轉 Method C。失敗 = formal test < 0.8536,或 presence BCE 的
  val recall 未超過 baseline 隱含 recall。

---

## 4. Method B — Reliability-Guided Bias Scale Head

**動機失敗機制**:over/under-correction + image-dependent bias + local uncertainty。靜態
`bias_logits` 對每張圖、每個區域用同一力道減。B **不重生 bias(IABR 的錯)**,而是學一個
**reliability 引導的低維 scale** 去縮放既有 bias。

- **插入位置**:bias 減法(model.py:477):
  `output = output_q − s(x) · bias_logits`,`s(x) = 1 + tanh(γ)·h_ψ(r(x))`。
- **input features**:reliability 訊號 `r(x)` = `output_q` 的 per-class/per-pixel
  信心度:top1−top2 margin、entropy、以及今晚用的 peakedness。低信心 → 讓 bias 校正的
  力道被學到的 `s` 調整。image-level 另取 `z_global`。
- **trainable parameters**:tiny head `h_ψ`(reliability 特徵 → scale 偏移)+ 純量 `γ`。
  **關鍵約束**:`s` 限制為 **per-image 純量 / per-class 向量 / rank-1 平滑空間場**,
  **不得是自由 dense 場**(這正是 IABR `local_gate` 漂移的來源)。容量被結構性壓死。
- **zero/identity-init**:`γ=0` → `s≡1` → `output = output_q − bias_logits` = baseline。
- **loss(無 pixel mask)**:沿用 baseline 的 **image-level gumbel region-prototype CE**
  (縮放 bias 若改善 present-class 的區域原型一致性,loss 下降)+ baseline-preserving
  reg `λ_reg·‖s−1‖²`(限制 6)。即用與 baseline **相同的 image-level 訊號** 訓練,只優化
  reliability-scale。
- **避免 drift**:低維 `s`(非 dense)+ 縮放既有 bias(有界偏離)+ `γ` zero-init + `‖s−1‖`
  reg。容量遠低於 IABR。**P5 尤其關鍵**:B 與 IABR 最像,必須嚴盯 train-eval↑/test↓ 反轉,
  一出現立即中止。
- **避免壓掉真實類別**:`s` 只調 bias 力道且被 reg 拉回 1;硬約束 `s ∈ [1−ε, 1+ε]`。
- **最小實驗流程**:訓 `h_ψ`+`γ`,image-level loss + `‖s−1‖` reg,50 epoch。identity
  `γ=0`→0.8536;每 5 epoch formal test 並比對 train/test gap 方向。
- **test-time 對照**:test-time 版 = 固定全域 bias scale(我們已跑過 `fusion_gamma` 掃描的
  同型手法)。B 學 reliability-conditioned scale vs 固定 scale。
- **預期改善資料集**:邊界多、局部不確定性強的 **Cityscapes**;over/under-correction 明顯的
  資料集。
- **風險 / 失敗判準**:若 `s` 被 reg 壓成恆等 1 → 無增益但無害(可接受的 null);若出現
  IABR 式反轉 → 立即中止。失敗 = formal test < 0.8536 或 train/test 反轉。

---

> **[2026-07-07 UPDATE] Method C premise REFUTED** by a test-time diagnostic
> (tools/analyze_photometric_consistency.py, official ckpt, VOC val, K=5 views).
> Cross-view instability is dominated by signal MAGNITUDE: present classes flicker
> MORE (p95 cov present 0.0748 vs absent 0.0462), absent classes sit pinned at a
> stable near-zero floor. AUC of cov as an absent-detector = 0.186 (reverse of the
> hypothesis). Combining with peak-height wrecks precision (0.69→0.075). "Hallucination
> flickers" is FALSE on this checkpoint — hallucinations are photometrically STABLE.
> Method C is dropped. Only faint signal: z_global-cosine cross-view cov AUC≈0.65 (too
> weak to build on). Report: docs/diagnostics/photometric_consistency_premise.md.
> Consequence: pursue A/B instead; but first cheaply verify A's premise (see §7 note).

## 5. Method C — Photometric / Prompt Consistency-Trained Gate  [PREMISE REFUTED — see note above]

**動機失敗機制**:photometric instability +(經由一致性)class hallucination。**真實在場
類別在光度擾動 / prompt 改寫下穩定被偵測,幻覺類別閃爍。** C 用 **一致性** 當自監督訊號
訓練 presence/reliability gate。三者中最深、最 novel,且天生 dataset-agnostic(限制 8)。

- **插入位置**:與 A 相同的 soft presence gate(`output_q + tanh(γ)·a_c`),或 B 的
  reliability scale。**C 的區別在訊號/loss,不在插入點**。
- **input features**:multi-view —— 同一影像的 K 個 **光度擾動版**(brightness/contrast/
  color jitter;**不做幾何變換** 以保像素對應)過 **frozen** CLIP → K 組 `output_q` /
  `z_global`;per-class 一致性 = 各 view 的 presence/prediction 一致程度。可選 prompt
  改寫 view(每類多個 template → 文字嵌入變異)。
- **trainable parameters**:small gate head(per-class 一致性特徵 → gate)+ `γ` zero-init。
- **zero/identity-init**:`γ=0` → baseline。
- **loss(無 pixel mask,甚至可不用 tag)**:**consistency loss** —— 懲罰各 view 間 per-class
  presence 的不一致;訓練 gate 去 **降權不一致(閃爍=幻覺)類別、保留一致(在場)類別**。
  可加輕量 image-level BCE 當 aux。baseline-preserving reg on `γ`。
- **避免 drift**:一致性目標 **well-posed**(最優解是「對光度不變」,不獎勵退化式抑制)——
  這是三者中最強的抗漂移論證,遠優於 fit pseudo-label。加 zero-init + reg。
- **避免壓掉真實類別**:一致(present)類別依定義被保留,只降權閃爍類別;soft gate。
- **最小實驗流程**:K=4 光度 view,consistency loss,50 epoch。identity `γ=0`→0.8536。
- **test-time 對照**:**純 test-time 版完全不訓練** —— 推論時跑 K view、量一致性、soft
  gate(即先前提過的 training-free consistency gate)。C(訓練版)學「一致性→gate」的映射
  vs 手設規則。乾淨的 train-vs-test-time ablation。
- **預期改善資料集**:光度變異大的 **Cityscapes**;以及 **跨資料集 transfer**(一致性
  dataset-agnostic → 對限制 8 / 泛化主張最有利)。
- **風險 / 失敗判準**:K 次 forward → 推論成本 ×K;一致性未必與 presence 強相關。失敗 =
  gate 打平 baseline,或一致性訊號對 presence 的 precision/recall 過低。

---

## 6. 三方對照

| | A Presence Calibration | B Reliability Bias Scale | C Consistency Gate |
|---|---|---|---|
| 主打失敗機制 | hallucination / image-dep bias | over-under correction / local uncertainty | photometric instability / hallucination |
| 插入點 | `output_q` 後加 log-prior | bias 減法的 scale | 同 A(訊號不同) |
| 主訊號 | `z_global`(獨立) | `output_q` reliability | multi-view 一致性 |
| loss | image-level presence BCE | image-level gumbel CE + ‖s−1‖ | photometric consistency |
| 容量 | per-class 純量 | 低維 scale | per-class gate |
| 抗漂移強度 | 高 | 中(最接近 IABR,嚴盯 P5) | 最高(well-posed 一致性) |
| novelty | 中 | 中 | 高(縫合兩機制) |
| 推論成本 | ×1 | ×1 | ×K |
| 最有利資料集 | COCO/ADE | Cityscapes | Cityscapes / cross-dataset |
| 建議順序 | 先做(最便宜) | 次之 | 旗艦 |

---

## 7. 共同實驗協定與跨資料集計畫

**單一資料集流程(每個方法)**:
1. identity check:`γ=0` 全量 formal test 必須 = **0.8536 整**(新路徑無 bug 的門檻)。
2. smoke:2 iter,finite loss + grad 只落在新 module(+ 視需要 baseline 7 參數)。
3. 訓練 50 epoch,每 5 epoch 跑 formal `test.py`(**P5 drift tripwire**)。
4. formal test vs baseline 0.8536、vs 對應 **test-time 版**、vs **oracle 0.8998**(捕捉了
   多少 headroom)。
5. 落 `research_notes.md` + `updated.md` + commit(每實驗新 SAVE_DIR,不覆蓋權重)。

**跨資料集(限制 8)**:官方 Context/ADE/COCO/Cityscapes rectification 權重上游皆有;各資料集
text 嵌入與 image-level pseudo-label 亦可由既有工具產生。方法本身無 VOC 特化常數(閾值類別數/
解析度相對)。主張:同一組超參數直接移轉五資料集,增益在 VOC 小(近飽和)、在多類別/多光度
資料集大。

**成功總判準(呼應「> 0.8536」硬指標)**:任一方法在 VOC formal test 須 ≥ 0.8536(不得低於
baseline),且在至少一個更難資料集顯著為正;否則歸為乾淨負結果(仍入論文消融)並轉下一方法。

---

## 8. 與現有資產的關係

- test-time SFP+DTLR(0.8590)是 **正交的測試期線**,可與上述任一訓練版 **疊加**(訓練期
  presence/bias 校準 + 測試期 logit 精修),形成「訓練期 + 測試期」雙軸故事。
- 今晚的 diagnostics(image-dependent residual r=0.73、oracle +0.0462、hard-gate 脆性)是
  三個方法共同的 **動機證據**,直接進論文 method-motivation 段。
- 三方法皆不碰 CLIP encoder / 不替換 feature / 不用 pixel GT / zero-init,完全符合 §0 限制。
