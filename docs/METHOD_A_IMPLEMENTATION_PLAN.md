# Method A — Implementation Plan (Trainable Soft Presence Calibration Head)

> 主線方法實作計畫(2026-07-07,UTF-8)。從 ReCLIP++ baseline 出發,學一個
> **校準過的 image-level class-presence log-prior**,soft 加到 `output_q` 上,壓抑
> hallucination。設計依據:diagnostics 顯示 anti-hallucination oracle headroom
> = **+0.0462**(0.8536→0.8998),operator 已定(soft gate),缺的是 high-recall 的
> presence 估計 —— A 用 baseline 未使用的 `z_global` 學這個估計。

## 0. 硬性限制(實作時的紅線)

- 不修改 CLIP image/text encoder(`self.vit`、`self.text_encoder` body 全 frozen)。
- 不使用 distillation。
- 不使用 pixel-level GT mask。**只用 image-level 類別標籤 `gt_cls`**(baseline 既有)。
- 不替換 `feat` / `v`(唯讀 `z_global`、`output_q` 統計)。
- 新增 module 必須 zero-init → 初始逐位元等價 baseline。
- baseline-preserving reg,不允許自由大幅改 `bias_logits`(A 不碰 bias,只加 presence
  prior 到 `output_q`,更保守)。
- 不修改 `model/model.py`、`tools/train.py`、`tools/test.py`(全部靠新 module + config)。

---

## 1. 插入位置(model.py 的哪一步)

baseline `model/model.py:RECLIPPP.forward`(inference)相關行:

```
output_q = F.conv2d(feat, text_emb)  # 468  [B, HW, C]  cosine logits(pre-bias)
...
bias_logits = pe @ prompt.t()         # 476
output = output_q - bias_logits       # 477
```

**插入點 = 468 之後、477 之前**,把 `output_q` 換成校準版:

```
output_q' = output_q + tanh(gamma) * scale * gate_c        # per-class, broadcast 到 HW
output    = output_q' - bias_logits                        # 其餘完全不變
```

放在 bias 減法**之前**,讓 rectification 與 decoder 一起吸收 presence prior。`z_global`
在 baseline `forward` 的 line 459 已回傳但未使用 → 直接取用,零額外 CLIP 計算。

---

## 2. 需要新增的 module / class

新檔 `model/model_presence.py`,沿用 `model_sfp_dtlr.py` / `model_iabr.py` 的包裝法:

```python
from model.model import RECLIPPP as _BaseRECLIPPP
from model.model import ReCLIP            # re-export(load_model_classes 需要)

class PresenceHead(nn.Module):
    """z_global-based soft class-presence log-prior. Zero-init via gamma."""
    def __init__(self, mode="zglobal"):     # mode: "zglobal" | "zglobal_dense"
        super().__init__()
        self.tau   = nn.Parameter(torch.tensor(0.0))   # threshold(learnable)
        self.temp  = nn.Parameter(torch.tensor(1.0))   # temperature
        self.scale = nn.Parameter(torch.tensor(1.0))   # gate strength on log-presence
        self.gamma = nn.Parameter(torch.tensor(0.0))   # ZERO-INIT master gate → identity
        if mode == "zglobal_dense":
            # tiny shared MLP over [image_text_c, max_c, mean_c, margin_c] → presence logit
            self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start neutral
        self.mode = mode

    def presence_logit(self, image_text, dense_stats=None):
        # image_text: [B, C] = <norm(z_global), norm(text_c)>
        if self.mode == "zglobal":
            return (image_text - self.tau) * self.temp            # [B, C]
        feats = torch.stack([image_text, *dense_stats], dim=-1)   # [B, C, 4]
        return self.mlp(feats).squeeze(-1)                        # [B, C]

    def forward(self, image_text, dense_stats=None):
        pl   = self.presence_logit(image_text, dense_stats)       # [B, C]
        pres = torch.sigmoid(pl)                                  # soft presence prob
        gate = torch.log(pres.clamp_min(1e-4))                    # log-domain (<=0)
        return torch.tanh(self.gamma) * self.scale * gate, pres   # add-term, presence prob

class RECLIPPP(_BaseRECLIPPP):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        mode = _cfg_get(getattr(self.cfg.MODEL, "PRESENCE", None), "MODE", "zglobal")
        self.presence_head = PresenceHead(mode)
    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token,
                training=False, img_metas=None, return_feat=False):
        # 複製 baseline forward 到 output_q 之後,插入:
        #   z = F.normalize(z_global, dim=-1); t = F.normalize(text_emb, dim=-1)
        #   image_text = z @ t.t()                        # [B, C]
        #   (dense mode) max_c/mean_c/margin_c from output_q
        #   add_term, pres = self.presence_head(image_text, dense_stats)
        #   output_q = output_q + add_term[..broadcast..]
        # 其餘(bias 減法、decoder)與 baseline 完全相同。
        # training 分支:回傳 baseline region-proto loss + BCE(pres, gt_cls) + reg
        ...
```

Config block(註冊到 `config/configs.py`,仿 `MODEL.SFP_DTLR` 前例):
```
cfg.MODEL.PRESENCE = edict()
cfg.MODEL.PRESENCE.MODE      = "zglobal"   # "zglobal" | "zglobal_dense"
cfg.MODEL.PRESENCE.BCE_W     = 1.0         # image-level presence BCE 權重
cfg.MODEL.PRESENCE.NEG_POS_W = 0.2         # 非對稱:壓 present 的懲罰 >> 留 absent
cfg.MODEL.PRESENCE.REG_W     = 0.1         # baseline-preserving reg 權重
```

**新增 trainable 參數**(minimal zglobal 版):`tau, temp, scale, gamma`(4 純量);
dense 版另加一個 4→8→1 shared MLP。全部與類別數無關(跨資料集通用)。CLIP、`proj`、
baseline 的 7 參數預設 frozen(可選 co-train,見 ablation)。

---

## 3. Zero-init / identity-init 設計

- `gamma = 0` → `tanh(gamma) = 0` → `add_term = 0` → `output_q' ≡ output_q` → **逐位元
  等價 baseline**。
- dense MLP 末層 zero-init → 即使 gamma 被優化,起點 presence_logit 也中性。
- baseline 官方權重載入時,`presence_head.*` 為 checkpoint 缺鍵(`strict=False`,如
  IABR 前例),取 init 值,gamma=0 → identity。
- **這是硬門檻**:見 §6 identity check。

---

## 4. Loss function(不用 pixel GT)

training forward 回傳:

```
L = L_region                      # baseline 既有的 gumbel region-prototype CE(image-level 驅動)
  + BCE_W * L_presence            # image-level multi-label BCE:BCE(pres_c, tag_c)
  + REG_W * ||tanh(gamma)*scale*gate||^2   # baseline-preserving(限制 6)
```

- `L_presence` = 非對稱 BCE:`tag_c ∈ {0,1}` 來自 `gt_cls`(image-level 存在標籤,**非
  pixel mask**)。壓抑 present class(false negative)的懲罰以 `1/NEG_POS_W` 加權 →
  偏 high-recall,避免壓掉真實類別(§5)。
- `L_region` 保留 → decoder 與 presence prior 一致演化(或凍 decoder,見 ablation)。
- `L_reg` → gate 只在訊號夠強時偏離 baseline。
- **無任何 pixel-level 監督**;`L_region` 與 baseline 相同,`L_presence` 只用 image-level tag。

---

## 5. 如何避免真實類別被 suppress

- **soft log-gate**(非 −inf):漏判的 present class 只被輕微衰減、不歸零(diagnostics
  證明硬 gate 在 recall<1 時災難性)。
- **非對稱 BCE**(NEG_POS_W=0.2):壓 present 的代價 5×,學習偏 high-recall。
- **baseline-preserving reg** + `gamma` zero-init:預設貼近 baseline。
- `pres_c` 有下界 clamp(1e-4)→ log-gate 有界,不會無限壓。

---

## 6. Identity check(硬門檻,建構後第一件事)

```
# 用官方 baseline 權重、gamma=0(初始),跑全量 formal test:
& <PY> tools/test.py --cfg config/voc_test_presence_identity_cfg.yaml \
      --model RECLIPPP --model_module model.model_presence
# 通過條件:the mIOU:0.8536 整(逐位元等價 baseline);missing keys = 僅 presence_head.*
```
不等於 0.8536 → 插入路徑有 bug,先修再訓練。(此教訓來自本專案:fusion parity bug、
IABR identity 都靠這關把關。)

---

## 7. Training command

```
# 新 SAVE_DIR,不覆蓋任何權重;從官方 baseline 權重起訓(presence_head 為新參數)
& C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe tools/train.py \
    --cfg config/voc_train_presence_cfg.yaml \
    --model RECLIPPP --model_module model.model_presence
```
config `voc_train_presence_cfg.yaml`:複製 `voc_train_iabr_cfg.yaml`,SAVE_DIR
`experiments/voc_presence/`,加 `MODEL.PRESENCE` block;`LOAD_PATH` 指官方權重當初始化
(若 train.py 支援 resume/init)或依 train.py 慣例。50 epoch。
**先 smoke**:`tools/smoke_test_fusion.py` 或等效,2 iter,assert finite loss + grad 落在
`presence_head.*`(+ 視 ablation 決定是否含 baseline 7 參數)。

---

## 8. Formal test protocol

1. identity check(§6)= 0.8536 整 —— 未過不訓練。
2. 訓練中每 5 epoch 跑 formal `test.py`(**drift tripwire**):若 formal test < 0.8536
   而 train loss 續降 → 判定 drift(IABR 式),立即中止。
3. 訓練完 formal test,對照四個基準:
   - baseline **0.8536**(硬指標:必須 ≥,否則失敗)
   - test-time soft-gate 版(§9 ablation 3;訓練必須贏過手調才有價值)
   - oracle **0.8998**(捕捉了多少 headroom = (m−0.8536)/(0.8998−0.8536))
4. 落 `research_notes.md` §11 + `updated.md` + commit(新 SAVE_DIR)。

---

## 9. Ablation 設計

| # | 名稱 | 設定 | 回答的問題 |
|---|---|---|---|
| 1 | **z_global only** | `PRESENCE.MODE=zglobal`(tau/temp/scale/gamma 4 純量) | 純獨立 image-level 訊號夠不夠 |
| 2 | **z_global + dense stats** | `MODE=zglobal_dense`(+max/mean/margin,4→8→1 MLP) | 加 dense 統計(peak-height,recall 0.928)有無互補增益 |
| 3 | **trainable vs test-time** | 同 gate 但 tau/temp/scale **手調不訓練** vs 學出來 | 「訓練」相對「手調」有無加值 |
| (4) | **B conservative ablation** | 額外在 bias 上加低維 reliability scale `s=1+tanh(γ_b)·h(r)`,`‖s−1‖` reg | reliability-scaled bias 是否再加值(B 只做這個,不獨立主線) |

test-time 版(ablation 3 的對照)= 直接啟用 `model_feature_fusion` 那個停用的
`class_gate`(soft、log-domain),手調 threshold/temp,跑 test.py,無訓練。

---

## 10. Fail conditions(明確失敗判準)

- **F1 identity ≠ 0.8536** → 實作 bug,修。
- **F2 formal test < 0.8536** → 傷害 baseline。若 train-eval↑ 而 test↓ = IABR 式 drift → 中止該 run。
- **F3 trainable ≈ test-time 手調**(差 < 0.001)→ 訓練無加值;若兩者**皆**只打平
  baseline → z_global presence 訊號在 VOC 太弱(符合目前 diagnostics 模式)→ **不再於
  VOC 加碼,轉難資料集**(COCO-Stuff/ADE,見決策段的預期)。
- **F4 presence BCE 的 val recall 未超過** baseline 隱含 recall(0.93)→ 估計器沒學到東西。
- **預期誠實聲明**:依現有 diagnostics(hard-gate 打平、photometric 否證),VOC 上
  **最可能結果是打平 ~0.8536**。因此 VOC run 的角色是 **premise gate**,真正的成敗判定
  在難資料集。VOC 打平**不代表方向失敗**,代表戰場要換。

---

## 11. 實作順序(建議)

1. 寫 `model/model_presence.py`(zglobal 版)+ configs.py block + 3 個 config
   (train / test / identity)。**不改** model.py/train.py/test.py。
2. identity check(§6)→ 必須 0.8536。
3. smoke(2 iter)→ finite loss + grad 落點正確。
4. 訓練 ablation 1(zglobal),drift tripwire 全程監控。
5. formal test → 對照 baseline / test-time / oracle。
6. 依結果:過 → 做 ablation 2/3 +(B)ablation 4,並準備跨資料集;打平 → 依 F3 轉難資料集。
7. 全程落檔 + commit(push mine)。
