# Method A -- Trainable Soft Presence Calibration Head on the ReCLIP++ baseline.
#
# Design: docs/METHOD_A_IMPLEMENTATION_PLAN.md. The ViT/text encoders and the whole
# baseline are REUSED and FROZEN; the only change vs baseline RECLIPPP.forward is a
# per-class, image-level presence log-prior added to output_q (model.py:468) before the
# static bias subtraction (model.py:477):
#
#     output_q' = output_q + tanh(gamma) * scale * log(sigmoid((<z_global,text>-tau)*temp))
#
# gamma is zero-initialized so the added term is 0 at init: the official ReCLIP++
# checkpoint must reproduce its exact baseline mIoU (identity check,
# config/voc_test_presence_identity_cfg.yaml).
#
# Constraints honored: CLIP image/text encoders untouched; no distillation; no pixel-GT
# mask (presence supervised by image-level tags gt_cls, exactly like the baseline
# region-prototype loss); feat/v not replaced; zero-init identity; baseline-preserving
# reg. The baseline is loaded from PRESENCE.INIT_FROM and FROZEN so only the tiny head
# trains -> structurally cannot drift (unlike IABR/fusion which retrained from scratch).

import os

import torch
import torch.nn.functional as F
from torch import nn

from model.model import RECLIPPP as _BaseRECLIPPP
from model.model import ReCLIP  # noqa: F401  re-exported for load_model_classes


def _cfg_get(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


class PresenceHead(nn.Module):
    """Soft, calibrated, image-level class-presence log-prior.

    Input signal is `<norm(z_global), norm(text_c)>` -- the global CLIP image embedding
    (unused by the baseline forward) matched to each class text embedding, i.e. a
    presence estimate INDEPENDENT of the dense prediction. Optional 'zglobal_dense' mode
    also consumes per-class dense statistics of output_q.

    Zero-init: gamma=0 -> tanh(gamma)=0 -> add_term==0 -> exact baseline.
    """

    def __init__(self, mode="zglobal"):
        super().__init__()
        if mode not in ("zglobal", "zglobal_dense"):
            raise ValueError(f"unknown PRESENCE.MODE: {mode}")
        self.mode = mode
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))  # ZERO-INIT master gate -> identity
        if mode == "zglobal_dense":
            self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _presence_logit(self, image_text, dense):
        if self.mode == "zglobal":
            return (image_text - self.tau) * self.temp
        feats = torch.stack(
            [image_text, dense["max"], dense["mean"], dense["margin"]], dim=-1
        )  # [B, C, 4]
        return self.mlp(feats).squeeze(-1)  # [B, C]

    def forward(self, z_global, text_emb, output_q):
        image_text = (
            F.normalize(z_global, dim=-1, eps=1e-6)
            @ F.normalize(text_emb, dim=-1, eps=1e-6).t()
        )  # [B, C]
        dense = None
        if self.mode == "zglobal_dense":
            oq = output_q  # [B, HW, C]
            mx = oq.max(dim=1).values  # [B, C]
            mn = oq.mean(dim=1)        # [B, C]
            dense = {"max": mx, "mean": mn, "margin": mx - mn}
        pres = torch.sigmoid(self._presence_logit(image_text, dense))  # [B, C] in (0,1)
        gate = torch.log(pres.clamp_min(1e-4))                         # <= 0 (soft, bounded)
        add_term = torch.tanh(self.gamma) * self.scale * gate          # [B, C]
        return add_term, pres


class RECLIPPP(_BaseRECLIPPP):
    def __init__(self, cfg, clip_model, rank):
        super().__init__(cfg, clip_model, rank)
        pcfg = getattr(cfg.MODEL, "PRESENCE", None)
        self.presence_mode = str(_cfg_get(pcfg, "MODE", "zglobal"))
        self.presence_bce_w = float(_cfg_get(pcfg, "BCE_W", 1.0))
        self.presence_neg_pos_w = float(_cfg_get(pcfg, "NEG_POS_W", 0.2))
        self.presence_reg_w = float(_cfg_get(pcfg, "REG_W", 0.1))
        init_from = str(_cfg_get(pcfg, "INIT_FROM", ""))

        # Load the official baseline and FREEZE the entire baseline; only the presence
        # head (created after the freeze) will train. train.py builds from scratch with
        # no init/resume, so without this the baseline would be untrained.
        if init_from and os.path.isfile(init_from):
            sd = torch.load(init_from, map_location="cpu")
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=False)
        for p in self.parameters():
            p.requires_grad = False
        self.presence_head = PresenceHead(self.presence_mode)  # trainable

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token,
                training=False, img_metas=None, return_feat=False):
        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)
        batch_size = image.shape[0]
        image = image.to(device)
        v, shape, z_global, k, positional_embedding = self.vit(image, train=False, img_metas=img_metas)
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(
            batch_size, -1, cnum)

        # --- Method A: soft presence calibration (zero-init gamma -> exact identity) ---
        add_term, pres = self.presence_head(z_global, gt_cls_text_embeddings, output_q)  # [B,C],[B,C]
        output_q = output_q + add_term.unsqueeze(1)  # broadcast over HW; +0 at init

        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        feature = torch.cat((feat, output), dim=1)
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # baseline region-prototype loss (image-level tags, NO pixel GT)
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(
                batch_size, shape[0], shape[1], -1)
            region_loss = 0
            for j in range(batch_size):
                masked_image_features = []
                if len(gt_cls[j]) == 0:
                    continue
                for i in gt_cls[j]:
                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)
                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j]).to(device)
                region_loss += F.cross_entropy(similarity_img, labels)
            region_loss = region_loss / batch_size

            # image-level presence BCE (asymmetric -> high recall) + baseline-preserving reg
            tgt = torch.zeros(batch_size, cnum, device=device)
            for j in range(batch_size):
                if len(gt_cls[j]) > 0:
                    tgt[j, torch.tensor(gt_cls[j], device=device, dtype=torch.long)] = 1.0
            bce = F.binary_cross_entropy(pres, tgt, reduction="none")
            w = torch.where(tgt > 0.5, 1.0 / max(self.presence_neg_pos_w, 1e-6), 1.0)
            bce = (bce * w).mean()
            reg = (add_term ** 2).mean()
            total = region_loss + self.presence_bce_w * bce + self.presence_reg_w * reg
            return output, total

        return output
