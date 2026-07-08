# LGAK-MVP model wrapper on the frozen ReCLIP++ baseline.
#
# Non-destructive (constraint 3 / review F5): model/model.py is NOT edited. The forward below is
# a verbatim copy of RECLIPPP.forward (model/model.py:451-511) with exactly two LGAK changes,
# both marked "# LGAK":
#   (1) a TextGatedConvRefiner refines `feat` and the refined feature feeds output_q ONLY (F1=A);
#   (2) the decoder concat keeps the ORIGINAL `feat` (unchanged decoder input distribution).
# Everything else -- ViT, proj, text encoder, pe_proj, decoder, the region-prototype training
# loss -- is reused and FROZEN. Only the LGAK module trains (created after the freeze, like
# model/model_presence.py). Baseline loaded from cfg.MODEL.LGAK.INIT_FROM because tools/train.py
# builds from scratch with no resume. Review + protocol: docs/LGAK_IMPLEMENTATION_REVIEW.md.

import os

import torch
import torch.nn.functional as F
from torch import nn

from model.model import RECLIPPP as _BaseRECLIPPP
from model.model import ReCLIP  # noqa: F401  re-exported for load_model_classes
from model.lgak import TextGatedConvRefiner


def _cfg_get(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


class RECLIPPP(_BaseRECLIPPP):
    def __init__(self, cfg, clip_model, rank):
        super().__init__(cfg, clip_model, rank)
        lcfg = getattr(cfg.MODEL, "LGAK", None)
        init_from = str(_cfg_get(lcfg, "INIT_FROM", ""))
        kernel = int(_cfg_get(lcfg, "KERNEL", 3))
        hidden = int(_cfg_get(lcfg, "HIDDEN", 128))
        alpha_trainable = bool(_cfg_get(lcfg, "ALPHA_TRAINABLE", True))

        # Load the official baseline and FREEZE the entire baseline; only LGAK (created after
        # the freeze) will train. Without INIT_FROM, tools/train.py would start the baseline
        # untrained (it has no resume path).
        if init_from and os.path.isfile(init_from):
            sd = torch.load(init_from, map_location="cpu")
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=False)
        for p in self.parameters():
            p.requires_grad = False
        self.lgak = TextGatedConvRefiner(channels=cfg.MODEL.TEXT_CHANNEL,
                                         kernel_size=kernel, hidden=hidden)  # trainable
        if not alpha_trainable:
            self.lgak.alpha.requires_grad = False

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

        # --- LGAK (F1=A): refined feat feeds output_q ONLY; decoder keeps original feat ---
        feat_refined = self.lgak(feat, gt_cls_text_embeddings)
        output_q = F.conv2d(feat_refined, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(
            batch_size, -1, cnum)

        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        feature = torch.cat((feat, output), dim=1)  # LGAK (F1=A): ORIGINAL feat, not refined
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # Baseline region-prototype loss (image-level tags gt_cls, NO pixel GT). LGAK receives
            # gradient through output -> output_q -> feat_refined (straight-through gumbel mask).
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(
                batch_size, shape[0], shape[1], -1)
            loss = 0
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
                loss += F.cross_entropy(similarity_img, labels)
            return output, loss / batch_size

        return output
