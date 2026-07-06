# IABR (Image-Adaptive Bias Rectification) ReCLIP++ model.
#
# Ported from othermodel_guide/model_1/model_author_reclippp_iabr.py (629 lines,
# per docs/othermodel_guide/author_variants_analysis.md section 2.5). The ViT
# encoder and text encoder are reused verbatim from model.model (IABR does not
# touch them). The only change vs our baseline RECLIPPP is the bias-subtraction
# step in forward() -- model.py:475-477:
#
#     pe = self.pe_proj(positional_embedding)...
#     bias_logits = pe @ prompt.t()
#     output = torch.sub(output_q, bias_logits)...
#
# IABR makes the bias map image-adaptive:
#
#     rectified_logits = clip_logits - S(c, x, y; I) * bias_logits
#
# S is built from a global CLIP-similarity branch (iabr_gamma_global,
# iabr_class_amp) and a local uncertainty branch (iabr_gamma_local,
# iabr_local_gate). All new parameters are zero-initialized (iabr_local_gate's
# last conv layer is explicitly zero-init) so S == 1 everywhere at init:
# official ReCLIP++ checkpoints must reproduce their exact baseline behavior
# (verified by the identity check in config/voc_test_iabr_identity_cfg.yaml).

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.model import TextEncoder, VisionTransformer, ReCLIP  # noqa: F401 (ReCLIP re-exported)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RECLIPPP(nn.Module):
    def __init__(self, cfg, clip_model, rank):
        super(RECLIPPP, self).__init__()
        self.vit = VisionTransformer(clip_model=clip_model,
                                     input_resolution=224,
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=768)

        self.clip = clip_model
        self.k = cfg.DATASET.K
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)
        self.logit_scale = clip_model.logit_scale
        for p in self.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        if cfg.MODEL.TRAINING:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)

            # decoder
            self.decoder_conv2 = nn.Conv2d(512 + self.cnum, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)

        else:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

        # ------------------------------------------------------------------
        # IABR: Image-Adaptive Bias Rectification.
        # Reference: othermodel_guide/model_1/model_author_reclippp_iabr.py:483-512
        #
        # ReCLIP++ computes:  rectified_logits = clip_logits - bias_logits
        # IABR modulates the bias map with global class evidence and local
        # uncertainty:        rectified_logits = clip_logits - S(c,x,y;I) * bias_logits
        #
        # S = 1 when iabr_gamma_global == iabr_gamma_local == 0 (all new
        # params below are zero-init, and iabr_local_gate's last conv layer
        # is explicitly zero-init), therefore official ReCLIP++ checkpoints
        # keep their original behavior at init.
        # ------------------------------------------------------------------
        self.iabr_gamma_global = nn.Parameter(torch.tensor(0.0))
        self.iabr_gamma_local = nn.Parameter(torch.tensor(0.0))
        self.iabr_class_amp = nn.Parameter(torch.zeros(self.cnum))
        self.iabr_global_k = 2.0  # fixed constant, not learnable (reference iabr.py:499)

        # Local uncertainty branch. Input channels:
        #   feat: 512 L2-normalized visual channels
        #   max_prob, margin, entropy: 3 uncertainty channels
        # Output is a class-agnostic spatial bias-scale residual.
        self.iabr_local_gate = nn.Sequential(
            nn.Conv2d(512 + 3, 128, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=True),
        )
        # zero init makes local branch no-op at initialization (reference iabr.py:511-512)
        nn.init.zeros_(self.iabr_local_gate[-1].weight)
        nn.init.zeros_(self.iabr_local_gate[-1].bias)

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
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

        # ori
        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        # reference prompt
        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()  # model.py:475-476

        # ------------------------------------------------------------------
        # IABR bias-scale modulation replaces model.py:477's plain
        # `torch.sub(output_q, bias_logits)`. Ported faithfully from
        # othermodel_guide/model_1/model_author_reclippp_iabr.py:548-584.
        # ------------------------------------------------------------------
        if bias_logits.shape[0] == 1 and batch_size > 1:
            bias_logits = bias_logits.expand(batch_size, -1, -1)  # iabr.py:548-549

        base_rect = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, cnum, shape[0], shape[1])  # iabr.py:552

        # Global branch: CLIP global image feature vs zeroshot text -> per-class support (iabr.py:554-565)
        z = z_global.float()
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        t = gt_cls_text_embeddings.float()
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        global_score = z @ t.t()  # [B, C]
        global_score = (global_score - global_score.mean(dim=1, keepdim=True)) / global_score.std(dim=1, keepdim=True).clamp_min(1e-6)
        global_support = torch.sigmoid(self.iabr_global_k * global_score)
        global_signed = 1.0 - 2.0 * global_support  # unsupported -> positive, supported -> negative
        class_amp = 1.0 + 0.5 * torch.tanh(self.iabr_class_amp).view(1, cnum)
        global_delta = global_signed * class_amp  # [B, C]

        # Local branch: uncertainty of the un-modulated rectified logits (iabr.py:567-574)
        prob = F.softmax(base_rect.detach(), dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values
        max_prob = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) / math.log(float(cnum))
        local_in = torch.cat([feat.detach(), max_prob, margin, entropy], dim=1)
        local_delta = torch.tanh(self.iabr_local_gate(local_in))  # [B,1,H,W]

        # Fuse into a bounded scale factor (iabr.py:576-581)
        gamma_g = torch.tanh(self.iabr_gamma_global)
        gamma_l = torch.tanh(self.iabr_gamma_local)
        scale = 1.0
        scale = scale + gamma_g * global_delta.view(batch_size, cnum, 1, 1)
        scale = scale + gamma_l * local_delta
        scale = scale.clamp(0.50, 1.50)

        bias_map = bias_logits.permute(0, 2, 1).reshape(batch_size, cnum, shape[0], shape[1])
        output = output_q.permute(0, 2, 1).reshape(batch_size, cnum, shape[0], shape[1]) - scale * bias_map  # iabr.py:583-584

        feature = torch.cat((feat, output), dim=1)
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # gumbel softmax
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0], shape[1], -1)

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

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)
