# LGAK-MVP -- Language-Guided Adaptive Kernel refinement for frozen-CLIP dense features.
# Design + review: docs/LGAK_IMPLEMENTATION_REVIEW.md, docs/NEW_DIRECTION_LGAK_RESEARCH_PLAN.md.
#
# MVP = "Text-Gated Residual Depthwise Conv". Inserted on the projected, unit-normalized CLIP
# feature (model.py:463), feeding output_q ONLY (review F1=A). Re-normalized so the cosine-
# similarity calibration downstream holds at alpha>0 (review F2). alpha is zero-initialized so
# alpha=0 reproduces the baseline EXACTLY (review F4 / constraint 7).
#
# MVP scope only: NO offset sampling, NO kernel mixture, NO multi-layer, NO class-specific
# [B,K,C,H,W] expansion, NO per-image / GT-derived text conditioning (dataset-global mean only).

import torch
import torch.nn.functional as F
from torch import nn


class TextGatedConvRefiner(nn.Module):
    """Text-gated residual depthwise-conv refiner.

        F_refine = PWConv(DWConv(F))
        g        = 1 + MLP(mean_c(T))            # per-channel gate, neutral (=1) at init
        F_out    = normalize(F + alpha * g * F_refine)

    F : [B, C, H, W] unit-normalized CLIP feature (C = TEXT_CHANNEL = 512).
    T : [num_cls, C] full class text-embedding set (dataset-global; mean over classes).

    alpha = nn.Parameter(0): the residual is exactly 0 at init. In eval with alpha==0 the module
    returns F unchanged (exact identity, no re-normalize float noise); in training it always runs
    the full path so the cosine calibration holds and gradient flows (only alpha receives it while
    alpha==0 -- the documented F4 slow start).
    """

    def __init__(self, channels=512, kernel_size=3, hidden=128):
        super().__init__()
        self.channels = channels
        self.dwconv = nn.Conv2d(channels, channels, kernel_size,
                                padding=kernel_size // 2, groups=channels, bias=False)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels)
        )
        # Neutral gate at init: last layer zero -> MLP(mean_T)=0 -> g = 1 + 0 = 1.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(1))  # ZERO-INIT master gate -> identity

    def forward(self, feat, text_feat):
        # feat: [B, C, H, W] (unit-norm).  text_feat: [num_cls, C] (full class set).
        mean_t = text_feat.mean(dim=0)                        # [C] dataset-global text summary
        g = 1.0 + self.gate_mlp(mean_t)                       # [C], neutral (1) at init
        g = g.view(1, self.channels, 1, 1)                    # per-channel broadcast
        refine = self.pwconv(self.dwconv(feat))               # [B, C, H, W]
        # Exact identity in eval when alpha==0; full (re-normalized) path otherwise.
        if (not self.training) and float(self.alpha) == 0.0:
            return feat
        out = feat + self.alpha * g * refine
        out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)  # F2: keep unit norm
        return out
