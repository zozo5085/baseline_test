"""
Test-time logit-refinement wrapper around the UNMODIFIED ReCLIP++ baseline
(model.model.RECLIPPP).

Ported from othermodel_guide/model_1/model_lrab_v1_voc_final_862.py:
  - Stage 1: PG-CP-SFP logit purification  (source 862:795-994,  sfp_logit_purify)
  - Stage 2: SP-DTLR domain-transform edge-preserving refinement
             (source 862:480-583  DomainTransformRecursiveFilter,
              source 862:997-1137 sfp_domain_transform_logit_refine)
  - SFP outlier score (attention-based) (source 862:321-351 compute_sfp_score)

Both refinement stages are parameter-free (see DomainTransformRecursiveFilter's
own docstring, source 862:484-486) and operate purely on the dense per-pixel
class logits produced by the baseline decoder, i.e. the same `output` tensor
`model.model.RECLIPPP.forward` returns when called with training=False,
return_feat=False (see model/model.py:451-511, tensor built at model.py:477-482).
That tensor is exactly what source 862's forward calls sfp_logit_purify /
sfp_domain_transform_logit_refine on (source 862:1678-1689), so it is the
correct hook point: we call the UNMODIFIED base forward to get `output`, then
refine it here, without touching model/model.py at all.

EXCLUDED ON PURPOSE: Stage 3, PSAR-AP attribute-residual correction
(source 862:1358-1615, function sfp_attribute_residual_refine; enabled by
sfp_attr_enable at source 862:700 and invoked at source 862:1695-1701). It is
a hand-crafted VOC-specific hack that only ever touches chair/diningtable
logits (source 862:706, sfp_attr_apply_classes=(8, 10), class-wise eta 40/32
at source 862:702-704) using a hand-written attribute-phrase bank
(source 862:746-769). It is VOC-class-overfit and not part of this port.
TODO: revisit only if a dataset-agnostic attribute bank is designed later.

This wrapper adds ZERO trainable parameters (DomainTransformRecursiveFilter
has none), so the baseline checkpoint should load with the same effective
keys as model.model.RECLIPPP; tools/test.py already loads non-"model.model"
modules with strict=False and prints missing/unexpected key counts.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.model import RECLIPPP as _BaseRECLIPPP
from model.model import ReCLIP  # re-exported: load_model_classes(module) expects it


def _cfg_get(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


class DomainTransformRecursiveFilter(nn.Module):
    """
    Domain-transform recursive edge-preserving filter.

    This module is parameter-free. It is only used at test time through
    SFP-selected logit refinement, so it does not change the training
    checkpoint format.

    Verbatim port of
    othermodel_guide/model_1/model_lrab_v1_voc_final_862.py:480-583.
    """

    def __init__(self, sigma_s=30.0, sigma_r=0.30, num_iterations=1):
        super().__init__()
        self.sigma_s = float(sigma_s)
        self.sigma_r = float(sigma_r)
        self.num_iterations = int(num_iterations)

    def forward(self, img, joint_image):
        """
        img:
            [B, C, H, W], logits or probability maps to be filtered.

        joint_image:
            [B, 3, Hin, Win], RGB guide image. It is resized to the logit
            resolution and normalized to [0, 1] per image.
        """
        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        guide = joint_image.to(device=device, dtype=dtype)

        if guide.shape[-2:] != (H, W):
            guide = F.interpolate(
                guide,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        guide_min = guide.amin(dim=(2, 3), keepdim=True)
        guide_max = guide.amax(dim=(2, 3), keepdim=True)
        guide = (guide - guide_min) / (guide_max - guide_min).clamp_min(1e-6)

        grad_x = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])
        grad_y = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])

        grad_x = grad_x.sum(dim=1, keepdim=True)
        grad_y = grad_y.sum(dim=1, keepdim=True)

        ct_x = 1.0 + (self.sigma_s / max(self.sigma_r, 1e-6)) * grad_x
        ct_y = 1.0 + (self.sigma_s / max(self.sigma_r, 1e-6)) * grad_y

        ct_x = F.pad(ct_x, (1, 0, 0, 0), value=1.0)
        ct_y = F.pad(ct_y, (0, 0, 1, 0), value=1.0)

        out = img.clone()
        num_iter = max(int(self.num_iterations), 1)

        for i in range(num_iter):
            sigma_h = self.sigma_s * (
                math.sqrt(3.0) * (2.0 ** (num_iter - i - 1))
            ) / math.sqrt((4.0 ** num_iter) - 1.0)
            sigma_h = max(sigma_h, 1e-6)

            feedback = math.exp(-math.sqrt(2.0) / sigma_h)
            log_feedback = math.log(max(feedback, 1e-12))

            a_x = torch.exp(ct_x * log_feedback)
            a_y = torch.exp(ct_y * log_feedback)

            out = self._filter_horizontal(out, a_x)
            out = self._filter_vertical(out, a_y)

        return out

    def _filter_horizontal(self, img, a):
        B, C, H, W = img.shape
        out = img.clone()

        for w in range(1, W):
            out[:, :, :, w] = img[:, :, :, w] + a[:, :, :, w] * (
                out[:, :, :, w - 1] - img[:, :, :, w]
            )

        for w in range(W - 2, -1, -1):
            out[:, :, :, w] = out[:, :, :, w] + a[:, :, :, w + 1] * (
                out[:, :, :, w + 1] - out[:, :, :, w]
            )

        return out

    def _filter_vertical(self, img, a):
        B, C, H, W = img.shape
        out = img.clone()

        for h in range(1, H):
            out[:, :, h, :] = img[:, :, h, :] + a[:, :, h, :] * (
                out[:, :, h - 1, :] - img[:, :, h, :]
            )

        for h in range(H - 2, -1, -1):
            out[:, :, h, :] = out[:, :, h, :] + a[:, :, h + 1, :] * (
                out[:, :, h + 1, :] - out[:, :, h, :]
            )

        return out


class RECLIPPP(_BaseRECLIPPP):
    """
    Same architecture and same parameters as model.model.RECLIPPP. Adds
    test-time-only, parameter-free PG-CP-SFP + SP-DTLR logit refinement on
    top of the base forward's dense output logits.
    """

    def __init__(self, cfg, clip_model, rank):
        super().__init__(cfg, clip_model, rank)

        # --- MODEL.SFP_DTLR config block (config/configs.py) ---
        # Defensive lookup so a config WITHOUT the SFP_DTLR block still works
        # and is numerically identical to the legacy (VOC-tuned) behavior --
        # every default below is the verbatim value this file used to
        # hard-code. See docs/othermodel_guide/sfp_dtlr_generalization_review.md
        # for why TOP_FRACTION / DTLR_SIGMA_S_REL / DTLR_STRUCTURE_CLASSES
        # exist as dataset-relative overrides.
        sfp_cfg = _cfg_get(cfg.MODEL, "SFP_DTLR", None)

        # --- CP-SFP / PG-CP-SFP settings (source 862:615-623) ---
        self.sfp_enable = True
        self.sfp_topk = int(_cfg_get(sfp_cfg, "TOPK", 800))
        # TOP_FRACTION > 0 replaces the absolute TOPK cap with
        # k = ceil(TOP_FRACTION * valid_token_count), making selectivity
        # resolution-invariant (review finding F1).
        self.sfp_top_fraction = float(_cfg_get(sfp_cfg, "TOP_FRACTION", -1.0))
        self.sfp_min_score = -1e9
        self.sfp_logit_beta = float(_cfg_get(sfp_cfg, "LOGIT_BETA", 0.55))
        self.sfp_conf_thd = float(_cfg_get(sfp_cfg, "CONF_THD", 0.97))
        self.sfp_conf_scale = float(_cfg_get(sfp_cfg, "CONF_SCALE", 10.0))

        # Dataset-agnostic reliability gate (entropy-normalized). Removes the
        # class-count confound of the absolute max-prob CONF_THD / PROXY_CONF_THD
        # gates: max-softmax shrinks mechanically as #classes C grows, so a fixed
        # 0.97 cutoff flags a systematically larger token fraction on 59-class
        # Context than on 20-class VOC -- an operating-point shift unrelated to true
        # per-token reliability. H_norm = entropy(softmax(logits*CONF_SCALE))/log(C)
        # lies in [0,1] for any C, so a single frozen threshold means the same
        # degree of uncertainty on every dataset. ENTROPY_GATE off => original
        # max-prob gates (VOC behavior untouched). The two taus are the normalized-
        # entropy equivalents of the VOC max-prob thresholds at the VOC class count
        # (worst-case entropy of a distribution whose max-prob equals 0.97 / 0.95 at
        # C=20), then frozen and applied unchanged to all datasets.
        self.sfp_entropy_gate = bool(_cfg_get(sfp_cfg, "ENTROPY_GATE", False))
        self.sfp_entropy_tau_unrel = float(_cfg_get(sfp_cfg, "ENTROPY_TAU_UNREL", 0.0745))
        self.sfp_entropy_tau_rel = float(_cfg_get(sfp_cfg, "ENTROPY_TAU_REL", 0.1154))

        # Margin-aware SFP selection (source 862:625-632) -- left at the
        # shipped 862 defaults (disabled; ranking-only if ever enabled).
        self.sfp_margin_enable = False
        self.sfp_margin_lambda = 0.30
        self.sfp_margin_hard_enable = False
        self.sfp_margin_thd = 0.20

        self.sfp_debug_export = False
        self.sfp_debug_maps = {}
        self.sfp_last_stats_batch = []
        self.sfp_last_outlier_mask = None

        # Proxy-Guided CP-SFP (source 862:646-650).
        # PROXY_ENABLE / DTLR_ENABLE / CPSFP_UPDATE are component-ablation switches
        # (Table IV). Defaults True == shipped behavior, zero change to existing configs.
        self.sfp_proxy_enable = bool(_cfg_get(sfp_cfg, "PROXY_ENABLE", True))
        self.sfp_proxy_lambda = float(_cfg_get(sfp_cfg, "PROXY_LAMBDA", 2.00))
        self.sfp_proxy_conf_thd = float(_cfg_get(sfp_cfg, "PROXY_CONF_THD", 0.95))
        self.sfp_proxy_kernel = int(_cfg_get(sfp_cfg, "PROXY_KERNEL", 5))
        self.sfp_cpsfp_update = bool(_cfg_get(sfp_cfg, "CPSFP_UPDATE", True))

        # SFP-selected Domain-Transform Logit Refinement (source 862:661-668).
        self.sfp_dtlr_enable = bool(_cfg_get(sfp_cfg, "DTLR_ENABLE", True))
        self.sfp_dtlr_beta = float(_cfg_get(sfp_cfg, "DTLR_BETA", 1.20))
        self.sfp_dtlr_sigma_s = float(_cfg_get(sfp_cfg, "DTLR_SIGMA_S", 70.0))
        # DTLR_SIGMA_S_REL > 0 replaces the absolute sigma_s with
        # sigma_s = DTLR_SIGMA_S_REL * token_grid_width (review finding F6).
        self.sfp_dtlr_sigma_s_rel = float(_cfg_get(sfp_cfg, "DTLR_SIGMA_S_REL", -1.0))
        self.sfp_dtlr_sigma_r = float(_cfg_get(sfp_cfg, "DTLR_SIGMA_R", 1.50))
        self.sfp_dtlr_num_iter = int(_cfg_get(sfp_cfg, "DTLR_NUM_ITER", 1))
        self.sfp_dtlr_boundary_only = False

        # Structure-preserving DTLR protection (source 862:670-690).
        self.sfp_dtlr_structure_protect_enable = True
        self.sfp_dtlr_structure_gain_thd = float(
            _cfg_get(sfp_cfg, "DTLR_STRUCTURE_GAIN_THD", 0.00)
        )
        # bottle, chair, diningtable (VOC indices) by default; empty list
        # disables structure protection (structure_mask stays all-False).
        self.sfp_dtlr_structure_classes = tuple(
            _cfg_get(sfp_cfg, "DTLR_STRUCTURE_CLASSES", [4, 8, 10])
        )
        self.sfp_dtlr_class_beta_enable = False
        self.sfp_dtlr_class_beta_classes = (4, 8, 10)
        self.sfp_dtlr_class_beta_scale = 0.75

        # Parameter-free edge-preserving filter used by Stage 2.
        self.sfp_dtlr_filter = DomainTransformRecursiveFilter(
            sigma_s=self.sfp_dtlr_sigma_s,
            sigma_r=self.sfp_dtlr_sigma_r,
            num_iterations=self.sfp_dtlr_num_iter,
        )

        # --- Hook point for the SFP outlier score ---
        # source 862 gets the "unreliable token" attention score from a
        # modified VisionTransformer/Transformer/ResidualAttentionBlock that
        # caches attention weights and threads a penultimate-block attention
        # map (`sfp_attn`, source 862:285-289) out of the vit forward.
        # We reproduce the exact same signal (penultimate ResidualAttention-
        # Block, average_attn_weights=True, need_weights=True is already
        # requested by the UNMODIFIED model.model.py:30) via a forward hook
        # on that block's nn.MultiheadAttention submodule instead of
        # duplicating the vit classes -- this way model/model.py is not
        # touched and the checkpoint's vit.* keys are untouched too.
        # self.vit.transformer.resblock is a plain python list of the same
        # block instances used by resblocks (nn.Sequential); resblock[-2] is
        # the last plain ResidualAttentionBlock (index 10 of 12, i.e. CLIP
        # layer 11), matching source 862:287's `self.resblock[-2]`.
        self._sfp_attn_cache = None
        penultimate_block = self.vit.transformer.resblock[-2]
        penultimate_block.attn.register_forward_hook(self._capture_sfp_attn)

        self._sfp_dtlr_print_done = False

    def _capture_sfp_attn(self, module, inputs, output):
        # nn.MultiheadAttention forward returns (attn_output, attn_weights)
        # when need_weights=True; attn_weights is [B, L, L] batch-first
        # regardless of module.batch_first, already averaged over heads
        # since average_attn_weights defaults to True.
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self._sfp_attn_cache = output[1].detach()
        else:
            self._sfp_attn_cache = None

    def compute_sfp_score(self, attn_weight, output_h, output_w):
        """
        SFP-lite SOM score.

        attn_weight:
            [B, L, L] or [B, heads, L, L]

        score:
            [B, H, W]

        SFP idea:
            outlier if Attn_cls,i > Attn_i,i

        Verbatim port of source 862:321-351.
        """
        if attn_weight is None:
            return None

        if attn_weight.dim() == 4:
            attn_weight = attn_weight.mean(dim=1)

        # attn_weight: [B, L, L]
        cls_to_patch = attn_weight[:, 0, 1:]  # [B, HW]
        patch_self = torch.diagonal(
            attn_weight[:, 1:, 1:],
            dim1=-2,
            dim2=-1
        )  # [B, HW]

        sfp_score = cls_to_patch - patch_self
        sfp_score = sfp_score.reshape(attn_weight.shape[0], output_h, output_w)

        return sfp_score

    def sfp_logit_purify(self, output, sfp_score):
        """
        CP-SFP / PG-CP-SFP logit purification.

        output:    [B, C, H, W]
        sfp_score: [B, Hs, Ws]

        The selected SFP outlier mask is cached in self.sfp_last_outlier_mask
        for later stages.

        Verbatim port of source 862:795-994.
        """
        self.sfp_last_outlier_mask = None
        self.sfp_last_stats_batch = []

        if (not getattr(self, "sfp_enable", False)) or sfp_score is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        sfp_score = sfp_score.to(device=device, dtype=dtype)
        if sfp_score.shape[-2:] != (H, W):
            sfp_score = F.interpolate(
                sfp_score.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        with torch.no_grad():
            prob = torch.softmax(output * float(self.sfp_conf_scale), dim=1)
            conf = prob.max(dim=1)[0]  # [B, H, W]

            # Class-count-invariant reliability coordinate: normalized entropy in
            # [0,1]. Computed always (cheap); only used as the gate when ENTROPY_GATE.
            p32 = prob.float().clamp_min(1e-12)
            ent = -(p32 * p32.log()).sum(dim=1)                       # [B,H,W], nats
            h_norm = ent / math.log(max(int(prob.shape[1]), 2))       # [B,H,W] in [0,1]

            # Class ambiguity: small top-1/top-2 margin means the token is
            # semantically uncertain, even if the max confidence is not very low.
            top2_prob = torch.topk(prob, k=2, dim=1).values  # [B, 2, H, W]
            margin = top2_prob[:, 0] - top2_prob[:, 1]       # [B, H, W], in [0, 1]

            flat_score = sfp_score.reshape(B, -1)
            flat_conf = conf.reshape(B, -1)
            flat_h_norm = h_norm.reshape(B, -1)
            flat_margin = margin.reshape(B, -1)

            # Original valid region: SFP-valid and low-confidence.
            # Margin-aware mode only changes ranking by default.
            # Optional hard margin gate can be enabled for stricter ambiguity filtering.
            outlier_flat = torch.zeros_like(flat_score, dtype=torch.bool)

            margin_enable = bool(getattr(self, "sfp_margin_enable", False))
            margin_lambda = float(getattr(self, "sfp_margin_lambda", 0.30))
            margin_hard_enable = bool(getattr(self, "sfp_margin_hard_enable", False))
            margin_thd = float(getattr(self, "sfp_margin_thd", 0.20))

            if margin_enable:
                # Normalize SFP score per image so that lambda has a stable meaning.
                score_min = flat_score.amin(dim=1, keepdim=True)
                score_max = flat_score.amax(dim=1, keepdim=True)
                flat_score_rank = (flat_score - score_min) / (score_max - score_min).clamp_min(1e-6)

                # Higher means more suspicious:
                #   high SFP score + small semantic margin.
                rank_score = flat_score_rank + margin_lambda * (1.0 - flat_margin)
            else:
                rank_score = flat_score

            for b in range(B):
                if getattr(self, "sfp_entropy_gate", False):
                    # Unreliable = high normalized entropy (class-count-invariant).
                    conf_unreliable = flat_h_norm[b] > float(self.sfp_entropy_tau_unrel)
                else:
                    conf_unreliable = flat_conf[b] < float(self.sfp_conf_thd)
                valid = (
                    (flat_score[b] > float(self.sfp_min_score)) &
                    conf_unreliable
                )

                if margin_enable and margin_hard_enable:
                    valid = valid & (flat_margin[b] < margin_thd)

                if not valid.any():
                    continue

                valid_count = int(valid.sum().item())
                top_fraction = float(getattr(self, "sfp_top_fraction", -1.0))
                if top_fraction > 0:
                    k = min(valid_count, int(math.ceil(top_fraction * valid_count)))
                else:
                    k = min(int(self.sfp_topk), valid_count)
                score_b = rank_score[b].masked_fill(~valid, float("-inf"))
                topk_idx = torch.topk(score_b, k=k, dim=0).indices
                outlier_flat[b, topk_idx] = True

            outlier_mask = outlier_flat.reshape(B, H, W)
            self.sfp_last_outlier_mask = outlier_mask.detach().float()

            if getattr(self, "sfp_debug_export", False):
                self.sfp_debug_maps["sfp_score"] = sfp_score.detach().float().cpu()
                self.sfp_debug_maps["sfp_outlier_mask"] = outlier_mask.detach().float().cpu()
                self.sfp_debug_maps["sfp_confidence"] = conf.detach().float().cpu()
                self.sfp_debug_maps["sfp_margin"] = margin.detach().float().cpu()

        if not getattr(self, "sfp_cpsfp_update", True):
            # Ablation (Table IV "selection+DTLR only"): the outlier mask above is
            # computed and cached for DTLR, but the CP-SFP rewrite itself is skipped.
            return output

        update_mask = outlier_mask.to(device=device, dtype=dtype).unsqueeze(1)
        keep_mask = 1.0 - update_mask

        if update_mask.sum() < 1:
            return output

        # 8-neighbor CP-SFP target.
        kernel = torch.ones((1, 1, 3, 3), device=device, dtype=dtype)
        kernel[:, :, 1, 1] = 0.0

        neigh_sum = F.conv2d(
            output * keep_mask,
            kernel.expand(C, 1, 3, 3),
            padding=1,
            groups=C
        )
        neigh_count = F.conv2d(keep_mask, kernel, padding=1).clamp_min(1.0)
        neigh_mean = neigh_sum / neigh_count

        refined_target = neigh_mean
        proxy_available = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # Optional PG-CP-SFP local high-confidence proxy.
        if getattr(self, "sfp_proxy_enable", False):
            proxy_kernel_size = int(getattr(self, "sfp_proxy_kernel", 5))
            if proxy_kernel_size not in (3, 5):
                raise ValueError(
                    f"Unsupported sfp_proxy_kernel={proxy_kernel_size}. Use 3 or 5."
                )

            proxy_padding = proxy_kernel_size // 2
            proxy_kernel = torch.ones(
                (1, 1, proxy_kernel_size, proxy_kernel_size),
                device=device,
                dtype=dtype
            )

            if getattr(self, "sfp_entropy_gate", False):
                # Reliable proxy source = low normalized entropy (class-count-invariant).
                high_conf = (h_norm < float(self.sfp_entropy_tau_rel)).to(dtype).unsqueeze(1)
            else:
                high_conf = (conf > float(self.sfp_proxy_conf_thd)).to(dtype).unsqueeze(1)
            proxy_source_mask = high_conf * keep_mask

            proxy_sum = F.conv2d(
                output * proxy_source_mask,
                proxy_kernel.expand(C, 1, proxy_kernel_size, proxy_kernel_size),
                padding=proxy_padding,
                groups=C
            )
            proxy_count = F.conv2d(
                proxy_source_mask,
                proxy_kernel,
                padding=proxy_padding
            )

            proxy_available = (proxy_count > 0).to(dtype)
            proxy_mean = proxy_sum / proxy_count.clamp_min(1.0)

            proxy_lambda = float(getattr(self, "sfp_proxy_lambda", 2.0))
            refined_target = neigh_mean + proxy_lambda * proxy_available * (
                proxy_mean - neigh_mean
            )

        output_clean = output * (1.0 - update_mask) + refined_target * update_mask
        beta = float(self.sfp_logit_beta)
        output_new = (1.0 - beta) * output + beta * output_clean

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["cpsfp_delta"] = (output_new - output).abs().mean(dim=1).detach().float().cpu()
            self.sfp_debug_maps["pred_before_cpsfp"] = output.argmax(dim=1).detach().cpu()
            self.sfp_debug_maps["pred_after_cpsfp"] = output_new.argmax(dim=1).detach().cpu()

        with torch.no_grad():
            num_outliers = outlier_mask.float().sum(dim=(1, 2))
            ratio = num_outliers / float(H * W)
            diff = (output_new - output).abs()
            proxy_used = (proxy_available * update_mask).sum(dim=(1, 2, 3))
            update_count = update_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            proxy_available_ratio = proxy_used / update_count

            for b in range(B):
                self.sfp_last_stats_batch.append({
                    "H": int(H),
                    "W": int(W),
                    "num_tokens": int(H * W),
                    "outliers": float(num_outliers[b].detach().cpu()),
                    "ratio": float(ratio[b].detach().cpu()),
                    "score_min": float(sfp_score[b].min().detach().cpu()),
                    "score_max": float(sfp_score[b].max().detach().cpu()),
                    "score_mean": float(sfp_score[b].mean().detach().cpu()),
                    "conf_min": float(conf[b].min().detach().cpu()),
                    "conf_max": float(conf[b].max().detach().cpu()),
                    "diff_mean": float(diff[b].mean().detach().cpu()),
                    "diff_max": float(diff[b].max().detach().cpu()),
                    "proxy_enable": int(getattr(self, "sfp_proxy_enable", False)),
                    "proxy_lambda": float(getattr(self, "sfp_proxy_lambda", 0.0)),
                    "proxy_conf_thd": float(getattr(self, "sfp_proxy_conf_thd", 0.0)),
                    "proxy_kernel": int(getattr(self, "sfp_proxy_kernel", 0)),
                    "proxy_available_ratio": float(proxy_available_ratio[b].detach().cpu()),

                    # Confound evidence + calibration hooks (cheap scalars).
                    "entropy_gate": int(getattr(self, "sfp_entropy_gate", False)),
                    "h_norm_mean": float(h_norm[b].mean().detach().cpu()),
                    "unrel_frac_conf": float((conf[b] < float(self.sfp_conf_thd)).float().mean().detach().cpu()),
                    "unrel_frac_ent": float((h_norm[b] > float(self.sfp_entropy_tau_unrel)).float().mean().detach().cpu()),
                    "rel_frac_conf": float((conf[b] > float(self.sfp_proxy_conf_thd)).float().mean().detach().cpu()),
                    "rel_frac_ent": float((h_norm[b] < float(self.sfp_entropy_tau_rel)).float().mean().detach().cpu()),

                    "margin_enable": int(getattr(self, "sfp_margin_enable", False)),
                    "margin_lambda": float(getattr(self, "sfp_margin_lambda", 0.0)),
                    "margin_hard_enable": int(getattr(self, "sfp_margin_hard_enable", False)),
                    "margin_thd": float(getattr(self, "sfp_margin_thd", 0.0)),
                    "margin_min": float(margin[b].min().detach().cpu()),
                    "margin_max": float(margin[b].max().detach().cpu()),
                    "margin_mean": float(margin[b].mean().detach().cpu()),
                    "selected_margin_mean": float(
                        margin[b][outlier_mask[b]].mean().detach().cpu()
                    ) if outlier_mask[b].any() else 0.0,
                })

        return output_new

    def sfp_domain_transform_logit_refine(self, output, image):
        """
        SFP-selected Domain-Transform Logit Refinement.

        The domain-transform filter is computed on the current logits, but only
        SFP-selected unreliable tokens are updated. High-confidence tokens are
        preserved because the update mask is exactly the cached SFP outlier mask.

        Verbatim port of source 862:997-1137 (Stage 4 SFP-FBLS branch, which
        is disabled in source by default at 862:653, is not ported here).
        """
        if not getattr(self, "sfp_dtlr_enable", False):
            return output

        outlier_mask = getattr(self, "sfp_last_outlier_mask", None)
        if outlier_mask is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        outlier_mask = outlier_mask.to(device=device, dtype=dtype)

        if outlier_mask.shape[-2:] != (H, W):
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        selected = outlier_mask.unsqueeze(1)

        if selected.sum() < 1:
            return output

        # Keep parameters editable from __init__ without rebuilding the module.
        sigma_s_rel = float(getattr(self, "sfp_dtlr_sigma_s_rel", -1.0))
        if sigma_s_rel > 0:
            # Grid-relative sigma_s (review finding F6): scales with the
            # token grid width instead of drifting as an absolute constant.
            sigma_s = sigma_s_rel * float(W)
        else:
            sigma_s = float(getattr(self, "sfp_dtlr_sigma_s", 30.0))
        self.sfp_dtlr_filter.sigma_s = sigma_s
        self.sfp_dtlr_filter.sigma_r = float(getattr(self, "sfp_dtlr_sigma_r", 0.30))
        self.sfp_dtlr_filter.num_iterations = int(getattr(self, "sfp_dtlr_num_iter", 1))

        filtered = self.sfp_dtlr_filter(output, image)

        update_mask = selected
        reject_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        if getattr(self, "sfp_dtlr_boundary_only", False):
            pred = output.argmax(dim=1, keepdim=True)

            boundary = torch.zeros_like(pred, dtype=dtype)
            boundary[:, :, 1:, :] += (pred[:, :, 1:, :] != pred[:, :, :-1, :]).to(dtype)
            boundary[:, :, :-1, :] += (pred[:, :, :-1, :] != pred[:, :, 1:, :]).to(dtype)
            boundary[:, :, :, 1:] += (pred[:, :, :, 1:] != pred[:, :, :, :-1]).to(dtype)
            boundary[:, :, :, :-1] += (pred[:, :, :, :-1] != pred[:, :, :, 1:]).to(dtype)

            boundary = (boundary > 0).to(dtype)
            boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
            update_mask = update_mask * boundary

        # Structure-preserving protection:
        # if DTLR wants to flip a structure-sensitive class to another class,
        # only allow the update when filtered confidence is sufficiently higher.
        if getattr(self, "sfp_dtlr_structure_protect_enable", False):
            scale = float(getattr(self, "sfp_conf_scale", 10.0))

            orig_prob = torch.softmax(output * scale, dim=1)
            filt_prob = torch.softmax(filtered * scale, dim=1)

            orig_conf, orig_cls = orig_prob.max(dim=1, keepdim=True)  # [B,1,H,W]
            filt_conf, filt_cls = filt_prob.max(dim=1, keepdim=True)

            structure_classes = getattr(self, "sfp_dtlr_structure_classes", (1, 8, 10, 15))
            structure_mask = torch.zeros_like(orig_cls, dtype=torch.bool)
            for cls_id in structure_classes:
                structure_mask = structure_mask | (orig_cls == int(cls_id))

            class_flip = filt_cls != orig_cls
            conf_gain = filt_conf - orig_conf
            gain_thd = float(getattr(self, "sfp_dtlr_structure_gain_thd", 0.03))

            reject = structure_mask & class_flip & (conf_gain < gain_thd)
            reject_mask = reject.to(dtype)
            update_mask = update_mask * (1.0 - reject_mask)

        if update_mask.sum() < 1:
            return output

        beta = float(getattr(self, "sfp_dtlr_beta", 0.15))
        delta = filtered - output

        if bool(getattr(self, "sfp_dtlr_class_beta_enable", False)):
            with torch.no_grad():
                pred_cls = output.argmax(dim=1, keepdim=True)
                cls_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                for cls_idx in getattr(self, "sfp_dtlr_class_beta_classes", (4, 8, 10)):
                    cls_mask = cls_mask | (pred_cls == int(cls_idx))

            beta_map = output.new_full((output.shape[0], 1, output.shape[2], output.shape[3]), beta)
            beta_map = torch.where(
                cls_mask,
                beta_map * float(getattr(self, "sfp_dtlr_class_beta_scale", 0.75)),
                beta_map,
            )
        else:
            beta_map = beta

        output_new = output + beta_map * update_mask * delta

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["dtlr_update_mask"] = update_mask.detach().float().cpu()
            self.sfp_debug_maps["dtlr_reject_mask"] = reject_mask.detach().float().cpu()
            self.sfp_debug_maps["dtlr_delta"] = (output_new - output).abs().mean(dim=1).detach().float().cpu()
            self.sfp_debug_maps["pred_before_dtlr"] = output.argmax(dim=1).detach().cpu()
            self.sfp_debug_maps["pred_after_dtlr"] = output_new.argmax(dim=1).detach().cpu()

        return output_new

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
        # Unmodified baseline forward. When training=True or return_feat=True
        # the return signature is not a plain [B,C,H,W] logits tensor (it is
        # (output, loss) or (output[0], feat[0], shape) respectively, see
        # model/model.py:484-511) -- refinement only applies to the plain
        # test-time logits path, so those cases pass through untouched.
        output = super().forward(
            image, gt_cls, zeroshot_weights, cls_name_token,
            training=training, img_metas=img_metas, return_feat=return_feat,
        )

        if training or return_feat:
            return output

        H, W = output.shape[-2], output.shape[-1]
        sfp_score = self.compute_sfp_score(self._sfp_attn_cache, H, W)

        print("[SFP-DTLR] active")  # one-time-per-image marker for smoke tests

        output = self.sfp_logit_purify(output, sfp_score)
        if getattr(self, "sfp_dtlr_enable", False):
            output = self.sfp_domain_transform_logit_refine(output, image.to(self.device))

        return output
