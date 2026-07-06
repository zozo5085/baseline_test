from collections import OrderedDict
import math

import clip
import torch
from torch import nn
import torch.nn.functional as F

from model.model import TextEncoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _cfg_get(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


def normalize_feature_map(x, eps=1e-6):
    # x: [B,C,H,W]. Channel-wise standardization equalizes L6/L9/L12 scales.
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(eps)
    x = (x - mean) / std
    return F.normalize(x, dim=1, eps=eps)


def normalize_spatial_map(x, eps=1e-6):
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def spatial_gradient_norm(x, eps=1e-6):
    # Training-free boundary proxy from spatial changes in the semantic anchor.
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx.abs().mean(dim=1, keepdim=True), (0, 1, 0, 0))
    dy = F.pad(dy.abs().mean(dim=1, keepdim=True), (0, 0, 0, 1))
    return normalize_spatial_map(dx + dy, eps=eps)


def cosine_logits(feat, text_features):
    feat = F.normalize(feat, dim=1, eps=1e-6)
    text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    return F.conv2d(feat, text_features[:, :, None, None])


def build_uncertainty_gate(logits, temp=10.0, eps=1e-6):
    prob = F.softmax(logits * float(temp), dim=1)
    top2 = torch.topk(prob, k=2, dim=1)
    top1 = top2.values[:, 0:1]
    top2v = top2.values[:, 1:2]
    margin = top1 - top2v
    entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1, keepdim=True)
    entropy = entropy / math.log(max(logits.shape[1], 2))
    uncertainty = normalize_spatial_map(entropy, eps=eps) * (1.0 - normalize_spatial_map(margin, eps=eps))
    return uncertainty.clamp(0.0, 1.0), margin, entropy


def map_mean(x):
    return float(x.detach().float().mean().cpu())


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask
        self.last_attn_weight = None

    def forward(self, x: torch.Tensor, collect_attn: bool = False):
        y = self.ln_1(x)
        attn_out, attn_weight = self.attn(
            y, y, y,
            need_weights=collect_attn,
            average_attn_weights=True,
            attn_mask=None,
        )
        self.last_attn_weight = attn_weight.detach() if collect_attn and attn_weight is not None else None
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.last_attn_weight = None
        self._initialize_weights(clip_model)

    def forward(self, x: torch.Tensor, collect_attn: bool = False):
        y = self.ln_1(x)
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        B, L, C3 = qkv.shape
        qkv = qkv.view(B, L, 3, C3 // 3).permute(2, 0, 1, 3).reshape(3 * B, L, C3 // 3)
        qkv = F.linear(qkv, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = qkv.tensor_split(3, dim=0)
        v = v + x
        v = v + self.mlp(self.ln_2(v))

        y2 = self.ln_1(x)
        attn_out, attn_weight = self.attn(
            y2, y2, y2,
            need_weights=collect_attn,
            average_attn_weights=True,
            attn_mask=None,
        )
        self.last_attn_weight = attn_weight.detach() if collect_attn and attn_weight is not None else None
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, q, k, v

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False


class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = []
        for i in range(layers - 1):
            blocks.append(ResidualAttentionBlock(width, heads, attn_mask))
        blocks.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self.resblocks = nn.ModuleList(blocks)
        self._initialize_weights(clip_model)

    def forward(self, x: torch.Tensor, capture_layers=None, collect_attn: bool = False):
        capture_layers = set(capture_layers or [])
        hidden = {}
        attn = {}
        q = k = v = None
        for idx, block in enumerate(self.resblocks, start=1):
            capture = idx in capture_layers and collect_attn
            if idx == self.layers:
                x, q, k, v = block(x, collect_attn=capture)
            else:
                x = block(x, collect_attn=capture)
            if idx in capture_layers:
                hidden[idx] = x
                if getattr(block, "last_attn_weight", None) is not None:
                    attn[idx] = block.last_attn_weight
        return x, q, k, v, hidden, attn

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblocks[i]._initialize_weights(clip_model, i)


class DFFFusion2d(nn.Module):
    """
    Lightweight FPN-like multi-layer fusion with dynamic per-pixel layer gates.

    Inputs are CLIP visual feature maps from layers 6/9/12, all [B,768,H,W].
    Output keeps the same channel count so the original CLIP projection remains usable.
    """
    def __init__(self, channels=768, num_layers=3, init_gamma=0.0):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_layers)])
        self.gate = nn.Sequential(
            nn.Conv2d(channels * num_layers, channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_layers, 1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        self.last_gate = None

    def forward(self, maps, base):
        if len(maps) != len(self.laterals):
            return base
        lateral = [proj(x) for proj, x in zip(self.laterals, maps)]
        gate = torch.softmax(self.gate(torch.cat(lateral, dim=1)), dim=1)
        self.last_gate = gate.detach()
        stack = torch.stack(lateral, dim=1)
        fused = (stack * gate.unsqueeze(2)).sum(dim=1)
        fused = self.refine(fused)
        return base + self.gamma.to(dtype=base.dtype, device=base.device) * fused


class DFF2dBlock(nn.Module):
    """
    DFF2d block adapted from PlugNPlay-Modules/DFF2d.py.

    It fuses two same-shape 2D feature maps with channel attention followed by
    a spatial attention mask. We keep the output channel count unchanged.
    """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.conv_redu = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()
        self.last_spatial_gate = None

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        channel_gate = self.conv_atten(self.avg_pool(output))
        output = output * channel_gate
        output = self.conv_redu(output)

        spatial_gate = self.nonlin(self.conv1(x) + self.conv2(skip))
        self.last_spatial_gate = spatial_gate.detach()
        return output * spatial_gate


class DFF2dFusion(nn.Module):
    """
    CLIP-preserving L6/L9/L12 fusion.

    L12 remains the semantic anchor. DFF2d produces a residual detail branch
    from L9 and L6, then a small learnable gamma injects it back into L12.
    """
    def __init__(self, channels=768, init_gamma=0.01):
        super().__init__()
        self.l6_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.l9_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.l12_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.dff_9_12 = DFF2dBlock(channels)
        self.dff_6_mid = DFF2dBlock(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        self.last_gate = None
        self.last_preserve_cos = None

    def forward(self, maps, base):
        by_layer = {layer: feat for layer, feat in maps}
        f12 = self.l12_proj(by_layer.get(12, base))
        f9 = self.l9_proj(by_layer.get(9, base))
        f6 = self.l6_proj(by_layer.get(6, base))
        mid = self.dff_9_12(f12, f9)
        detail = self.dff_6_mid(mid, f6)
        detail = self.refine(detail)
        fused = base + self.gamma.to(dtype=base.dtype, device=base.device) * detail

        gates = []
        for gate in (self.dff_9_12.last_spatial_gate, self.dff_6_mid.last_spatial_gate):
            if gate is not None:
                gates.append(gate)
        self.last_gate = torch.cat(gates, dim=1).detach() if gates else None
        preserve_cos = (F.normalize(fused.float(), dim=1) * F.normalize(base.float(), dim=1)).sum(
            dim=1, keepdim=True
        )
        self.last_preserve_cos = map_mean(preserve_cos)
        return fused


class VisionTransformer(nn.Module):
    def __init__(self, cfg, clip_model: clip, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dilation = [1, 1]
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = torch.load('utils/cls_token.pt')
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(clip_model, width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)

        ff_cfg = _cfg_get(cfg.MODEL, "FEATURE_FUSION", None)
        self.fusion_enabled = bool(_cfg_get(ff_cfg, "ENABLE", False))
        self.fusion_layers = list(_cfg_get(ff_cfg, "LAYERS", [6, 9, 12]))
        self.fusion_mode = str(_cfg_get(ff_cfg, "MODE", "l12_only"))
        init_gamma = float(_cfg_get(ff_cfg, "INIT_GAMMA", 0.0))
        if self.fusion_mode == "dff2d":
            self.feature_fusion = DFF2dFusion(channels=width, init_gamma=init_gamma)
        else:
            self.feature_fusion = DFFFusion2d(
                channels=width,
                num_layers=len(self.fusion_layers),
                init_gamma=init_gamma,
            )
        self._initialize_weights(clip_model)

    def _patch_embed(self, x):
        B = x.shape[0]
        input_h, input_w = x.size()[-2:]
        stride_h = stride_w = self.patch_size
        kernel_h = kernel_w = self.patch_size
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + kernel_h - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + kernel_w - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        x = x.to(device)
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, output_h, output_w

    def _pos_embed(self, output_h, output_w):
        positional_embedding = self.positional_embedding.unsqueeze(0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size
        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, positional_embedding.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w), mode='bicubic',
                                         align_corners=False)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    def _tokens_to_map(self, tokens, output_h, output_w):
        patch = self.ln_post(tokens)[:, 1:]
        B, _, C = patch.shape
        return patch.reshape(B, output_h, output_w, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, train=False, img_metas=None, return_debug=False):
        x, output_h, output_w = self._patch_embed(x)
        positional_embedding = self._pos_embed(output_h, output_w)
        x = self.ln_pre(x + positional_embedding)

        capture_layers = sorted(set(self.fusion_layers))
        x, q, k, v, hidden, attn = self.transformer(
            x,
            capture_layers=capture_layers,
            collect_attn=return_debug,
        )

        x = self.ln_post(x)
        v = self.ln_post(v)
        cls_token = x[:, 0]

        q = q[:, 1:]
        k = k[:, 1:]
        base_v = v[:, 1:].reshape(v.shape[0], output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()

        layer_maps = {}
        for layer in capture_layers:
            if layer == 12:
                layer_maps[layer] = base_v
            elif layer in hidden:
                layer_maps[layer] = self._tokens_to_map(hidden[layer], output_h, output_w)

        fused_v = base_v
        if self.fusion_enabled and self.fusion_mode in {"trainable_fusion", "dff2d"}:
            if self.fusion_mode == "dff2d":
                maps = [(layer, layer_maps[layer]) for layer in self.fusion_layers if layer in layer_maps]
            else:
                maps = [layer_maps[layer] for layer in self.fusion_layers if layer in layer_maps]
            fused_v = self.feature_fusion(maps, base_v)

        z_global = cls_token @ self.proj if self.proj is not None else cls_token
        debug = None
        if return_debug or self.fusion_enabled:
            debug = {
                "shape": (output_h, output_w),
                "base_v": base_v,
                "fused_v": fused_v,
                "layer_maps": layer_maps,
                "attn": attn,
                "fusion_gate": getattr(self.feature_fusion, "last_gate", None),
                "preserve_cos": getattr(self.feature_fusion, "last_preserve_cos", None),
            }

        return [fused_v, (output_h, output_w), z_global, k, positional_embedding[:, 1:, :], debug]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        for p in self.parameters():
            p.requires_grad = False


class RECLIPPP(nn.Module):
    def __init__(self, cfg, clip_model, rank, zeroshot_weights=None):
        super(RECLIPPP, self).__init__()
        self.vit = VisionTransformer(
            cfg=cfg,
            clip_model=clip_model,
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            output_dim=768,
        )
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
        ff_cfg = getattr(cfg.MODEL, "FEATURE_FUSION", None)
        self.fusion_enabled = bool(_cfg_get(ff_cfg, "ENABLE", False))
        self.fusion_mode = str(_cfg_get(ff_cfg, "MODE", "l12_only"))
        self.fusion_gamma9 = float(_cfg_get(ff_cfg, "GAMMA9", 0.20))
        self.fusion_gamma6 = float(_cfg_get(ff_cfg, "GAMMA6", 0.05))
        self.fusion_gate_temp = float(_cfg_get(ff_cfg, "GATE_TEMP", 10.0))
        self.preserve_loss_weight = float(_cfg_get(ff_cfg, "PRESERVE_LOSS_WEIGHT", 0.0))
        class_gate_cfg = getattr(cfg.MODEL, "CLASS_GATE", None)
        self.class_gate_enabled = bool(_cfg_get(class_gate_cfg, "ENABLE", False))
        self.class_gate_threshold = float(_cfg_get(class_gate_cfg, "THRESHOLD", 0.20))
        self.class_gate_temp = float(_cfg_get(class_gate_cfg, "TEMP", 10.0))
        self.class_gate_log_bias_scale = float(_cfg_get(class_gate_cfg, "LOG_BIAS_SCALE", 1.0))

        if cfg.MODEL.TRAINING:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(512 + self.cnum, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)
        else:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

        if self.fusion_enabled and self.fusion_mode in {"trainable_fusion", "dff2d"}:
            for p in self.vit.feature_fusion.parameters():
                p.requires_grad = True

    def apply_safe_layer_fusion(self, layer_maps, text_features):
        f12 = layer_maps.get(12)
        if f12 is None:
            raise RuntimeError("safe layer fusion requires layer 12 feature map")

        f12n = normalize_feature_map(f12)
        f9n = normalize_feature_map(layer_maps.get(9, f12))
        f6n = normalize_feature_map(layer_maps.get(6, f12))

        # Layer 12 stays semantic anchor. L9/L6 only inject residual spatial cues.
        anchor_feat = self.proj(f12n)
        anchor_logits = cosine_logits(anchor_feat, text_features)
        g9, margin, entropy = build_uncertainty_gate(anchor_logits, temp=self.fusion_gate_temp)
        boundary = spatial_gradient_norm(anchor_logits)
        g6 = (g9 * boundary).clamp(0.0, 1.0)

        a9 = f9n - f12n
        a6 = f6n - f12n

        if self.fusion_mode == "l12_only" or not self.fusion_enabled:
            fused = f12n
        elif self.fusion_mode == "l9_l12":
            fused = f12n + self.fusion_gamma9 * g9 * a9
        elif self.fusion_mode == "l6_l12":
            fused = f12n + self.fusion_gamma6 * g6 * a6
        elif self.fusion_mode == "l6_l9_l12":
            fused = f12n + self.fusion_gamma9 * a9 + self.fusion_gamma6 * a6
        elif self.fusion_mode == "safe_l6_l9_l12":
            fused = f12n + self.fusion_gamma9 * g9 * a9 + self.fusion_gamma6 * g6 * a6
        elif self.fusion_mode == "trainable_fusion":
            return None, {}
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        fused = normalize_feature_map(fused)
        stats = {
            "gate9": g9.detach(),
            "gate6": g6.detach(),
            "boundary_proxy": boundary.detach(),
            "anchor_logits": anchor_logits.detach(),
            "anchor_margin": margin.detach(),
            "anchor_entropy": entropy.detach(),
            "cos_fused_f12": map_mean((fused * f12n).sum(dim=1, keepdim=True)),
            "cos_f9_f12": map_mean((f9n * f12n).sum(dim=1, keepdim=True)),
            "cos_f6_f12": map_mean((f6n * f12n).sum(dim=1, keepdim=True)),
        }
        return fused, stats

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False, return_debug=False, debug_fusion=False):
        cnum = zeroshot_weights.shape[0]
        gt_cls_text_embeddings = zeroshot_weights.to(self.device)
        batch_size = image.shape[0]
        image = image.to(self.device)

        v, shape, z_global, k, positional_embedding, debug = self.vit(
            image,
            train=False,
            img_metas=img_metas,
            return_debug=return_debug or debug_fusion,
        )
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        fusion_stats = {}
        if self.fusion_enabled and self.fusion_mode not in {"trainable_fusion", "dff2d"}:
            if debug is None or "layer_maps" not in debug:
                raise RuntimeError("safe layer fusion requires captured layer maps")
            fused_v, fusion_stats = self.apply_safe_layer_fusion(debug["layer_maps"], gt_cls_text_embeddings)
            v = fused_v
            debug["fused_v"] = fused_v
            debug.update(fusion_stats)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        logit_scale = self.logit_scale.exp()

        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(
            batch_size, -1, cnum
        )
        class_gate = None
        if self.class_gate_enabled:
            image_text = F.normalize(z_global, dim=-1, eps=1e-6) @ F.normalize(
                gt_cls_text_embeddings, dim=-1, eps=1e-6
            ).t()
            class_gate = torch.sigmoid((image_text - self.class_gate_threshold) * self.class_gate_temp)
            output_q = output_q + self.class_gate_log_bias_scale * torch.log(
                class_gate.clamp_min(1e-4)
            ).unsqueeze(1)

        prompt = self.text_encoder(cls_name_token)
        # Must match model.model exactly: global Frobenius norm, NOT per-class L2.
        # Per-class normalization rescales bias_logits and collapsed mIoU 0.8451->0.2838
        # with the baseline checkpoint (parity bug found 2026-07-07).
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        feature = torch.cat((feat, output), dim=1)
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_debug:
            debug = debug or {}
            debug["projected_feat"] = feat
            debug["logits"] = output
            debug["fusion_mode"] = self.fusion_mode
            debug["class_gate"] = class_gate.detach() if class_gate is not None else None
            return output, debug

        if return_feat:
            return output[0], feat[0], shape

        if training:
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(
                batch_size, shape[0], shape[1], -1
            )
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
                labels = torch.tensor(gt_cls[j]).to(self.device)
                loss += F.cross_entropy(similarity_img, labels)
            loss = loss / batch_size
            if self.preserve_loss_weight > 0 and debug is not None:
                base_v = debug.get("base_v")
                fused_v = debug.get("fused_v")
                if base_v is not None and fused_v is not None:
                    preserve = 1.0 - (
                        F.normalize(fused_v.float(), dim=1) * F.normalize(base_v.float(), dim=1)
                    ).sum(dim=1).mean()
                    loss = loss + self.preserve_loss_weight * preserve
            return output, loss

        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(
            clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
            requires_grad=False,
        )


ReCLIP = RECLIPPP
