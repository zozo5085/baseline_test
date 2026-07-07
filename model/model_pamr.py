"""
PAMR (Pixel-Adaptive Mask Refinement, Araslanov & Roth, "Single-Stage Semantic
Segmentation from Image Labels", CVPR 2020) as a training-free, parameter-free
test-time logit refiner wrapped around the UNMODIFIED ReCLIP++ baseline
(model.model.RECLIPPP).

This is the standard training-free refiner benchmark in CLIP unsupervised
segmentation literature; it exists here to give a fair, same-hook-point
head-to-head comparison against model/model_sfp_dtlr.py's SFP+DTLR refiner
on the same official checkpoint.

Wrapping pattern copied from model/model_sfp_dtlr.py:
  - subclass model.model.RECLIPPP, re-export ReCLIP so
    tools/test.py's load_model_classes(module) keeps working;
  - call super().forward(...) to obtain the baseline `output` dense logits
    tensor [B, cnum, H, W] (model/model.py:451-511, tensor built at
    model.py:477-482) -- H, W are the TOKEN GRID resolution, not the input
    image resolution;
  - refine only the plain test-time logits path (training=False,
    return_feat=False); the training / return_feat return signatures are
    passed through untouched (model/model.py:484-511);
  - add ZERO trainable parameters, so the official checkpoint loads with the
    same effective keys as model.model.RECLIPPP. All PAMR conv kernels are
    registered as non-persistent buffers (persistent=False) so they never
    appear in this module's state_dict and therefore never show up as
    "missing keys" when loading a checkpoint that doesn't have them --
    tools/test.py already loads non-"model.model" modules with
    strict=False and prints missing/unexpected key counts.

PAMR algorithm (faithful to the paper / the paper's official release,
visinf/1-stage-wseg, models/mods/pamr.py):
  1. From the RGB guide image (resized to the logits' token-grid resolution),
     compute, for every dilation d in MODEL.PAMR.DILATIONS, the 8-neighbour
     (3x3 minus center) affinity kernel: local Gaussian-like weight
     softmax(-|RGB_i - RGB_j| / (0.1 * local_std_i)), where local_std_i is
     the per-pixel, per-dilation local standard deviation over the 3x3
     neighbourhood (LocalStDev). All dilations' neighbour candidates are
     concatenated into one pool and a SINGLE softmax is taken jointly over
     that whole pool -- this is what aggregates/averages information across
     dilation scales (more distant/higher-dilation neighbours compete
     directly against closer ones in the same softmax).
  2. The affinity is computed ONCE from the image and reused every
     iteration. Starting from probs = softmax(baseline logits, dim=class),
     MODEL.PAMR.NUM_ITER times: replace every pixel's class-probability
     vector with the affinity-weighted average of its (dilated) neighbours'
     probability vectors (LocalAffinityCopy gathers neighbour values via a
     "copy" convolution kernel; the mask, i.e. probability map, is what gets
     diffused -- NOT the raw logits, per the paper).
  3. The diffused probabilities are converted back to a logits-shaped
     tensor via log(probs) before being returned, so that downstream code
     in tools/test.py (which does F.softmax(output * 10, ...) for PD class
     gating, then F.interpolate, then F.softmax + argmax for the final
     prediction) sees the same [B, cnum, H, W] contract the baseline
     forward returns. log(.) is the natural, shift-consistent inverse of
     softmax (softmax(log(p)) == p up to normalization, and it is
     shift-invariant / rank-preserving under the downstream's later
     softmax + argmax), so this round-trip does not silently change the
     final argmax prediction on its own -- only the PAMR diffusion does.

MODEL.PAMR.NUM_ITER: 0 is treated as an EXACT IDENTITY: the baseline
`output` tensor is returned completely untouched (no softmax/log
round-trip at all), so a checkpoint evaluated with NUM_ITER: 0 must
reproduce the official baseline mIoU bit-for-bit. This is the mandatory
sanity gate for this wrapper.
"""
import torch
import torch.nn.functional as F
from torch import nn

from model.model import RECLIPPP as _BaseRECLIPPP
from model.model import ReCLIP  # re-exported: load_model_classes(module) expects it

# 3x3 neighbourhood minus the center pixel, in row-major (r, c) order.
_NEIGHBOR_OFFSETS = [
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
]


def _cfg_get(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


class _LocalAffinity(nn.Module):
    """
    Base class for the three PAMR local-neighbourhood operators. For every
    dilation in `dilations`, convolves a fixed (non-trainable) 3x3-minus-
    center kernel over the input, then concatenates the per-dilation results
    along a new "candidate neighbour" dimension P.

    Verbatim-equivalent port of the official PAMR implementation
    (Araslanov & Roth, CVPR 2020; visinf/1-stage-wseg, models/mods/pamr.py).
    """

    def __init__(self, dilations):
        super().__init__()
        self.dilations = list(dilations)
        weight = self._init_kernel()
        # persistent=False: this is a fixed, parameter-free convolution
        # kernel, not a learned weight -- it must NOT appear in this
        # module's state_dict, or checkpoint loading would report it as a
        # spurious "missing key" even though the wrapper adds zero
        # trainable parameters.
        self.register_buffer("kernel", weight, persistent=False)

    def _init_kernel(self):
        raise NotImplementedError

    def forward(self, x):
        # x: [B, K, H, W] -> per-(batch,channel) plane convolution.
        B, K, H, W = x.shape
        x = x.reshape(B * K, 1, H, W)

        aff_per_dilation = []
        for d in self.dilations:
            x_pad = F.pad(x, [d, d, d, d], mode="replicate")
            x_aff = F.conv2d(x_pad, self.kernel.to(dtype=x.dtype), dilation=d)
            aff_per_dilation.append(x_aff)

        # [B*K, num_dilations * num_neighbors, H, W]
        x_aff = torch.cat(aff_per_dilation, dim=1)
        return x_aff.reshape(B, K, -1, H, W)


class _LocalAffinityAbs(_LocalAffinity):
    """|center - neighbour| for each of the 8 neighbours, per dilation."""

    def _init_kernel(self):
        weight = torch.zeros(len(_NEIGHBOR_OFFSETS), 1, 3, 3)
        for i, (r, c) in enumerate(_NEIGHBOR_OFFSETS):
            weight[i, 0, 1, 1] = 1.0
            weight[i, 0, r, c] = -1.0
        return weight

    def forward(self, x):
        return torch.abs(super().forward(x))


class _LocalAffinityCopy(_LocalAffinity):
    """Gathers (copies) each of the 8 neighbours' values, per dilation."""

    def _init_kernel(self):
        weight = torch.zeros(len(_NEIGHBOR_OFFSETS), 1, 3, 3)
        for i, (r, c) in enumerate(_NEIGHBOR_OFFSETS):
            weight[i, 0, r, c] = 1.0
        return weight


class _LocalStDev(_LocalAffinity):
    """Per-pixel local standard deviation over the full 3x3 window (incl. center)."""

    def _init_kernel(self):
        weight = torch.zeros(9, 1, 3, 3)
        for idx in range(9):
            r, c = divmod(idx, 3)
            weight[idx, 0, r, c] = 1.0
        return weight

    def forward(self, x):
        x = super().forward(x)  # [B, K, num_dilations * 9, H, W]
        return x.std(dim=2, keepdim=True)


class PAMRRefiner(nn.Module):
    """
    Parameter-free PAMR probability-diffusion module.

    forward(image, probs):
        image: [B, 3, H, W]  RGB guide, already resized to `probs`' H, W.
        probs: [B, C, H, W]  class probabilities (softmax of the baseline
               logits) to be diffused.
        returns: [B, C, H, W] refined class probabilities.
    """

    def __init__(self, num_iter=10, dilations=(1, 2, 4, 8, 12, 24)):
        super().__init__()
        self.num_iter = int(num_iter)
        dilations = list(dilations)
        self.aff_x = _LocalAffinityAbs(dilations)
        self.aff_mask = _LocalAffinityCopy(dilations)
        self.aff_std = _LocalStDev(dilations)

    def forward(self, image, probs):
        # Affinity is computed ONCE from the (fixed) guide image and reused
        # across all diffusion iterations.
        x_std = self.aff_std(image)                       # [B,3,1,H,W]
        x = -self.aff_x(image) / (1e-8 + 0.1 * x_std)      # [B,3,P,H,W]
        x = x.mean(dim=1, keepdim=True)                    # [B,1,P,H,W]
        x = F.softmax(x, dim=2)                            # joint softmax over all dilations' neighbours

        mask = probs
        for _ in range(max(self.num_iter, 0)):
            m = self.aff_mask(mask)          # [B,C,P,H,W], gathered neighbour probs
            mask = (m * x).sum(dim=2)        # [B,C,H,W], affinity-weighted average

        return mask


class RECLIPPP(_BaseRECLIPPP):
    """
    Same architecture and same parameters as model.model.RECLIPPP. Adds
    test-time-only, parameter-free PAMR logit refinement on top of the base
    forward's dense output logits, for a fair head-to-head comparison
    against model/model_sfp_dtlr.py's SFP+DTLR refiner.
    """

    def __init__(self, cfg, clip_model, rank):
        super().__init__(cfg, clip_model, rank)

        # --- MODEL.PAMR config block (config/configs.py) ---
        # Defensive lookup so a config WITHOUT the PAMR block still works,
        # falling back to the paper's standard settings (10 iterations,
        # dilations 1/2/4/8/12/24 -- visinf/1-stage-wseg's shipped default).
        pamr_cfg = _cfg_get(cfg.MODEL, "PAMR", None)
        self.pamr_num_iter = int(_cfg_get(pamr_cfg, "NUM_ITER", 10))
        self.pamr_dilations = tuple(_cfg_get(pamr_cfg, "DILATIONS", [1, 2, 4, 8, 12, 24]))
        # "token" (default): refine at the baseline token-grid resolution -- the
        # same hook point SFP+DTLR uses, but PAMR is crippled at ~21px grids.
        # "image": upsample the baseline logits to the ORIGINAL image resolution,
        # compute PAMR affinity from the full-res RGB image and diffuse there
        # (where PAMR was designed to run), then downsample back to the token
        # grid so tools/test.py's [B,cnum,H,W] contract is unchanged.
        self.pamr_resolution = str(_cfg_get(pamr_cfg, "RESOLUTION", "token")).lower()
        self.pamr_eps = 1e-6

        # Parameter-free; built even when NUM_ITER == 0 for a stable
        # state_dict / module tree, but never invoked on the identity path.
        self.pamr = PAMRRefiner(
            num_iter=self.pamr_num_iter,
            dilations=self.pamr_dilations,
        )

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

        if self.pamr_num_iter <= 0:
            # Mandatory sanity gate: NUM_ITER=0 must be an exact identity in
            # BOTH resolution modes -- the baseline `output` is returned
            # completely untouched (no upsample/softmax/log round-trip).
            print("[PAMR] identity (NUM_ITER=0)")  # one-time-per-image marker for smoke tests
            return output

        Hg, Wg = output.shape[-2], output.shape[-1]  # token-grid shape test.py expects
        guide = image.to(device=output.device, dtype=output.dtype)

        if self.pamr_resolution == "image":
            # Run PAMR where it is designed to operate: full image resolution.
            Hin, Win = guide.shape[-2], guide.shape[-1]
            logits_up = F.interpolate(
                output, size=(Hin, Win), mode="bilinear", align_corners=False
            )
            probs = torch.softmax(logits_up, dim=1)
            refined_probs = self.pamr(guide, probs)            # full-res diffusion
            refined_probs = F.interpolate(
                refined_probs, size=(Hg, Wg), mode="bilinear", align_corners=False
            )
            marker = "[PAMR] active (image-res)"
        else:
            # token-grid resolution (default): downsample the guide to the grid.
            guide_g = guide
            if guide_g.shape[-2:] != (Hg, Wg):
                guide_g = F.interpolate(
                    guide_g, size=(Hg, Wg), mode="bilinear", align_corners=False
                )
            probs = torch.softmax(output, dim=1)
            refined_probs = self.pamr(guide_g, probs)
            marker = "[PAMR] active (token-res)"

        refined_probs = refined_probs.clamp_min(self.pamr_eps)
        # log(.) is the natural inverse of softmax: it hands tools/test.py's
        # downstream (which softmaxes `output` again, at various temperatures,
        # before the final argmax) a logits-shaped tensor whose ranking
        # matches the diffused probabilities exactly.
        refined_logits = torch.log(refined_probs)

        print(marker)  # one-time-per-image marker for smoke tests

        return refined_logits
