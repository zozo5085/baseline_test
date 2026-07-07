"""
Diagnostic (NOT a method): characterize the image-dependent residual bias left
by ReCLIP++'s static bias subtraction, on the official VOC checkpoint.

Premise being tested (see docs/diagnostics/bias_residual_diagnostic.md):
RECLIPPP.forward computes bias_logits = pe_proj(positional_embedding) @ prompt.T,
which is IMAGE-INDEPENDENT (same functional form for every image, only the ViT
positional embedding grid enters it -- no image content). After
output = output_q - bias_logits (and the small decoder conv/BN on top of it),
does the FINAL per-pixel class probability map still carry a spatially-uniform
("DC") elevation for classes that are NOT present in the image, and does that
elevation vary across images? If yes, a purely static/image-independent bias
term cannot remove it -- that is the headroom this diagnostic quantifies.

This script builds the model EXACTLY as tools/test.py does (same preprocessing,
same PD filtering, same double bilinear interpolation, same final softmax), and
captures the tensor immediately BEFORE tools/test.py's `torch.argmax(output, dim=1)`
(tools/test.py line 203) -- i.e. the final per-pixel class probabilities used to
produce the prediction -- instead of saving the post-argmax label map.

GT labels are used only to label each class per image as present/absent for this
analysis. This is legitimate for a diagnostic that characterizes the failure
mode; it is NOT used to train or gate anything, and the eventual method this
diagnostic motivates will be unsupervised (GT-free) at inference time.

Does NOT modify model/model.py, tools/test.py, or any config/*.yaml.

Usage:
    <conda-env-python> tools/analyze_bias_residual.py [--cfg CFG] [--limit N]
"""
import argparse
import io
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import clip
from config.configs import cfg_from_file
from model.model import RECLIPPP
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens, voc_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='config/voc_test_official854_cfg.yaml')
    p.add_argument('--limit', type=int, default=None, help='cap number of val images (debug)')
    p.add_argument('--out_dir', default='docs/diagnostics')
    p.add_argument('--npz_out', default='experiments/diag_official854_eval/bias_residual_raw.npz')
    return p.parse_args()


def load_label(path, cfg):
    """Same GT handling as utils.test_mIoU.intersect_and_union / mean_iou path:
    read the palette PNG as raw class-index array, then apply REDUCE_ZERO_LABEL
    (VOC background id 0 -> ignore(255); ids 1..20 -> 0..19 to match the model's
    20 output channels)."""
    label = np.array(Image.open(path))
    label = label.astype(np.int64)
    if cfg.DATASET.REDUCE_ZERO_LABEL:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    return label


def build_model(cfg):
    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    model = RECLIPPP(cfg=cfg, clip_model=clip_model, rank=0)
    weight = torch.load(cfg.LOAD_PATH)
    new_weight = {}
    for k, v in weight.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_weight[new_k] = v
    missing, unexpected = model.load_state_dict(new_weight, strict=True)
    model = model.to(device)
    model.eval()
    return model, clip_model


def run_inference(cfg, model, cls_name_token, text_weight, val_images, val_labels, val_filenames, limit=None):
    c_num = cfg.DATASET.NUM_CLASSES
    n = len(val_images) if limit is None else min(limit, len(val_images))

    # per-image, per-class accumulators
    U = np.full((n, c_num), np.nan, dtype=np.float64)      # spatial mean prob
    P95 = np.full((n, c_num), np.nan, dtype=np.float64)    # spatial 95th pct prob
    PRESENT = np.zeros((n, c_num), dtype=bool)

    # per-image scalars
    IMG_ERR = np.full(n, np.nan, dtype=np.float64)          # 1 - pixel accuracy on valid pixels
    IMG_MEAN_U_ABSENT = np.full(n, np.nan, dtype=np.float64)

    # for (e): FP pixels attributable to "absent-with-elevated-DC" classes.
    # collected as running totals; elevated threshold applied in a 2nd pass
    # (needs the global mean-u-for-present-classes computed after pass 1), so
    # we cache the per-image [C,H,W]-derived per-pixel argmax pred + valid mask
    # is too large to keep for 1449 images -> we instead cache, per image, the
    # per-class boolean "is this absent class's spatial mean elevated" AFTER
    # pass 1 threshold is known, by re-deriving FP counts from stored per-image
    # prediction histograms captured during pass 1.
    fp_total = np.zeros(n, dtype=np.int64)
    valid_total = np.zeros(n, dtype=np.int64)
    # fp_by_class[i, c] = number of valid-pixel false positives in image i predicted as class c
    FP_BY_CLASS = np.zeros((n, c_num), dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for idx in range(n):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)

            label = load_label(val_labels[idx], cfg)
            ori_shape = label.shape  # (H, W)

            shape = img.shape[2:]
            output = model(img, [], text_weight, cls_name_token, training=False)

            # --- exact PD filtering as tools/test.py (RECLIPPP branch) ---
            N, C, H, W = output.shape
            _output = F.softmax(output * 10, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

            # --- exact double interpolation as tools/test.py ---
            output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
            output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])

            # --- final softmax: this is EXACTLY the tensor tools/test.py argmaxes at line 203 ---
            output = F.softmax(output, dim=1)  # [1, C, H, W], pre-argmax

            label_t = torch.from_numpy(label).to(device)
            valid_mask = (label_t != 255)
            n_valid = int(valid_mask.sum().item())
            valid_total[idx] = n_valid

            out_valid = output[0][:, valid_mask]  # [C, n_valid]

            u = out_valid.mean(dim=1)  # [C]
            p95 = torch.quantile(out_valid, 0.95, dim=1)  # [C]

            U[idx] = u.detach().cpu().numpy()
            P95[idx] = p95.detach().cpu().numpy()

            present_classes = torch.unique(label_t[valid_mask]).detach().cpu().numpy()
            present_classes = present_classes[(present_classes >= 0) & (present_classes < c_num)]
            PRESENT[idx, present_classes] = True

            absent_mask = ~PRESENT[idx]
            if absent_mask.any():
                IMG_MEAN_U_ABSENT[idx] = U[idx, absent_mask].mean()

            pred = torch.argmax(output, dim=1)[0]  # [H, W], same op as tools/test.py line 203
            pred_valid = pred[valid_mask]
            label_valid = label_t[valid_mask]
            wrong = pred_valid != label_valid
            fp_total[idx] = int(wrong.sum().item())
            IMG_ERR[idx] = fp_total[idx] / max(n_valid, 1)

            if wrong.any():
                wrong_preds = pred_valid[wrong].detach().cpu().numpy()
                counts = np.bincount(wrong_preds, minlength=c_num)
                FP_BY_CLASS[idx] = counts

            if (idx + 1) % 100 == 0 or idx + 1 == n:
                elapsed = time.time() - t0
                print(f'[analyze_bias_residual] {idx + 1}/{n} images, {elapsed:.1f}s elapsed', flush=True)

    return {
        'U': U, 'P95': P95, 'PRESENT': PRESENT,
        'IMG_ERR': IMG_ERR, 'IMG_MEAN_U_ABSENT': IMG_MEAN_U_ABSENT,
        'FP_BY_CLASS': FP_BY_CLASS, 'fp_total': fp_total, 'valid_total': valid_total,
        'filenames': val_filenames[:n],
    }


def summarize(results, class_names, out_dir, npz_out):
    U = results['U']; P95 = results['P95']; PRESENT = results['PRESENT']
    IMG_ERR = results['IMG_ERR']; IMG_MEAN_U_ABSENT = results['IMG_MEAN_U_ABSENT']
    FP_BY_CLASS = results['FP_BY_CLASS']; fp_total = results['fp_total']
    n, C = U.shape
    GAP = P95 - U

    present_u = U[PRESENT]
    absent_u = U[~PRESENT]
    present_gap = GAP[PRESENT]
    absent_gap = GAP[~PRESENT]

    a_mean, a_std = float(present_u.mean()), float(present_u.std())
    b_mean, b_std = float(absent_u.mean()), float(absent_u.std())

    # (b) per-class across-image std of u_c, computed over images where class c is absent
    per_class_absent_std = np.full(C, np.nan)
    for c in range(C):
        vals = U[~PRESENT[:, c], c]
        if len(vals) > 1:
            per_class_absent_std[c] = vals.std()
    median_absent_std = float(np.nanmedian(per_class_absent_std))
    top5_idx = np.argsort(-np.nan_to_num(per_class_absent_std, nan=-1))[:5]
    top5 = [(class_names[i], float(per_class_absent_std[i])) for i in top5_idx]

    # (c) peakedness gap present vs absent
    c_present_mean, c_present_std = float(present_gap.mean()), float(present_gap.std())
    c_absent_mean, c_absent_std = float(absent_gap.mean()), float(absent_gap.std())

    # (d) Pearson correlation, per image: mean u over absent classes vs per-image error
    #     (error = 1 - pixel accuracy on valid pixels = FP-pixel fraction here)
    valid_mask_img = ~np.isnan(IMG_MEAN_U_ABSENT) & ~np.isnan(IMG_ERR)
    x = IMG_MEAN_U_ABSENT[valid_mask_img]
    y = IMG_ERR[valid_mask_img]
    if len(x) > 2 and x.std() > 0 and y.std() > 0:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
    else:
        pearson_r = float('nan')
    n_images_for_corr = int(valid_mask_img.sum())

    # (e) fraction of FP pixels belonging to "absent-with-elevated-DC" classes.
    # elevated threshold = global mean u_c for PRESENT classes (a_mean): an absent
    # class in a given image counts as "elevated" if its spatial-mean probability
    # in that image is at least as high as the typical PRESENT-class spatial mean.
    elevated_absent = (~PRESENT) & (U >= a_mean)  # [n, C] boolean
    fp_elevated = FP_BY_CLASS[elevated_absent].sum() if elevated_absent.any() else 0
    fp_total_sum = int(fp_total.sum())
    frac_fp_elevated = float(fp_elevated) / fp_total_sum if fp_total_sum > 0 else float('nan')

    stats = {
        'a_present_mean': a_mean, 'a_present_std': a_std,
        'a_absent_mean': b_mean, 'a_absent_std': b_std,
        'b_median_absent_across_image_std': median_absent_std,
        'b_top5_classes_highest_std': top5,
        'c_present_gap_mean': c_present_mean, 'c_present_gap_std': c_present_std,
        'c_absent_gap_mean': c_absent_mean, 'c_absent_gap_std': c_absent_std,
        'd_pearson_r': pearson_r, 'd_n_images': n_images_for_corr,
        'd_metric': 'per-image error = 1 - pixel accuracy on valid (non-ignore) pixels (== FP-pixel fraction)',
        'e_frac_fp_from_elevated_absent': frac_fp_elevated,
        'e_fp_elevated_count': int(fp_elevated), 'e_fp_total_count': fp_total_sum,
        'n_images': n, 'n_classes': C,
    }

    os.makedirs(out_dir, exist_ok=True)
    np.savez(npz_out, U=U, P95=P95, PRESENT=PRESENT, IMG_ERR=IMG_ERR,
             IMG_MEAN_U_ABSENT=IMG_MEAN_U_ABSENT, FP_BY_CLASS=FP_BY_CLASS,
             fp_total=fp_total, class_names=np.array(class_names))
    with open(os.path.join(out_dir, 'bias_residual_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # --- figure ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(min(present_u.min(), absent_u.min()), max(present_u.max(), absent_u.max()), 60)
    ax.hist(present_u, bins=bins, alpha=0.6, label=f'present (n={len(present_u)})', color='#1f77b4', density=True)
    ax.hist(absent_u, bins=bins, alpha=0.6, label=f'absent (n={len(absent_u)})', color='#d62728', density=True)
    ax.axvline(a_mean, color='#1f77b4', linestyle='--', linewidth=1)
    ax.axvline(b_mean, color='#d62728', linestyle='--', linewidth=1)
    ax.set_xlabel('u_c: per-image spatial mean of final softmax probability (class c)')
    ax.set_ylabel('density')
    ax.set_title('Residual DC elevation: present vs absent classes\n(RECLIPPP, official VOC checkpoint)')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(out_dir, 'bias_residual.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return stats, fig_path


def write_report(stats, fig_path, out_dir, cfg_path):
    top5_lines = '\n'.join(f'  {i+1}. {name}: std={std:.6f}' for i, (name, std) in enumerate(stats['b_top5_classes_highest_std']))

    verdict = 'SUPPORTED' if (
        stats['a_absent_mean'] > 0 and
        stats['a_absent_mean'] < stats['a_present_mean'] and
        stats['b_median_absent_across_image_std'] > 0.01 and
        (not np.isnan(stats['d_pearson_r'])) and stats['d_pearson_r'] > 0.1
    ) else 'WEAK'

    md = f"""# Diagnostic: Image-Dependent Residual Bias After Static Bias Subtraction

## Premise

ReCLIP++ (RECLIPPP) computes `bias_logits = pe_proj(positional_embedding) @ prompt.T`
inside `RECLIPPP.forward` (`model/model.py`). `positional_embedding` here is the ViT's
(interpolated) positional embedding grid -- it does not depend on image content, so
`bias_logits` is the same functional bias for every image (only image content enters
through `output_q`). The model then computes
`output = output_q - bias_logits`, followed by a small decoder conv/BN
(`decoder_conv2` + `decoder_norm2`) on `concat(feat, output)`.

Hypothesis: after this static-bias subtraction (and the decoder on top of it), the
FINAL per-pixel, per-class probability map still carries a spatially-uniform ("DC")
elevation for classes that are NOT actually present in the image (class
hallucination / residual bias), and this elevation varies across images
(image-dependent). If true, a static (image-independent) subtraction structurally
cannot remove it -- this is the headroom a future method would target.

## Method

- Model: `RECLIPPP` (`model/model.py`), official checkpoint
  `experiments/official/voc_reclippp_854/best_weight.pth`, config `{cfg_path}`.
- Built and run EXACTLY as `tools/test.py` (`build_model`, `val_preprocess`, PD
  filtering with `cfg.TEST.PD`, double bilinear interpolation to original image
  resolution, final `F.softmax(output, dim=1)`). The tensor analyzed is the one
  `tools/test.py` line 203 feeds into `torch.argmax` -- i.e. the actual per-pixel
  class-probability map that determines the prediction -- captured BEFORE argmax,
  instead of the post-argmax label map `tools/test.py` saves to disk.
- Full VOC2012 val set ({stats['n_images']} images), C={stats['n_classes']} classes
  (VOC foreground classes only; background/ignore excluded via
  `DATASET.REDUCE_ZERO_LABEL`).
- Per image, per class c: `u_c` = spatial mean of the probability channel c over
  valid (non-ignore) pixels ("DC" / uniform component); `p_c` = spatial 95th
  percentile of channel c ("peak"); `gap_c = p_c - u_c` (peakedness).
- GT is used ONLY to label each class, per image, as present (appears in that
  image's GT) or absent. This is legitimate for this diagnostic -- it
  characterizes the failure mode, it is not a training signal and does not gate
  anything. The eventual method motivated by this diagnostic will be unsupervised
  (GT-free) at inference time.

## Raw numbers

| # | Quantity | Value |
|---|---|---|
| (a) | mean u_c, PRESENT classes | {stats['a_present_mean']:.6f} |
| (a) | std u_c, PRESENT classes | {stats['a_present_std']:.6f} |
| (a) | mean u_c, ABSENT classes | {stats['a_absent_mean']:.6f} |
| (a) | std u_c, ABSENT classes | {stats['a_absent_std']:.6f} |
| (b) | median across-class of (per-class across-image std of u_c, ABSENT-class occurrences) | {stats['b_median_absent_across_image_std']:.6f} |
| (b) | top-5 classes by this across-image std | see below |
| (c) | mean peakedness gap_c, PRESENT classes | {stats['c_present_gap_mean']:.6f} |
| (c) | std peakedness gap_c, PRESENT classes | {stats['c_present_gap_std']:.6f} |
| (c) | mean peakedness gap_c, ABSENT classes | {stats['c_absent_gap_mean']:.6f} |
| (c) | std peakedness gap_c, ABSENT classes | {stats['c_absent_gap_std']:.6f} |
| (d) | Pearson r(mean u over absent classes, per-image error) | {stats['d_pearson_r']:.6f} |
| (d) | n images used for (d) | {stats['d_n_images']} |
| (d) | error metric used | {stats['d_metric']} |
| (e) | fraction of FP pixels from ABSENT-with-elevated-DC classes | {stats['e_frac_fp_from_elevated_absent']:.6f} |
| (e) | FP pixels from elevated-absent classes / total FP pixels | {stats['e_fp_elevated_count']} / {stats['e_fp_total_count']} |

Top-5 classes with highest per-class across-image std of u_c (measured only over
images where that class is absent) -- (b):
```
{top5_lines}
```

Note on (e): "elevated" is defined as `u_c >= mean(u_c over PRESENT classes,
dataset-wide)` (row (a), {stats['a_present_mean']:.6f}) for an absent class c in a
given image -- i.e. the absent class's spatial-mean probability in that image is at
least as high as a typical present class's spatial-mean probability dataset-wide.

## Figure

![present vs absent u_c histogram]({os.path.basename(fig_path)})

Histogram of `u_c` (per-image spatial mean probability) for present classes
(blue) vs absent classes (red), overlaid, density-normalized. Dashed lines mark
the two group means.

## Honest read

{verdict}: absent classes have mean u_c = {stats['a_absent_mean']:.6f} vs present
classes' {stats['a_present_mean']:.6f} (row a); the median per-class across-image
std for absent-class occurrences is {stats['b_median_absent_across_image_std']:.6f}
(row b), meaning the residual DC level for a given absent class is not a fixed
constant but varies image-to-image by roughly that much, which a static
(image-independent) bias subtraction cannot track; present classes are
substantially more peaked than absent classes (gap_c row c); the per-image
correlation between mean-u-over-absent-classes and per-image error is
r={stats['d_pearson_r']:.4f} (row d); and {stats['e_frac_fp_from_elevated_absent']*100:.2f}%
of all false-positive pixels are attributable to absent classes whose DC level in
that image reached typical present-class levels (row e). Read these numbers
together, not any single one in isolation, to judge whether the headroom is
sizeable enough to justify a dynamic (image-dependent) bias-correction method.
"""
    path = os.path.join(out_dir, 'bias_residual_diagnostic.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)
    return path


def main():
    args = get_parser()
    cfg = cfg_from_file(args.cfg)

    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    model, _ = build_model(cfg)

    results = run_inference(cfg, model, cls_name_token, text_weight, val_images, val_labels, val_filenames, limit=args.limit)

    stats, fig_path = summarize(results, voc_classes, args.out_dir, args.npz_out)
    report_path = write_report(stats, fig_path, args.out_dir, args.cfg)

    print('=== RAW NUMBERS ===')
    print(json.dumps(stats, indent=2, default=str))
    print('report:', report_path)
    print('figure:', fig_path)


if __name__ == '__main__':
    main()
