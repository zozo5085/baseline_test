"""Diagnostic metrics for the journal generalization audit (Framing A).

Computes, from SAVED argmax prediction .pt maps + GT pngs (no model re-run):
  - boundary-band error : misclassification rate in a dilated band around GT boundaries (lower better)
  - small-object accuracy: pixel accuracy on small GT connected components (higher better)
  - false-positive class count: mean number of classes predicted (>= min area) but absent in GT (lower better)

reduce_zero_label is applied to GT to match the prediction label space (0..C-1, 255 ignore),
identical to the eval mIoU. Usage:
  python tools/diag_metrics.py --pred_dir experiments/context_conv_eval/base_notta \
    --gt_dir "D:/ReCLIPv3/datasets/PASCAL VOC/VOC2010/SegmentationClassContext" \
    --num_classes 59 --stride 5 --label "Context base no-TTA"
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def reduce_zero(gt):
    gt = gt.astype(np.int64).copy()
    gt[gt == 0] = 255
    gt = gt - 1
    gt[gt == 254] = 255
    return gt


def boundary_band(gt, r):
    b = np.zeros(gt.shape, bool)
    d = gt[:-1, :] != gt[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = gt[:, :-1] != gt[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    if r > 1:
        b = ndimage.binary_dilation(b, iterations=r - 1)
    return b


def small_mask(gt, T):
    m = np.zeros(gt.shape, bool)
    for c in np.unique(gt):
        if c == 255:
            continue
        cc, n = ndimage.label(gt == c)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel())
        small_ids = np.where(sizes < T)[0]
        small_ids = small_ids[small_ids != 0]
        if small_ids.size:
            m |= np.isin(cc, small_ids)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--gt_suffix', default='.png')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--band_r', type=int, default=3)
    ap.add_argument('--small_T', type=int, default=1024)
    ap.add_argument('--fp_min_frac', type=float, default=0.001)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, '*.pt')))[::args.stride]
    tot_band = wrong_band = tot_small = corr_small = 0
    fp_sum = 0.0
    n = miss = 0
    for pf in preds:
        name = os.path.splitext(os.path.basename(pf))[0]
        gtp = os.path.join(args.gt_dir, name + args.gt_suffix)
        if not os.path.exists(gtp):
            miss += 1
            continue
        pred = torch.load(pf).cpu().numpy().astype(np.int64)
        gt = reduce_zero(np.array(Image.open(gtp)))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred.astype(np.int32)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(np.int64)
        valid = gt != 255
        band = boundary_band(gt, args.band_r) & valid
        tot_band += int(band.sum())
        wrong_band += int(((pred != gt) & band).sum())
        sm = small_mask(gt, args.small_T) & valid
        tot_small += int(sm.sum())
        corr_small += int(((pred == gt) & sm).sum())
        H, W = gt.shape
        min_px = max(50, int(args.fp_min_frac * H * W))
        gt_cls = set(np.unique(gt[valid]).tolist())
        pc, cnt = np.unique(pred, return_counts=True)
        pred_cls = set(pc[cnt >= min_px].tolist())
        fp_sum += len(pred_cls - gt_cls)
        n += 1
    print(f"[{args.label}] N={n} miss_gt={miss}")
    print(f"  boundary_band_error = {wrong_band / max(tot_band, 1):.4f}  (band_r={args.band_r})")
    print(f"  small_object_acc    = {corr_small / max(tot_small, 1):.4f}  (T<{args.small_T}px)")
    print(f"  fp_class_count_img  = {fp_sum / max(n, 1):.4f}")


if __name__ == '__main__':
    main()
