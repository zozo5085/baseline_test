"""Paired per-image bootstrap significance for mIoU deltas (pure analysis).

mIoU here is additive over per-image (intersect, union) class histograms, so a
paired bootstrap is exact w.r.t. the eval semantics: compute each image's
intersect/union arrays once per method (identical to utils/test_mIoU.py --
num_classes = NUM_CLASSES+1 bins, reduce_zero_label, IoU nan->0, mIoU =
IoU.sum()/NUM_CLASSES), then resample IMAGE INDICES with replacement (the same
indices for base and variant), recompute both mIoUs from the resampled sums,
and read the delta distribution.

Reports: observed mIoU (must reproduce the recorded numbers verbatim -- built-in
sanity), observed delta, 95% percentile CI, one-sided and two-sided bootstrap
p-values. Fixed seed -> reproducible.

Usage:
  python tools/bootstrap_significance.py --cfg config/voc_test_lgak_identity_cfg.yaml \
      --base experiments/lgak_id_baseline \
      --variants flip=experiments/voc_diag_flip,sfp_gen=experiments/voc_sfp_dtlr_gen_official_eval \
      --out experiments/bootstrap_significance/voc.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.preprocess import read_file_list
from utils.test_mIoU import intersect_and_union


def per_image_arrays(pred_dir, val_filenames, val_labels, num_classes, reduce_zero_label):
    N = len(val_filenames)
    I = np.zeros((N, num_classes), dtype=np.float64)
    U = np.zeros((N, num_classes), dtype=np.float64)
    for i, (name, gt) in enumerate(zip(val_filenames, val_labels)):
        pred_path = os.path.join(pred_dir, name + '.pt')
        ai, au, _, _ = intersect_and_union(
            pred_path, gt, num_classes, ignore_index=255,
            label_map=dict(), reduce_zero_label=reduce_zero_label)
        I[i] = ai.cpu().numpy()
        U[i] = au.cpu().numpy()
        if (i + 1) % 500 == 0 or i + 1 == N:
            print('  %s: %d/%d' % (os.path.basename(pred_dir.rstrip('/\\')), i + 1, N))
    return I, U


def miou_from_sums(I_sum, U_sum, c_num):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = I_sum / U_sum
    iou = np.nan_to_num(iou, nan=0.0)
    return float(iou.sum() / c_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--variants', required=True,
                    help='comma list name=pred_dir')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = cfg_from_file(args.cfg)
    c_num = cfg.DATASET.NUM_CLASSES
    num_classes = c_num + 1
    rzl = cfg.DATASET.REDUCE_ZERO_LABEL

    _, val_filenames, _, _, _, val_labels, _, _ = read_file_list(cfg)
    N = len(val_filenames)
    print('images=%d num_classes=%d(+1 bins) reduce_zero_label=%s' % (N, c_num, rzl))

    print('[base] %s' % args.base)
    I_b, U_b = per_image_arrays(args.base, val_filenames, val_labels, num_classes, rzl)
    base_miou = miou_from_sums(I_b.sum(0), U_b.sum(0), c_num)
    print('[base] observed mIoU = %.4f' % base_miou)

    rng = np.random.default_rng(args.seed)
    idx_mat = rng.integers(0, N, size=(args.n_boot, N))

    results = {'cfg': args.cfg, 'base': args.base, 'images': N,
               'n_boot': args.n_boot, 'seed': args.seed,
               'base_miou': base_miou, 'variants': {}}

    # Precompute base bootstrap mIoUs once (same idx_mat reused for every variant).
    base_boot = np.empty(args.n_boot, dtype=np.float64)
    for b in range(args.n_boot):
        idx = idx_mat[b]
        base_boot[b] = miou_from_sums(I_b[idx].sum(0), U_b[idx].sum(0), c_num)

    for spec in args.variants.split(','):
        vname, vdir = spec.split('=', 1)
        print('[variant:%s] %s' % (vname, vdir))
        I_v, U_v = per_image_arrays(vdir, val_filenames, val_labels, num_classes, rzl)
        v_miou = miou_from_sums(I_v.sum(0), U_v.sum(0), c_num)
        delta_obs = v_miou - base_miou

        deltas = np.empty(args.n_boot, dtype=np.float64)
        for b in range(args.n_boot):
            idx = idx_mat[b]
            deltas[b] = miou_from_sums(I_v[idx].sum(0), U_v[idx].sum(0), c_num) - base_boot[b]

        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_le0 = float((deltas <= 0).mean())
        p_ge0 = float((deltas >= 0).mean())
        p_two = min(1.0, 2 * min(p_le0, p_ge0))
        results['variants'][vname] = {
            'pred_dir': vdir, 'miou': v_miou, 'delta': delta_obs,
            'ci95': [float(lo), float(hi)],
            'p_one_sided_le0': p_le0, 'p_one_sided_ge0': p_ge0,
            'p_two_sided': p_two,
        }
        print('  mIoU=%.4f delta=%+.4f 95%%CI=[%+.4f, %+.4f] p(two-sided)=%s'
              % (v_miou, delta_obs, lo, hi,
                 ('<%g' % (2.0 / args.n_boot)) if p_two == 0 else '%.4f' % p_two))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('written: %s' % args.out)


if __name__ == '__main__':
    main()
