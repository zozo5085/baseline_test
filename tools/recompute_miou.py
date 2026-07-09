"""Recompute mIoU from already-saved per-image .pt argmax predictions.

No model, no GPU: rebuilds the (pred paths, GT paths) lists exactly as the eval
scripts do (read_file_list on the cfg) and calls the same mean_iou. Verifies a
recorded number from its SAVE_DIR without re-running inference.

Usage:
  python tools/recompute_miou.py --cfg config/voc_test_pamr_official_fullres_cfg.yaml
"""
import argparse
import os
import sys

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.test_mIoU import mean_iou
from utils.preprocess import read_file_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--save_dir', default=None, help='override cfg.SAVE_DIR')
    args = ap.parse_args()

    cfg = cfg_from_file(args.cfg)
    if args.save_dir:
        cfg.SAVE_DIR = args.save_dir

    _, val_filenames, _, _, _, val_labels, results_iou, _ = read_file_list(cfg)
    missing = [p for p in results_iou if not os.path.exists(p)]
    print('preds expected=%d missing=%d save_dir=%s' % (len(results_iou), len(missing), cfg.SAVE_DIR))
    if missing:
        print('first missing:', missing[0])
        sys.exit(1)

    c_num = cfg.DATASET.NUM_CLASSES
    iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255,
                   nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
    avg = iou['IoU'].sum() / c_num
    print('avg:%.4f' % avg)
    print('the mIOU:%.4f' % avg)


if __name__ == '__main__':
    main()
