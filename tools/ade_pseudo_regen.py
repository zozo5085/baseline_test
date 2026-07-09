"""Regenerate ADE20K pseudo image-level labels locally (sliding-window top-1 voting).

Why: the shipped text/ade_pseudo_label_ReCLIPPP.json does not correspond to this
machine's training-image enumeration (recall 0.235 ~= chance, invariant to image
shifts -- see tools/ade_integrity_check.py + docs). Same failure class as the
Context lesson; the fix there was local regeneration with corrected text
(sliding-window), which this script reproduces for ADE with the batched encoder.

Recipe = tools/pseudo_class.py ReCLIPPP mode: window = CROP_SIZE/6, step = window/2,
CLIP top-1 class per crop, image pseudo set = union of crop votes (fallback: global
top-K). Differences: crops are batched through encode_image (fast), no sleep, output
lists sorted, live GT recall/precision reporting. Line i of the output corresponds to
image i of os.listdir(images/training) on THIS machine -- the same enumeration
utils/preprocess.read_file_list uses at train time.

Usage:
  python tools/ade_pseudo_regen.py --cfg config/ade_train_converged_cfg.yaml \
      --out text/ade_pseudo_label.json [--limit 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import clip
from config.configs import cfg_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--encode_bs', type=int, default=256)
    args = ap.parse_args()

    cfg = cfg_from_file(args.cfg)
    crop = cfg.DATASET.CROP_SIZE
    w = int(crop[0] / 6)
    s = int(w / 2)
    K = int(cfg.DATASET.K)

    model, preprocess = clip.load('ViT-B/16')
    model = model.to(device).eval()
    text = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)
    text = text / text.norm(dim=-1, keepdim=True)

    img_dir = cfg.DATASET.DATAROOT + 'images/training/'
    ann_dir = cfg.DATASET.DATAROOT + 'annotations/training/'
    names = [n[:-4] for n in os.listdir(img_dir) if n[-3:] == 'jpg']
    if args.limit:
        names = names[:args.limit]
    print('train images: %d  window=%d step=%d K=%d' % (len(names), w, s, K))

    if os.path.exists(args.out):
        print('REFUSING to overwrite existing %s' % args.out)
        sys.exit(1)

    rec_sum = prec_sum = 0.0
    nchk = 0
    import time
    t0 = time.time()
    with torch.no_grad(), open(args.out, 'a') as fout:
        for i, name in enumerate(names):
            img = Image.open(img_dir + name + '.jpg').convert('RGB')
            width, height = img.size
            crops = []
            for y in range(0, max(height - w, 1), s):
                for x in range(0, max(width - w, 1), s):
                    crops.append(preprocess(img.crop((x, y, x + w, y + w))))
            if not crops:
                crops = [preprocess(img)]
            batch = torch.stack(crops)
            votes = set()
            logits_pool = []
            for b0 in range(0, len(batch), args.encode_bs):
                feat = model.encode_image(batch[b0:b0 + args.encode_bs].to(device))
                feat = feat / feat.norm(dim=1, keepdim=True)
                logits = feat @ text.t()
                logits_pool.append(logits)
                votes.update(logits.argmax(dim=1).tolist())
            if not votes:
                votes = set(torch.cat(logits_pool).mean(0).topk(K).indices.tolist())
            temp = sorted(int(v) for v in votes)
            fout.write(json.dumps(temp))
            fout.write('\n')

            arr = np.array(Image.open(ann_dir + name + '.png'))
            gt = set(int(v) - 1 for v in np.unique(arr) if v != 0)
            if gt:
                nchk += 1
                rec_sum += len(gt & votes) / len(gt)
                prec_sum += len(gt & votes) / max(len(votes), 1)
            if (i + 1) % 500 == 0 or i + 1 == len(names):
                el = time.time() - t0
                print('%d/%d  %.2f img/s  ETA %.1f min  recall %.3f  precision %.3f'
                      % (i + 1, len(names), (i + 1) / el,
                         (len(names) - i - 1) / ((i + 1) / el) / 60,
                         rec_sum / max(nchk, 1), prec_sum / max(nchk, 1)), flush=True)

    print('DONE. mean GT recall %.3f  precision %.3f  (Context fixed ref: recall 0.579)'
          % (rec_sum / max(nchk, 1), prec_sum / max(nchk, 1)))
    print('written: %s (%d lines)' % (args.out, len(names)))


if __name__ == '__main__':
    main()
