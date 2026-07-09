"""Qualitative before/after figure for the journal (flip-TTA benefit).

Picks the Context val images where flip-TTA most reduces pixel error vs the no-TTA baseline,
and renders a grid: input | GT | baseline pred | flip pred | flip-vs-baseline diff.
diff map: green = flip fixed a baseline error, red = flip introduced an error, gray = unchanged/ignore.

Usage:
  python tools/diag_figure.py --base experiments/context_conv_eval/base_notta \
    --flip experiments/context_conv_eval/base_flip \
    --gt_dir "D:/ReCLIPv3/datasets/PASCAL VOC/VOC2010/SegmentationClassContext" \
    --img_dir "D:/ReCLIPv3/datasets/PASCAL VOC/VOC2010/JPEGImages" \
    --num_classes 59 --topk 3 --cand 250 --out <path.png>
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reduce_zero(gt):
    gt = gt.astype(np.int64).copy()
    gt[gt == 0] = 255
    gt = gt - 1
    gt[gt == 254] = 255
    return gt


def color_labels(lbl, C, seed=7):
    rng = np.random.RandomState(seed)
    pal = rng.randint(45, 225, (C, 3)).astype(np.uint8)
    out = np.zeros((*lbl.shape, 3), np.uint8)
    m = (lbl >= 0) & (lbl < C)
    out[m] = pal[lbl[m]]
    out[lbl == 255] = (55, 55, 55)
    return out


def load_pred(pf, shape):
    p = torch.load(pf).cpu().numpy().astype(np.int64)
    if p.shape != shape:
        p = np.array(Image.fromarray(p.astype(np.int32)).resize((shape[1], shape[0]), Image.NEAREST)).astype(np.int64)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--flip', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--cand', type=int, default=250)
    ap.add_argument('--reclippp', default='ReCLIP++')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cand = sorted(glob.glob(os.path.join(args.base, '*.pt')))[:args.cand]
    scored = []
    for pf in cand:
        name = os.path.splitext(os.path.basename(pf))[0]
        gtp = os.path.join(args.gt_dir, name + '.png')
        ff = os.path.join(args.flip, name + '.pt')
        if not (os.path.exists(gtp) and os.path.exists(ff)):
            continue
        gt = reduce_zero(np.array(Image.open(gtp)))
        valid = gt != 255
        b = load_pred(pf, gt.shape)
        f = load_pred(ff, gt.shape)
        be = ((b != gt) & valid).sum()
        fe = ((f != gt) & valid).sum()
        scored.append((be - fe, name, gt, b, f, valid))
    scored.sort(key=lambda x: -x[0])
    picks = scored[:args.topk]

    C = args.num_classes
    fig, axes = plt.subplots(len(picks), 5, figsize=(15, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    titles = ['input', 'GT', f'{args.reclippp} baseline', 'baseline + flip-TTA', 'flip vs baseline']
    for r, (delta, name, gt, b, f, valid) in enumerate(picks):
        img = np.array(Image.open(os.path.join(args.img_dir, name + '.jpg')).convert('RGB'))
        diff = np.full((*gt.shape, 3), 210, np.uint8)
        fixed = valid & (b != gt) & (f == gt)
        broke = valid & (b == gt) & (f != gt)
        diff[fixed] = (30, 175, 30)
        diff[broke] = (200, 40, 40)
        panels = [img, color_labels(gt, C), color_labels(b, C), color_labels(f, C), diff]
        for c, (p, t) in enumerate(zip(panels, titles)):
            axes[r, c].imshow(p)
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(t, fontsize=11)
        axes[r, 0].set_ylabel(f'{name}\n(-{delta} err px)', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"saved {args.out} ; picks: {[p[1] for p in picks]} deltas={[int(p[0]) for p in picks]}")


if __name__ == '__main__':
    main()
