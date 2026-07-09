"""ADE20K data-integrity gate (pre-registered in GENERALIZATION_PROTOCOL.md section 8.2).

The Context lesson: shipped GT / text-embedding / pseudo-label class orders silently
disagreed (top1-in-GT 0.094) and every training was garbage. Run this BEFORE any ADE
training. Checks:

  1. repo ade_classes order vs the dataset's objectInfo150.txt (official index order)
  2. text-embedding recipe verification: rebuild VOC weights with the prompt-ensemble
     recipe (utils.preprocess 'open' branch) and compare to the shipped
     text/voc_ViT16_clip_text.pth (per-class cosine; min must be ~1.0)
  3. (--write) generate text/ade_ViT16_clip_text.pth in ade_classes order
  4. GT annotation value range on N validation masks (expect {0..150}, 0 = ignore)
  5. top1-in-GT rate on N validation images with the ADE text weights
     (Context reference: broken 0.094 -> fixed 0.873)
  6. pseudo-label sanity vs GT on N training images (line count == #train imgs,
     index range, mean recall of GT classes; Context reference: broken 0.25 -> fixed 0.579)

Usage:
  python tools/ade_integrity_check.py --dataroot "D:/ReCLIPv3/datasets/ADEChallengeData2016/" \
      --n_sample 300 [--write]
"""
import argparse
import os
import sys
import json

import numpy as np
import torch
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import clip
from utils.preprocess import ade_classes, prompt_templates

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_zeroshot(model, classnames):
    with torch.no_grad():
        ws = []
        for classname in classnames:
            texts = [t.format(classname) for t in prompt_templates]
            texts = clip.tokenize(texts).to(device)
            emb = model.encode_text(texts)
            emb /= emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            emb /= emb.norm()
            ws.append(emb)
        return torch.stack(ws, dim=1).permute(1, 0).float().cpu()  # [C,512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--n_sample', type=int, default=300)
    ap.add_argument('--write', action='store_true',
                    help='write text/ade_ViT16_clip_text.pth (only after checks pass)')
    ap.add_argument('--pseudo_json', default='text/ade_pseudo_label_ReCLIPPP.json')
    args = ap.parse_args()

    fails = []

    # ---- 1. class order vs objectInfo150.txt ----
    info = os.path.join(args.dataroot, 'objectInfo150.txt')
    official = []
    with open(info, 'r', encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            official.append(parts[-1].strip())
    print('[1] objectInfo150 entries: %d; repo ade_classes: %d' % (len(official), len(ade_classes)))
    mism = 0
    for i, (o, r) in enumerate(zip(official, ade_classes)):
        first_syn = o.split(',')[0].strip().lower()
        rr = r.strip().lower()
        if first_syn != rr:
            mism += 1
            if mism <= 10:
                print('    idx %d: official "%s" vs repo "%s"' % (i, o, r))
    print('[1] first-synonym mismatches: %d / %d' % (mism, len(ade_classes)))
    if len(official) != len(ade_classes):
        fails.append('class count mismatch')

    # ---- 2. recipe verification on VOC ----
    model, preprocess = clip.load('ViT-B/16')
    model = model.to(device).eval()
    from utils.preprocess import voc_classes
    voc_shipped = torch.load('text/voc_ViT16_clip_text.pth', map_location='cpu').float()
    voc_re = build_zeroshot(model, voc_classes)
    print('[2] shipped voc shape %s / rebuilt %s' % (tuple(voc_shipped.shape), tuple(voc_re.shape)))
    if voc_shipped.shape != voc_re.shape:
        fails.append('voc text shape mismatch (recipe orientation?)')
    else:
        a = voc_shipped / voc_shipped.norm(dim=-1, keepdim=True)
        b = voc_re / voc_re.norm(dim=-1, keepdim=True)
        cos = (a * b).sum(-1)
        print('[2] per-class cosine shipped-vs-rebuilt: min %.6f mean %.6f' % (cos.min(), cos.mean()))
        if cos.min() < 0.995:
            fails.append('voc recipe cosine < 0.995 -> recipe NOT confirmed')

    # ---- 3. ADE text weights ----
    ade_w = build_zeroshot(model, ade_classes)
    print('[3] ade text weights built: %s' % (tuple(ade_w.shape),))
    if args.write:
        torch.save(ade_w, 'text/ade_ViT16_clip_text.pth')
        print('[3] written text/ade_ViT16_clip_text.pth')

    # ---- 4. GT value range ----
    val_img_dir = os.path.join(args.dataroot, 'images', 'validation')
    val_ann_dir = os.path.join(args.dataroot, 'annotations', 'validation')
    val_names = sorted(n[:-4] for n in os.listdir(val_img_dir) if n.endswith('.jpg'))
    step = max(1, len(val_names) // args.n_sample)
    sample = val_names[::step][:args.n_sample]
    vmax, vmin, bad = 0, 999, 0
    for n in sample:
        arr = np.array(Image.open(os.path.join(val_ann_dir, n + '.png')))
        vmax = max(vmax, int(arr.max())); vmin = min(vmin, int(arr.min()))
        if arr.max() > 150:
            bad += 1
    print('[4] GT value range over %d masks: min %d max %d; masks with value>150: %d'
          % (len(sample), vmin, vmax, bad))
    if vmax > 150 or bad:
        fails.append('GT values out of 0..150')

    # ---- 5. top1-in-GT ----
    ade_w_dev = (ade_w / ade_w.norm(dim=-1, keepdim=True)).to(device)
    hit = 0
    with torch.no_grad():
        for n in sample:
            img = preprocess(Image.open(os.path.join(val_img_dir, n + '.jpg')).convert('RGB'))
            feat = model.encode_image(img.unsqueeze(0).to(device)).float()
            feat /= feat.norm(dim=-1, keepdim=True)
            top1 = int((feat @ ade_w_dev.t()).argmax())
            arr = np.array(Image.open(os.path.join(val_ann_dir, n + '.png')))
            gt_cls = set(int(v) - 1 for v in np.unique(arr) if v != 0)
            if top1 in gt_cls:
                hit += 1
    rate = hit / len(sample)
    print('[5] top1-in-GT rate: %.3f (%d/%d)  [Context ref: broken 0.094 / fixed 0.873]'
          % (rate, hit, len(sample)))
    if rate < 0.5:
        fails.append('top1-in-GT rate < 0.5 -> class-order misalignment suspected')

    # ---- 6. pseudo labels vs GT ----
    train_img_dir = os.path.join(args.dataroot, 'images', 'training')
    train_ann_dir = os.path.join(args.dataroot, 'annotations', 'training')
    train_names = [n[:-4] for n in os.listdir(train_img_dir) if n.endswith('.jpg')]
    pseudo = []
    with open(args.pseudo_json, 'r') as f:
        for line in f:
            line = line.strip()[1:-1]
            if line.endswith(','):
                line = line[:-1]
            if ',' in line:
                pseudo.append(sorted(map(int, line.split(','))))
            elif line:
                pseudo.append([int(line)])
            else:
                pseudo.append([])
    flat = [c for p in pseudo for c in p]
    print('[6] pseudo lines %d vs train imgs %d; class idx range %d..%d'
          % (len(pseudo), len(train_names), min(flat), max(flat)))
    if len(pseudo) != len(train_names):
        fails.append('pseudo line count != train image count')
    if flat and (min(flat) < 0 or max(flat) > 149):
        fails.append('pseudo class index out of 0..149')
    tstep = max(1, len(train_names) // args.n_sample)
    rec, prec, nchk = 0.0, 0.0, 0
    for i in range(0, len(train_names), tstep):
        arr = np.array(Image.open(os.path.join(train_ann_dir, train_names[i] + '.png')))
        gt_cls = set(int(v) - 1 for v in np.unique(arr) if v != 0)
        ps = set(pseudo[i])
        if not gt_cls:
            continue
        nchk += 1
        rec += len(gt_cls & ps) / len(gt_cls)
        if ps:
            prec += len(gt_cls & ps) / len(ps)
    print('[6] over %d train imgs: mean GT-class recall %.3f / pseudo precision %.3f'
          % (nchk, rec / max(nchk, 1), prec / max(nchk, 1)))
    if nchk and rec / nchk < 0.35:
        fails.append('pseudo recall < 0.35 -> pseudo/GT misalignment suspected (Context broken ref 0.25)')

    print('\n===== ADE INTEGRITY: %s =====' % ('FAIL' if fails else 'PASS'))
    for msg in fails:
        print('  FAIL:', msg)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
