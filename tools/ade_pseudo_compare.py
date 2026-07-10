"""Compare two ADE20K pseudo-label files against the CURRENT training enumeration + GT.

Purpose (2026-07-10): decide whether an old pseudo file (external repo) is usable as
training input, vs a locally regenerated one (tools/ade_pseudo_regen.py) built from the
verified text/ade_ViT16_clip_text.pth and THIS machine's image enumeration.

Per file:
  - full-file stats: line count, parse failures, empty lines/lists, class idx min/max,
    mean classes per image
  - eval on the FIRST N enumerated training images (N = smoke line count):
    mean GT-class recall, pseudo precision, and ORDER CONSISTENCY = recall when the
    image<->line alignment is shifted by +1/-1 (a correctly ordered file collapses
    under shift; an order-broken file barely moves)

Writes a markdown report. Read-only w.r.t. inputs; refuses to overwrite the report
unless --force.

Usage:
  python tools/ade_pseudo_compare.py --cfg config/ade_train_converged_cfg.yaml \
    --old D:/ReCLIPPP2026/text/ade_pseudo_label.json \
    --new text/ade_pseudo_label_regen_smoke.json \
    --out_md docs/ADE20K_PSEUDO_COMPARISON.md
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file


def load_pseudo(path):
    rows, bad = [], 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                rows.append(None); bad += 1
                continue
            try:
                v = json.loads(line)
                assert isinstance(v, list) and all(isinstance(x, int) for x in v)
                rows.append(v)
            except Exception:
                rows.append(None); bad += 1
    return rows, bad


def file_stats(rows, bad):
    flat = [c for r in rows if r for c in r]
    empties = sum(1 for r in rows if r is not None and len(r) == 0)
    return {
        'lines': len(rows),
        'parse_failures': bad,
        'empty_lists': empties,
        'idx_min': min(flat) if flat else None,
        'idx_max': max(flat) if flat else None,
        'mean_classes': (sum(len(r) for r in rows if r) / max(sum(1 for r in rows if r), 1)),
    }


def eval_against_gt(rows, gts, shift=0):
    N = len(gts)
    rec = prec = 0.0
    n = 0
    for i in range(N):
        j = i + shift
        if j < 0 or j >= len(rows) or rows[j] is None:
            continue
        gt = gts[i]
        if not gt:
            continue
        ps = set(rows[j])
        n += 1
        rec += len(gt & ps) / len(gt)
        prec += len(gt & ps) / max(len(ps), 1)
    return (rec / max(n, 1), prec / max(n, 1), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out_md', required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if os.path.exists(args.out_md) and not args.force:
        print('REFUSING to overwrite %s (use --force)' % args.out_md)
        sys.exit(1)

    cfg = cfg_from_file(args.cfg)
    img_dir = cfg.DATASET.DATAROOT + 'images/training/'
    ann_dir = cfg.DATASET.DATAROOT + 'annotations/training/'
    names = [n[:-4] for n in os.listdir(img_dir) if n[-3:] == 'jpg']
    names2 = [n[:-4] for n in os.listdir(img_dir) if n[-3:] == 'jpg']
    enum_stable = names == names2
    enum_md5 = hashlib.md5('\n'.join(names).encode()).hexdigest()

    old_rows, old_bad = load_pseudo(args.old)
    new_rows, new_bad = load_pseudo(args.new)
    n_eval = sum(1 for r in new_rows if r is not None)

    gts = []
    for i in range(min(n_eval, len(names))):
        arr = np.array(Image.open(ann_dir + names[i] + '.png'))
        gts.append(set(int(v) - 1 for v in np.unique(arr) if v != 0))

    so, sn = file_stats(old_rows, old_bad), file_stats(new_rows, new_bad)
    rows_out = []
    for label, rows in (('old', old_rows), ('new', new_rows)):
        r0 = eval_against_gt(rows, gts, 0)
        rp = eval_against_gt(rows, gts, +1)
        rm = eval_against_gt(rows, gts, -1)
        rows_out.append((label, r0, rp, rm))

    def fmt(x):
        return 'n/a' if x is None else str(x)

    md = []
    md.append('# ADE20K pseudo-label comparison — old external vs regenerated (2026-07-10)')
    md.append('')
    md.append('Old (A) = `%s`' % args.old)
    md.append('New (B) = `%s` (tools/ade_pseudo_regen.py, text = verified `%s`)' % (args.new, cfg.DATASET.TEXT_WEIGHT))
    md.append('Enumeration: `os.listdir(%s)` -> %d names, repeat-listing stable = %s, md5(list) = `%s`'
              % (img_dir, len(names), enum_stable, enum_md5))
    md.append('Eval set = first %d enumerated training images (paired for both files); GT = annotations/training.' % len(gts))
    md.append('')
    md.append('| metric | A (old) | B (regen smoke) |')
    md.append('|---|---|---|')
    md.append('| line count | %s | %s |' % (so['lines'], sn['lines']))
    md.append('| parse failures | %s | %s |' % (so['parse_failures'], sn['parse_failures']))
    md.append('| empty lists | %s | %s |' % (so['empty_lists'], sn['empty_lists']))
    md.append('| class idx range | %s..%s | %s..%s |' % (fmt(so['idx_min']), fmt(so['idx_max']), fmt(sn['idx_min']), fmt(sn['idx_max'])))
    md.append('| mean classes / image | %.2f | %.2f |' % (so['mean_classes'], sn['mean_classes']))
    ro, rn = rows_out[0], rows_out[1]
    md.append('| mean GT recall (aligned) | %.3f | %.3f |' % (ro[1][0], rn[1][0]))
    md.append('| pseudo precision (aligned) | %.3f | %.3f |' % (ro[1][1], rn[1][1]))
    md.append('| recall @ image shift +1 | %.3f | %.3f |' % (ro[2][0], rn[2][0]))
    md.append('| recall @ image shift -1 | %.3f | %.3f |' % (ro[3][0], rn[3][0]))
    md.append('')
    md.append('**Order-consistency test**: a pseudo file whose line order matches the current enumeration')
    md.append('must LOSE most of its recall when the alignment is shifted by one image; a file whose recall')
    md.append('is invariant under shift has no per-image correspondence (order broken / different enumeration).')
    md.append('')
    # Threshold note: the shifted recall is the file's own chance level (common classes
    # like wall/sky cover part of any image), so "order-consistent" means a clear drop
    # from aligned to shifted, not a drop to zero. High-coverage files (many classes
    # per image) have a high chance level.
    def verdict(r):
        drop = r[1][0] - max(r[2][0], r[3][0])
        return 'ORDER-BROKEN (shift-invariant)' if drop < 0.05 else ('order-consistent (drop %.3f under shift)' % drop)
    md.append('- A (old): recall %.3f vs shifted %.3f/%.3f -> **%s**' % (ro[1][0], ro[2][0], ro[3][0], verdict(ro)))
    md.append('- B (regen): recall %.3f vs shifted %.3f/%.3f -> **%s**' % (rn[1][0], rn[2][0], rn[3][0], verdict(rn)))
    md.append('')

    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md[-8:]))
    print('written:', args.out_md)


if __name__ == '__main__':
    main()
