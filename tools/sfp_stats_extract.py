"""Extract SFP per-image stats (flagged fractions) over a val set.

model/model_sfp_dtlr.py collects a per-forward stats dict in
`model.sfp_last_stats_batch` (reset at every sfp_logit_purify call) but nothing
in the repo dumps it. This runs the frozen model (no-TTA, single scale, no PD /
no mIoU -- stats are collected inside the forward before any of that) over the
val list and aggregates the mean of every numeric stat key. The keys that feed
the journal flagged-fraction table:

  ratio           : selected outlier fraction actually rewritten (after TOPK / TOP_FRACTION cap)
  unrel_frac_conf : fraction flagged unreliable by the max-prob gate  (conf < CONF_THD)
  unrel_frac_ent  : fraction flagged unreliable by the entropy gate   (H_norm > tau_unrel)
  rel_frac_conf   : proxy-source fraction under the max-prob gate     (conf > PROXY_CONF_THD)
  rel_frac_ent    : proxy-source fraction under the entropy gate      (H_norm < tau_rel)

Both gate variants are computed on every forward regardless of ENTROPY_GATE, so
one run per dataset yields both columns.

Usage:
  python tools/sfp_stats_extract.py --cfg config/voc_test_sfp_dtlr_gen_official_cfg.yaml \
      --out experiments/sfp_stats_extract/voc_gen.json
  python tools/sfp_stats_extract.py --cfg config/context_test_sfp_dtlr_gen_cfg.yaml \
      --load_path experiments/context_vanilla_converged/best_weight.pth \
      --out experiments/sfp_stats_extract/context_gen.json
"""
import argparse
import json
import os
import sys
import importlib

import torch
import clip
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_module', default='model.model_sfp_dtlr')
    ap.add_argument('--load_path', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = cfg_from_file(args.cfg)
    if args.load_path:
        cfg.LOAD_PATH = args.load_path

    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    module = importlib.import_module(args.model_module)
    model_cls = getattr(module, 'RECLIPPP')

    _, val_filenames, _, _, val_images, val_labels, _, _ = read_file_list(cfg)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    try:
        model = model_cls(cfg=cfg, clip_model=clip_model, rank=0, zeroshot_weights=text_weight)
    except TypeError:
        model = model_cls(cfg=cfg, clip_model=clip_model, rank=0)

    weight = torch.load(cfg.LOAD_PATH)
    new_weight = {(k[7:] if k.startswith("module.") else k): v for k, v in weight.items()}
    missing, unexpected = model.load_state_dict(new_weight, strict=False)
    print("[Load] missing keys:", len(missing), " unexpected keys:", len(unexpected))
    model = model.to(device)
    model.eval()

    if args.limit and args.limit > 0:
        val_images = val_images[:args.limit]
        val_labels = val_labels[:args.limit]

    sums = {}
    n_stats = 0
    n_empty = 0
    with torch.no_grad():
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)
            model(img, [], text_weight, cls_name_token, training=False)
            batch = getattr(model, 'sfp_last_stats_batch', [])
            if not batch:
                n_empty += 1
            for stat in batch:
                n_stats += 1
                for k, v in stat.items():
                    if isinstance(v, (int, float)):
                        sums[k] = sums.get(k, 0.0) + float(v)
            if (idx + 1) % 200 == 0 or idx + 1 == len(val_images):
                print('progress: {}/{}'.format(idx + 1, len(val_images)))

    means = {k: v / max(n_stats, 1) for k, v in sums.items()}
    result = {
        'cfg': args.cfg,
        'load_path': cfg.LOAD_PATH,
        'images': len(val_images),
        'stats_entries': n_stats,
        'empty_forwards': n_empty,
        'means': means,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print('\n===== SFP STATS SUMMARY =====')
    print('images=%d stats_entries=%d empty=%d' % (len(val_images), n_stats, n_empty))
    for k in ('ratio', 'unrel_frac_conf', 'unrel_frac_ent',
              'rel_frac_conf', 'rel_frac_ent', 'proxy_available_ratio', 'h_norm_mean'):
        if k in means:
            print('%-22s = %.4f' % (k, means[k]))
    print('written: %s' % args.out)


if __name__ == '__main__':
    main()
