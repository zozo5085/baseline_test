"""Table V runtime benchmark (journal Framing A).

Times the model forward (batch 1, eval mode, CUDA-synchronized) over N val images,
reporting mean ms/img, FPS, and peak VRAM. Scope: val_preprocess'd image -> model()
-> softmax (+ optional flip second view, averaged); excludes disk I/O and the final
resize-to-label interpolation, which are identical across methods.

Usage:
  python tools/bench_runtime.py --cfg config/... --model_module model.model \
    [--load_path W] [--flip] [--n 50] --label "VOC base"
"""
import argparse
import importlib
import os
import sys
import time

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import torch
import clip

from config.configs import cfg_from_file
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_module', default='model.model')
    ap.add_argument('--load_path', default=None)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    device = torch.device('cuda')
    cfg = cfg_from_file(args.cfg)
    if args.load_path:
        cfg.LOAD_PATH = args.load_path

    clip_model, _ = clip.load("ViT-B/16", device=device)
    module = importlib.import_module(args.model_module)
    model = getattr(module, 'RECLIPPP')(cfg=cfg, clip_model=clip_model, rank=device).to(device)
    weight = torch.load(cfg.LOAD_PATH, map_location='cpu')
    weight = {(k[7:] if k.startswith('module.') else k): v for k, v in weight.items()}
    model.load_state_dict(weight, strict=False)
    model.eval()

    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    _, _, _, _, val_images, _, _, _ = read_file_list(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    imgs = val_images[:args.warmup + args.n]
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i, ip in enumerate(imgs):
            with open(ip, 'rb') as f:
                buf = f.read()
            img = val_preprocess(cfg, buf).unsqueeze(0).to(device)
            views = [(img, False), (torch.flip(img, dims=[3]), True)] if args.flip else [(img, False)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob = None
            for v, flipped in views:
                out = model(v, [], text_weight, cls_name_token, training=False)
                out = torch.softmax(out, dim=1)
                if flipped:
                    out = torch.flip(out, dims=[3])
                prob = out if prob is None else prob + out
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(dt)
    ms = 1000.0 * sum(times) / len(times)
    vram = torch.cuda.max_memory_allocated() / 2**20
    print(f"[{args.label}] ms/img={ms:.1f}  FPS={1000.0/ms:.2f}  peakVRAM={vram:.0f}MiB  "
          f"total_params={total_params/1e6:.1f}M  n={len(times)}")


if __name__ == '__main__':
    main()
