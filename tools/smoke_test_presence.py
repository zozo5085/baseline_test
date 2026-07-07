"""Quick smoke test for Method A (presence-calibration head).

Builds the presence model from a config, prints trainable params + PRESENCE.MODE,
runs a few train iterations, asserts loss finite and gradients reach the trainable
params. Also prints the training-set size (len(train_images)) for ETA estimation.

Usage:
    python tools/smoke_test_presence.py --cfg config/voc_train_presence_cfg.yaml --iters 3
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
from utils.preprocess import preprocess, read_file_list, prepare_dataset_cls_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--model_module', default='model.model_presence')
    parser.add_argument('--iters', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = cfg_from_file(args.cfg)
    print(f"presence mode: {cfg.MODEL.PRESENCE.MODE}, init_from: {cfg.MODEL.PRESENCE.INIT_FROM}")

    clip_model, _ = clip.load("ViT-B/16", device=device)
    module = importlib.import_module(args.model_module)
    model_cls = getattr(module, 'RECLIPPP')
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    try:
        model = model_cls(cfg=cfg, clip_model=clip_model, rank=device, zeroshot_weights=text_weight).to(device)
    except TypeError:
        model = model_cls(cfg=cfg, clip_model=clip_model, rank=device).to(device)
    model.train()

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(c for _, c in trainable)
    print(f"trainable params: {n_trainable:,}")
    for n, c in trainable:
        print(f"  {n}: {c:,}")

    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    _, _, train_images, train_labels, _, _, _, pseudo_classes = read_file_list(cfg)
    print(f"train_images: {len(train_images)}")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)

    done = 0
    t0 = None
    for idx in range(len(train_images)):
        gt = [int(t) if isinstance(t, int) else int(t.item()) for t in pseudo_classes[idx]]
        if len(gt) == 0:
            continue
        with open(train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        img = img.unsqueeze(0)

        if done == 1:
            t0 = time.time()  # start timing after the first (warm-up) iter

        output, loss = model(img.to(device), [gt], text_weight, cls_name_token, training=True, img_metas=[img_metas])
        assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
        optimizer.zero_grad()
        loss.backward()
        grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
                      if p.requires_grad and p.grad is not None}
        n_with_grad = len(grad_norms)
        total_grad = sum(grad_norms.values())
        optimizer.step()
        print(f"iter {done}: img={os.path.basename(train_images[idx])} loss={loss.item():.5f} "
              f"params_with_grad={n_with_grad} total_grad_norm={total_grad:.5f}")
        assert total_grad > 0, "no gradient flowed to trainable params"
        done += 1
        if done >= args.iters:
            break

    if t0 is not None and done > 1:
        dt = (time.time() - t0) / (done - 1)
        print(f"approx {dt:.3f} s/iter (fwd+bwd, no data-loader sleep)")

    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
