"""Smoke test for LGAK-MVP (text-gated conv refiner).

Builds model.model_lgak from a train config, asserts ONLY lgak.* params are trainable
(grad only on LGAK), runs a few train iterations, and logs per iter: loss (finite),
which params got grad, alpha value + grad, gate g mean/std, and feature norm before/after
LGAK. At alpha=0 only alpha is expected to receive gradient (the documented F4 slow start);
the DW/PW conv + MLP receive ~0 until alpha leaves 0.

Usage:
    python tools/smoke_test_lgak.py --cfg config/voc_train_lgak_mvp_cfg.yaml --iters 2
"""
import argparse
import importlib
import os
import sys

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
    parser.add_argument('--model_module', default='model.model_lgak')
    parser.add_argument('--iters', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = cfg_from_file(args.cfg)
    print(f"LGAK init_from: {cfg.MODEL.LGAK.INIT_FROM}  kernel: {cfg.MODEL.LGAK.KERNEL}  "
          f"hidden: {cfg.MODEL.LGAK.HIDDEN}  alpha_trainable: {cfg.MODEL.LGAK.ALPHA_TRAINABLE}")

    clip_model, _ = clip.load("ViT-B/16", device=device)
    module = importlib.import_module(args.model_module)
    model = getattr(module, 'RECLIPPP')(cfg=cfg, clip_model=clip_model, rank=device).to(device)
    model.train()
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    # grad-only-on-LGAK: assert nothing outside lgak.* is trainable.
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    non_lgak = [n for n, _ in trainable if not n.startswith('lgak.')]
    assert not non_lgak, f"non-LGAK params are trainable (baseline not frozen): {non_lgak}"
    print(f"trainable params ({sum(c for _, c in trainable):,}), all under lgak.:")
    for n, c in trainable:
        print(f"  {n}: {c:,}")

    # forward hook: feature norms before/after LGAK + gate stats.
    cap = {}

    def hook(mod, inp, out):
        feat, text = inp[0], inp[1]
        g = 1.0 + mod.gate_mlp(text.mean(dim=0))
        cap['in_norm'] = feat.norm(dim=1).mean().item()
        cap['out_norm'] = out.norm(dim=1).mean().item()
        cap['delta_norm'] = (out - feat).norm(dim=1).mean().item()
        cap['g_mean'] = g.mean().item()
        cap['g_std'] = g.std().item()

    model.lgak.register_forward_hook(hook)

    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    _, _, train_images, train_labels, _, _, _, pseudo_classes = read_file_list(cfg)
    print(f"train_images: {len(train_images)}")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)

    done = 0
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

        alpha_before = float(model.lgak.alpha)
        output, loss = model(img.to(device), [gt], text_weight, cls_name_token, training=True, img_metas=[img_metas])
        assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
        optimizer.zero_grad()
        loss.backward()
        grad_params = {n: p.grad.norm().item() for n, p in model.named_parameters()
                       if p.requires_grad and p.grad is not None}
        leak = [n for n in grad_params if not n.startswith('lgak.')]
        assert not leak, f"gradient leaked to non-LGAK params: {leak}"
        assert grad_params, "no gradient reached any LGAK param"
        alpha_grad = model.lgak.alpha.grad.norm().item() if model.lgak.alpha.grad is not None else None
        conv_mlp_grad = sum(v for n, v in grad_params.items() if ('conv' in n or 'mlp' in n))
        optimizer.step()
        print(f"iter {done}: loss={loss.item():.5f}  feat_norm in={cap['in_norm']:.4f} out={cap['out_norm']:.4f} "
              f"delta={cap['delta_norm']:.6e}  g_mean={cap['g_mean']:.4f} g_std={cap['g_std']:.4f}  "
              f"alpha {alpha_before:.6e}->{float(model.lgak.alpha):.6e} (grad={alpha_grad})  "
              f"conv+mlp_grad={conv_mlp_grad:.3e}  n_grad_params={len(grad_params)}")
        done += 1
        if done >= args.iters:
            break

    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
