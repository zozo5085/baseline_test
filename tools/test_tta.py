"""Test-time augmentation (multi-scale + horizontal flip) eval wrapper.

Inference-only. Does NOT touch CLIP weights or features — it only runs the
frozen model on rescaled / flipped inputs and averages the per-view softmax
probabilities at the ORIGINAL image resolution before argmax. Everything else
(TEST.PD hard class-prune, per-image .pt saving, mIoU) is identical to
tools/test.py so the number is directly comparable to the 0.8536 baseline.

Examples:
    # identity check (single scale, no flip) -> must reproduce the plain test.py number
    python tools/test_tta.py --cfg config/voc_test_official854_cfg.yaml \
        --model RECLIPPP --model_module model.model --scales 1.0

    # flip-only TTA
    python tools/test_tta.py --cfg config/voc_test_official854_cfg.yaml \
        --model RECLIPPP --model_module model.model --scales 1.0 --flip

    # multi-scale + flip
    python tools/test_tta.py --cfg config/voc_test_official854_cfg.yaml \
        --model RECLIPPP --model_module model.model --scales 1.0,1.25,1.5 --flip

    # quick smoke on first 20 images
    python tools/test_tta.py --cfg ... --scales 1.0,1.25 --flip --limit 20
"""
import argparse
import torch
import clip
import torch.nn.functional as F
import numpy as np
from PIL import Image
import importlib
from pathlib import Path
import sys
import os

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from config.configs import cfg_from_file
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_preprocess = clip.load("ViT-B/16")
clip_model = clip_model.to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='config/voc_test_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name', default='RECLIPPP', type=str)
    parser.add_argument('--model_module', default='model.model')
    parser.add_argument('--fusion_mode', default=None,
                        choices=['l12_only', 'l9_l12', 'l6_l12', 'l6_l9_l12',
                                 'safe_l6_l9_l12', 'trainable_fusion', 'dff2d'])
    parser.add_argument('--class_gate', action='store_true')
    parser.add_argument('--scales', default='1.0', type=str,
                        help='comma-separated scale factors relative to DATASET.SCALE short side, e.g. 1.0,1.25,1.5')
    parser.add_argument('--flip', action='store_true', help='add horizontal-flip views')
    parser.add_argument('--limit', type=int, default=0, help='if >0, only evaluate the first N images (smoke)')
    parser.add_argument('--save_dir', default=None, help='override cfg.SAVE_DIR (fresh dir per run)')
    parser.add_argument('--load_path', default=None, help='override cfg.LOAD_PATH (e.g. a run1 best_weight)')
    parser.add_argument('--nonstrict', action='store_true',
                        help='force strict=False load (for ckpts that omit the frozen CLIP backbone)')
    parser.add_argument('--sfp_disable', default='',
                        help='comma list of SFP components to disable for ablation: dtlr,proxy,cpsfp')
    return parser.parse_args()


def load_model_classes(module_name):
    module = importlib.import_module(module_name)
    return getattr(module, 'RECLIPPP'), getattr(module, 'ReCLIP', None)


def build_model(model_cls, cfg, clip_model, rank, text_weight):
    try:
        return model_cls(cfg=cfg, clip_model=clip_model, rank=rank, zeroshot_weights=text_weight)
    except TypeError:
        return model_cls(cfg=cfg, clip_model=clip_model, rank=rank)


def apply_fusion_args(cfg, args):
    if args.fusion_mode is not None:
        cfg.MODEL.FEATURE_FUSION.ENABLE = args.fusion_mode != 'l12_only'
        cfg.MODEL.FEATURE_FUSION.MODE = args.fusion_mode
    if args.class_gate:
        cfg.MODEL.CLASS_GATE.ENABLE = True


def infer_one_view(model, img, cfg, args, c_num, text_weight, cls_name_token, ori_shape):
    """Run the frozen model on one view; return softmax prob at ori_shape [1,C,H0,W0]."""
    shape = img.shape[2:]
    output = model(img, [], text_weight, cls_name_token, training=False)
    N, C, H, W = output.shape
    # TEST.PD hard class-prune — identical to tools/test.py:186-197
    if args.model_name == 'RECLIPPP':
        _output = F.softmax(output * 10, dim=1)
        max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
        selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
        output[selected_cls] = -100
    else:
        _output = F.softmax(output * 100, dim=1)
        max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
        selected_cls = (max_cls_conf < cfg.TEST.ReCLIP_PD)[:, :, None, None].expand(N, C, H, W)
        output[selected_cls] = -100
    output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
    output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])
    return F.softmax(output, dim=1)


def test():
    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)
    apply_fusion_args(cfg, args)
    if args.save_dir:
        cfg.SAVE_DIR = args.save_dir
    if args.load_path:
        cfg.LOAD_PATH = args.load_path
    if getattr(args, 'sfp_disable', ''):
        _ablate = {'dtlr': 'DTLR_ENABLE', 'proxy': 'PROXY_ENABLE', 'cpsfp': 'CPSFP_UPDATE'}
        for tok in args.sfp_disable.split(','):
            tok = tok.strip().lower()
            if tok:
                setattr(cfg.MODEL.SFP_DTLR, _ablate[tok], False)
                print(f'[ablate] {_ablate[tok]} = False')
    Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    RECLIPPP, ReCLIP = load_model_classes(args.model_module)

    _, val_filenames, _, _, val_images, val_labels, results_iou, _ = read_file_list(cfg)
    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    model = build_model(RECLIPPP if args.model_name == 'RECLIPPP' else ReCLIP, cfg, clip_model, 0, text_weight)

    weight = torch.load(cfg.LOAD_PATH)
    new_weight = {(k[7:] if k.startswith("module.") else k): v for k, v in weight.items()}
    strict_load = (args.model_module == "model.model") and not args.nonstrict
    missing, unexpected = model.load_state_dict(new_weight, strict=strict_load)
    if not strict_load:
        print("[Load] missing keys:", len(missing), " unexpected keys:", len(unexpected))
    model = model.to(device)

    if args.limit and args.limit > 0:
        n = args.limit
        val_filenames, val_images, val_labels, results_iou = \
            val_filenames[:n], val_images[:n], val_labels[:n], results_iou[:n]

    scales = [float(s) for s in args.scales.split(',') if s.strip()]
    flip_opts = [False, True] if args.flip else [False]
    orig_scale = list(cfg.DATASET.SCALE)
    long_cap, base_short = max(orig_scale), min(orig_scale)
    print(f"[TTA] scales={scales} flip={args.flip} views/img={len(scales) * len(flip_opts)} "
          f"images={len(val_images)} base_short={base_short}")

    c_num = cfg.DATASET.NUM_CLASSES
    model.eval()
    with torch.no_grad():
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            label = Image.open(val_labels[idx])
            ori_shape = tuple((label.size[1], label.size[0]))

            probs = None
            n_views = 0
            for s in scales:
                cfg.DATASET.SCALE = [long_cap, max(1, int(round(base_short * s)))]
                img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)
                for do_flip in flip_opts:
                    view = torch.flip(img, dims=[3]) if do_flip else img
                    prob = infer_one_view(model, view, cfg, args, c_num, text_weight, cls_name_token, ori_shape)
                    if do_flip:
                        prob = torch.flip(prob, dims=[3])
                    probs = prob if probs is None else probs + prob
                    n_views += 1
            cfg.DATASET.SCALE = orig_scale

            probs = probs / n_views
            pred = torch.argmax(probs, dim=1).squeeze(dim=0)
            torch.save(pred, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
            if (idx + 1) % 100 == 0 or idx + 1 == len(val_images):
                print('test progress: {}/{}'.format(idx + 1, len(val_images)))

        iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255,
                       nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
        print(iou['IoU'])
        avg = iou['IoU'].sum() / c_num
        print('avg:%.4f' % (avg))
        print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (len(val_images), len(val_images), avg))


if __name__ == '__main__':
    test()
