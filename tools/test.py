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
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_test_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='RECLIPPP', type=str)
    parser.add_argument('--model_module', default='model.model',
                        help='Python module that provides RECLIPPP/ReCLIP classes.')
    parser.add_argument('--fusion_mode', default=None,
                        choices=['l12_only', 'l9_l12', 'l6_l12', 'l6_l9_l12',
                                 'safe_l6_l9_l12', 'trainable_fusion', 'dff2d'])
    parser.add_argument('--fusion_gamma9', type=float, default=None)
    parser.add_argument('--fusion_gamma6', type=float, default=None)
    parser.add_argument('--fusion_gate_temp', type=float, default=None)
    parser.add_argument('--class_gate', action='store_true')
    parser.add_argument('--debug_fusion', action='store_true')
    parser.add_argument('--dump_layer_maps', action='store_true')
    parser.add_argument('--dump_gate_maps', action='store_true')
    parser.add_argument('--debug_out_dir', default='outputs/debug_layers')
    args = parser.parse_args()
    return args


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
    if args.fusion_gamma9 is not None:
        cfg.MODEL.FEATURE_FUSION.GAMMA9 = float(args.fusion_gamma9)
    if args.fusion_gamma6 is not None:
        cfg.MODEL.FEATURE_FUSION.GAMMA6 = float(args.fusion_gamma6)
    if args.fusion_gate_temp is not None:
        cfg.MODEL.FEATURE_FUSION.GATE_TEMP = float(args.fusion_gate_temp)
    if args.class_gate:
        cfg.MODEL.CLASS_GATE.ENABLE = True


def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-6)


def heat_color(arr):
    x = normalize01(arr)
    r = np.clip(1.5 * x - 0.25, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0, 1)
    b = np.clip(1.25 - 1.5 * x, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def save_debug_map(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heat_color(arr)).save(path)


def dump_debug_maps(debug, image_id, out_dir):
    out_dir = Path(out_dir)
    h, w = debug.get("shape", (None, None))
    for layer, feat in sorted(debug.get("layer_maps", {}).items()):
        norm = torch.linalg.vector_norm(feat[0].float(), dim=0).detach().cpu().numpy()
        save_debug_map(norm, out_dir / f"{image_id}_l{layer}_norm.png")
        attn = debug.get("attn", {}).get(layer)
        if attn is not None and h is not None:
            a = attn[0, 0, 1:1 + h * w].reshape(h, w).detach().cpu().numpy()
            save_debug_map(a, out_dir / f"{image_id}_l{layer}_attn.png")
    if "gate9" in debug:
        save_debug_map(debug["gate9"][0, 0].detach().cpu().numpy(), out_dir / f"{image_id}_gate9.png")
    if "gate6" in debug:
        save_debug_map(debug["gate6"][0, 0].detach().cpu().numpy(), out_dir / f"{image_id}_gate6.png")
    if "fused_v" in debug:
        norm = torch.linalg.vector_norm(debug["fused_v"][0].float(), dim=0).detach().cpu().numpy()
        save_debug_map(norm, out_dir / f"{image_id}_fused_norm.png")


def test():
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    apply_fusion_args(cfg, args)
    Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    RECLIPPP, ReCLIP = load_model_classes(args.model_module)

    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    if args.model_name == 'RECLIPPP':
        model = build_model(RECLIPPP, cfg, clip_model, 0, text_weight)
    else:
        model = build_model(ReCLIP, cfg, clip_model, 0, text_weight)
    weight = torch.load(cfg.LOAD_PATH)
    new_weight = {}
    for key, value in weight.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_weight[new_key] = value

    strict_load = args.model_module == "model.model"
    missing, unexpected = model.load_state_dict(new_weight, strict=strict_load)
    if not strict_load:
        print("[Load] missing keys:", len(missing))
        print("[Load] unexpected keys:", len(unexpected))
    model = model.to(device)

    c_num = cfg.DATASET.NUM_CLASSES
    model.eval()
    with torch.no_grad():
        idx = 0
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)

            label = Image.open(val_labels[idx])
            ori_shape = tuple((label.size[1], label.size[0]))
            label = np.asarray(label).copy()
            label[label == 0] = 255

            gt_cls = []
            shape = img.shape[2:]
            if args.debug_fusion or args.dump_layer_maps or args.dump_gate_maps:
                output, debug = model(
                    img,
                    gt_cls,
                    text_weight,
                    cls_name_token,
                    training=False,
                    return_debug=True,
                    debug_fusion=args.debug_fusion,
                )
                if args.debug_fusion:
                    print(
                        "[Fusion]",
                        val_filenames[idx],
                        "mode=", debug.get("fusion_mode"),
                        "cos_fused_f12=", debug.get("cos_fused_f12"),
                        "cos_f9_f12=", debug.get("cos_f9_f12"),
                        "cos_f6_f12=", debug.get("cos_f6_f12"),
                        "gate9_mean=", float(debug["gate9"].mean().cpu()) if "gate9" in debug else "",
                        "gate6_mean=", float(debug["gate6"].mean().cpu()) if "gate6" in debug else "",
                        "class_gate_mean=", float(debug["class_gate"].mean().cpu()) if debug.get("class_gate") is not None else "",
                    )
                if args.dump_layer_maps or args.dump_gate_maps:
                    dump_debug_maps(debug, val_filenames[idx], args.debug_out_dir)
            else:
                output = model(img, gt_cls, text_weight, cls_name_token, training=False)

            # pd
            N, C, H, W = output.shape
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

            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1).squeeze(dim=0)
            torch.save(output, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
            if (idx + 1) % 100 == 0 or idx + 1 == len(val_images):
                print('test progress: {}/{}'.format(idx + 1, len(val_images)))

        iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255, nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
        print(iou['IoU'])
        avg = iou['IoU'].sum() / c_num
        print('avg:%.4f' % (avg))
        print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (idx, len(val_images), avg))


if __name__ == '__main__':
    test()
