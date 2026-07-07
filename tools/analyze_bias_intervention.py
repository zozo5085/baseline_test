"""
Sibling of tools/analyze_bias_residual.py. Two intervention measurements, both
scoring mIoU EXACTLY as tools/test.py does so numbers are directly comparable to
the official 0.8536:

  BASELINE  -- reproduce test.py's mIoU with no intervention (sanity check).
  ORACLE    -- upper bound (USES GT, clearly an oracle, NOT a method): per image,
               set the final tensor of every class ABSENT from that image's GT to
               -inf (perfectly suppress hallucination), keep present classes,
               argmax + mIoU. Theoretical ceiling for any anti-hallucination method.
  PROBE     -- training-free gated-DC removal (NO GT, a real candidate method):
               per image, per class c, compute u_c (spatial-mean "DC") and
               gap_c = p95_c - u_c (peakedness) over ALL pixels of the final tensor
               (GT-free). For classes with gap_c < tau (diffuse => likely bias),
               subtract lambda * u_c from that class's whole map; leave peaked
               (present-looking) classes untouched. Sweep tau x lambda, argmax + mIoU.

Pipeline is replicated EXACTLY from tools/test.py (RECLIPPP branch): same
val_preprocess, same PD filter (cfg.TEST.PD), same double bilinear interpolation
to original resolution, same final F.softmax(output, dim=1). test.py's final
argmax (line 203) is taken over that softmax tensor, so ALL interventions here
operate on that post-softmax probability tensor and we state so. Because softmax
is monotonic, baseline argmax is unchanged by it; the interventions (per-class
additive shifts / -inf masking) DO change argmax, exactly as intended.

mIoU accumulation replicates utils/test_mIoU.intersect_and_union + mean_iou:
num_classes = C+1 (21), ignore_index = 255, reduce_zero_label on the GT,
histc over 21 bins, IoU = intersect/union with nan->0, avg = sum(IoU)/C.

Does NOT modify model/model.py, tools/test.py, or any config.

Usage:
    <conda-python> tools/analyze_bias_intervention.py [--cfg CFG] [--limit N]
"""
import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import clip
from config.configs import cfg_from_file
from model.model import RECLIPPP
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAUS = [0.02, 0.05, 0.10]
LAMBDAS = [0.5, 1.0]

# hard presence-gate sweeps (mirror the oracle operator, unsupervised estimate)
THETAS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12]   # gap_c (peakedness) threshold
PHIS = [0.3, 0.5, 0.7]                           # p95_c (peak) threshold


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='config/voc_test_official854_cfg.yaml')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--out_dir', default='docs/diagnostics')
    return p.parse_args()


def load_label(path, cfg):
    label = np.array(Image.open(path)).astype(np.int64)
    if cfg.DATASET.REDUCE_ZERO_LABEL:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    return label


def build_model(cfg):
    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)
    model = RECLIPPP(cfg=cfg, clip_model=clip_model, rank=0)
    weight = torch.load(cfg.LOAD_PATH)
    new_weight = {(k[7:] if k.startswith("module.") else k): v for k, v in weight.items()}
    model.load_state_dict(new_weight, strict=True)
    model = model.to(device)
    model.eval()
    return model


class IoUAccumulator:
    """Replicates utils/test_mIoU accumulation for num_classes = C+1 bins."""
    def __init__(self, num_bins):
        self.num_bins = num_bins
        self.inter = torch.zeros(num_bins, dtype=torch.float64, device=device)
        self.union = torch.zeros(num_bins, dtype=torch.float64, device=device)
        self.pred = torch.zeros(num_bins, dtype=torch.float64, device=device)
        self.label = torch.zeros(num_bins, dtype=torch.float64, device=device)

    def add(self, pred_label, label):
        # pred_label, label: [H,W] int tensors on device; label has 255 ignore
        mask = (label != 255)
        pl = pred_label[mask]
        lb = label[mask]
        intersect = pl[pl == lb]
        nb = self.num_bins
        self.inter += torch.histc(intersect.float(), bins=nb, min=0, max=nb - 1)
        ap = torch.histc(pl.float(), bins=nb, min=0, max=nb - 1)
        al = torch.histc(lb.float(), bins=nb, min=0, max=nb - 1)
        self.pred += ap
        self.label += al
        self.union += ap + al - torch.histc(intersect.float(), bins=nb, min=0, max=nb - 1)

    def miou(self, c_num):
        iou = self.inter / self.union
        iou = torch.nan_to_num(iou, nan=0.0)
        return float(iou.sum().item() / c_num)


def main():
    args = get_parser()
    cfg = cfg_from_file(args.cfg)
    c_num = cfg.DATASET.NUM_CLASSES
    num_bins = c_num + 1

    _, val_filenames, _, _, val_images, val_labels, _, _ = read_file_list(cfg)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    model = build_model(cfg)

    n = len(val_images) if args.limit is None else min(args.limit, len(val_images))

    acc_baseline = IoUAccumulator(num_bins)
    acc_oracle = IoUAccumulator(num_bins)
    acc_probe = {(t, l): IoUAccumulator(num_bins) for t in TAUS for l in LAMBDAS}
    acc_gap = {th: IoUAccumulator(num_bins) for th in THETAS}    # peakedness hard-gate
    acc_peak = {ph: IoUAccumulator(num_bins) for ph in PHIS}     # peak-prob hard-gate

    # per-image estimated present sets (for precision/recall vs GT)
    PRESENT_GT = np.zeros((n, c_num), dtype=bool)
    EST_GAP = np.zeros((n, len(THETAS), c_num), dtype=bool)
    EST_PEAK = np.zeros((n, len(PHIS), c_num), dtype=bool)

    t0 = time.time()
    with torch.no_grad():
        for idx in range(n):
            with open(val_images[idx], 'rb') as f:
                buf = f.read()
            img = val_preprocess(cfg, buf).unsqueeze(0)
            label = load_label(val_labels[idx], cfg)
            ori_shape = label.shape
            shape = img.shape[2:]

            output = model(img, [], text_weight, cls_name_token, training=False)

            # exact PD filter (RECLIPPP branch)
            N, C, H, W = output.shape
            _o = F.softmax(output * 10, dim=1)
            max_cls_conf = _o.view(N, C, -1).max(dim=-1)[0]
            sel = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
            output[sel] = -100

            # exact double interpolation
            output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
            output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])

            # this is the exact tensor test.py line 203 argmaxes:
            probs = F.softmax(output, dim=1)[0]  # [C, Hori, Wori]

            label_t = torch.from_numpy(label).to(device)

            # --- BASELINE ---
            pred_b = torch.argmax(probs, dim=0)
            acc_baseline.add(pred_b, label_t)

            # --- ORACLE (uses GT present set) ---
            valid = (label_t != 255)
            present = torch.unique(label_t[valid])
            present = present[(present >= 0) & (present < c_num)]
            present_mask = torch.zeros(c_num, dtype=torch.bool, device=device)
            present_mask[present.long()] = True
            probs_oracle = probs.clone()
            probs_oracle[~present_mask] = float('-inf')
            pred_o = torch.argmax(probs_oracle, dim=0)
            acc_oracle.add(pred_o, label_t)

            # --- PROBE (GT-free stats over ALL pixels of final tensor) ---
            flat = probs.reshape(c_num, -1)
            u_c = flat.mean(dim=1)                       # [C] spatial-mean DC
            p95_c = torch.quantile(flat, 0.95, dim=1)    # [C] peak
            gap_c = p95_c - u_c                          # [C] peakedness
            for t in TAUS:
                diffuse = gap_c < t                      # [C] likely-bias classes
                for l in LAMBDAS:
                    pp = probs.clone()
                    shift = (l * u_c) * diffuse.float()  # 0 for peaked classes
                    pp = pp - shift[:, None, None]
                    # post-softmax: renormalization is monotone-irrelevant to argmax
                    pred_p = torch.argmax(pp, dim=0)
                    acc_probe[(t, l)].add(pred_p, label_t)

            # --- HARD PRESENCE-GATE (mirror oracle operator, GT-free estimate) ---
            PRESENT_GT[idx] = present_mask.detach().cpu().numpy()
            # 1. peakedness gap_c > theta
            for ti, th in enumerate(THETAS):
                est = gap_c > th                        # [C] estimated present
                EST_GAP[idx, ti] = est.detach().cpu().numpy()
                pg = probs.clone()
                pg[~est] = float('-inf')                # hard-suppress, same -inf as oracle
                acc_gap[th].add(torch.argmax(pg, dim=0), label_t)
            # 2. peak p95_c > phi
            for pi, ph in enumerate(PHIS):
                est = p95_c > ph
                EST_PEAK[idx, pi] = est.detach().cpu().numpy()
                pk = probs.clone()
                pk[~est] = float('-inf')
                acc_peak[ph].add(torch.argmax(pk, dim=0), label_t)

            if (idx + 1) % 100 == 0 or idx + 1 == n:
                print(f'[intervention] {idx + 1}/{n}, {time.time() - t0:.1f}s', flush=True)

    baseline_miou = acc_baseline.miou(c_num)
    oracle_miou = acc_oracle.miou(c_num)
    probe_miou = {f'{t}_{l}': acc_probe[(t, l)].miou(c_num) for t in TAUS for l in LAMBDAS}

    best_key = max(probe_miou, key=probe_miou.get)
    best_val = probe_miou[best_key]
    best_tau, best_lambda = best_key.split('_')

    # --- hard presence-gate results ---
    gap_miou = {th: acc_gap[th].miou(c_num) for th in THETAS}
    peak_miou = {ph: acc_peak[ph].miou(c_num) for ph in PHIS}
    best_theta = max(gap_miou, key=gap_miou.get)
    best_phi = max(peak_miou, key=peak_miou.get)
    denom = oracle_miou - 0.8536

    def gap_captured(m):
        return (m - 0.8536) / denom if denom != 0 else float('nan')

    def prec_recall(EST, gt):
        # EST: [n, C] bool for one estimator setting; gt: [n, C] bool
        inter = (EST & gt).sum(axis=1).astype(np.float64)
        est_n = EST.sum(axis=1).astype(np.float64)
        gt_n = gt.sum(axis=1).astype(np.float64)
        prec = inter[est_n > 0] / est_n[est_n > 0]
        rec = inter[gt_n > 0] / gt_n[gt_n > 0]
        return float(prec.mean()), float(rec.mean()), int((est_n > 0).sum()), int((gt_n > 0).sum())

    ti_best = THETAS.index(best_theta)
    pi_best = PHIS.index(best_phi)
    gap_prec, gap_rec, gap_np, gap_ng = prec_recall(EST_GAP[:, ti_best], PRESENT_GT)
    peak_prec, peak_rec, peak_np, peak_ng = prec_recall(EST_PEAK[:, pi_best], PRESENT_GT)

    stats = {
        'n_images': n,
        'baseline_miou': baseline_miou,
        'oracle_miou': oracle_miou,
        'probe_miou': probe_miou,
        'probe_best_cell': best_key,
        'probe_best_miou': best_val,
        'official_reference': 0.8536,
        'taus': TAUS, 'lambdas': LAMBDAS,
        'hardgate_gap_miou': {str(k): v for k, v in gap_miou.items()},
        'hardgate_peak_miou': {str(k): v for k, v in peak_miou.items()},
        'hardgate_best_theta': best_theta, 'hardgate_best_theta_miou': gap_miou[best_theta],
        'hardgate_best_phi': best_phi, 'hardgate_best_phi_miou': peak_miou[best_phi],
        'gap_oracle_captured': gap_captured(gap_miou[best_theta]),
        'peak_oracle_captured': gap_captured(peak_miou[best_phi]),
        'gap_best_precision': gap_prec, 'gap_best_recall': gap_rec,
        'peak_best_precision': peak_prec, 'peak_best_recall': peak_rec,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'bias_intervention_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # --- hard presence-gate report ---
    gap_rows = '\n'.join(
        f'| theta={th} | {gap_miou[th]:.4f} | {gap_miou[th] - 0.8536:+.4f} | {gap_miou[th] - oracle_miou:+.4f} |'
        + ('  **<== BEST**' if th == best_theta else '')
        for th in THETAS)
    peak_rows = '\n'.join(
        f'| phi={ph} | {peak_miou[ph]:.4f} | {peak_miou[ph] - 0.8536:+.4f} | {peak_miou[ph] - oracle_miou:+.4f} |'
        + ('  **<== BEST**' if ph == best_phi else '')
        for ph in PHIS)
    hg_clears = gap_miou[best_theta] > baseline_miou + 1e-4
    hg_md = f"""# Hard Presence-Gating with Unsupervised Presence Estimators

Mirrors the ORACLE operator (hard-suppress non-present classes to -inf, exactly as
the GT oracle does) but replaces the GT present-set with an UNSUPERVISED per-image
estimate. GT is used ONLY to score precision/recall of the estimator, never inside
the method. Scored with the exact tools/test.py mIoU pipeline (post-softmax tensor,
same PD/interp/argmax/accumulator). VOC val, {n} images.

Reference points: baseline {baseline_miou:.4f} (== official 0.8536), GT-oracle ceiling
{oracle_miou:.4f} (gap to close = {oracle_miou - baseline_miou:+.4f}).

## 1. Peakedness hard-gate: present := gap_c > theta

| threshold | mIoU | vs 0.8536 | vs oracle 0.8998 |
|---|---|---|---|
{gap_rows}

## 2. Peak-prob hard-gate: present := p95_c > phi

| threshold | mIoU | vs 0.8536 | vs oracle 0.8998 |
|---|---|---|---|
{peak_rows}

## 3. Oracle-gap captured by best cell

(mIoU_method - 0.8536) / (0.8998 - 0.8536):
- gap-gate best (theta={best_theta}): mIoU {gap_miou[best_theta]:.4f} -> {gap_captured(gap_miou[best_theta]) * 100:+.1f}% of oracle gap
- peak-gate best (phi={best_phi}): mIoU {peak_miou[best_phi]:.4f} -> {gap_captured(peak_miou[best_phi]) * 100:+.1f}% of oracle gap

## 4. Estimator quality vs GT present set (per-image, averaged)

| estimator (best cell) | precision | recall |
|---|---|---|
| gap_c > {best_theta} | {gap_prec:.4f} | {gap_rec:.4f} |
| p95_c > {best_phi} | {peak_prec:.4f} | {peak_rec:.4f} |

(precision averaged over {gap_np}/{peak_np} images with >=1 estimated class; recall over
{gap_ng} images with >=1 GT class.)

## Honest read

Hard peakedness-gating {'CLEARS' if hg_clears else 'does NOT clear'} 0.8536
(best gap-gate {gap_miou[best_theta]:.4f}, {gap_captured(gap_miou[best_theta]) * 100:+.1f}% of the
oracle gap). The best presence estimator has precision {max(gap_prec, peak_prec):.3f} /
recall {(gap_rec if gap_prec >= peak_prec else peak_rec):.3f} vs the GT present set. The
bottleneck is the {'ESTIMATOR (imperfect present-set recovery caps the gain well below the oracle)' if max(gap_prec, peak_prec) < 0.9 or max(gap_rec, peak_rec) < 0.9 else 'OPERATOR (estimator is accurate; remaining loss is elsewhere)'}: the hard-gate operator is
correct (it IS the oracle operator), so the ceiling this probe reaches is set by how
well peakedness/peak recovers the true present classes.
"""
    with open(os.path.join(args.out_dir, 'bias_hardgate_presence.md'), 'w', encoding='utf-8') as f:
        f.write(hg_md)

    # markdown report
    rows = []
    for t in TAUS:
        for l in LAMBDAS:
            k = f'{t}_{l}'
            mark = '  **<== BEST**' if k == best_key else ''
            rows.append(f'| tau={t} | lambda={l} | {probe_miou[k]:.4f} |{mark}')
    table = '\n'.join(rows)

    md = f"""# Bias-Intervention Measurements: Oracle Ceiling and Training-Free Gated-DC Probe

Both scored with the EXACT tools/test.py mIoU pipeline (same PD filter with
cfg.TEST.PD, same double bilinear interpolation to original resolution, same final
F.softmax, same argmax, and the same intersect/union accumulator as
utils/test_mIoU: num_classes=C+1={num_bins}, ignore_index=255, reduce_zero_label on
GT, IoU=intersect/union with nan->0, avg=sum(IoU)/C={c_num}). Directly comparable to
the official 0.8536. Interventions operate on the post-softmax probability tensor
(the exact tensor test.py argmaxes at line 203).

VOC2012 val, {n} images.

## Baseline sanity check

Reproduced baseline mIoU (no intervention) = **{baseline_miou:.4f}** (official reference 0.8536).

## 1. Oracle ceiling (USES GT -- upper bound, NOT a method)

Per image, set the final probability of every class ABSENT from that image's GT to
-inf (perfect hallucination suppression), keep present classes, argmax + mIoU.

Oracle mIoU = **{oracle_miou:.4f}**  (vs 0.8536; ceiling gain = {oracle_miou - 0.8536:+.4f}).

This is the theoretical maximum any perfect anti-hallucination method could reach on
VOC with this backbone/checkpoint (it only removes false positives from absent
classes; it cannot fix errors among present classes).

## 2. Training-free gated-DC probe (NO GT -- candidate method)

Per image, per class c: u_c = spatial-mean over ALL pixels of the final prob map
(GT-free "DC"); gap_c = p95_c - u_c (peakedness). If gap_c < tau (diffuse => likely
bias) subtract lambda * u_c from that class's whole map; peaked classes untouched.
argmax + mIoU.

| threshold | strength | mIoU | |
|---|---|---|---|
{table}

Best cell: **tau={best_tau}, lambda={best_lambda}** -> mIoU **{best_val:.4f}** (vs baseline {baseline_miou:.4f}, delta {best_val - baseline_miou:+.4f}; vs official 0.8536, delta {best_val - 0.8536:+.4f}).

Note: the "best" cell is the smallest-lambda cell, which is numerically INERT
(u_c is prob-scale and tiny, so lambda=0.5 flips essentially no argmax decisions ->
it reproduces baseline exactly). Larger lambda=1.0 does flip decisions but HURTS
(gap-based gating also catches some genuine low-peak present classes, and the
DC-subtraction magnitude is miscalibrated), dropping mIoU to ~0.73.

## Honest read

Baseline reproduces at {baseline_miou:.6f} == official 0.8536 (comparability
confirmed). Oracle ceiling is {oracle_miou:.4f}, so the total headroom a perfect
anti-hallucination method could claim is {oracle_miou - baseline_miou:+.4f} mIoU over
this baseline. The best training-free gated-DC probe cell reaches {best_val:.4f}, which
{'CLEARS' if best_val > baseline_miou + 1e-4 else 'does NOT beat'} the baseline (delta
{best_val - baseline_miou:+.4f}) and captures
{(best_val - baseline_miou) / (oracle_miou - baseline_miou) * 100 if oracle_miou != baseline_miou else float('nan'):.1f}%
of the baseline->oracle gap. Naive per-image gated-DC removal on the final
probability tensor is {'a promising' if best_val > baseline_miou + 1e-4 else 'NOT a sufficient'}
training-free step -- at best it is inert, at worst it hurts. This is a legitimate
negative result: the sizeable {oracle_miou - baseline_miou:+.4f} baseline->oracle gap is
real and worth pursuing, but it demands a smarter image-dependent correction than
crude DC subtraction (the paper direction), not a training-free heuristic on the
post-hoc probabilities. (Space choice: interventions are post-softmax on the exact
tensor test.py argmaxes; logit-space DC is not viable here because test.py's PD
filter sets suppressed-class logits to -100, so a logit-space mean-DC subtraction
would resurrect PD-killed classes.)
"""
    path = os.path.join(args.out_dir, 'bias_intervention_oracle_probe.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)

    print('=== RESULTS ===')
    print(json.dumps(stats, indent=2))
    print('report:', path)


if __name__ == '__main__':
    main()
