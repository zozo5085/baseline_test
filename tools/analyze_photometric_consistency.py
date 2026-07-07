"""
Method-C premise diagnostic (test-time only, NO training, NO GT in the method):
do HALLUCINATED (GT-absent) classes flicker MORE under mild photometric perturbation
than truly-PRESENT classes? If yes, cross-view instability is a usable presence
signal that COMPLEMENTS peak-height (p95>0.3, which scored recall 0.928 / prec 0.692
earlier). If present and absent classes are equally (in)consistent, the premise dies.

Views: K=4 mild photometric jitter views + the original (5 total). Jitter is applied
to the RAW RGB (0-255) AFTER cv2 resize+BGR2RGB but BEFORE the model's IMG_NORM
mean/std subtraction, so it mimics real photometric variation. NO geometric transform
(pixel correspondence preserved). View 0 = identity: its preprocessing tensor is
verified equal to utils.preprocess.val_preprocess (so its prediction reproduces the
baseline).

Per view: frozen model -> exact tools/test.py pipeline (PD filter, double bilinear
interp to original resolution, final softmax). Two per-class image-level presence
scores:
  p95_c  = 95th-percentile of the per-class probability over space (the "peak" that
           scored recall 0.928).
  g_c    = cos(z_global, text_c): global CLIP image-token vs class text embedding
           (z_global captured from model.vit via a forward hook -- no model.py edit).

Cross-view stats per (image,class): m_c=mean_k, s_c=std_k, cov_c=s_c/(m_c+eps).
GT present/absent used ONLY to score the signal, never inside any rule.

Does NOT modify model/model.py, tools/test.py, or config.
Usage: <conda-python> tools/analyze_photometric_consistency.py [--cfg CFG] [--limit N]
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
import cv2
from cv2 import IMREAD_COLOR
from torchvision.transforms import functional as TF

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

import clip
from config.configs import cfg_from_file
from model.model import RECLIPPP
from utils.preprocess import read_file_list, prepare_dataset_cls_tokens, val_preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6
K_JITTER = 4
PEAK_THR = 0.3
TAUS = [0.1, 0.2, 0.3, 0.5]


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='config/voc_test_official854_cfg.yaml')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--out_dir', default='docs/diagnostics')
    p.add_argument('--seed', type=int, default=1234)
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


def preprocess_jitter(cfg, buf, jitter):
    """Replicates utils.preprocess.val_preprocess EXACTLY, but injects a photometric
    jitter on the RGB 0-255 float image immediately before mean/std normalization.
    jitter = (brightness, contrast, saturation, hue); (1,1,1,0) is identity."""
    img_scale = cfg.DATASET.SCALE
    img_np = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(img_np, IMREAD_COLOR)  # BGR
    h, w = img.shape[:2]
    max_long_edge = max(img_scale)
    max_short_edge = min(img_scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
    resized_img = cv2.resize(img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)

    img = resized_img.copy().astype(np.float32)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # now RGB 0-255

    b, c, s, hue = jitter
    if not (b == 1.0 and c == 1.0 and s == 1.0 and hue == 0.0):
        t = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # [3,H,W] in [0,1] RGB
        t = TF.adjust_brightness(t, b)
        t = TF.adjust_contrast(t, c)
        t = TF.adjust_saturation(t, s)
        t = TF.adjust_hue(t, hue)
        img = (t.clamp(0, 1) * 255.0).permute(1, 2, 0).numpy().astype(np.float32)

    mean = np.array(cfg.DATASET.IMG_NORM_CFG.MEAN, dtype=np.float32)
    std = np.array(cfg.DATASET.IMG_NORM_CFG.STD, dtype=np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = np.ascontiguousarray(img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return torch.from_numpy(img.transpose(2, 0, 1))


def sample_views(rng):
    views = [(1.0, 1.0, 1.0, 0.0)]  # identity first
    for _ in range(K_JITTER):
        b = 1.0 + rng.uniform(-0.3, 0.3)
        c = 1.0 + rng.uniform(-0.3, 0.3)
        s = 1.0 + rng.uniform(-0.3, 0.3)
        hue = rng.uniform(-0.05, 0.05)
        views.append((b, c, s, hue))
    return views


def auc_pos_high(scores, is_pos):
    """AUC that a random positive (is_pos=True) has a HIGHER score than a random
    negative. Rank formula with tie-averaging (no sklearn dependency)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_pos = np.asarray(is_pos, dtype=bool)
    finite = np.isfinite(scores)
    scores = scores[finite]; is_pos = is_pos[finite]
    n_pos = int(is_pos.sum()); n_neg = int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return (ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    args = get_parser()
    cfg = cfg_from_file(args.cfg)
    c_num = cfg.DATASET.NUM_CLASSES

    _, val_filenames, _, _, val_images, val_labels, _, _ = read_file_list(cfg)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).float()
    text_norm = text_weight / text_weight.norm(dim=1, keepdim=True)
    model = build_model(cfg)

    # hook to capture z_global from model.vit forward (no model.py edit)
    captured = {}

    def hook(m, i, o):
        captured['vit'] = o
    model.vit.register_forward_hook(hook)

    n = len(val_images) if args.limit is None else min(args.limit, len(val_images))

    PRESENT = np.zeros((n, c_num), dtype=bool)
    P95_orig = np.full((n, c_num), np.nan)
    P95_m = np.full((n, c_num), np.nan); P95_s = np.full((n, c_num), np.nan); P95_cov = np.full((n, c_num), np.nan)
    G_orig = np.full((n, c_num), np.nan)
    G_m = np.full((n, c_num), np.nan); G_s = np.full((n, c_num), np.nan); G_cov = np.full((n, c_num), np.nan)

    sanity_max_diff = 0.0
    t0 = time.time()
    with torch.no_grad():
        for idx in range(n):
            with open(val_images[idx], 'rb') as f:
                buf = f.read()
            label = load_label(val_labels[idx], cfg)
            ori_shape = label.shape
            label_t = torch.from_numpy(label).to(device)
            valid = (label_t != 255)
            present = torch.unique(label_t[valid])
            present = present[(present >= 0) & (present < c_num)]
            PRESENT[idx, present.detach().cpu().numpy()] = True

            rng = np.random.RandomState(args.seed + idx)
            views = sample_views(rng)

            p95_views = np.zeros((len(views), c_num))
            g_views = np.zeros((len(views), c_num))
            for kv, jit in enumerate(views):
                img = preprocess_jitter(cfg, buf, jit).unsqueeze(0)
                if kv == 0 and idx < 3:
                    ref = val_preprocess(cfg, buf).unsqueeze(0)
                    sanity_max_diff = max(sanity_max_diff, float((img - ref).abs().max()))

                output = model(img, [], text_weight, cls_name_token, training=False)
                z_global = captured['vit'][2]  # [1,512]
                zg = z_global / (z_global.norm(dim=1, keepdim=True) + EPS)
                g_c = (zg @ text_norm.t())[0]  # [C] cosine

                N, C, H, W = output.shape
                _o = F.softmax(output * 10, dim=1)
                mcc = _o.view(N, C, -1).max(dim=-1)[0]
                sel = (mcc < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
                output[sel] = -100
                output = F.interpolate(output, img.shape[2:], None, 'bilinear', False).reshape(1, c_num, img.shape[2], img.shape[3])
                output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])
                probs = F.softmax(output, dim=1)[0]
                p95 = torch.quantile(probs.reshape(c_num, -1), 0.95, dim=1)

                p95_views[kv] = p95.detach().cpu().numpy()
                g_views[kv] = g_c.detach().cpu().numpy()

            P95_orig[idx] = p95_views[0]
            P95_m[idx] = p95_views.mean(0); P95_s[idx] = p95_views.std(0)
            P95_cov[idx] = P95_s[idx] / (P95_m[idx] + EPS)
            G_orig[idx] = g_views[0]
            G_m[idx] = g_views.mean(0); G_s[idx] = g_views.std(0)
            G_cov[idx] = np.abs(G_s[idx]) / (np.abs(G_m[idx]) + EPS)

            if (idx + 1) % 100 == 0 or idx + 1 == n:
                print(f'[photo] {idx + 1}/{n}, {time.time() - t0:.1f}s, sanity_max_diff={sanity_max_diff:.2e}', flush=True)

    pres = PRESENT.reshape(-1)
    absn = ~pres

    def ms(arr):
        a = arr.reshape(-1)
        return float(np.nanmean(a[pres])), float(np.nanstd(a[pres])), float(np.nanmean(a[absn])), float(np.nanstd(a[absn]))

    p95_s_stats = ms(P95_s); p95_cov_stats = ms(P95_cov)
    g_s_stats = ms(G_s); g_cov_stats = ms(G_cov)

    # (b) AUC: absent = positive; instability (high s / high cov) should score absent higher
    auc_p95_cov = auc_pos_high(P95_cov.reshape(-1), absn)
    auc_p95_s = auc_pos_high(P95_s.reshape(-1), absn)
    auc_g_cov = auc_pos_high(G_cov.reshape(-1), absn)
    auc_g_s = auc_pos_high(G_s.reshape(-1), absn)

    # (c) complementarity (using p95 peak-height on the ORIGINAL view)
    p95o = P95_orig; cov = P95_cov
    peak_keep = p95o > PEAK_THR
    recall_miss = PRESENT & (~peak_keep)      # present but peak misses
    prec_fail = (~PRESENT) & peak_keep        # absent but peak keeps
    comp = {}
    for tau in TAUS:
        low_cov = cov < tau
        high_cov = cov >= tau
        rm = recall_miss
        pf = prec_fail
        frac_miss_lowcov = float((rm & low_cov).sum() / max(rm.sum(), 1))
        frac_fail_highcov = float((pf & high_cov).sum() / max(pf.sum(), 1))
        comp[tau] = {'frac_recallmiss_lowcov': frac_miss_lowcov,
                     'frac_precfail_highcov': frac_fail_highcov}
    n_recall_miss = int(recall_miss.sum()); n_prec_fail = int(prec_fail.sum())

    # (d) combined rule present := (p95>0.3) OR (cov<tau); per-image precision/recall
    def pr_of(est):
        inter = (est & PRESENT).sum(1).astype(np.float64)
        en = est.sum(1).astype(np.float64); gn = PRESENT.sum(1).astype(np.float64)
        prec = float((inter[en > 0] / en[en > 0]).mean())
        rec = float((inter[gn > 0] / gn[gn > 0]).mean())
        return prec, rec
    peak_alone_pr = pr_of(peak_keep)
    combined = {}
    for tau in TAUS:
        est = peak_keep | (cov < tau)
        combined[tau] = pr_of(est)

    stats = {
        'n_images': n, 'n_views': 1 + K_JITTER, 'sanity_max_diff_view0_vs_valpreprocess': sanity_max_diff,
        'a_p95_s_present_mean': p95_s_stats[0], 'a_p95_s_present_std': p95_s_stats[1],
        'a_p95_s_absent_mean': p95_s_stats[2], 'a_p95_s_absent_std': p95_s_stats[3],
        'a_p95_cov_present_mean': p95_cov_stats[0], 'a_p95_cov_present_std': p95_cov_stats[1],
        'a_p95_cov_absent_mean': p95_cov_stats[2], 'a_p95_cov_absent_std': p95_cov_stats[3],
        'a_zglobal_s_present_mean': g_s_stats[0], 'a_zglobal_s_absent_mean': g_s_stats[2],
        'a_zglobal_cov_present_mean': g_cov_stats[0], 'a_zglobal_cov_absent_mean': g_cov_stats[2],
        'b_auc_p95_cov': auc_p95_cov, 'b_auc_p95_s': auc_p95_s,
        'b_auc_zglobal_cov': auc_g_cov, 'b_auc_zglobal_s': auc_g_s,
        'c_n_recall_miss': n_recall_miss, 'c_n_prec_fail': n_prec_fail, 'c_by_tau': comp,
        'd_peak_alone_precision': peak_alone_pr[0], 'd_peak_alone_recall': peak_alone_pr[1],
        'd_combined_by_tau': {str(t): {'precision': combined[t][0], 'recall': combined[t][1]} for t in TAUS},
        'taus': TAUS, 'peak_thr': PEAK_THR,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'photometric_consistency_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # figure: cov histogram present vs absent
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = np.clip(P95_cov[PRESENT], 0, 3)
    ac = np.clip(P95_cov[~PRESENT], 0, 3)
    bins = np.linspace(0, 3, 60)
    ax.hist(pc, bins=bins, alpha=0.6, density=True, color='#1f77b4', label=f'present (n={len(pc)})')
    ax.hist(ac, bins=bins, alpha=0.6, density=True, color='#d62728', label=f'absent (n={len(ac)})')
    ax.set_xlabel('cov_c = std_k(p95) / (mean_k(p95)+eps), clipped to [0,3]')
    ax.set_ylabel('density')
    ax.set_title('Cross-view instability (cov of p95): present vs absent classes')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, 'photometric_consistency_cov.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    comp_lines = '\n'.join(
        f'| {t} | {comp[t]["frac_recallmiss_lowcov"]:.3f} | {comp[t]["frac_precfail_highcov"]:.3f} |'
        for t in TAUS)
    comb_lines = '\n'.join(
        f'| tau={t} | {combined[t][0]:.4f} | {combined[t][1]:.4f} |'
        for t in TAUS)

    supported = (auc_p95_cov > 0.6) or (p95_cov_stats[2] > 1.5 * p95_cov_stats[0])
    verdict = 'SUPPORTED' if (auc_p95_cov > 0.65) else ('WEAK' if supported else 'REFUTED')

    md = f"""# Method-C Premise: Photometric Cross-View Consistency as a Presence Signal

Test-time only, no training, GT used only to SCORE the signal (never inside a rule).
{n} VOC val images, {1 + K_JITTER} views each (1 original + {K_JITTER} mild photometric
jitters on raw RGB before IMG_NORM: brightness/contrast/saturation +-0.3, hue +-0.05,
no geometric transform). Same official checkpoint, exact tools/test.py pipeline.
Sanity: max abs diff of view-0 preprocessing vs val_preprocess = {sanity_max_diff:.2e}
(so the original view reproduces the baseline prediction).

Signals per (image,class): p95_c (peak of dense prob over space) and
g_c=cos(z_global,text_c). Cross-view: m=mean_k, s=std_k, cov=s/(m+eps).

## (a) Instability of PRESENT vs ABSENT classes

| signal | stat | PRESENT mean(std) | ABSENT mean(std) |
|---|---|---|---|
| p95 | s (abs std) | {p95_s_stats[0]:.4f} ({p95_s_stats[1]:.4f}) | {p95_s_stats[2]:.4f} ({p95_s_stats[3]:.4f}) |
| p95 | cov | {p95_cov_stats[0]:.4f} ({p95_cov_stats[1]:.4f}) | {p95_cov_stats[2]:.4f} ({p95_cov_stats[3]:.4f}) |
| z_global | s | {g_s_stats[0]:.4f} | {g_s_stats[2]:.4f} |
| z_global | cov | {g_cov_stats[0]:.4f} | {g_cov_stats[2]:.4f} |

(premise holds if ABSENT instability > PRESENT, especially in scale-invariant cov.)

## (b) AUC of instability as an absent-vs-present discriminator (all image,class pairs)

| discriminator | AUC (absent=positive, higher instability => absent) |
|---|---|
| cov(p95) | {auc_p95_cov:.4f} |
| s(p95) | {auc_p95_s:.4f} |
| cov(z_global) | {auc_g_cov:.4f} |
| s(z_global) | {auc_g_s:.4f} |

(0.5 = no separation; >0.5 = absent classes more unstable.)

## (c) Complementarity with peak-height (p95>{PEAK_THR}) -- the key value

Recall failures (present but peak MISSES): n={n_recall_miss}.
Precision failures (absent but peak KEEPS): n={n_prec_fail}.

| tau | frac of recall-misses with LOW cov (<tau) [consistency would KEEP] | frac of precision-fails with HIGH cov (>=tau) [consistency would FLAG] |
|---|---|---|
{comp_lines}

## (d) Combined rule: present := (p95>{PEAK_THR}) OR (cov<tau)

Peak-alone: precision {peak_alone_pr[0]:.4f}, recall {peak_alone_pr[1]:.4f}.

| rule | precision | recall |
|---|---|---|
{comb_lines}

![cov histogram](photometric_consistency_cov.png)

## Honest read

Premise **{verdict}**: absent-class cov(p95) mean = {p95_cov_stats[2]:.4f} vs present
{p95_cov_stats[0]:.4f}; the scale-invariant AUC of cov(p95) separating absent-from-present
is {auc_p95_cov:.4f} (0.5 = useless). z_global cov AUC = {auc_g_cov:.4f}. On
complementarity (what matters): of the {n_recall_miss} present classes peak-height
misses, the low-cov fraction (tau=0.2) is {comp[0.2]['frac_recallmiss_lowcov']:.3f} (these
consistency could rescue); of the {n_prec_fail} absent classes peak-height wrongly keeps,
the high-cov fraction (tau=0.2) is {comp[0.2]['frac_precfail_highcov']:.3f} (these
consistency could flag). The combined OR rule vs peak-alone
({peak_alone_pr[0]:.3f}/{peak_alone_pr[1]:.3f}) {'RAISES recall without wrecking precision' if any(combined[t][1] > peak_alone_pr[1] + 0.005 and combined[t][0] > peak_alone_pr[0] - 0.03 for t in TAUS) else 'does NOT beat peak-alone (precision collapses because confidently-absent classes also have low cov)'}.
Judge these together: cross-view photometric consistency is
{'a promising complementary signal' if verdict == 'SUPPORTED' else ('a weak, marginal signal' if verdict == 'WEAK' else 'not a usable presence signal here')}.
"""
    with open(os.path.join(args.out_dir, 'photometric_consistency_premise.md'), 'w', encoding='utf-8') as f:
        f.write(md)

    print('=== RESULTS ===')
    print(json.dumps(stats, indent=2))
    print('report:', os.path.join(args.out_dir, 'photometric_consistency_premise.md'))
    print('figure:', fig_path)


if __name__ == '__main__':
    main()
