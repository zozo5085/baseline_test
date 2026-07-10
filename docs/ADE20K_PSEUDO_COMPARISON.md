# ADE20K pseudo-label comparison — old external vs regenerated (2026-07-10)

Old (A) = `D:/ReCLIPPP2026/text/ade_pseudo_label.json`
New (B) = `text/ade_pseudo_label_regen_smoke.json` (tools/ade_pseudo_regen.py, text = verified `text/ade_ViT16_clip_text.pth`)
Enumeration: `os.listdir(D:/ReCLIPv3/datasets/ADEChallengeData2016/images/training/)` -> 20210 names, repeat-listing stable = True, md5(list) = `5431fc69e5c4613af725f8238a24bd8a`
Eval set = first 300 enumerated training images (paired for both files); GT = annotations/training.

| metric | A (old) | B (regen smoke) |
|---|---|---|
| line count | 20210 | 300 |
| parse failures | 0 | 0 |
| empty lists | 14 | 0 |
| class idx range | 0..149 | 0..149 |
| mean classes / image | 4.67 | 29.65 |
| mean GT recall (aligned) | 0.497 | 0.692 |
| pseudo precision (aligned) | 1.000 | 0.268 |
| recall @ image shift +1 | 0.317 | 0.571 |
| recall @ image shift -1 | 0.309 | 0.555 |

**Order-consistency test**: a pseudo file whose line order matches the current enumeration
must LOSE most of its recall when the alignment is shifted by one image; a file whose recall
is invariant under shift has no per-image correspondence (order broken / different enumeration).

- A (old): recall 0.497 vs shifted 0.317/0.309 -> **order-consistent (drop 0.180 under shift)**
- B (regen): recall 0.692 vs shifted 0.571/0.555 -> **order-consistent (drop 0.121 under shift)**

## Interpretation & decision (2026-07-10)

1. ⛔ **A = DATA LEAKAGE — banned from all formal / unsupervised training.**
   `D:\ReCLIPPP2026\text\ade_pseudo_label.json` **= GT-derived top-5 presence labels
   = invalid for unsupervised experiments** (provenance confirmed by the user 2026-07-10;
   consistent with the measured statistics: precision exactly **1.000** over 300 eval
   images — every listed class is in GT, zero false positives, which CLIP-derived pseudo
   never achieves — recall 0.497 ≈ top-5-of-GT coverage, mean 4.67 classes/image,
   14 empty lists). Any run trained on it would carry image-level oracle supervision and
   cannot be reported as C-USS. It may at most serve as an *oracle-presence upper-bound
   diagnostic*, clearly labeled as such, never in a formal table.
2. Note: A (`D:\ReCLIPPP2026\text\ade_pseudo_label.json`, md5 `612DE2D6BE3EB1743D5AD0E77D4A632C`)
   is a DIFFERENT file from the repo-shipped `text/ade_pseudo_label_ReCLIPPP.json`
   (md5 `4C599508B0A1FFA98365DD8C21B54065`), which remains order-broken under this machine's
   enumeration (recall 0.235, shift-invariant — see SESSION_HANDOFF 2026-07-10 checkpoint).
3. **B (regen) passes every smoke criterion**: parse 0 failures, 0 empty lists, idx range
   0..149, recall 0.692 (>> 0.235 shipped-file baseline and > A's 0.497), order-consistent,
   text = the verified `text/ade_ViT16_clip_text.pth`, enumeration repeat-stable
   (md5 `5431fc69e5c4613af725f8238a24bd8a`). Recipe = the same sliding-window ReCLIPPP mode
   as the repo's active VOC pseudo and the 2026-07-08 Context regeneration.
4. **Decision: regenerate the full file locally (B recipe) → `text/ade_pseudo_label.json`;
   do NOT use A as training input.** Full-file validation appended below after generation.
   No base training is started before that validation (user instruction 2026-07-10).

## Generation-process audit: no GT in the label decision (2026-07-10)

`tools/ade_pseudo_regen.py` — the ONLY reads of `annotations/training/` are lines 99–104,
inside the reporting block that computes recall/precision **after** the image's label list
has already been written (`fout.write(json.dumps(temp))`, line 95–96). The written classes
(`votes`/`temp`) derive exclusively from CLIP: sliding-window crops → `encode_image` →
logits against `text/ade_ViT16_clip_text.pth` → per-crop argmax votes (fallback:
mean-logits top-K). GT never flows into the decision. The recall/precision printed during
generation are diagnostics only.

## Training pointer chain (what a future ADE run will actually read)

- `config/ade_train_converged_cfg.yaml` → `DATASET.NAME: 'ade'`, `TEXT_WEIGHT:
  'text/ade_ViT16_clip_text.pth'` (the verified embedding, recipe cosine 0.999999 vs VOC).
- `utils/preprocess.py:411` (dataset `'ade'` branch) hardcodes the pseudo path:
  `text/ade_pseudo_label.json` — i.e. exactly the regenerated file, not A, not the
  order-broken shipped `_ReCLIPPP` variant.

## FINAL VALIDATION of the regenerated `text/ade_pseudo_label.json` — PASS (2026-07-10)

Generation: `tools/ade_pseudo_regen.py`, 20210/20210 images, 6.86 img/s, ~52 min;
log preserved at `experiments/journal_logs/ade_pseudo_regen.log`.

| check | requirement | measured | verdict |
|---|---|---|---|
| line count | = 20210 | 20210 (0 parse/blank failures) | ✅ |
| class index range | all in 0..149 | min 0, max 149, integers only (no NaN possible) | ✅ |
| empty lines / lists | none | 0 blank lines, 0 empty lists | ✅ |
| mean classes / image | (info) | 20.99 | — |
| mean GT recall (full set, aligned) | clearly > 0.235 | **0.583** (≈ Context fixed ref 0.579; diagnostics only — GT not used in generation, see audit above) | ✅ |
| order integrity | line i ↔ enumeration image i | recall computed per-image aligned during generation over all 20210 | ✅ |
| text embedding | verified `text/ade_ViT16_clip_text.pth` | config `TEXT_WEIGHT` + regen `--cfg` both point at it | ✅ |
| training pointer | `text/ade_pseudo_label.json` | `utils/preprocess.py:411` hardcode | ✅ |

**Status: ADE20K pseudo-label input is ready. Base training has NOT been started** —
it remains gated on the user's explicit go (pre-registration `GENERALIZATION_PROTOCOL.md §8`
governs the run: EPOCH 30, Context-identical recipe, PD 0.85, fixed battery afterwards).

