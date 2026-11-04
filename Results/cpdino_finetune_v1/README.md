# cpdino (ViT-L) fine-tuning on porcine brightfield, split `porcine_bf_v1`

Same procedure as cpdino-vitb (`experiments/cpdino_vitb_finetune_v1/`), with `--base-model cpdino`.
Scripts and runbook: `experiments/cpdino_finetune_v1/`. Run on the Cairo cluster (Tesla V100 32 GB,
torch 2.14.1+cu126, cellpose 4.2.1.1) on 2026-11-04.

## Setup

- Split `porcine_bf_v1` (fingerprint `22d6fd1c60e2`): 18 train / 4 val / 6 test whole images.
- Cellpose `train_seg` defaults: AdamW, lr 1e-5, weight decay 0.1, 100 epochs, batch 1, 384 x 384 crops,
  `min_train_masks 0`. Peak GPU memory 9.51 GB; 14-21 s per epoch (23-35 min per run; the slower run shared the GPU).
- Settings compared on val: raw (`none`) vs IMES CLAHE (`imes_clahe`).
- **Seeds: 1 per setting (seed 0). The protocol asks for 3**, so there is no SD and no paired comparison. Time on the
  shared GPU was limited. Seeds 1 and 2 were not run.
- Decision rule (fixed before the results): lower |fat % error| on the differentiated val images wins; a difference
  < 0.5 pp is a tie and goes to CLAHE.

## Validation (4 images: 3 differentiated, 1 control; used for the decision)

| Differentiated val | \|fat % error\| (pp) | Dice | F1@0.5 | F1@0.7 | count err |
|---|---|---|---|---|---|
| zero-shot cpdino | 4.51 | 0.758 | 0.536 | 0.407 | 1.846 |
| fine-tuned raw (s0) | 1.94 | 0.880 | 0.822 | 0.671 | -0.070 |
| fine-tuned CLAHE (s0) | 1.95 | 0.880 | 0.816 | 0.662 | -0.068 |

Control val image, false objects: zero-shot 88, raw 1, CLAHE 4.
Rule: 1.94 vs 1.95 pp, difference 0.01 pp < 0.5 pp, tie, **CLAHE chosen** (`cpdino-ft-clahe-v1`).

## Test (6 images: 5 differentiated, 1 control; used once, after the choice)

| Differentiated test | \|fat % error\| (pp) | signed fat % error | Dice | F1@0.5 | F1@0.7 | count err |
|---|---|---|---|---|---|---|
| zero-shot cpdino | 3.72 | -0.53 | 0.737 | 0.562 | 0.408 | 0.387 |
| fine-tuned CLAHE (s0) | 2.13 | 1.09 | 0.853 | 0.793 | 0.613 | -0.046 |

Control test image, false objects: zero-shot 59, fine-tuned 3.

For reference, cpdino-vitb on the same test images (3 seeds, CLAHE): \|fat % error\| 5.80 -> 2.79 +/- 0.24 pp,
Dice 0.68 -> 0.84, F1@0.7 0.38 -> 0.60. cpdino looks slightly better here, but this is one seed against a
3-seed mean, so no difference is claimed.

## Notes

- The test set was used once (zero-shot cpdino and CLAHE seed 0 only).
- Raw and CLAHE are indistinguishable on val with one seed; raw is marginally ahead on a few secondary metrics
  (control false objects 1 vs 4). The choice follows the pre-set rule.
- `cellpose-env` installs a CUDA 13 torch by default, which does not run on the V100 / driver 570 (CUDA 12.8).
  `torch==2.14.1+cu126` and `torchvision==0.29.1+cu126` were installed into the venv by hand, with `UV_NO_SYNC=1`.
  A Linux cu126 pin in `cellpose-env` is still to be discussed with the team.
- Files: `val_*.csv` and `test_*.csv` (`summary`, `per_image`, `seeds_summary`), and `train_config_*.json`.
  Weights, masks and images are not committed.
